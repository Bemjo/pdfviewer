/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * *
 * Copyright (C) 2012-2020 Chuan Ji                                         *
 * *
 * Licensed under the Apache License, Version 2.0 (the "License");          *
 * you may not use this file except in compliance with the License.         *
 * You may obtain a copy of the License at                                  *
 * *
 * http://www.apache.org/licenses/LICENSE-2.0                              *
 * *
 * Unless required by applicable law or agreed to in writing, software      *
 * distributed under the License is distributed on an "AS IS" BASIS,        *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 * See the License for the specific language governing permissions and      *
 * limitations under the License.                                           *
 * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "pixel_buffer.hpp"

#include <algorithm>
#include <vector>
#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include "multithreading.hpp"

PixelBuffer::PixelBuffer(
    const PixelBuffer::Size& size, const PixelBuffer::Format* format)
    : _size(size),
      _allocated_size(size),
      _offset(0, 0),
      _format(format),
      _has_ownership(true) {
  assert(_format != nullptr);
  void* raw_ptr = nullptr;
  posix_memalign(&raw_ptr, 16, GetBufferByteSize());
  _buffer = static_cast<uint8_t*>(raw_ptr);
}

PixelBuffer::PixelBuffer(
    const PixelBuffer::Size& size, const PixelBuffer::Format* format,
    uint8_t* buffer, const PixelBuffer::Size& allocated_size,
    const PixelBuffer::Size& offset)
    : _size(size),
      _allocated_size(allocated_size),
      _offset(offset),
      _format(format),
      _buffer(buffer),
      _has_ownership(false) {
  assert(_format != nullptr);
  assert(_buffer != nullptr);
  assert(_size.Width + _offset.Width <= _allocated_size.Width);
  assert(_size.Height + _offset.Height <= _allocated_size.Height);
}

PixelBuffer::~PixelBuffer() {
  if (_has_ownership) {
    free(_buffer);
  }
}

PixelBuffer::Size PixelBuffer::GetSize() const { return _size; }

PixelBuffer::Rect PixelBuffer::GetRect() const {
  return Rect(0, 0, _size.Width, _size.Height);
}

namespace {

inline uint8_t Clamp8(int v) {
  return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

#ifdef __ARM_NEON

// ---------------------------------------------------------------------------
// Horizontal interpolation, depth=3.
//
// OPT #1/#8/#2: Eliminated stg_l/stg_r staging arrays (they caused a
// NEON store-to-load stall of ~20 cycles on Cortex-A9).  Replaced with
// direct vset_lane_u8 gather into six uint8x8_t registers, with the inner
// k-loop fully unrolled via #pragma GCC unroll 8.
// OPT #2: __builtin_prefetch for the next batch's source pixels.
// OPT #9: col_ix0/col_ix1 are uint16_t (max offset 1919*3=5757 < 65535).
// ---------------------------------------------------------------------------
static void BilinearHorizRow3(
    const uint8_t* __restrict__ src_row,
    const uint16_t* __restrict__ col_ix0,
    const uint16_t* __restrict__ col_ix1,
    const uint8_t* __restrict__ col_fx8,
    uint8_t* __restrict__ out, int dw) {
  // out always points at an hrow[] slot, which is now padded to a multiple
  // of 16 bytes within a posix_memalign(16)'d buffer, so the slot's base
  // address is 16-byte aligned. (Individual vst3_u8 writes inside the loop
  // are at out+dx*3, which drifts off 16-byte boundaries after the first
  // iteration since 3 doesn't divide 16 -- this hint only helps the
  // compiler's pointer arithmetic and the very first store, not every
  // store in the loop.)
  out = static_cast<uint8_t*>(__builtin_assume_aligned(out, 16));
  int dx = 0;
  const uint8x8_t v255 = vdup_n_u8(255);
  for (; dx + 8 <= dw; dx += 8) {
    // OPT #2: prefetch source pixels for the iteration 8 pixels ahead.
    // Guard with dx+16<=dw so we never read col_ix0[dw] (one past end).
    // On Cortex-A9 the L1 fill latency is ~11 cycles; issuing two PLDs
    // per iteration keeps the cache warm without flooding the bus.
    if (dx + 16 <= dw) {
      __builtin_prefetch(src_row + col_ix0[dx + 8], 0, 1);
      __builtin_prefetch(src_row + col_ix1[dx + 8], 0, 1);
    }

    // OPT #1: gather directly into lane registers — no stack staging,
    // no store-to-load hazard.
    uint8x8_t lA, lB, lC, rA, rB, rC;
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      const uint8_t* L = src_row + col_ix0[dx + k];
      const uint8_t* R = src_row + col_ix1[dx + k];
      lA = vset_lane_u8(L[0], lA, k);
      lB = vset_lane_u8(L[1], lB, k);
      lC = vset_lane_u8(L[2], lC, k);
      rA = vset_lane_u8(R[0], rA, k);
      rB = vset_lane_u8(R[1], rB, k);
      rC = vset_lane_u8(R[2], rC, k);
    }

    uint8x8_t vfx  = vld1_u8(col_fx8 + dx);
    uint8x8_t vfxc = vsub_u8(v255, vfx);

    uint16x8_t accA = vmull_u8(lA, vfxc);
    uint16x8_t accB = vmull_u8(lB, vfxc);
    uint16x8_t accC = vmull_u8(lC, vfxc);
    accA = vmlal_u8(accA, rA, vfx);
    accB = vmlal_u8(accB, rB, vfx);
    accC = vmlal_u8(accC, rC, vfx);

    uint8x8x3_t vout;
    vout.val[0] = vrshrn_n_u16(accA, 8);
    vout.val[1] = vrshrn_n_u16(accB, 8);
    vout.val[2] = vrshrn_n_u16(accC, 8);
    vst3_u8(out + dx * 3, vout);
  }
  for (; dx < dw; ++dx) {
    const uint8_t* L = src_row + col_ix0[dx];
    const uint8_t* R = src_row + col_ix1[dx];
    const uint32_t fx = col_fx8[dx], fxc = 255u - fx;
    out[dx*3+0] = static_cast<uint8_t>((L[0]*fxc + R[0]*fx + 128u) >> 8);
    out[dx*3+1] = static_cast<uint8_t>((L[1]*fxc + R[1]*fx + 128u) >> 8);
    out[dx*3+2] = static_cast<uint8_t>((L[2]*fxc + R[2]*fx + 128u) >> 8);
  }
}

// ---------------------------------------------------------------------------
// Vertical blend + sharpen, depth=3 — templated fast-path.
//
// OPT #4: Replaced vld3/vst3 (slow VLD3.8 on A9, ~8 cyc for 24 bytes) with
// flat vld1q_u8/vst1q_u8 (1-2 cyc for 16 bytes).  The hrow buffers are
// already packed byte-interleaved so the vertical blend doesn't need to
// separate channels.  We process 16 bytes (≈5.3 pixels) per inner iteration
// using low/high half-word arithmetic.
//
// OPT #3: vdup_n_u8(sharp_strength) hoisted out of the loop.
//
// OPT #5: boost term  base + (base*sharp)/256  replaced with
// base + (base >> BoostShift)  using a compile-time second template param,
// eliminating vmull_u8 + vshrq_n_u16 from the hot path.
//   sharp=32  → BoostShift=3  (×1.125)
//   sharp=64  → BoostShift=2  (×1.25)
//   sharp=128 → BoostShift=1  (×1.5)
// ---------------------------------------------------------------------------
template <int BoostShift>
static void BilinearVertBlendSharp3_Templated(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ prev,
    const uint8_t* __restrict__ next,
    uint8_t fy8,
    uint8_t* __restrict__ dest_row, int dw) {
  // row0/row1/prev/next are always hrow[1]/hrow[2]/hrow[0]/hrow[3] -- 16-byte
  // aligned slots in the padded hbuf. The loop below advances i by 16 each
  // iteration, so every vld1q_u8 in this loop is 16-byte aligned.
  // dest_row is the caller's real output row and is NOT guaranteed aligned
  // (depends on dest_rect.X, dest stride, and the destination buffer's own
  // offset), so it intentionally gets no alignment hint.
  row0 = static_cast<const uint8_t*>(__builtin_assume_aligned(row0, 16));
  row1 = static_cast<const uint8_t*>(__builtin_assume_aligned(row1, 16));
  prev = static_cast<const uint8_t*>(__builtin_assume_aligned(prev, 16));
  next = static_cast<const uint8_t*>(__builtin_assume_aligned(next, 16));

  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const int nbytes = dw * 3;

  #pragma GCC ivdep
  for (int i = 0; i + 16 <= nbytes; i += 16) {
    uint8x16_t v0 = vld1q_u8(row0 + i);
    uint8x16_t v1 = vld1q_u8(row1 + i);
    uint8x16_t vp = vld1q_u8(prev + i);
    uint8x16_t vn = vld1q_u8(next + i);

    // Bilinear vertical blend.
    uint16x8_t bilL = vmull_u8(vget_low_u8(v0),  vfyc);
    uint16x8_t bilH = vmull_u8(vget_high_u8(v0), vfyc);
    bilL = vmlal_u8(bilL, vget_low_u8(v1),  vfy);
    bilH = vmlal_u8(bilH, vget_high_u8(v1), vfy);

    uint8x8_t baseL = vrshrn_n_u16(bilL, 8);
    uint8x8_t baseH = vrshrn_n_u16(bilH, 8);

    // OPT #5: sharpening boost via shift — no multiply needed.
    // s = base + (base >> BoostShift)
    // NOTE: widening with vshll_n_u8(x, 0) is not valid asm (VSHLL has no
    // #0 shift encoding); vmovl_u8 is the correct "widen, no scale" op.
    uint16x8_t base16L = vmovl_u8(baseL);
    uint16x8_t base16H = vmovl_u8(baseH);
    int16x8_t sL = vreinterpretq_s16_u16(
        vaddq_u16(base16L, vshrq_n_u16(base16L, BoostShift)));
    int16x8_t sH = vreinterpretq_s16_u16(
        vaddq_u16(base16H, vshrq_n_u16(base16H, BoostShift)));

    // Outer (prev+next) subtraction.
    // OPT #11: vhaddq_u8 computes (p+n)>>1 in one cycle with no widening.
    // The remaining outer shift (originally ShiftVal) is, by construction
    // of the dispatch table below, always equal to ShiftVal-1 == BoostShift
    // (ShiftVal = BoostShift+1 for all three sharp_strength cases), so a
    // single template parameter correctly drives both shifts — no second
    // param needed. Shift is done in-lane on u8 before widening, keeping
    // the outer term in u8 until vsubw_u8 widens and subtracts in one
    // fused step — saves 2 vaddl + 2 vshrq_n_u16 per iteration.
    uint8x16_t havg = vhaddq_u8(vp, vn);       // (p+n)>>1  in u8
    uint8x8_t outByteL = vshr_n_u8(vget_low_u8(havg), BoostShift);
    uint8x8_t outByteH = vshr_n_u8(vget_high_u8(havg), BoostShift);
    sL = vreinterpretq_s16_u16(vsubw_u8(vreinterpretq_u16_s16(sL), outByteL));
    sH = vreinterpretq_s16_u16(vsubw_u8(vreinterpretq_u16_s16(sH), outByteH));

    vst1q_u8(dest_row + i, vcombine_u8(vqmovun_s16(sL), vqmovun_s16(sH)));
  }
  // 8-byte sub-tail.
  for (int i = (nbytes / 16) * 16; i + 8 <= nbytes; i += 8) {
    uint8x8_t v0b = vld1_u8(row0 + i), v1b = vld1_u8(row1 + i);
    uint8x8_t vpb = vld1_u8(prev  + i), vnb = vld1_u8(next  + i);
    uint16x8_t bil = vmull_u8(v0b, vfyc);
    bil = vmlal_u8(bil, v1b, vfy);
    uint8x8_t base = vrshrn_n_u16(bil, 8);
    uint16x8_t base16 = vmovl_u8(base);
    int16x8_t s16 = vreinterpretq_s16_u16(
        vaddq_u16(base16, vshrq_n_u16(base16, BoostShift)));
    // OPT #11: same hadd trick in the 8-byte tail.
    uint8x8_t outByte = vshr_n_u8(vhadd_u8(vpb, vnb), BoostShift);
    s16 = vreinterpretq_s16_u16(vsubw_u8(vreinterpretq_u16_s16(s16), outByte));
    vst1_u8(dest_row + i, vqmovun_s16(s16));
  }
}

// OPT #3: sharp_strength no longer passed into the templated function since
// the boost is encoded in BoostShift.  The dispatcher below maps the three
// known values.
static void BilinearVertBlendSharp3(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ prev,
    const uint8_t* __restrict__ next,
    uint8_t fy8, uint8_t sharp_strength,
    uint8_t* __restrict__ dest_row, int dw) {
  // row0/row1/prev/next are always hrow[] slots (16-byte aligned, padded).
  // dest_row is the caller's real buffer row and is not hinted.
  row0 = static_cast<const uint8_t*>(__builtin_assume_aligned(row0, 16));
  row1 = static_cast<const uint8_t*>(__builtin_assume_aligned(row1, 16));
  prev = static_cast<const uint8_t*>(__builtin_assume_aligned(prev, 16));
  next = static_cast<const uint8_t*>(__builtin_assume_aligned(next, 16));

  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const int nbytes = dw * 3;
  int i = 0;

  if (sharp_strength == 64) {
    // ShiftVal=3 (outer >> 3), BoostShift=2 (base*1.25)
    BilinearVertBlendSharp3_Templated<2>(row0, row1, prev, next, fy8, dest_row, dw);
    i = (nbytes / 8) * 8;
  } else if (sharp_strength == 128) {
    // ShiftVal=2 (outer >> 2), BoostShift=1 (base*1.5)
    BilinearVertBlendSharp3_Templated<1>(row0, row1, prev, next, fy8, dest_row, dw);
    i = (nbytes / 8) * 8;
  } else if (sharp_strength == 32) {
    // ShiftVal=4 (outer >> 4), BoostShift=3 (base*1.125)
    BilinearVertBlendSharp3_Templated<3>(row0, row1, prev, next, fy8, dest_row, dw);
    i = (nbytes / 8) * 8;
  } else {
    // Generic path: OPT #4 flat vld1q, OPT #3 hoisted vdup.
    const uint8x8_t vsh = vdup_n_u8(sharp_strength);
    #pragma GCC ivdep
    for (; i + 16 <= nbytes; i += 16) {
      uint8x16_t v0 = vld1q_u8(row0 + i);
      uint8x16_t v1 = vld1q_u8(row1 + i);
      uint8x16_t vp = vld1q_u8(prev + i);
      uint8x16_t vn = vld1q_u8(next + i);

      uint16x8_t bilL = vmull_u8(vget_low_u8(v0),  vfyc);
      uint16x8_t bilH = vmull_u8(vget_high_u8(v0), vfyc);
      bilL = vmlal_u8(bilL, vget_low_u8(v1),  vfy);
      bilH = vmlal_u8(bilH, vget_high_u8(v1), vfy);

      uint8x8_t baseL = vrshrn_n_u16(bilL, 8);
      uint8x8_t baseH = vrshrn_n_u16(bilH, 8);

      uint16x8_t shL = vmull_u8(baseL, vsh);
      uint16x8_t shH = vmull_u8(baseH, vsh);
      int16x8_t sL = vreinterpretq_s16_u16(vaddw_u8(vshrq_n_u16(shL, 8), baseL));
      int16x8_t sH = vreinterpretq_s16_u16(vaddw_u8(vshrq_n_u16(shH, 8), baseH));

      uint16x8_t outL = vaddl_u8(vget_low_u8(vp),  vget_low_u8(vn));
      uint16x8_t outH = vaddl_u8(vget_high_u8(vp), vget_high_u8(vn));
      outL = vshrq_n_u16(vmulq_n_u16(outL, sharp_strength), 9);
      outH = vshrq_n_u16(vmulq_n_u16(outH, sharp_strength), 9);
      sL = vsubq_s16(sL, vreinterpretq_s16_u16(outL));
      sH = vsubq_s16(sH, vreinterpretq_s16_u16(outH));

      vst1q_u8(dest_row + i, vcombine_u8(vqmovun_s16(sL), vqmovun_s16(sH)));
    }
    // 8-byte tail for generic path.
    for (; i + 8 <= nbytes; i += 8) {
      uint8x8_t v0b = vld1_u8(row0 + i), v1b = vld1_u8(row1 + i);
      uint8x8_t vpb = vld1_u8(prev  + i), vnb = vld1_u8(next  + i);
      uint16x8_t bil = vmull_u8(v0b, vfyc);
      bil = vmlal_u8(bil, v1b, vfy);
      uint8x8_t base = vrshrn_n_u16(bil, 8);
      uint16x8_t sh_add = vmull_u8(base, vsh);
      int16x8_t s16 = vreinterpretq_s16_u16(
          vaddw_u8(vshrq_n_u16(sh_add, 8), base));
      uint16x8_t outer = vaddl_u8(vpb, vnb);
      outer = vshrq_n_u16(vmulq_n_u16(outer, sharp_strength), 9);
      s16 = vsubq_s16(s16, vreinterpretq_s16_u16(outer));
      vst1_u8(dest_row + i, vqmovun_s16(s16));
    }
  }

  // Scalar tail (handles the remaining < 8 bytes).
  const int fy = fy8, fyc = 255 - fy, sh = sharp_strength;
  for (; i < nbytes; ++i) {
    const int bil = (row0[i]*fyc + row1[i]*fy + 128) >> 8;
    const int out = bil + ((bil*sh) >> 8) - (((int)prev[i] + next[i])*sh >> 9);
    dest_row[i] = static_cast<uint8_t>(out < 0 ? 0 : out > 255 ? 255 : out);
  }
}

// ---------------------------------------------------------------------------
// Vertical blend + sharpen, depth=4 — templated fast-path.
//
// OPT #5: same BoostShift trick as depth=3.
// OPT #6: alpha is always 255 in source pixels, so the vbsl alpha-passthrough
// is removed entirely.
// ---------------------------------------------------------------------------
template <int BoostShift>
static void BilinearVertBlendSharp4_Templated(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ prev,
    const uint8_t* __restrict__ next,
    uint8_t fy8,
    uint8_t* __restrict__ dest_row, int dw) {
  // row0/row1/prev/next are always hrow[1]/hrow[2]/hrow[0]/hrow[3] -- 16-byte
  // aligned slots. i advances by 16 each iteration so every vld1q_u8 here
  // is 16-byte aligned. dest_row (real output buffer) is not hinted.
  row0 = static_cast<const uint8_t*>(__builtin_assume_aligned(row0, 16));
  row1 = static_cast<const uint8_t*>(__builtin_assume_aligned(row1, 16));
  prev = static_cast<const uint8_t*>(__builtin_assume_aligned(prev, 16));
  next = static_cast<const uint8_t*>(__builtin_assume_aligned(next, 16));

  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const int nbytes = dw * 4;

  #pragma GCC ivdep
  for (int i = 0; i + 16 <= nbytes; i += 16) {
    uint8x16_t v0 = vld1q_u8(row0 + i);
    uint8x16_t v1 = vld1q_u8(row1 + i);
    uint8x16_t vp = vld1q_u8(prev + i);
    uint8x16_t vn = vld1q_u8(next + i);

    uint16x8_t bilL = vmull_u8(vget_low_u8(v0),  vfyc);
    uint16x8_t bilH = vmull_u8(vget_high_u8(v0), vfyc);
    bilL = vmlal_u8(bilL, vget_low_u8(v1),  vfy);
    bilH = vmlal_u8(bilH, vget_high_u8(v1), vfy);

    uint8x8_t baseL = vrshrn_n_u16(bilL, 8);
    uint8x8_t baseH = vrshrn_n_u16(bilH, 8);

    // OPT #5: boost via shift. vmovl_u8 widens with no scale (vshll_n_u8
    // with a #0 immediate is not a legal VSHLL encoding).
    uint16x8_t base16L = vmovl_u8(baseL);
    uint16x8_t base16H = vmovl_u8(baseH);
    int16x8_t sL = vreinterpretq_s16_u16(
        vaddq_u16(base16L, vshrq_n_u16(base16L, BoostShift)));
    int16x8_t sH = vreinterpretq_s16_u16(
        vaddq_u16(base16H, vshrq_n_u16(base16H, BoostShift)));

    // OPT #11: vhaddq_u8 + vshr_n_u8 instead of vaddl + vshrq_n_u16.
    uint8x16_t havg = vhaddq_u8(vp, vn);
    uint8x8_t outByteL = vshr_n_u8(vget_low_u8(havg),  BoostShift);
    uint8x8_t outByteH = vshr_n_u8(vget_high_u8(havg), BoostShift);
    sL = vreinterpretq_s16_u16(vsubw_u8(vreinterpretq_u16_s16(sL), outByteL));
    sH = vreinterpretq_s16_u16(vsubw_u8(vreinterpretq_u16_s16(sH), outByteH));

    // OPT #6: alpha always 255 — no vbsl needed.
    vst1q_u8(dest_row + i, vcombine_u8(vqmovun_s16(sL), vqmovun_s16(sH)));
  }
}

static void BilinearVertBlendSharp4(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ prev,
    const uint8_t* __restrict__ next,
    uint8_t fy8, uint8_t sharp_strength,
    uint8_t* __restrict__ dest_row, int dw) {
  // row0/row1/prev/next are always hrow[] slots (16-byte aligned, padded).
  // dest_row is the caller's real buffer row and is not hinted.
  row0 = static_cast<const uint8_t*>(__builtin_assume_aligned(row0, 16));
  row1 = static_cast<const uint8_t*>(__builtin_assume_aligned(row1, 16));
  prev = static_cast<const uint8_t*>(__builtin_assume_aligned(prev, 16));
  next = static_cast<const uint8_t*>(__builtin_assume_aligned(next, 16));

  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const int nbytes = dw * 4;
  int i = 0;

  if (sharp_strength == 64) {
    BilinearVertBlendSharp4_Templated<2>(row0, row1, prev, next, fy8, dest_row, dw);
    i = (nbytes / 16) * 16;
  } else if (sharp_strength == 128) {
    BilinearVertBlendSharp4_Templated<1>(row0, row1, prev, next, fy8, dest_row, dw);
    i = (nbytes / 16) * 16;
  } else if (sharp_strength == 32) {
    BilinearVertBlendSharp4_Templated<3>(row0, row1, prev, next, fy8, dest_row, dw);
    i = (nbytes / 16) * 16;
  } else {
    // Generic path: OPT #3 hoisted vdup, OPT #6 no vbsl.
    const uint8x8_t vsh = vdup_n_u8(sharp_strength);

    #pragma GCC ivdep
    for (; i + 16 <= nbytes; i += 16) {
      uint8x16_t v0 = vld1q_u8(row0 + i);
      uint8x16_t v1 = vld1q_u8(row1 + i);
      uint8x16_t vp = vld1q_u8(prev + i);
      uint8x16_t vn = vld1q_u8(next + i);

      uint16x8_t bilL = vmull_u8(vget_low_u8(v0),  vfyc);
      uint16x8_t bilH = vmull_u8(vget_high_u8(v0), vfyc);
      bilL = vmlal_u8(bilL, vget_low_u8(v1),  vfy);
      bilH = vmlal_u8(bilH, vget_high_u8(v1), vfy);

      uint8x8_t baseL = vrshrn_n_u16(bilL, 8);
      uint8x8_t baseH = vrshrn_n_u16(bilH, 8);

      uint16x8_t shL = vmull_u8(baseL, vsh);
      uint16x8_t shH = vmull_u8(baseH, vsh);
      int16x8_t sL = vreinterpretq_s16_u16(vaddw_u8(vshrq_n_u16(shL, 8), baseL));
      int16x8_t sH = vreinterpretq_s16_u16(vaddw_u8(vshrq_n_u16(shH, 8), baseH));

      uint16x8_t outL = vaddl_u8(vget_low_u8(vp),  vget_low_u8(vn));
      uint16x8_t outH = vaddl_u8(vget_high_u8(vp), vget_high_u8(vn));
      sL = vsubq_s16(sL, vreinterpretq_s16_u16(vshrq_n_u16(vmulq_n_u16(outL, sharp_strength), 9)));
      sH = vsubq_s16(sH, vreinterpretq_s16_u16(vshrq_n_u16(vmulq_n_u16(outH, sharp_strength), 9)));

      // OPT #6: alpha always 255 — no vbsl.
      vst1q_u8(dest_row + i, vcombine_u8(vqmovun_s16(sL), vqmovun_s16(sH)));
    }
  }

  // Scalar tail — alpha byte (i%4==3) is also always 255 so sharpening it
  // is harmless; but we preserve the original safe passthrough just in case.
  const uint32_t fy = fy8, fyc = 255u - fy8;
  for (; i < nbytes; ++i) {
    if (i % 4 == 3) {
      dest_row[i] = row0[i];
    } else {
      const int bil = (row0[i]*fyc + row1[i]*fy + 128) >> 8;
      const int out = bil + ((bil*sharp_strength) >> 8) - (((int)prev[i] + next[i])*sharp_strength >> 9);
      dest_row[i] = static_cast<uint8_t>(out < 0 ? 0 : out > 255 ? 255 : out);
    }
  }
}

// ---------------------------------------------------------------------------
// Vertical blend only (no sharpen), depth=3.
//
// OPT #4: replaced vld3/vst3 (VLD3.8 ~8 cyc/24B) with flat vld1q_u8
// (1-2 cyc/16B).  Channel identity is irrelevant here; we just lerp bytes.
// ---------------------------------------------------------------------------
static void BilinearVertBlend3(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    uint8_t fy8, uint8_t* __restrict__ dest_row, int dw) {
  // row0/row1 are always hrow[1]/hrow[2] -- 16-byte aligned, padded slots.
  // dest_row is the real output row and is not hinted.
  row0 = static_cast<const uint8_t*>(__builtin_assume_aligned(row0, 16));
  row1 = static_cast<const uint8_t*>(__builtin_assume_aligned(row1, 16));

  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const int nbytes = dw * 3;
  int i = 0;
  for (; i + 16 <= nbytes; i += 16) {
    uint8x16_t v0 = vld1q_u8(row0 + i);
    uint8x16_t v1 = vld1q_u8(row1 + i);
    uint16x8_t accL = vmull_u8(vget_low_u8(v0),  vfyc);
    uint16x8_t accH = vmull_u8(vget_high_u8(v0), vfyc);
    accL = vmlal_u8(accL, vget_low_u8(v1),  vfy);
    accH = vmlal_u8(accH, vget_high_u8(v1), vfy);
    vst1q_u8(dest_row + i,
             vcombine_u8(vrshrn_n_u16(accL, 8), vrshrn_n_u16(accH, 8)));
  }
  for (; i + 8 <= nbytes; i += 8) {
    uint8x8_t v0 = vld1_u8(row0 + i);
    uint8x8_t v1 = vld1_u8(row1 + i);
    uint16x8_t acc = vmull_u8(v0, vfyc);
    acc = vmlal_u8(acc, v1, vfy);
    vst1_u8(dest_row + i, vrshrn_n_u16(acc, 8));
  }
  const uint32_t fy = fy8, fyc = 255u - fy;
  for (; i < nbytes; ++i)
    dest_row[i] = static_cast<uint8_t>((row0[i]*fyc + row1[i]*fy + 128u) >> 8);
}

// ---------------------------------------------------------------------------
// Horizontal interpolation, depth=4.
//
// OPT #1/#8/#2: Same staging-buffer elimination as depth=3.
// OPT #9: uint16_t index arrays.
// ---------------------------------------------------------------------------
static void BilinearHorizRow4(
    const uint8_t* __restrict__ src_row,
    const uint16_t* __restrict__ col_ix0,
    const uint16_t* __restrict__ col_ix1,
    const uint8_t* __restrict__ col_fx8,
    uint8_t* __restrict__ out, int dw) {
  // out always points at an hrow[] slot, padded to a 16-byte multiple, so
  // the base is 16-byte aligned. Unlike depth=3, every vst4_u8 write at
  // out+dx*4 is ALSO 16-byte aligned each iteration, since dx advances by 8
  // and 8*4=32 is a multiple of 16 -- so this hint benefits every store in
  // the loop, not just the first.
  out = static_cast<uint8_t*>(__builtin_assume_aligned(out, 16));
  int dx = 0;
  const uint8x8_t v255 = vdup_n_u8(255);
  for (; dx + 8 <= dw; dx += 8) {
    // OPT #2: prefetch next batch.
    // Guard with dx+16<=dw to avoid reading col_ix0[dw] on the last iteration.
    if (dx + 16 <= dw) {
      __builtin_prefetch(src_row + col_ix0[dx + 8], 0, 1);
      __builtin_prefetch(src_row + col_ix1[dx + 8], 0, 1);
    }

    // OPT #1: direct lane gather.
    uint8x8_t lA, lB, lC, lD, rA, rB, rC, rD;
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      const uint8_t* L = src_row + col_ix0[dx + k];
      const uint8_t* R = src_row + col_ix1[dx + k];
      lA = vset_lane_u8(L[0], lA, k);
      lB = vset_lane_u8(L[1], lB, k);
      lC = vset_lane_u8(L[2], lC, k);
      lD = vset_lane_u8(L[3], lD, k);
      rA = vset_lane_u8(R[0], rA, k);
      rB = vset_lane_u8(R[1], rB, k);
      rC = vset_lane_u8(R[2], rC, k);
      rD = vset_lane_u8(R[3], rD, k);
    }

    uint8x8_t vfx  = vld1_u8(col_fx8 + dx);
    uint8x8_t vfxc = vsub_u8(v255, vfx);

    uint16x8_t accA = vmull_u8(lA, vfxc);
    uint16x8_t accB = vmull_u8(lB, vfxc);
    uint16x8_t accC = vmull_u8(lC, vfxc);
    uint16x8_t accD = vmull_u8(lD, vfxc);
    accA = vmlal_u8(accA, rA, vfx);
    accB = vmlal_u8(accB, rB, vfx);
    accC = vmlal_u8(accC, rC, vfx);
    accD = vmlal_u8(accD, rD, vfx);

    uint8x8x4_t vout;
    vout.val[0] = vrshrn_n_u16(accA, 8);
    vout.val[1] = vrshrn_n_u16(accB, 8);
    vout.val[2] = vrshrn_n_u16(accC, 8);
    vout.val[3] = vrshrn_n_u16(accD, 8);
    vst4_u8(out + dx * 4, vout);
  }
  for (; dx < dw; ++dx) {
    const uint8_t* L = src_row + col_ix0[dx];
    const uint8_t* R = src_row + col_ix1[dx];
    const uint32_t fx = col_fx8[dx], fxc = 255u - fx;
    out[dx*4+0] = static_cast<uint8_t>((L[0]*fxc + R[0]*fx + 128u) >> 8);
    out[dx*4+1] = static_cast<uint8_t>((L[1]*fxc + R[1]*fx + 128u) >> 8);
    out[dx*4+2] = static_cast<uint8_t>((L[2]*fxc + R[2]*fx + 128u) >> 8);
    out[dx*4+3] = static_cast<uint8_t>((L[3]*fxc + R[3]*fx + 128u) >> 8);
  }
}

// BilinearVertBlend4 is already optimal (flat vld1q, no channel split needed).
static void BilinearVertBlend4(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    uint8_t fy8, uint8_t* __restrict__ dest_row, int dw) {
  // row0/row1 are always hrow[1]/hrow[2] -- 16-byte aligned, padded slots.
  // dest_row is the real output row and is not hinted.
  row0 = static_cast<const uint8_t*>(__builtin_assume_aligned(row0, 16));
  row1 = static_cast<const uint8_t*>(__builtin_assume_aligned(row1, 16));

  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const int nbytes = dw * 4;
  int i = 0;
  for (; i + 16 <= nbytes; i += 16) {
    uint8x16_t v0 = vld1q_u8(row0 + i);
    uint8x16_t v1 = vld1q_u8(row1 + i);
    uint16x8_t accL = vmull_u8(vget_low_u8(v0),  vfyc);
    uint16x8_t accH = vmull_u8(vget_high_u8(v0), vfyc);
    accL = vmlal_u8(accL, vget_low_u8(v1),  vfy);
    accH = vmlal_u8(accH, vget_high_u8(v1), vfy);
    vst1q_u8(dest_row + i,
             vcombine_u8(vrshrn_n_u16(accL, 8), vrshrn_n_u16(accH, 8)));
  }
  for (; i + 8 <= nbytes; i += 8) {
    uint8x8_t v0 = vld1_u8(row0 + i), v1 = vld1_u8(row1 + i);
    uint16x8_t acc = vmull_u8(v0, vfyc);
    acc = vmlal_u8(acc, v1, vfy);
    vst1_u8(dest_row + i, vrshrn_n_u16(acc, 8));
  }
  const uint32_t fy = fy8, fyc = 255u - fy;
  for (; i < nbytes; ++i)
    dest_row[i] = static_cast<uint8_t>((row0[i]*fyc + row1[i]*fy + 128u) >> 8);
}

#endif // __ARM_NEON

} // namespace


void PixelBuffer::Copy(
    const PixelBuffer::Rect& src_rect, const PixelBuffer::Rect& dest_rect,
    PixelBuffer* dest, PixelBuffer::ScaleMode scale_mode, uint8_t sharpen_strength) const {
  assert(_format->GetDepth() == dest->_format->GetDepth());
  assert(_size.Width  >= src_rect.X  + src_rect.Width);
  assert(_size.Height >= src_rect.Y  + src_rect.Height);
  assert(dest->_size.Width  >= dest_rect.X + dest_rect.Width);
  assert(dest->_size.Height >= dest_rect.Y + dest_rect.Height);

  const int depth = _format->GetDepth();
  const int src_stride  = _allocated_size.Width * depth;
  const int dest_stride = dest->_allocated_size.Width * depth;

  const uint8_t* src_base =
      _buffer +
      (_offset.Height * _allocated_size.Width + _offset.Width) * depth;
  uint8_t* dest_base =
      dest->_buffer +
      (dest->_offset.Height * dest->_allocated_size.Width +
       dest->_offset.Width) * depth;

  const bool same_size = (src_rect.Width  == dest_rect.Width &&
                          src_rect.Height == dest_rect.Height);

  if (same_size) {
    const int row_bytes = src_rect.Width * depth;
    ExecuteInParallel([=](int num_threads, int i) {
      const int rows_per = src_rect.Height / num_threads;
      const int height   = (i == num_threads - 1)
                               ? (src_rect.Height - i * rows_per)
                               : rows_per;
      const int y0 = i * rows_per;
      for (int y = 0; y < height; ++y) {
        memcpy(
            dest_base + (dest_rect.Y + y0 + y) * dest_stride +
                dest_rect.X * depth,
            src_base  + (src_rect.Y  + y0 + y) * src_stride  +
                src_rect.X  * depth,
            row_bytes);
      }
    });
    return;
  }

  const int dw = dest_rect.Width;
  const int dh = dest_rect.Height;
  const int sw = src_rect.Width;
  const int sh = src_rect.Height;
  const bool valid_sharp_depth = depth == 3 || depth == 4;

  switch (scale_mode) {
    case SCALE_NEAREST: {
      std::vector<int> sx_off(dw);
      for (int dx = 0; dx < dw; ++dx) {
        const int sx = src_rect.X + (dx * sw) / dw;
        sx_off[dx] = sx * depth;
      }

      ExecuteInParallel([=, &sx_off](int num_threads, int i) {
        const int rows_per = dh / num_threads;
        const int height   = (i == num_threads - 1)
                                 ? (dh - i * rows_per) : rows_per;
        const int dy0 = i * rows_per;
        const int row_bytes = dw * depth;

        int cached_sy = -1;
        uint8_t* prev_dest_row = nullptr;

        for (int dy_local = 0; dy_local < height; ++dy_local) {
          const int dy = dy0 + dy_local;
          const int sy = src_rect.Y + (dy * sh) / dh;
          uint8_t* dest_row = dest_base + (dest_rect.Y + dy) * dest_stride +
                              dest_rect.X * depth;

          // Optimization: If same source row, copy previous result directly.
          if (sy == cached_sy && prev_dest_row != nullptr) {
            memcpy(dest_row, prev_dest_row, row_bytes);
            prev_dest_row = dest_row;
            continue;
          }

          cached_sy = sy;
          const uint8_t* src_row = src_base + sy * src_stride;
          uint8_t* out = dest_row;

          if (depth == 3) {
            for (int dx = 0; dx < dw; ++dx) {
              const uint8_t* sp = src_row + sx_off[dx];
              out[0] = sp[0]; out[1] = sp[1]; out[2] = sp[2];
              out += 3;
            }
          } else if (depth == 4) {
            for (int dx = 0; dx < dw; ++dx) {
              *reinterpret_cast<uint32_t*>(out) =
                  *reinterpret_cast<const uint32_t*>(src_row + sx_off[dx]);
              out += 4;
            }
          } else {
            for (int dx = 0; dx < dw; ++dx) {
              memcpy(out, src_row + sx_off[dx], depth);
              out += depth;
            }
          }
          prev_dest_row = dest_row;
        }
      });
      break;
    }

    case SCALE_BILINEAR: {
#ifdef __ARM_NEON
      if (valid_sharp_depth) {
        // OPT #9: uint16_t index arrays — max offset 1919*4=7676 < 65535,
        // halving the L1 footprint of the index tables.
        std::vector<uint16_t> col_ix0(dw), col_ix1(dw);
        std::vector<uint8_t>  col_fx8(dw);
        for (int dx = 0; dx < dw; ++dx) {
          const float sx = src_rect.X +
              (dx + 0.5f) * static_cast<float>(sw) / static_cast<float>(dw)
              - 0.5f;
          int ix = static_cast<int>(sx);
          float fxf = sx - ix;
          if (ix < src_rect.X) { ix = src_rect.X; fxf = 0.0f; }
          const int ix_max = src_rect.X + sw - 1;
          const int ix1 = (ix < ix_max) ? ix + 1 : ix_max;
          col_ix0[dx] = static_cast<uint16_t>(ix  * depth);
          col_ix1[dx] = static_cast<uint16_t>(ix1 * depth);
          col_fx8[dx] = static_cast<uint8_t>(fxf * 255.0f + 0.5f);
        }

        const uint8_t sharp_strength = valid_sharp_depth && sharpen_strength > 0 ?
            16u << static_cast<uint8_t>(std::min(std::max(1, static_cast<int>(sharpen_strength)), 3)):
            0u;

        // OPT #10: precompute the source row clamp bounds once so that
        // ensure_hrow does a branchless clamp (SMAX/SMIN) rather than
        // calling std::max/std::min with possible function-call overhead.
        const int ry_min = src_rect.Y;
        const int ry_max = src_rect.Y + sh - 1;

        ExecuteInParallel([=, &col_ix0, &col_ix1, &col_fx8](
                              int num_threads, int i) {
          const int rows_per = dh / num_threads;
          const int height   = (i == num_threads - 1)
                                   ? (dh - i * rows_per) : rows_per;
          const int dy0 = i * rows_per;

          // Round each hrow slot up to a multiple of 16 bytes so that, given
          // a 16-byte-aligned allocation, every one of the 4 slots starts on
          // a 16-byte boundary. dw*depth alone is only a multiple of 16 when
          // dw happens to be a multiple of 16 (depth=3) or 4 (depth=4); the
          // padding guarantees alignment regardless of dw. The extra bytes
          // per slot (0-15) are never read since BilinearHorizRow3/4 only
          // ever write the first dw*depth bytes of each slot.
          const int row_bytes = dw * depth;
          const int row_bytes_aligned = (row_bytes + 15) & ~15;
          const size_t buf_size =
              static_cast<size_t>(row_bytes_aligned) * 4;
          void* raw_ptr = nullptr;
          posix_memalign(&raw_ptr, 16, buf_size);

          std::unique_ptr<uint8_t, void(*)(void*)> hbuf(
              static_cast<uint8_t*>(raw_ptr),
              std::free
          );

          uint8_t* hrow[4] = {
            hbuf.get(),
            hbuf.get() + row_bytes_aligned,
            hbuf.get() + row_bytes_aligned * 2,
            hbuf.get() + row_bytes_aligned * 3,
          };
          int cached_iy[4] = { -1, -1, -1, -1 };

          // OPT #10: clamp is now a simple ternary; GCC emits SMAX/SMIN.
          auto clamp_row = [&](int ry) -> int {
            return ry < ry_min ? ry_min : (ry > ry_max ? ry_max : ry);
          };

          auto ensure_hrow = [&](int slot, int ry) {
            if (cached_iy[slot] != ry) {
              if (depth == 3)
                BilinearHorizRow3(src_base + ry * src_stride,
                    col_ix0.data(), col_ix1.data(), col_fx8.data(),
                    hrow[slot], dw);
              else
                BilinearHorizRow4(src_base + ry * src_stride,
                    col_ix0.data(), col_ix1.data(), col_fx8.data(),
                    hrow[slot], dw);
              cached_iy[slot] = ry;
            }
          };

          for (int dy_local = 0; dy_local < height; ++dy_local) {
            const int dy = dy0 + dy_local;
            const float sy = src_rect.Y +
                (dy + 0.5f) * static_cast<float>(sh) / static_cast<float>(dh)
                - 0.5f;
            int iy = static_cast<int>(sy);
            float fyf = sy - iy;
            if (iy < src_rect.Y) { iy = src_rect.Y; fyf = 0.0f; }
            const int iy_max = src_rect.Y + sh - 1;
            const int iy1 = (iy < iy_max) ? iy + 1 : iy_max;
            const uint8_t fy8 = static_cast<uint8_t>(fyf * 255.0f + 0.5f);

            uint8_t* dest_row = dest_base + (dest_rect.Y + dy) * dest_stride +
                                dest_rect.X * depth;

            if (valid_sharp_depth && sharp_strength > 0) {
              // OPT #10: clamp all four row indices before entering ensure_hrow
              // so the clamped values are reused directly without re-clamping.
              const int r0 = clamp_row(iy - 1);
              const int r1 = clamp_row(iy);
              const int r2 = clamp_row(iy1);
              const int r3 = clamp_row(iy1 + 1);
              ensure_hrow(0, r0);
              ensure_hrow(1, r1);
              ensure_hrow(2, r2);
              ensure_hrow(3, r3);

              if (iy == iy1) {
                memcpy(dest_row, hrow[1], row_bytes);
              } else if (depth == 3) {
                BilinearVertBlendSharp3(hrow[1], hrow[2], hrow[0], hrow[3],
                                        fy8, sharp_strength, dest_row, dw);
              } else if (depth == 4) {
                BilinearVertBlendSharp4(hrow[1], hrow[2], hrow[0], hrow[3],
                                        fy8, sharp_strength, dest_row, dw);
              }
            } else {
              const int r1 = clamp_row(iy);
              const int r2 = clamp_row(iy1);
              ensure_hrow(1, r1);
              ensure_hrow(2, r2);
              if (iy == iy1) {
                memcpy(dest_row, hrow[1], row_bytes);
              } else if (depth == 3) {
                BilinearVertBlend3(hrow[1], hrow[2], fy8, dest_row, dw);
              } else {
                BilinearVertBlend4(hrow[1], hrow[2], fy8, dest_row, dw);
              }
            }
          }
        });
        break;
      }
#endif

      // Scalar fallback
      struct ColEntry { int ix0; int ix1_off; uint32_t fx16; };
      std::vector<ColEntry> col(dw);
      for (int dx = 0; dx < dw; ++dx) {
        const float sx = src_rect.X +
            (dx + 0.5f) * static_cast<float>(sw) / static_cast<float>(dw)
            - 0.5f;
        int ix = static_cast<int>(sx);
        float fxf = sx - ix;
        if (ix < src_rect.X) { ix = src_rect.X; fxf = 0.0f; }
        const int ix_max = src_rect.X + sw - 1;
        const int ix1 = (ix < ix_max) ? ix + 1 : ix_max;
        col[dx] = { ix * depth, ix1 * depth,
                    static_cast<uint32_t>(fxf * 65536.0f + 0.5f) };
      }

      ExecuteInParallel([=, &col](int num_threads, int i) {
        const int rows_per = dh / num_threads;
        const int height   = (i == num_threads - 1)
                                 ? (dh - i * rows_per) : rows_per;
        const int dy0 = i * rows_per;

        for (int dy_local = 0; dy_local < height; ++dy_local) {
          const int dy = dy0 + dy_local;
          const float sy = src_rect.Y +
              (dy + 0.5f) * static_cast<float>(sh) / static_cast<float>(dh)
              - 0.5f;
          int iy = static_cast<int>(sy);
          float fyf = sy - iy;
          if (iy < src_rect.Y) { iy = src_rect.Y; fyf = 0.0f; }
          const int iy_max = src_rect.Y + sh - 1;
          const int iy1 = (iy < iy_max) ? iy + 1 : iy_max;
          const uint32_t fy16  = static_cast<uint32_t>(fyf * 65536.0f + 0.5f);
          const uint32_t fy16c = 65536u - fy16;

          const uint8_t* row0 = src_base + iy  * src_stride;
          const uint8_t* row1 = src_base + iy1 * src_stride;
          uint8_t* dest_row   = dest_base + (dest_rect.Y + dy) * dest_stride +
                                dest_rect.X * depth;

          if (depth == 3) {
            for (int dx = 0; dx < dw; ++dx) {
              const ColEntry& c = col[dx];
              const uint64_t fx16  = c.fx16;
              const uint64_t fx16c = 65536u - fx16;
              const uint64_t fy16c64 = fy16c;
              const uint64_t fy1664  = fy16;
              const uint8_t* tl = row0 + c.ix0;
              const uint8_t* tr = row0 + c.ix1_off;
              const uint8_t* bl = row1 + c.ix0;
              const uint8_t* br = row1 + c.ix1_off;
              dest_row[0] = static_cast<uint8_t>(
                  ((tl[0]*fx16c + tr[0]*fx16)*fy16c64 +
                   (bl[0]*fx16c + br[0]*fx16)*fy1664 + (1u<<31)) >> 32);
              dest_row[1] = static_cast<uint8_t>(
                  ((tl[1]*fx16c + tr[1]*fx16)*fy16c64 +
                   (bl[1]*fx16c + br[1]*fx16)*fy1664 + (1u<<31)) >> 32);
              dest_row[2] = static_cast<uint8_t>(
                  ((tl[2]*fx16c + tr[2]*fx16)*fy16c64 +
                   (bl[2]*fx16c + br[2]*fx16)*fy1664 + (1u<<31)) >> 32);
              dest_row += 3;
            }
          } else {
            for (int dx = 0; dx < dw; ++dx) {
              const ColEntry& c = col[dx];
              const uint64_t fx16  = c.fx16;
              const uint64_t fx16c = 65536u - fx16;
              const uint64_t fy16c64 = fy16c;
              const uint64_t fy1664  = fy16;
              const uint8_t* tl = row0 + c.ix0;
              const uint8_t* tr = row0 + c.ix1_off;
              const uint8_t* bl = row1 + c.ix0;
              const uint8_t* br = row1 + c.ix1_off;
              for (int ch = 0; ch < depth; ++ch) {
                dest_row[ch] = static_cast<uint8_t>(
                    ((tl[ch]*fx16c + tr[ch]*fx16)*fy16c64 +
                     (bl[ch]*fx16c + br[ch]*fx16)*fy1664 + (1u<<31)) >> 32);
              }
              dest_row += depth;
            }
          }
        }
      });
      break;
    }
    default:
      break;
  }
}

uint8_t* PixelBuffer::GetRawBuffer() const {
  return _buffer;
}

int PixelBuffer::GetAllocatedStrideBytes() const {
  return _allocated_size.Width * _format->GetDepth();
}

int PixelBuffer::GetBufferByteSize() const {
  return _size.Width * _size.Height * _format->GetDepth();
}

uint8_t* PixelBuffer::GetPixelAddress(int x, int y) const {
  assert((x >= 0) && (x < _size.Width));
  assert((y >= 0) && (y < _size.Height));
  return _buffer +
         ((y + _offset.Height) * _allocated_size.Width + (x + _offset.Width)) *
             _format->GetDepth();
}
