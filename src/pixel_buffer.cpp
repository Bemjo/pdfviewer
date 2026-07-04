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
#include <cmath>
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

#if defined(__linux__)
#include <sched.h>
#endif

namespace {

// Returns the number of CPUs this thread is actually permitted to run on,
// per the kernel's scheduling affinity mask -- NOT the total number of
// physical cores. std::thread::hardware_concurrency() (and a naive
// GetDefaultNumThreads()) report the latter, which is wrong when the
// process has been pinned to fewer CPUs than exist: a cpuset cgroup,
// taskset, an isolcpus= kernel boot parameter, or an affinity mask
// inherited from a parent process/launcher can all restrict this. On such
// a system, spawning threads equal to the physical core count just makes
// them timeslice on the one CPU they're actually allowed to run on, paying
// real thread-creation and context-switch cost for zero actual
// parallelism -- frequently slower than running single-threaded. Queried
// once and cached: affinity restrictions imposed by the kernel or a
// launcher process are not expected to change during the process's life.
int GetEffectiveNumThreads() {
#if defined(__linux__)
  static const int cached = [] {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
      const int count = CPU_COUNT(&mask);
      if (count > 0) return count;
    }
    return 1;  // conservative fallback if the syscall is unavailable/fails
  }();
  return cached;
#else
  return 0;  // defer to ExecuteInParallel's own default on other platforms
#endif
}

}  // namespace

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

// ---- Ratio classification: detects whether a src:dst pair is an exact
// 2:3 or 1:2 ratio (in lowest terms), independent per axis. Horizontal and
// vertical are classified separately since the two passes are already
// independent in this codebase -- a call with a 2:3 horizontal scale and a
// 1:2 vertical scale (or a 2:3/general mix, etc.) correctly gets the fast
// path on whichever axis qualifies.
enum class FastRatio { kGeneral, k1to2, k2to3 };

static inline FastRatio ClassifyRatio(int src_dim, int dst_dim) {
  int a = src_dim, b = dst_dim;
  while (b != 0) { const int t = a % b; a = b; b = t; }
  const int g = a;
  const int p = src_dim / g, q = dst_dim / g;
  if (p == 1 && q == 2) return FastRatio::k1to2;
  if (p == 2 && q == 3) return FastRatio::k2to3;
  return FastRatio::kGeneral;
}

#ifdef __ARM_NEON

// 1:2 Depth 4 (NEON) - 1 load, 1 zip, 2 stores per 8 output pixels
static void NearestHorizFast4_1to2(const uint8_t* __restrict__ src_row, uint8_t* __restrict__ out, int dw) {
  out = static_cast<uint8_t*>(__builtin_assume_aligned(out, 16));
  const uint32_t* src32 = reinterpret_cast<const uint32_t*>(src_row);
  uint32_t* out32 = reinterpret_cast<uint32_t*>(out);
  int dx = 0;
  for (; dx + 8 <= dw; dx += 8) {
    uint32x4_t p = vld1q_u32(src32 + (dx / 2));
    // Zipping a register with itself perfectly duplicates the lanes
    uint32x4x2_t dup = vzipq_u32(p, p);
    vst1q_u32(out32 + dx + 0, dup.val[0]);
    vst1q_u32(out32 + dx + 4, dup.val[1]);
  }
  for (; dx < dw; ++dx) {
    out32[dx] = src32[dx / 2];
  }
}

// 1:2 Depth 3 (NEON) - Bulk 24-byte to 48-byte expansion
static void NearestHorizFast3_1to2(const uint8_t* __restrict__ src_row, uint8_t* __restrict__ out, int dw) {
  out = static_cast<uint8_t*>(__builtin_assume_aligned(out, 16));
  int dx = 0;
  for (; dx + 16 <= dw; dx += 16) {
    uint8x8x3_t p = vld3_u8(src_row + (dx / 2) * 3);

    uint8x8x2_t r_dup = vzip_u8(p.val[0], p.val[0]);
    uint8x8x2_t g_dup = vzip_u8(p.val[1], p.val[1]);
    uint8x8x2_t b_dup = vzip_u8(p.val[2], p.val[2]);

    uint8x16x3_t out_vec;
    out_vec.val[0] = vcombine_u8(r_dup.val[0], r_dup.val[1]);
    out_vec.val[1] = vcombine_u8(g_dup.val[0], g_dup.val[1]);
    out_vec.val[2] = vcombine_u8(b_dup.val[0], b_dup.val[1]);
    vst3q_u8(out + dx * 3, out_vec);
  }
  for (; dx < dw; ++dx) {
    const uint8_t* p = src_row + (dx / 2) * 3;
    out[dx*3+0] = p[0]; out[dx*3+1] = p[1]; out[dx*3+2] = p[2];
  }
}

// ---------------------------------------------------------------------------
// Opt #1: Per the Cortex-A9 NEON MPE TRM (DDI0409H, Table 3-9), VLD1 {Dd[x]}
// (single-lane load) costs 3 issue cycles with Result not ready until cycle 4
// and Writeback at cycle 8, and each lane-insert has a register-merge RAW
// dependency on the previous write to the same Dd. Chaining 8 of these into
// one accumulator is a genuine serialized NEON dependency chain.
//
// An alternative was evaluated: gather the 8 lanes into a scalar staging
// buffer via ldrb/strb, then issue one full-register VLD1 per channel. That
// removes the NEON-side register-merge hazard, but introduces a different
// one: per the Cortex-A9 TRM (Ch. 6, L1 memory system), the store buffer is
// a 4-slot *merging* buffer with documented LDR<-STR forwarding for the
// scalar core, but no documented fast path for a wide VLD1 consuming several
// preceding narrow STRB writes -- that pattern risks a store-buffer drain
// stall before the load can be satisfied, which could cost more than the
// register-merge hazard it removes. Absent a cycle-accurate A9 model or real
// hardware measurement (QEMU is functional-only and does not model timing),
// this tradeoff cannot be verified on paper, so the direct vld1_lane_u8 form
// below -- whose cost is fully characterized by the TRM table above -- is
// kept rather than substituting an unverified alternative hazard.
// ---------------------------------------------------------------------------
static void BilinearHorizRow3(
    const uint8_t* __restrict__ src_row,
    const uint16_t* __restrict__ col_ix0,
    const uint16_t* __restrict__ col_ix1,
    const uint8_t* __restrict__ col_fx8,
    uint8_t* __restrict__ out, int dw) {
  out = static_cast<uint8_t*>(__builtin_assume_aligned(out, 16));
  int dx = 0;
  const uint8x8_t v255 = vdup_n_u8(255);

  for (; dx + 8 <= dw; dx += 8) {
    if (dx + 16 <= dw) {
      __builtin_prefetch(src_row + col_ix0[dx + 8], 0, 1);
      __builtin_prefetch(src_row + col_ix1[dx + 8], 0, 1);
    }

    uint8x8_t lA = vdup_n_u8(0), lB = vdup_n_u8(0), lC = vdup_n_u8(0);
    uint8x8_t rA = vdup_n_u8(0), rB = vdup_n_u8(0), rC = vdup_n_u8(0);

    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      const uint8_t* L = src_row + col_ix0[dx + k];
      const uint8_t* R = src_row + col_ix1[dx + k];
      lA = vld1_lane_u8(L + 0, lA, k);
      lB = vld1_lane_u8(L + 1, lB, k);
      lC = vld1_lane_u8(L + 2, lC, k);
      rA = vld1_lane_u8(R + 0, rA, k);
      rB = vld1_lane_u8(R + 1, rB, k);
      rC = vld1_lane_u8(R + 2, rC, k);
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

template <int BoostShift>
static void BilinearVertBlendSharp3_Templated(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ prev,
    const uint8_t* __restrict__ next,
    uint8_t fy8,
    uint8_t* __restrict__ dest_row, int dw) {
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

    uint16x8_t bilL = vmull_u8(vget_low_u8(v0),  vfyc);
    uint16x8_t bilH = vmull_u8(vget_high_u8(v0), vfyc);
    bilL = vmlal_u8(bilL, vget_low_u8(v1),  vfy);
    bilH = vmlal_u8(bilH, vget_high_u8(v1), vfy);

    // Opt #2: Eliminated the vrshrn + vmovl narrow/widen bubble.
    // Shift directly within the 16-bit registers using vrshrq.
    uint16x8_t base16L = vrshrq_n_u16(bilL, 8);
    uint16x8_t base16H = vrshrq_n_u16(bilH, 8);

    int16x8_t sL = vreinterpretq_s16_u16(
        vaddq_u16(base16L, vshrq_n_u16(base16L, BoostShift)));
    int16x8_t sH = vreinterpretq_s16_u16(
        vaddq_u16(base16H, vshrq_n_u16(base16H, BoostShift)));

    uint8x16_t havg = vhaddq_u8(vp, vn);
    uint8x8_t outByteL = vshr_n_u8(vget_low_u8(havg), BoostShift);
    uint8x8_t outByteH = vshr_n_u8(vget_high_u8(havg), BoostShift);
    sL = vreinterpretq_s16_u16(vsubw_u8(vreinterpretq_u16_s16(sL), outByteL));
    sH = vreinterpretq_s16_u16(vsubw_u8(vreinterpretq_u16_s16(sH), outByteH));

    vst1q_u8(dest_row + i, vcombine_u8(vqmovun_s16(sL), vqmovun_s16(sH)));
  }
  for (int i = (nbytes / 16) * 16; i + 8 <= nbytes; i += 8) {
    uint8x8_t v0b = vld1_u8(row0 + i), v1b = vld1_u8(row1 + i);
    uint8x8_t vpb = vld1_u8(prev  + i), vnb = vld1_u8(next  + i);
    uint16x8_t bil = vmull_u8(v0b, vfyc);
    bil = vmlal_u8(bil, v1b, vfy);

    // Opt #2 Tail handling
    uint16x8_t base16 = vrshrq_n_u16(bil, 8);

    int16x8_t s16 = vreinterpretq_s16_u16(
        vaddq_u16(base16, vshrq_n_u16(base16, BoostShift)));
    uint8x8_t outByte = vshr_n_u8(vhadd_u8(vpb, vnb), BoostShift);
    s16 = vreinterpretq_s16_u16(vsubw_u8(vreinterpretq_u16_s16(s16), outByte));
    vst1_u8(dest_row + i, vqmovun_s16(s16));
  }
}

static void BilinearVertBlendSharp3(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ prev,
    const uint8_t* __restrict__ next,
    uint8_t fy8, uint8_t sharp_strength,
    uint8_t* __restrict__ dest_row, int dw) {
  row0 = static_cast<const uint8_t*>(__builtin_assume_aligned(row0, 16));
  row1 = static_cast<const uint8_t*>(__builtin_assume_aligned(row1, 16));
  prev = static_cast<const uint8_t*>(__builtin_assume_aligned(prev, 16));
  next = static_cast<const uint8_t*>(__builtin_assume_aligned(next, 16));
  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const int nbytes = dw * 3;
  int i = 0;

  if (sharp_strength > 0) {
    if (sharp_strength == 64) {
      BilinearVertBlendSharp3_Templated<2>(row0, row1, prev, next, fy8, dest_row, dw);
    } else if (sharp_strength == 128) {
      BilinearVertBlendSharp3_Templated<1>(row0, row1, prev, next, fy8, dest_row, dw);
    } else if (sharp_strength == 32) {
      BilinearVertBlendSharp3_Templated<3>(row0, row1, prev, next, fy8, dest_row, dw);
    }
    i = (nbytes / 8) * 8;
  } else {
    const uint16x8_t vsh16 = vmovl_u8(vdup_n_u8(sharp_strength));
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

      uint16x8_t base16L = vrshrq_n_u16(bilL, 8);
      uint16x8_t base16H = vrshrq_n_u16(bilH, 8);

      uint16x8_t shL = vmulq_u16(base16L, vsh16);
      uint16x8_t shH = vmulq_u16(base16H, vsh16);

      int16x8_t sL = vreinterpretq_s16_u16(vaddq_u16(vshrq_n_u16(shL, 8), base16L));
      int16x8_t sH = vreinterpretq_s16_u16(vaddq_u16(vshrq_n_u16(shH, 8), base16H));

      uint16x8_t outL = vaddl_u8(vget_low_u8(vp),  vget_low_u8(vn));
      uint16x8_t outH = vaddl_u8(vget_high_u8(vp), vget_high_u8(vn));
      outL = vshrq_n_u16(vmulq_n_u16(outL, sharp_strength), 9);
      outH = vshrq_n_u16(vmulq_n_u16(outH, sharp_strength), 9);
      sL = vsubq_s16(sL, vreinterpretq_s16_u16(outL));
      sH = vsubq_s16(sH, vreinterpretq_s16_u16(outH));

      vst1q_u8(dest_row + i, vcombine_u8(vqmovun_s16(sL), vqmovun_s16(sH)));
    }
    for (; i + 8 <= nbytes; i += 8) {
      uint8x8_t v0b = vld1_u8(row0 + i), v1b = vld1_u8(row1 + i);
      uint8x8_t vpb = vld1_u8(prev  + i), vnb = vld1_u8(next  + i);
      uint16x8_t bil = vmull_u8(v0b, vfyc);
      bil = vmlal_u8(bil, v1b, vfy);

      uint16x8_t base16 = vrshrq_n_u16(bil, 8);
      uint16x8_t sh_add = vmulq_u16(base16, vsh16);
      int16x8_t s16 = vreinterpretq_s16_u16(vaddq_u16(vshrq_n_u16(sh_add, 8), base16));

      uint16x8_t outer = vaddl_u8(vpb, vnb);
      outer = vshrq_n_u16(vmulq_n_u16(outer, sharp_strength), 9);
      s16 = vsubq_s16(s16, vreinterpretq_s16_u16(outer));
      vst1_u8(dest_row + i, vqmovun_s16(s16));
    }
  }

  const int fy = fy8, fyc = 255 - fy, sh = sharp_strength;
  for (; i < nbytes; ++i) {
    const int bil = (row0[i]*fyc + row1[i]*fy + 128) >> 8;
    const int out = bil + ((bil*sh) >> 8) - (((int)prev[i] + next[i])*sh >> 9);
    dest_row[i] = static_cast<uint8_t>(out < 0 ? 0 : out > 255 ? 255 : out);
  }
}

template <int BoostShift>
static void BilinearVertBlendSharp4_Templated(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ prev,
    const uint8_t* __restrict__ next,
    uint8_t fy8,
    uint8_t* __restrict__ dest_row, int dw) {
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

    // Opt #2
    uint16x8_t base16L = vrshrq_n_u16(bilL, 8);
    uint16x8_t base16H = vrshrq_n_u16(bilH, 8);

    int16x8_t sL = vreinterpretq_s16_u16(
        vaddq_u16(base16L, vshrq_n_u16(base16L, BoostShift)));
    int16x8_t sH = vreinterpretq_s16_u16(
        vaddq_u16(base16H, vshrq_n_u16(base16H, BoostShift)));

    uint8x16_t havg = vhaddq_u8(vp, vn);
    uint8x8_t outByteL = vshr_n_u8(vget_low_u8(havg),  BoostShift);
    uint8x8_t outByteH = vshr_n_u8(vget_high_u8(havg), BoostShift);
    sL = vreinterpretq_s16_u16(vsubw_u8(vreinterpretq_u16_s16(sL), outByteL));
    sH = vreinterpretq_s16_u16(vsubw_u8(vreinterpretq_u16_s16(sH), outByteH));

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
  row0 = static_cast<const uint8_t*>(__builtin_assume_aligned(row0, 16));
  row1 = static_cast<const uint8_t*>(__builtin_assume_aligned(row1, 16));
  prev = static_cast<const uint8_t*>(__builtin_assume_aligned(prev, 16));
  next = static_cast<const uint8_t*>(__builtin_assume_aligned(next, 16));
  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const int nbytes = dw * 4;
  int i = 0;

  if (sharp_strength > 0) {
    if (sharp_strength == 64) {
      BilinearVertBlendSharp4_Templated<2>(row0, row1, prev, next, fy8, dest_row, dw);
    } else if (sharp_strength == 128) {
      BilinearVertBlendSharp4_Templated<1>(row0, row1, prev, next, fy8, dest_row, dw);
    } else if (sharp_strength == 32) {
      BilinearVertBlendSharp4_Templated<3>(row0, row1, prev, next, fy8, dest_row, dw);
    }
    i = (nbytes / 16) * 16;
  } else {
    const uint16x8_t vsh16 = vmovl_u8(vdup_n_u8(sharp_strength));

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

      uint16x8_t base16L = vrshrq_n_u16(bilL, 8);
      uint16x8_t base16H = vrshrq_n_u16(bilH, 8);

      uint16x8_t shL = vmulq_u16(base16L, vsh16);
      uint16x8_t shH = vmulq_u16(base16H, vsh16);

      int16x8_t sL = vreinterpretq_s16_u16(vaddq_u16(vshrq_n_u16(shL, 8), base16L));
      int16x8_t sH = vreinterpretq_s16_u16(vaddq_u16(vshrq_n_u16(shH, 8), base16H));

      uint16x8_t outL = vaddl_u8(vget_low_u8(vp),  vget_low_u8(vn));
      uint16x8_t outH = vaddl_u8(vget_high_u8(vp), vget_high_u8(vn));
      sL = vsubq_s16(sL, vreinterpretq_s16_u16(vshrq_n_u16(vmulq_n_u16(outL, sharp_strength), 9)));
      sH = vsubq_s16(sH, vreinterpretq_s16_u16(vshrq_n_u16(vmulq_n_u16(outH, sharp_strength), 9)));

      vst1q_u8(dest_row + i, vcombine_u8(vqmovun_s16(sL), vqmovun_s16(sH)));
    }
  }

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

static void BilinearVertBlend3(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    uint8_t fy8, uint8_t* __restrict__ dest_row, int dw) {
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

// Opt #1: Same vld1_lane_u8 gather as BilinearHorizRow3 above -- see the note
// there for the TRM-grounded cost of this chain and why a staged scalar
// gather (which trades the NEON register-merge hazard for an unverified
// store-buffer hazard) was not adopted.
static void BilinearHorizRow4(
    const uint8_t* __restrict__ src_row,
    const uint16_t* __restrict__ col_ix0,
    const uint16_t* __restrict__ col_ix1,
    const uint8_t* __restrict__ col_fx8,
    uint8_t* __restrict__ out, int dw) {
  out = static_cast<uint8_t*>(__builtin_assume_aligned(out, 16));
  int dx = 0;
  const uint8x8_t v255 = vdup_n_u8(255);

  for (; dx + 8 <= dw; dx += 8) {
    if (dx + 16 <= dw) {
      __builtin_prefetch(src_row + col_ix0[dx + 8], 0, 1);
      __builtin_prefetch(src_row + col_ix1[dx + 8], 0, 1);
    }

    uint8x8_t lA = vdup_n_u8(0), lB = vdup_n_u8(0), lC = vdup_n_u8(0), lD = vdup_n_u8(0);
    uint8x8_t rA = vdup_n_u8(0), rB = vdup_n_u8(0), rC = vdup_n_u8(0), rD = vdup_n_u8(0);

    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      const uint8_t* L = src_row + col_ix0[dx + k];
      const uint8_t* R = src_row + col_ix1[dx + k];
      lA = vld1_lane_u8(L + 0, lA, k);
      lB = vld1_lane_u8(L + 1, lB, k);
      lC = vld1_lane_u8(L + 2, lC, k);
      lD = vld1_lane_u8(L + 3, lD, k);
      rA = vld1_lane_u8(R + 0, rA, k);
      rB = vld1_lane_u8(R + 1, rB, k);
      rC = vld1_lane_u8(R + 2, rC, k);
      rD = vld1_lane_u8(R + 3, rD, k);
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

static void BilinearVertBlend4(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    uint8_t fy8, uint8_t* __restrict__ dest_row, int dw) {
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

// =============================================================================
// Fast-path polyphase resampling for exact rational scale ratios (2:3 and
// 1:2). See the constants block above for the derivation. All tables and
// the algorithm below (chunk cycling, head/tail clamp boundaries, starting
// chunk index after the head) were verified bit-exact against an exact
// (zero-floating-point, Fraction-based) reference across randomized trials,
// including mixed horizontal/vertical ratio combinations, before being
// written here. Two things had to be discovered empirically during that
// verification and are worth documenting since they are easy to get wrong:
//
//  1. Bilinear's 2-tap kernel only ever needs left-edge clamping at dx=0.
//     Bicubic's 4-tap kernel reaches ix-1, which goes negative at dx=0 AND
//     dx=1 for ratio 2:3 (and dx=0,1,2 for ratio 1:2) -- so bicubic needs a
//     2- or 3-pixel clamped head, not a 1-pixel head. The right/bottom edge
//     is symmetric: bicubic needs a 2- or 3-pixel clamped tail.
//  2. Chunk index and phase are NOT the same sequence once the head skips
//     more than 1 pixel: chunk c's first lane sits at phase (8c) mod q, not
//     at phase c. Starting the interior loop at chunk index (head-1) mod q
//     -- which works fine for bilinear's 1-pixel head -- silently produces
//     a one-position-shifted result for bicubic's multi-pixel head. The
//     correct starting chunk is whichever one's phase equals (head) mod q;
//     for this codebase's two ratios that resolves to chunk 2 (2:3) and
//     chunk 0 (1:2), baked in below as verified constants.
// =============================================================================

// ============ Bilinear horizontal ============
static constexpr int8_t  kBilinRelIx_2to3_c0[8] = { 0, 1, 1, 2, 3, 3, 4, 5 };
static constexpr uint8_t kBilinFx8_2to3_c0[8]   = { 128, 43, 213, 128, 43, 213, 128, 43 };
static constexpr int8_t  kBilinRelIx_2to3_c1[8] = { 0, 1, 2, 2, 3, 4, 4, 5 };
static constexpr uint8_t kBilinFx8_2to3_c1[8]   = { 213, 128, 43, 213, 128, 43, 213, 128 };
static constexpr int8_t  kBilinRelIx_2to3_c2[8] = { 0, 0, 1, 2, 2, 3, 4, 4 };
static constexpr uint8_t kBilinFx8_2to3_c2[8]   = { 43, 213, 128, 43, 213, 128, 43, 213 };
static constexpr int8_t  kBilinRelIx_1to2_c0[8] = { 0, 0, 1, 1, 2, 2, 3, 3 };
static constexpr uint8_t kBilinFx8_1to2_c0[8]   = { 64, 191, 64, 191, 64, 191, 64, 191 };
static constexpr int8_t  kBilinRelIx_1to2_c1[8] = { 0, 0, 1, 1, 2, 2, 3, 3 };
static constexpr uint8_t kBilinFx8_1to2_c1[8]   = { 64, 191, 64, 191, 64, 191, 64, 191 };
static constexpr int8_t  kVertRelRow_2to3[3] = { 0, 1, 1 };
static constexpr uint8_t kVertFy8_2to3[3]    = { 128, 43, 213 };
static constexpr int16_t kVertW0_2to3[3] = { -256, -237, -47 };
static constexpr int16_t kVertW1_2to3[3] = { 2304, 3840, 540 };
static constexpr int16_t kVertW2_2to3[3] = { 2304, 540, 3840 };
static constexpr int16_t kVertW3_2to3[3] = { -256, -47, -237 };
static constexpr int8_t  kVertRelRow_1to2[2] = { 0, 0 };
static constexpr uint8_t kVertFy8_1to2[2]    = { 64, 191 };
static constexpr int16_t kVertW0_1to2[2] = { -288, -96 };
static constexpr int16_t kVertW1_1to2[2] = { 3552, 928 };
static constexpr int16_t kVertW2_1to2[2] = { 928, 3552 };
static constexpr int16_t kVertW3_1to2[2] = { -96, -288 };

// ---- Bilinear horizontal fast path, depth=3 ----
static void BilinearHorizRowFast3_2to3(
    const uint8_t* __restrict__ src_row, int sw,
    uint8_t* __restrict__ out, int dw) {
  out = static_cast<uint8_t*>(__builtin_assume_aligned(out, 16));
  out[0] = src_row[0]; out[1] = src_row[1]; out[2] = src_row[2];

  int dx = 1, chunk = 0;
  while (dx + 16 <= dw) {
    const int8_t* rel; const uint8_t* fx8t;
    if (chunk == 0)      { rel = kBilinRelIx_2to3_c0; fx8t = kBilinFx8_2to3_c0; }
    else if (chunk == 1) { rel = kBilinRelIx_2to3_c1; fx8t = kBilinFx8_2to3_c1; }
    else                 { rel = kBilinRelIx_2to3_c2; fx8t = kBilinFx8_2to3_c2; }
    const int base_ix = (4*dx - 1) / 6;

    uint8x8_t lA_lo=vdup_n_u8(0), lB_lo=vdup_n_u8(0), lC_lo=vdup_n_u8(0);
    uint8x8_t rA_lo=vdup_n_u8(0), rB_lo=vdup_n_u8(0), rC_lo=vdup_n_u8(0);

    uint8x8_t lA_hi=vdup_n_u8(0), lB_hi=vdup_n_u8(0), lC_hi=vdup_n_u8(0);
    uint8x8_t rA_hi=vdup_n_u8(0), rB_hi=vdup_n_u8(0), rC_hi=vdup_n_u8(0);

    #pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) {
      const uint8_t* L = src_row + (base_ix + rel[k]) * 3;
      const uint8_t* R = L + 3;
      lA_lo = vld1_lane_u8(L+0, lA_lo, k); lB_lo = vld1_lane_u8(L+1, lB_lo, k); lC_lo = vld1_lane_u8(L+2, lC_lo, k);
      rA_lo = vld1_lane_u8(R+0, rA_lo, k); rB_lo = vld1_lane_u8(R+1, rB_lo, k); rC_lo = vld1_lane_u8(R+2, rC_lo, k);
    }

    #pragma GCC unroll 4
    for (int k = 4; k < 8; ++k) {
      const uint8_t* L = src_row + (base_ix + rel[k]) * 3;
      const uint8_t* R = L + 3;
      lA_hi = vld1_lane_u8(L+0, lA_hi, k); lB_hi = vld1_lane_u8(L+1, lB_hi, k); lC_hi = vld1_lane_u8(L+2, lC_hi, k);
      rA_hi = vld1_lane_u8(R+0, rA_hi, k); rB_hi = vld1_lane_u8(R+1, rB_hi, k); rC_hi = vld1_lane_u8(R+2, rC_hi, k);
    }

    // Merge the independent chains (1 cycle latency)
    uint8x8_t lA = vorr_u8(lA_lo, lA_hi);
    uint8x8_t lB = vorr_u8(lB_lo, lB_hi);
    uint8x8_t lC = vorr_u8(lC_lo, lC_hi);
    uint8x8_t rA = vorr_u8(rA_lo, rA_hi);
    uint8x8_t rB = vorr_u8(rB_lo, rB_hi);
    uint8x8_t rC = vorr_u8(rC_lo, rC_hi);
    uint8x8_t vfx = vld1_u8(fx8t);
    uint8x8_t vfxc = vsub_u8(vdup_n_u8(255), vfx);
    uint16x8_t accA = vmull_u8(lA, vfxc); accA = vmlal_u8(accA, rA, vfx);
    uint16x8_t accB = vmull_u8(lB, vfxc); accB = vmlal_u8(accB, rB, vfx);
    uint16x8_t accC = vmull_u8(lC, vfxc); accC = vmlal_u8(accC, rC, vfx);
    uint8x8x3_t vout;
    vout.val[0] = vrshrn_n_u16(accA, 8);
    vout.val[1] = vrshrn_n_u16(accB, 8);
    vout.val[2] = vrshrn_n_u16(accC, 8);
    vst3_u8(out + dx*3, vout);
    dx += 8; chunk = (chunk + 1) % 3;
  }
  for (; dx < dw; ++dx) {
    const int num = 4*dx - 1;
    int ix = num / 6;
    const int ix_max = sw - 1;
    const int ix1 = (ix < ix_max) ? ix + 1 : ix_max;
    if (ix > ix_max) ix = ix_max;
    const uint8_t fx = kBilinFx8_2to3_c0[(dx - 1) % 3];
    const uint32_t fxc = 255u - fx;
    const uint8_t* L = src_row + ix  * 3;
    const uint8_t* R = src_row + ix1 * 3;
    out[dx*3+0] = static_cast<uint8_t>((L[0]*fxc + R[0]*fx + 128u) >> 8);
    out[dx*3+1] = static_cast<uint8_t>((L[1]*fxc + R[1]*fx + 128u) >> 8);
    out[dx*3+2] = static_cast<uint8_t>((L[2]*fxc + R[2]*fx + 128u) >> 8);
  }
}

static void BilinearHorizRowFast3_1to2(
    const uint8_t* __restrict__ src_row, int sw,
    uint8_t* __restrict__ out, int dw) {
  out = static_cast<uint8_t*>(__builtin_assume_aligned(out, 16));
  out[0] = src_row[0]; out[1] = src_row[1]; out[2] = src_row[2];

  int dx = 1;
  const int8_t* rel = kBilinRelIx_1to2_c0;   // only 1 distinct table needed
  const uint8_t* fx8t = kBilinFx8_1to2_c0;
  while (dx + 16 <= dw) {
    const int base_ix = (2*dx - 1) / 4;
uint8x8_t lA_lo=vdup_n_u8(0), lB_lo=vdup_n_u8(0), lC_lo=vdup_n_u8(0);
    uint8x8_t rA_lo=vdup_n_u8(0), rB_lo=vdup_n_u8(0), rC_lo=vdup_n_u8(0);

    uint8x8_t lA_hi=vdup_n_u8(0), lB_hi=vdup_n_u8(0), lC_hi=vdup_n_u8(0);
    uint8x8_t rA_hi=vdup_n_u8(0), rB_hi=vdup_n_u8(0), rC_hi=vdup_n_u8(0);

    #pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) {
      const uint8_t* L = src_row + (base_ix + rel[k]) * 3;
      const uint8_t* R = L + 3;
      lA_lo = vld1_lane_u8(L+0, lA_lo, k); lB_lo = vld1_lane_u8(L+1, lB_lo, k); lC_lo = vld1_lane_u8(L+2, lC_lo, k);
      rA_lo = vld1_lane_u8(R+0, rA_lo, k); rB_lo = vld1_lane_u8(R+1, rB_lo, k); rC_lo = vld1_lane_u8(R+2, rC_lo, k);
    }

    #pragma GCC unroll 4
    for (int k = 4; k < 8; ++k) {
      const uint8_t* L = src_row + (base_ix + rel[k]) * 3;
      const uint8_t* R = L + 3;
      lA_hi = vld1_lane_u8(L+0, lA_hi, k); lB_hi = vld1_lane_u8(L+1, lB_hi, k); lC_hi = vld1_lane_u8(L+2, lC_hi, k);
      rA_hi = vld1_lane_u8(R+0, rA_hi, k); rB_hi = vld1_lane_u8(R+1, rB_hi, k); rC_hi = vld1_lane_u8(R+2, rC_hi, k);
    }

    // Merge the independent chains (1 cycle latency)
    uint8x8_t lA = vorr_u8(lA_lo, lA_hi);
    uint8x8_t lB = vorr_u8(lB_lo, lB_hi);
    uint8x8_t lC = vorr_u8(lC_lo, lC_hi);
    uint8x8_t rA = vorr_u8(rA_lo, rA_hi);
    uint8x8_t rB = vorr_u8(rB_lo, rB_hi);
    uint8x8_t rC = vorr_u8(rC_lo, rC_hi);
    uint8x8_t vfx = vld1_u8(fx8t);
    uint8x8_t vfxc = vsub_u8(vdup_n_u8(255), vfx);
    uint16x8_t accA = vmull_u8(lA, vfxc); accA = vmlal_u8(accA, rA, vfx);
    uint16x8_t accB = vmull_u8(lB, vfxc); accB = vmlal_u8(accB, rB, vfx);
    uint16x8_t accC = vmull_u8(lC, vfxc); accC = vmlal_u8(accC, rC, vfx);
    uint8x8x3_t vout;
    vout.val[0] = vrshrn_n_u16(accA, 8);
    vout.val[1] = vrshrn_n_u16(accB, 8);
    vout.val[2] = vrshrn_n_u16(accC, 8);
    vst3_u8(out + dx*3, vout);
    dx += 8;
  }
  for (; dx < dw; ++dx) {
    const int num = 2*dx - 1;
    int ix = num / 4;
    const int ix_max = sw - 1;
    const int ix1 = (ix < ix_max) ? ix + 1 : ix_max;
    if (ix > ix_max) ix = ix_max;
    const uint8_t fx = kBilinFx8_1to2_c0[(dx - 1) % 2];
    const uint32_t fxc = 255u - fx;
    const uint8_t* L = src_row + ix  * 3;
    const uint8_t* R = src_row + ix1 * 3;
    out[dx*3+0] = static_cast<uint8_t>((L[0]*fxc + R[0]*fx + 128u) >> 8);
    out[dx*3+1] = static_cast<uint8_t>((L[1]*fxc + R[1]*fx + 128u) >> 8);
    out[dx*3+2] = static_cast<uint8_t>((L[2]*fxc + R[2]*fx + 128u) >> 8);
  }
}

// ---- Bilinear horizontal fast path, depth=4 ----
static void BilinearHorizRowFast4_2to3(
    const uint8_t* __restrict__ src_row, int sw,
    uint8_t* __restrict__ out, int dw) {
  out = static_cast<uint8_t*>(__builtin_assume_aligned(out, 16));
  out[0]=src_row[0]; out[1]=src_row[1]; out[2]=src_row[2]; out[3]=src_row[3];

  int dx = 1, chunk = 0;
  while (dx + 16 <= dw) {
    const int8_t* rel; const uint8_t* fx8t;
    if (chunk == 0)      { rel = kBilinRelIx_2to3_c0; fx8t = kBilinFx8_2to3_c0; }
    else if (chunk == 1) { rel = kBilinRelIx_2to3_c1; fx8t = kBilinFx8_2to3_c1; }
    else                 { rel = kBilinRelIx_2to3_c2; fx8t = kBilinFx8_2to3_c2; }
    const int base_ix = (4*dx - 1) / 6;

    uint8x8_t lA_lo=vdup_n_u8(0), lB_lo=vdup_n_u8(0), lC_lo=vdup_n_u8(0), lD_lo=vdup_n_u8(0);
    uint8x8_t rA_lo=vdup_n_u8(0), rB_lo=vdup_n_u8(0), rC_lo=vdup_n_u8(0), rD_lo=vdup_n_u8(0);

    uint8x8_t lA_hi=vdup_n_u8(0), lB_hi=vdup_n_u8(0), lC_hi=vdup_n_u8(0), lD_hi=vdup_n_u8(0);
    uint8x8_t rA_hi=vdup_n_u8(0), rB_hi=vdup_n_u8(0), rC_hi=vdup_n_u8(0), rD_hi=vdup_n_u8(0);

    #pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) {
      const uint8_t* L = src_row + (base_ix + rel[k]) * 4;
      const uint8_t* R = L + 4;
      lA_lo = vld1_lane_u8(L+0, lA_lo, k); lB_lo = vld1_lane_u8(L+1, lB_lo, k); lC_lo = vld1_lane_u8(L+2, lC_lo, k); lD_lo = vld1_lane_u8(L+3, lD_lo, k);
      rA_lo = vld1_lane_u8(R+0, rA_lo, k); rB_lo = vld1_lane_u8(R+1, rB_lo, k); rC_lo = vld1_lane_u8(R+2, rC_lo, k); rD_lo = vld1_lane_u8(R+3, rD_lo, k);
    }

    #pragma GCC unroll 4
    for (int k = 4; k < 8; ++k) {
      const uint8_t* L = src_row + (base_ix + rel[k]) * 4;
      const uint8_t* R = L + 4;
      lA_hi = vld1_lane_u8(L+0, lA_hi, k); lB_hi = vld1_lane_u8(L+1, lB_hi, k); lC_hi = vld1_lane_u8(L+2, lC_hi, k); lD_hi = vld1_lane_u8(L+3, lD_hi, k);
      rA_hi = vld1_lane_u8(R+0, rA_hi, k); rB_hi = vld1_lane_u8(R+1, rB_hi, k); rC_hi = vld1_lane_u8(R+2, rC_hi, k); rD_hi = vld1_lane_u8(R+3, rD_hi, k);
    }

    // Merge the independent chains (1 cycle latency)
    uint8x8_t lA = vorr_u8(lA_lo, lA_hi);
    uint8x8_t lB = vorr_u8(lB_lo, lB_hi);
    uint8x8_t lC = vorr_u8(lC_lo, lC_hi);
    uint8x8_t lD = vorr_u8(lD_lo, lD_hi);
    uint8x8_t rA = vorr_u8(rA_lo, rA_hi);
    uint8x8_t rB = vorr_u8(rB_lo, rB_hi);
    uint8x8_t rC = vorr_u8(rC_lo, rC_hi);
    uint8x8_t rD = vorr_u8(rD_lo, rD_hi);

    uint8x8_t vfx = vld1_u8(fx8t);
    uint8x8_t vfxc = vsub_u8(vdup_n_u8(255), vfx);
    uint16x8_t accA=vmull_u8(lA,vfxc); accA=vmlal_u8(accA,rA,vfx);
    uint16x8_t accB=vmull_u8(lB,vfxc); accB=vmlal_u8(accB,rB,vfx);
    uint16x8_t accC=vmull_u8(lC,vfxc); accC=vmlal_u8(accC,rC,vfx);
    uint16x8_t accD=vmull_u8(lD,vfxc); accD=vmlal_u8(accD,rD,vfx);
    uint8x8x4_t vout;
    vout.val[0]=vrshrn_n_u16(accA,8); vout.val[1]=vrshrn_n_u16(accB,8);
    vout.val[2]=vrshrn_n_u16(accC,8); vout.val[3]=vrshrn_n_u16(accD,8);
    vst4_u8(out + dx*4, vout);
    dx += 8; chunk = (chunk + 1) % 3;
  }
  for (; dx < dw; ++dx) {
    const int num = 4*dx - 1;
    int ix = num / 6;
    const int ix_max = sw - 1;
    const int ix1 = (ix < ix_max) ? ix + 1 : ix_max;
    if (ix > ix_max) ix = ix_max;
    const uint8_t fx = kBilinFx8_2to3_c0[(dx - 1) % 3];
    const uint32_t fxc = 255u - fx;
    const uint8_t* L = src_row + ix  * 4;
    const uint8_t* R = src_row + ix1 * 4;
    for (int ch = 0; ch < 4; ++ch)
      out[dx*4+ch] = static_cast<uint8_t>((L[ch]*fxc + R[ch]*fx + 128u) >> 8);
  }
}

static void BilinearHorizRowFast4_1to2(
    const uint8_t* __restrict__ src_row, int sw,
    uint8_t* __restrict__ out, int dw) {
  out = static_cast<uint8_t*>(__builtin_assume_aligned(out, 16));
  out[0]=src_row[0]; out[1]=src_row[1]; out[2]=src_row[2]; out[3]=src_row[3];

  int dx = 1;
  const int8_t* rel = kBilinRelIx_1to2_c0;
  const uint8_t* fx8t = kBilinFx8_1to2_c0;
  while (dx + 16 <= dw) {
    const int base_ix = (2*dx - 1) / 4;

    uint8x8_t lA_lo=vdup_n_u8(0), lB_lo=vdup_n_u8(0), lC_lo=vdup_n_u8(0), lD_lo=vdup_n_u8(0);
    uint8x8_t rA_lo=vdup_n_u8(0), rB_lo=vdup_n_u8(0), rC_lo=vdup_n_u8(0), rD_lo=vdup_n_u8(0);

    uint8x8_t lA_hi=vdup_n_u8(0), lB_hi=vdup_n_u8(0), lC_hi=vdup_n_u8(0), lD_hi=vdup_n_u8(0);
    uint8x8_t rA_hi=vdup_n_u8(0), rB_hi=vdup_n_u8(0), rC_hi=vdup_n_u8(0), rD_hi=vdup_n_u8(0);

    #pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) {
      const uint8_t* L = src_row + (base_ix + rel[k]) * 4;
      const uint8_t* R = L + 4;
      lA_lo = vld1_lane_u8(L+0, lA_lo, k); lB_lo = vld1_lane_u8(L+1, lB_lo, k); lC_lo = vld1_lane_u8(L+2, lC_lo, k); lD_lo = vld1_lane_u8(L+3, lD_lo, k);
      rA_lo = vld1_lane_u8(R+0, rA_lo, k); rB_lo = vld1_lane_u8(R+1, rB_lo, k); rC_lo = vld1_lane_u8(R+2, rC_lo, k); rD_lo = vld1_lane_u8(R+3, rD_lo, k);
    }

    #pragma GCC unroll 4
    for (int k = 4; k < 8; ++k) {
      const uint8_t* L = src_row + (base_ix + rel[k]) * 4;
      const uint8_t* R = L + 4;
      lA_hi = vld1_lane_u8(L+0, lA_hi, k); lB_hi = vld1_lane_u8(L+1, lB_hi, k); lC_hi = vld1_lane_u8(L+2, lC_hi, k); lD_hi = vld1_lane_u8(L+3, lD_hi, k);
      rA_hi = vld1_lane_u8(R+0, rA_hi, k); rB_hi = vld1_lane_u8(R+1, rB_hi, k); rC_hi = vld1_lane_u8(R+2, rC_hi, k); rD_hi = vld1_lane_u8(R+3, rD_hi, k);
    }

        // Merge the independent chains (1 cycle latency)
    uint8x8_t lA = vorr_u8(lA_lo, lA_hi);
    uint8x8_t lB = vorr_u8(lB_lo, lB_hi);
    uint8x8_t lC = vorr_u8(lC_lo, lC_hi);
    uint8x8_t lD = vorr_u8(lD_lo, lD_hi);
    uint8x8_t rA = vorr_u8(rA_lo, rA_hi);
    uint8x8_t rB = vorr_u8(rB_lo, rB_hi);
    uint8x8_t rC = vorr_u8(rC_lo, rC_hi);
    uint8x8_t rD = vorr_u8(rD_lo, rD_hi);

    uint8x8_t vfx = vld1_u8(fx8t);
    uint8x8_t vfxc = vsub_u8(vdup_n_u8(255), vfx);
    uint16x8_t accA=vmull_u8(lA,vfxc); accA=vmlal_u8(accA,rA,vfx);
    uint16x8_t accB=vmull_u8(lB,vfxc); accB=vmlal_u8(accB,rB,vfx);
    uint16x8_t accC=vmull_u8(lC,vfxc); accC=vmlal_u8(accC,rC,vfx);
    uint16x8_t accD=vmull_u8(lD,vfxc); accD=vmlal_u8(accD,rD,vfx);
    uint8x8x4_t vout;
    vout.val[0]=vrshrn_n_u16(accA,8); vout.val[1]=vrshrn_n_u16(accB,8);
    vout.val[2]=vrshrn_n_u16(accC,8); vout.val[3]=vrshrn_n_u16(accD,8);
    vst4_u8(out + dx*4, vout);
    dx += 8;
  }
  for (; dx < dw; ++dx) {
    const int num = 2*dx - 1;
    int ix = num / 4;
    const int ix_max = sw - 1;
    const int ix1 = (ix < ix_max) ? ix + 1 : ix_max;
    if (ix > ix_max) ix = ix_max;
    const uint8_t fx = kBilinFx8_1to2_c0[(dx - 1) % 2];
    const uint32_t fxc = 255u - fx;
    const uint8_t* L = src_row + ix  * 4;
    const uint8_t* R = src_row + ix1 * 4;
    for (int ch = 0; ch < 4; ++ch)
      out[dx*4+ch] = static_cast<uint8_t>((L[ch]*fxc + R[ch]*fx + 128u) >> 8);
  }
}

// ---- Vertical fast path: row index/weight lookup feeding the EXISTING
// blend functions (BilinearVertBlend3/4, BicubicVertBlend) completely
// unchanged. The vertical pass runs once per OUTPUT ROW, not once per
// pixel like the horizontal gather, so there's no hot-loop/lane-hazard
// concern here -- the only thing worth eliminating is the general path's
// upfront row_table/col-index vector construction (an O(dh) allocation and
// float-precompute pass done once per Copy() call). A single closed-form,
// always-clamped helper is simplest and correct uniformly for every row,
// interior or edge; there is no need for the horizontal side's separate
// chunk/phase machinery since nothing here is 8-lanes-per-iteration.
template<int p, int q>
static inline void BilinearVertRowFast(
    int dy, int sh, int* iy_prev, int* iy0, int* iy1, int* iy_next, uint8_t* fy8) {
  const int num = (2*dy + 1)*p - q;
  constexpr int twoq = 2*q;
  constexpr float twoq_inv = 1.0f / static_cast<float>(twoq);
  int iy = num * twoq_inv;
  int rem = num - iy*twoq;
  if (rem < 0) { iy -= 1; rem += twoq; }
  if (iy < 0) { iy = 0; rem = 0; }
  const int iy_max = sh - 1;
  // iy_prev/iy0/iy1/iy_next are all derived from the same raw iy, matching
  // the general path's r0=clamp_row(iy-1), r1=clamp_row(iy),
  // r2=clamp_row(iy1), r3=clamp_row(iy1+1) exactly (verified: for the
  // upscale ratios this fast path handles, raw iy never exceeds iy_max, so
  // there is no discrepancy between clamping iy directly vs. clamping an
  // already-clamped iy0 -- but deriving all four from the same raw iy here
  // keeps that invariant explicit rather than relying on it silently).
  *iy0 = (iy < iy_max) ? iy : iy_max;
  *iy1 = (iy < iy_max) ? iy + 1 : iy_max;
  if (iy > iy_max) *iy0 = *iy1 = iy_max;
  *iy_prev = (iy - 1 < 0) ? 0 : (iy - 1 > iy_max ? iy_max : iy - 1);
  *iy_next = (*iy1 + 1 > iy_max) ? iy_max : *iy1 + 1;
  const float t = static_cast<float>(rem) * twoq_inv;
  *fy8 = static_cast<uint8_t>(t * 255.0f + 0.5f);
}

// ---- Fast-path row-cache slot selector shared by bilinear and bicubic
// dispatch below: given a newly-needed source row index, either reuse an
// already-computed slot or recompute into the least-recently-replaced one.
// Mirrors the general path's ensure_hrow but simplified since the fast
// path's row advance pattern means at most 1 new row is ever needed per
// output row (the others are always carried over from the previous call).

// =============================================================================
// Fast-path Copy() dispatch for BILINEAR. Activates whenever both axes are
// an exact 2:3 or 1:2 ratio, sharpened or not. Falls through to the
// existing general path when either axis doesn't match that pattern.
// Sharpening reuses BilinearVertBlendSharp3/4 verbatim (unchanged) -- the
// only new work here is fetching two extra rows (prev/next) per output row
// and picking those row indices the same way the general path's
// clamp_row(iy-1)/clamp_row(iy1+1) does, verified to match exactly.
// =============================================================================
static bool CopyBilinearFastPath(
    const uint8_t* src_base, int src_stride,
    uint8_t* dest_base, int dest_stride, int depth,
    int sw, int sh, int dw, int dh,
    int src_rect_x, int src_rect_y,
    int dest_rect_x, int dest_rect_y,
    uint8_t sharp_strength) {
  const FastRatio hr = ClassifyRatio(sw, dw);
  const FastRatio vr = ClassifyRatio(sh, dh);
  if (hr == FastRatio::kGeneral || vr == FastRatio::kGeneral) return false;
  if (depth != 3 && depth != 4) return false;
  const int src_row_offset = src_rect_y * src_stride + src_rect_x * depth;

  ExecuteInParallel([=](int num_threads, int i) {
    const int rows_per = dh / num_threads;
    const int height = (i == num_threads - 1) ? (dh - i * rows_per) : rows_per;
    const int dy0 = i * rows_per;
    const int row_bytes = dw * depth;
    const int num_slots = (sharp_strength > 0) ? 4 : 2;

    const size_t row_stride = (static_cast<size_t>(row_bytes) + 15) & ~size_t(15);
    void* raw_ptr = nullptr;
    posix_memalign(&raw_ptr, 16, row_stride * num_slots);
    std::unique_ptr<uint8_t, void(*)(void*)> hbuf(static_cast<uint8_t*>(raw_ptr), std::free);
    uint8_t* hrow[4] = {
      hbuf.get(),
      hbuf.get() + row_stride,
      hbuf.get() + row_stride * 2,
      hbuf.get() + row_stride * 3,
    };
    int cached_iy[4] = { -1, -1, -1, -1 };

    auto ensure_row = [&](int slot, int iy) {
      if (cached_iy[slot] == iy) return;
      const uint8_t* src_row = src_base + src_row_offset + iy * src_stride;
      if (depth == 3) {
        if (hr == FastRatio::k2to3) BilinearHorizRowFast3_2to3(src_row, sw, hrow[slot], dw);
        else                        BilinearHorizRowFast3_1to2(src_row, sw, hrow[slot], dw);
      } else {
        if (hr == FastRatio::k2to3) BilinearHorizRowFast4_2to3(src_row, sw, hrow[slot], dw);
        else                        BilinearHorizRowFast4_1to2(src_row, sw, hrow[slot], dw);
      }
      cached_iy[slot] = iy;
    };

    for (int dy_local = 0; dy_local < height; ++dy_local) {
      const int dy = dy0 + dy_local;
      int iy_prev, iy0, iy1, iy_next; uint8_t fy8;
      if (vr == FastRatio::k2to3) BilinearVertRowFast<2, 3>(dy, sh, &iy_prev, &iy0, &iy1, &iy_next, &fy8);
      else if (vr == FastRatio::k1to2) BilinearVertRowFast<1, 2>(dy, sh, &iy_prev, &iy0, &iy1, &iy_next, &fy8);
      uint8_t* dest_row = dest_base + (dest_rect_y + dy) * dest_stride + dest_rect_x * depth;

      if (sharp_strength > 0) {
        ensure_row(0, iy_prev);
        ensure_row(1, iy0);
        ensure_row(2, iy1);
        ensure_row(3, iy_next);
        if (iy0 == iy1) {
          // No interpolation happening at this row (bottom-edge clamp
          // collapsed both taps to the same source row) -- nothing to
          // sharpen, matches the general path's identical shortcut.
          memcpy(dest_row, hrow[1], row_bytes);
        } else if (depth == 3) {
          BilinearVertBlendSharp3(hrow[1], hrow[2], hrow[0], hrow[3],
                                   fy8, sharp_strength, dest_row, dw);
        } else {
          BilinearVertBlendSharp4(hrow[1], hrow[2], hrow[0], hrow[3],
                                   fy8, sharp_strength, dest_row, dw);
        }
      } else {
        ensure_row(0, iy0);
        ensure_row(1, iy1);
        if (iy0 == iy1) {
          memcpy(dest_row, hrow[0], row_bytes);
        } else if (depth == 3) {
          BilinearVertBlend3(hrow[0], hrow[1], fy8, dest_row, dw);
        } else {
          BilinearVertBlend4(hrow[0], hrow[1], fy8, dest_row, dw);
        }
      }
    }
  }, GetEffectiveNumThreads());
  return true;
}

#endif // __ARM_NEON

// 2:3 Depth 4 (Scalar Unrolled - Optimal for A9 Integer Dual-Issue)
static void NearestHorizFast4_2to3(const uint8_t* __restrict__ src_row, uint8_t* __restrict__ out, int dw) {
  const uint32_t* src32 = reinterpret_cast<const uint32_t*>(src_row);
  uint32_t* out32 = reinterpret_cast<uint32_t*>(out);
  int dx = 0, sx = 0;
  for (; dx + 6 <= dw; dx += 6, sx += 4) {
    uint32_t p0 = src32[sx + 0];
    uint32_t p1 = src32[sx + 1];
    uint32_t p2 = src32[sx + 2];
    uint32_t p3 = src32[sx + 3];
    out32[dx+0] = p0; out32[dx+1] = p0; out32[dx+2] = p1;
    out32[dx+3] = p2; out32[dx+4] = p2; out32[dx+5] = p3;
  }
  for (; dx < dw; ++dx) {
    out32[dx] = src32[(dx * 2) / 3];
  }
}

// 2:3 Depth 3 (Scalar Unrolled - Optimal for A9 Integer Dual-Issue)
static void NearestHorizFast3_2to3(const uint8_t* __restrict__ src_row, uint8_t* __restrict__ out, int dw) {
  int dx = 0, sx = 0;
  for (; dx + 6 <= dw; dx += 6, sx += 4) {
    const uint8_t* p0 = src_row + (sx + 0) * 3;
    const uint8_t* p1 = src_row + (sx + 1) * 3;
    const uint8_t* p2 = src_row + (sx + 2) * 3;
    const uint8_t* p3 = src_row + (sx + 3) * 3;
    out[(dx+0)*3+0] = p0[0]; out[(dx+0)*3+1] = p0[1]; out[(dx+0)*3+2] = p0[2];
    out[(dx+1)*3+0] = p0[0]; out[(dx+1)*3+1] = p0[1]; out[(dx+1)*3+2] = p0[2];
    out[(dx+2)*3+0] = p1[0]; out[(dx+2)*3+1] = p1[1]; out[(dx+2)*3+2] = p1[2];

    out[(dx+3)*3+0] = p2[0]; out[(dx+3)*3+1] = p2[1]; out[(dx+3)*3+2] = p2[2];
    out[(dx+4)*3+0] = p2[0]; out[(dx+4)*3+1] = p2[1]; out[(dx+4)*3+2] = p2[2];
    out[(dx+5)*3+0] = p3[0]; out[(dx+5)*3+1] = p3[1]; out[(dx+5)*3+2] = p3[2];
  }
  for (; dx < dw; ++dx) {
    const uint8_t* p = src_row + ((dx * 2) / 3) * 3;
    out[dx*3+0] = p[0]; out[dx*3+1] = p[1]; out[dx*3+2] = p[2];
  }
}

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
    }, GetEffectiveNumThreads());
    return;
  }

  const int dw = dest_rect.Width;
  const int dh = dest_rect.Height;
  const int sw = src_rect.Width;
  const int sh = src_rect.Height;
  const bool valid_sharp_depth = depth == 3 || depth == 4;

  switch (scale_mode) {
    case SCALE_NEAREST: {
      const FastRatio hr = ClassifyRatio(sw, dw);
      std::vector<int> sx_off;

      // Only allocate the general offset map if we don't hit a horizontal fast path
      if (hr == FastRatio::kGeneral) {
        sx_off.resize(dw);
        for (int dx = 0; dx < dw; ++dx) {
          const int sx = src_rect.X + (dx * sw) / dw;
          sx_off[dx] = sx * depth;
        }
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

          // Vertical Nearest is perfectly optimized via simple row caching
          if (sy == cached_sy && prev_dest_row != nullptr) {
            memcpy(dest_row, prev_dest_row, row_bytes);
            prev_dest_row = dest_row;
            continue;
          }

          cached_sy = sy;
          const uint8_t* src_row = src_base + sy * src_stride;
          uint8_t* out = dest_row;
          bool handled = false;

          // For fast paths, we explicitly advance the pointer by the source X rect
          const uint8_t* src_row_fast = src_row + src_rect.X * depth;

          if (hr == FastRatio::k1to2) {
            if (depth == 4) {
#ifdef __ARM_NEON
              NearestHorizFast4_1to2(src_row_fast, out, dw); handled = true;
#endif
            } else if (depth == 3) {
#ifdef __ARM_NEON
              NearestHorizFast3_1to2(src_row_fast, out, dw); handled = true;
#endif
            }
          } else if (hr == FastRatio::k2to3) {
            if (depth == 4) {
              NearestHorizFast4_2to3(src_row_fast, out, dw); handled = true;
            } else if (depth == 3) {
              NearestHorizFast3_2to3(src_row_fast, out, dw); handled = true;
            }
          }

          // General Fallback
          if (!handled) {
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
          }
          prev_dest_row = dest_row;
        }
      }, GetEffectiveNumThreads());
      break;
    }

    case SCALE_BILINEAR: {
#ifdef __ARM_NEON
      if (valid_sharp_depth) {
        const uint8_t sharp_strength = sharpen_strength > 0 ?
            16u << static_cast<uint8_t>(std::min(std::max(1, static_cast<int>(sharpen_strength)), 3)):
            0u;

        // Fast path: exact 2:3 or 1:2 ratio on both axes, sharpened or not
        // -- see CopyBilinearFastPath for the polyphase derivation and
        // BilinearVertBlendSharp3/4 for the sharpening math (unchanged,
        // reused as-is). Falls through to the general path below when
        // either axis doesn't match.
        if (CopyBilinearFastPath(src_base, src_stride, dest_base, dest_stride,
                                  depth, sw, sh, dw, dh,
                                  src_rect.X, src_rect.Y,
                                  dest_rect.X, dest_rect.Y,
                                  sharp_strength)) {
          break;
        }

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

        const int ry_min = src_rect.Y;
        const int ry_max = src_rect.Y + sh - 1;

        // Opt #5: Thread memory allocation pulled OUT of the loop entirely.
        // We calculate memory requirements up-front, ensure memory row strides
        // are 16-byte aligned, and allocate a single slab for the thread pool to slice up.
        // max_workers is tied to the SAME effective thread count actually
        // passed to ExecuteInParallel below, so the allocation is provably
        // sized correctly for however many threads truly get spawned,
        // rather than trusting a generic upper bound.
        const int max_workers = std::min((int)dh, GetEffectiveNumThreads());
        const size_t row_stride = (dw * depth + 15) & ~15;
        const size_t buf_size = row_stride * 4;

        void* raw_ptr = nullptr;
        posix_memalign(&raw_ptr, 16, buf_size * max_workers);
        std::unique_ptr<uint8_t, void(*)(void*)> hbuf(
            static_cast<uint8_t*>(raw_ptr), std::free);

        ExecuteInParallel([=, &col_ix0, &col_ix1, &col_fx8](
                              int num_threads, int i) {
          const int rows_per = dh / num_threads;
          const int height   = (i == num_threads - 1)
                                   ? (dh - i * rows_per) : rows_per;
          const int dy0 = i * rows_per;

          uint8_t* thread_hbuf = static_cast<uint8_t*>(raw_ptr) + i * buf_size;
          uint8_t* hrow[4] = {
            thread_hbuf,
            thread_hbuf + row_stride,
            thread_hbuf + row_stride * 2,
            thread_hbuf + row_stride * 3,
          };
          int cached_iy[4] = { -1, -1, -1, -1 };

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
              const int r0 = clamp_row(iy - 1);
              const int r1 = clamp_row(iy);
              const int r2 = clamp_row(iy1);
              const int r3 = clamp_row(iy1 + 1);
              ensure_hrow(0, r0);
              ensure_hrow(1, r1);
              ensure_hrow(2, r2);
              ensure_hrow(3, r3);

              if (iy == iy1) {
                memcpy(dest_row, hrow[1], dw * depth);
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
                memcpy(dest_row, hrow[1], dw * depth);
              } else if (depth == 3) {
                BilinearVertBlend3(hrow[1], hrow[2], fy8, dest_row, dw);
              } else {
                BilinearVertBlend4(hrow[1], hrow[2], fy8, dest_row, dw);
              }
            }
          }
        }, GetEffectiveNumThreads());
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
      }, GetEffectiveNumThreads());
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