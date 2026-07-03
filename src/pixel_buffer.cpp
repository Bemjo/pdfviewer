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

#ifdef __ARM_NEON

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

// Opt #4: Process 8 horizontal pixels per iteration instead of 4 using Q6 scaling.
static constexpr int kBicubicLanes = 8;

// ---------------------------------------------------------------------------
// Opt #3: the vld1_lane_u8 chain below (12 registers x 8 chained inserts) is
// the worst case in this file for the Cortex-A9 NEON MPE register-merge RAW
// hazard documented in TRM Table 3-9 (VLD1{Dd[x]}: Source = destination
// register itself, Result@4, WB@8, vs plain VLD1{Dd}: 2 issue cycles,
// Result/WB@7, no inter-lane dependency). A staged scalar-gather alternative
// (ldrb/strb into a scratch buffer, then one vld1_u8 per channel) was
// evaluated to remove this chain, but was not adopted: per the Cortex-A9 TRM
// (Ch. 6, L1 memory system), the store buffer is a 4-slot merging buffer
// with documented LDR<-STR forwarding on the scalar side, but no documented
// forwarding path for a wide VLD1 consuming several preceding narrow STRB
// writes -- risking a store-buffer drain stall that could cost more than the
// hazard it removes. This cannot be resolved on paper (QEMU does not model
// timing); the vld1_lane_u8 form is kept since its cost is fully
// characterized by the TRM table above.
// Opt #4: Accumulator runs in 16-bit format using Q6 weights.
// ---------------------------------------------------------------------------
static void BicubicHorizRow3(
    const uint8_t* __restrict__ src_row,
    const uint16_t* __restrict__ col_ix0,
    const uint16_t* __restrict__ col_ix1,
    const uint16_t* __restrict__ col_ix2,
    const uint16_t* __restrict__ col_ix3,
    const int16_t* __restrict__ col_w0,
    const int16_t* __restrict__ col_w1,
    const int16_t* __restrict__ col_w2,
    const int16_t* __restrict__ col_w3,
    int16_t* __restrict__ out, int dw) {
  out = static_cast<int16_t*>(__builtin_assume_aligned(out, 16));
  int dx = 0;

  for (; dx + kBicubicLanes <= dw; dx += kBicubicLanes) {
    if (dx + 2 * kBicubicLanes <= dw) {
      __builtin_prefetch(src_row + col_ix0[dx + kBicubicLanes], 0, 1);
      __builtin_prefetch(src_row + col_ix3[dx + kBicubicLanes], 0, 1);
    }

    uint8x8x3_t t0, t1, t2, t3;
    t0.val[0] = t0.val[1] = t0.val[2] = vdup_n_u8(0);
    t1.val[0] = t1.val[1] = t1.val[2] = vdup_n_u8(0);
    t2.val[0] = t2.val[1] = t2.val[2] = vdup_n_u8(0);
    t3.val[0] = t3.val[1] = t3.val[2] = vdup_n_u8(0);

    #pragma GCC unroll 8
    for (int k = 0; k < kBicubicLanes; ++k) {
      const int i = dx + k;

      const uint8_t* p0 = src_row + col_ix0[i];
      t0.val[0] = vld1_lane_u8(p0 + 0, t0.val[0], k);
      t0.val[1] = vld1_lane_u8(p0 + 1, t0.val[1], k);
      t0.val[2] = vld1_lane_u8(p0 + 2, t0.val[2], k);

      const uint8_t* p1 = src_row + col_ix1[i];
      t1.val[0] = vld1_lane_u8(p1 + 0, t1.val[0], k);
      t1.val[1] = vld1_lane_u8(p1 + 1, t1.val[1], k);
      t1.val[2] = vld1_lane_u8(p1 + 2, t1.val[2], k);

      const uint8_t* p2 = src_row + col_ix2[i];
      t2.val[0] = vld1_lane_u8(p2 + 0, t2.val[0], k);
      t2.val[1] = vld1_lane_u8(p2 + 1, t2.val[1], k);
      t2.val[2] = vld1_lane_u8(p2 + 2, t2.val[2], k);

      const uint8_t* p3 = src_row + col_ix3[i];
      t3.val[0] = vld1_lane_u8(p3 + 0, t3.val[0], k);
      t3.val[1] = vld1_lane_u8(p3 + 1, t3.val[1], k);
      t3.val[2] = vld1_lane_u8(p3 + 2, t3.val[2], k);
    }

    int16x8_t w0 = vld1q_s16(col_w0 + dx);
    int16x8_t w1 = vld1q_s16(col_w1 + dx);
    int16x8_t w2 = vld1q_s16(col_w2 + dx);
    int16x8_t w3 = vld1q_s16(col_w3 + dx);

    #define CUBIC_TAP8(IDX, OUTVAR) \
      do { \
        int16x8_t P0 = vreinterpretq_s16_u16(vmovl_u8(t0.val[IDX])); \
        int16x8_t P1 = vreinterpretq_s16_u16(vmovl_u8(t1.val[IDX])); \
        int16x8_t P2 = vreinterpretq_s16_u16(vmovl_u8(t2.val[IDX])); \
        int16x8_t P3 = vreinterpretq_s16_u16(vmovl_u8(t3.val[IDX])); \
        int16x8_t acc = vmulq_s16(P0, w0); \
        acc = vmlaq_s16(acc, P1, w1); \
        acc = vmlaq_s16(acc, P2, w2); \
        acc = vmlaq_s16(acc, P3, w3); \
        OUTVAR = vrshrq_n_s16(acc, 2); \
      } while (0)

    int16x8_t outA, outB, outC;
    CUBIC_TAP8(0, outA);
    CUBIC_TAP8(1, outB);
    CUBIC_TAP8(2, outC);
    #undef CUBIC_TAP8

    int16x8x3_t out_struct;
    out_struct.val[0] = outA;
    out_struct.val[1] = outB;
    out_struct.val[2] = outC;
    vst3q_s16(out + dx * 3, out_struct);
  }
  for (; dx < dw; ++dx) {
    const uint8_t* P0 = src_row + col_ix0[dx];
    const uint8_t* P1 = src_row + col_ix1[dx];
    const uint8_t* P2 = src_row + col_ix2[dx];
    const uint8_t* P3 = src_row + col_ix3[dx];
    const int w0 = col_w0[dx], w1 = col_w1[dx], w2 = col_w2[dx], w3 = col_w3[dx];
    for (int ch = 0; ch < 3; ++ch) {
      const int acc = P0[ch]*w0 + P1[ch]*w1 + P2[ch]*w2 + P3[ch]*w3;
      out[dx*3 + ch] = static_cast<int16_t>((acc + 2) >> 2);
    }
  }
}

// See Opt #3 note above BicubicHorizRow3 for the TRM-grounded cost of the
// vld1_lane_u8 chain below and why the staged-gather alternative was not
// adopted. This is the worst-case lane-chain count in this file: 4 taps x 4
// channels x 8 chained inserts = 16 registers built one lane at a time.
static void BicubicHorizRow4(
    const uint8_t* __restrict__ src_row,
    const uint16_t* __restrict__ col_ix0,
    const uint16_t* __restrict__ col_ix1,
    const uint16_t* __restrict__ col_ix2,
    const uint16_t* __restrict__ col_ix3,
    const int16_t* __restrict__ col_w0,
    const int16_t* __restrict__ col_w1,
    const int16_t* __restrict__ col_w2,
    const int16_t* __restrict__ col_w3,
    int16_t* __restrict__ out, int dw) {
  out = static_cast<int16_t*>(__builtin_assume_aligned(out, 16));
  int dx = 0;

  for (; dx + kBicubicLanes <= dw; dx += kBicubicLanes) {
    if (dx + 2 * kBicubicLanes <= dw) {
      __builtin_prefetch(src_row + col_ix0[dx + kBicubicLanes], 0, 1);
      __builtin_prefetch(src_row + col_ix3[dx + kBicubicLanes], 0, 1);
    }

    uint8x8x4_t t0, t1, t2, t3;
    t0.val[0] = t0.val[1] = t0.val[2] = t0.val[3] = vdup_n_u8(0);
    t1.val[0] = t1.val[1] = t1.val[2] = t1.val[3] = vdup_n_u8(0);
    t2.val[0] = t2.val[1] = t2.val[2] = t2.val[3] = vdup_n_u8(0);
    t3.val[0] = t3.val[1] = t3.val[2] = t3.val[3] = vdup_n_u8(0);

    #pragma GCC unroll 8
    for (int k = 0; k < kBicubicLanes; ++k) {
      const int i = dx + k;

      const uint8_t* p0 = src_row + col_ix0[i];
      t0.val[0] = vld1_lane_u8(p0 + 0, t0.val[0], k);
      t0.val[1] = vld1_lane_u8(p0 + 1, t0.val[1], k);
      t0.val[2] = vld1_lane_u8(p0 + 2, t0.val[2], k);
      t0.val[3] = vld1_lane_u8(p0 + 3, t0.val[3], k);

      const uint8_t* p1 = src_row + col_ix1[i];
      t1.val[0] = vld1_lane_u8(p1 + 0, t1.val[0], k);
      t1.val[1] = vld1_lane_u8(p1 + 1, t1.val[1], k);
      t1.val[2] = vld1_lane_u8(p1 + 2, t1.val[2], k);
      t1.val[3] = vld1_lane_u8(p1 + 3, t1.val[3], k);

      const uint8_t* p2 = src_row + col_ix2[i];
      t2.val[0] = vld1_lane_u8(p2 + 0, t2.val[0], k);
      t2.val[1] = vld1_lane_u8(p2 + 1, t2.val[1], k);
      t2.val[2] = vld1_lane_u8(p2 + 2, t2.val[2], k);
      t2.val[3] = vld1_lane_u8(p2 + 3, t2.val[3], k);

      const uint8_t* p3 = src_row + col_ix3[i];
      t3.val[0] = vld1_lane_u8(p3 + 0, t3.val[0], k);
      t3.val[1] = vld1_lane_u8(p3 + 1, t3.val[1], k);
      t3.val[2] = vld1_lane_u8(p3 + 2, t3.val[2], k);
      t3.val[3] = vld1_lane_u8(p3 + 3, t3.val[3], k);
    }

    int16x8_t w0 = vld1q_s16(col_w0 + dx);
    int16x8_t w1 = vld1q_s16(col_w1 + dx);
    int16x8_t w2 = vld1q_s16(col_w2 + dx);
    int16x8_t w3 = vld1q_s16(col_w3 + dx);

    #define CUBIC_TAP8(IDX, OUTVAR) \
      do { \
        int16x8_t P0 = vreinterpretq_s16_u16(vmovl_u8(t0.val[IDX])); \
        int16x8_t P1 = vreinterpretq_s16_u16(vmovl_u8(t1.val[IDX])); \
        int16x8_t P2 = vreinterpretq_s16_u16(vmovl_u8(t2.val[IDX])); \
        int16x8_t P3 = vreinterpretq_s16_u16(vmovl_u8(t3.val[IDX])); \
        int16x8_t acc = vmulq_s16(P0, w0); \
        acc = vmlaq_s16(acc, P1, w1); \
        acc = vmlaq_s16(acc, P2, w2); \
        acc = vmlaq_s16(acc, P3, w3); \
        OUTVAR = vrshrq_n_s16(acc, 2); \
      } while (0)

    int16x8_t outA, outB, outC, outD;
    CUBIC_TAP8(0, outA);
    CUBIC_TAP8(1, outB);
    CUBIC_TAP8(2, outC);
    CUBIC_TAP8(3, outD);
    #undef CUBIC_TAP8

    int16x8x4_t out_struct;
    out_struct.val[0] = outA;
    out_struct.val[1] = outB;
    out_struct.val[2] = outC;
    out_struct.val[3] = outD;
    vst4q_s16(out + dx * 4, out_struct);
  }
  for (; dx < dw; ++dx) {
    const uint8_t* P0 = src_row + col_ix0[dx];
    const uint8_t* P1 = src_row + col_ix1[dx];
    const uint8_t* P2 = src_row + col_ix2[dx];
    const uint8_t* P3 = src_row + col_ix3[dx];
    const int w0 = col_w0[dx], w1 = col_w1[dx], w2 = col_w2[dx], w3 = col_w3[dx];
    for (int ch = 0; ch < 4; ++ch) {
      const int acc = P0[ch]*w0 + P1[ch]*w1 + P2[ch]*w2 + P3[ch]*w3;
      out[dx*4 + ch] = static_cast<int16_t>((acc + 2) >> 2);
    }
  }
}

static void BicubicVertBlend(
    const int16_t* __restrict__ row0,
    const int16_t* __restrict__ row1,
    const int16_t* __restrict__ row2,
    const int16_t* __restrict__ row3,
    int16_t w0, int16_t w1, int16_t w2, int16_t w3,
    uint8_t* __restrict__ dest_row, int nbytes) {
  row0 = static_cast<const int16_t*>(__builtin_assume_aligned(row0, 16));
  row1 = static_cast<const int16_t*>(__builtin_assume_aligned(row1, 16));
  row2 = static_cast<const int16_t*>(__builtin_assume_aligned(row2, 16));
  row3 = static_cast<const int16_t*>(__builtin_assume_aligned(row3, 16));
  const int16x4_t vw0 = vdup_n_s16(w0);
  const int16x4_t vw1 = vdup_n_s16(w1);
  const int16x4_t vw2 = vdup_n_s16(w2);
  const int16x4_t vw3 = vdup_n_s16(w3);
  int i = 0;
  #pragma GCC ivdep
  for (; i + 8 <= nbytes; i += 8) {
    int16x4_t r0lo = vld1_s16(row0 + i),     r0hi = vld1_s16(row0 + i + 4);
    int16x4_t r1lo = vld1_s16(row1 + i),     r1hi = vld1_s16(row1 + i + 4);
    int16x4_t r2lo = vld1_s16(row2 + i),     r2hi = vld1_s16(row2 + i + 4);
    int16x4_t r3lo = vld1_s16(row3 + i),     r3hi = vld1_s16(row3 + i + 4);

    int32x4_t accLo = vmull_s16(r0lo, vw0);
    accLo = vmlal_s16(accLo, r1lo, vw1);
    accLo = vmlal_s16(accLo, r2lo, vw2);
    accLo = vmlal_s16(accLo, r3lo, vw3);

    int32x4_t accHi = vmull_s16(r0hi, vw0);
    accHi = vmlal_s16(accHi, r1hi, vw1);
    accHi = vmlal_s16(accHi, r2hi, vw2);
    accHi = vmlal_s16(accHi, r3hi, vw3);

    int16x4_t sLo = vrshrn_n_s32(accLo, 16);
    int16x4_t sHi = vrshrn_n_s32(accHi, 16);
    uint8x8_t result = vqmovun_s16(vcombine_s16(sLo, sHi));
    vst1_u8(dest_row + i, result);
  }
  for (; i < nbytes; ++i) {
    const int acc = static_cast<int>(row0[i]) * w0 +
                    static_cast<int>(row1[i]) * w1 +
                    static_cast<int>(row2[i]) * w2 +
                    static_cast<int>(row3[i]) * w3;
    const int val = (acc + 32768) >> 16;
    dest_row[i] = Clamp8(val);
  }
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
// ============ Bicubic horizontal ============
static constexpr int8_t  kCubicRelIx_2to3_c0[8] = { 0, 1, 1, 2, 3, 3, 4, 5 };
static constexpr int16_t kCubicW0_2to3_c0[8]    = { -4, -4, -1, -4, -4, -1, -4, -4 };
static constexpr int16_t kCubicW1_2to3_c0[8]    = { 36, 60, 8, 36, 60, 8, 36, 60 };
static constexpr int16_t kCubicW2_2to3_c0[8]    = { 36, 8, 60, 36, 8, 60, 36, 8 };
static constexpr int16_t kCubicW3_2to3_c0[8]    = { -4, -1, -4, -4, -1, -4, -4, -1 };
static constexpr int8_t  kCubicRelIx_2to3_c1[8] = { 0, 1, 2, 2, 3, 4, 4, 5 };
static constexpr int16_t kCubicW0_2to3_c1[8]    = { -1, -4, -4, -1, -4, -4, -1, -4 };
static constexpr int16_t kCubicW1_2to3_c1[8]    = { 8, 36, 60, 8, 36, 60, 8, 36 };
static constexpr int16_t kCubicW2_2to3_c1[8]    = { 60, 36, 8, 60, 36, 8, 60, 36 };
static constexpr int16_t kCubicW3_2to3_c1[8]    = { -4, -4, -1, -4, -4, -1, -4, -4 };
static constexpr int8_t  kCubicRelIx_2to3_c2[8] = { 0, 0, 1, 2, 2, 3, 4, 4 };
static constexpr int16_t kCubicW0_2to3_c2[8]    = { -4, -1, -4, -4, -1, -4, -4, -1 };
static constexpr int16_t kCubicW1_2to3_c2[8]    = { 60, 8, 36, 60, 8, 36, 60, 8 };
static constexpr int16_t kCubicW2_2to3_c2[8]    = { 8, 60, 36, 8, 60, 36, 8, 60 };
static constexpr int16_t kCubicW3_2to3_c2[8]    = { -1, -4, -4, -1, -4, -4, -1, -4 };
static constexpr int8_t  kCubicRelIx_1to2_c0[8] = { 0, 0, 1, 1, 2, 2, 3, 3 };
static constexpr int16_t kCubicW0_1to2_c0[8]    = { -5, -2, -5, -2, -5, -2, -5, -2 };
static constexpr int16_t kCubicW1_1to2_c0[8]    = { 56, 15, 56, 15, 56, 15, 56, 15 };
static constexpr int16_t kCubicW2_1to2_c0[8]    = { 15, 56, 15, 56, 15, 56, 15, 56 };
static constexpr int16_t kCubicW3_1to2_c0[8]    = { -2, -5, -2, -5, -2, -5, -2, -5 };
static constexpr int8_t  kCubicRelIx_1to2_c1[8] = { 0, 0, 1, 1, 2, 2, 3, 3 };
static constexpr int16_t kCubicW0_1to2_c1[8]    = { -5, -2, -5, -2, -5, -2, -5, -2 };
static constexpr int16_t kCubicW1_1to2_c1[8]    = { 56, 15, 56, 15, 56, 15, 56, 15 };
static constexpr int16_t kCubicW2_1to2_c1[8]    = { 15, 56, 15, 56, 15, 56, 15, 56 };
static constexpr int16_t kCubicW3_1to2_c1[8]    = { -2, -5, -2, -5, -2, -5, -2, -5 };
// ============ Vertical (shared bilinear Q8 / bicubic Q12) ============
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

    uint8x8_t lA=vdup_n_u8(0), lB=vdup_n_u8(0), lC=vdup_n_u8(0);
    uint8x8_t rA=vdup_n_u8(0), rB=vdup_n_u8(0), rC=vdup_n_u8(0);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      const uint8_t* L = src_row + (base_ix + rel[k]) * 3;
      const uint8_t* R = L + 3;
      lA = vld1_lane_u8(L+0, lA, k); lB = vld1_lane_u8(L+1, lB, k); lC = vld1_lane_u8(L+2, lC, k);
      rA = vld1_lane_u8(R+0, rA, k); rB = vld1_lane_u8(R+1, rB, k); rC = vld1_lane_u8(R+2, rC, k);
    }
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
    uint8x8_t lA=vdup_n_u8(0), lB=vdup_n_u8(0), lC=vdup_n_u8(0);
    uint8x8_t rA=vdup_n_u8(0), rB=vdup_n_u8(0), rC=vdup_n_u8(0);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      const uint8_t* L = src_row + (base_ix + rel[k]) * 3;
      const uint8_t* R = L + 3;
      lA = vld1_lane_u8(L+0, lA, k); lB = vld1_lane_u8(L+1, lB, k); lC = vld1_lane_u8(L+2, lC, k);
      rA = vld1_lane_u8(R+0, rA, k); rB = vld1_lane_u8(R+1, rB, k); rC = vld1_lane_u8(R+2, rC, k);
    }
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

    uint8x8_t lA=vdup_n_u8(0),lB=vdup_n_u8(0),lC=vdup_n_u8(0),lD=vdup_n_u8(0);
    uint8x8_t rA=vdup_n_u8(0),rB=vdup_n_u8(0),rC=vdup_n_u8(0),rD=vdup_n_u8(0);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      const uint8_t* L = src_row + (base_ix + rel[k]) * 4;
      const uint8_t* R = L + 4;
      lA=vld1_lane_u8(L+0,lA,k); lB=vld1_lane_u8(L+1,lB,k); lC=vld1_lane_u8(L+2,lC,k); lD=vld1_lane_u8(L+3,lD,k);
      rA=vld1_lane_u8(R+0,rA,k); rB=vld1_lane_u8(R+1,rB,k); rC=vld1_lane_u8(R+2,rC,k); rD=vld1_lane_u8(R+3,rD,k);
    }
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
    uint8x8_t lA=vdup_n_u8(0),lB=vdup_n_u8(0),lC=vdup_n_u8(0),lD=vdup_n_u8(0);
    uint8x8_t rA=vdup_n_u8(0),rB=vdup_n_u8(0),rC=vdup_n_u8(0),rD=vdup_n_u8(0);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      const uint8_t* L = src_row + (base_ix + rel[k]) * 4;
      const uint8_t* R = L + 4;
      lA=vld1_lane_u8(L+0,lA,k); lB=vld1_lane_u8(L+1,lB,k); lC=vld1_lane_u8(L+2,lC,k); lD=vld1_lane_u8(L+3,lD,k);
      rA=vld1_lane_u8(R+0,rA,k); rB=vld1_lane_u8(R+1,rB,k); rC=vld1_lane_u8(R+2,rC,k); rD=vld1_lane_u8(R+3,rD,k);
    }
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

// ---- Bicubic horizontal fast path, depth=3 ----
//
// Clamped scalar single-pixel helper for the head (2 or 3 pixels, ratio
// dependent -- see the note above the constants block for why bicubic's
// 4-tap kernel needs more than bilinear's 1-pixel head) and tail regions.
// This is cold code (at most 3 pixels per row), so a plain float weight
// computation here is fine -- it mirrors the general path's own per-pixel
// formula exactly, just evaluated at a handful of positions instead of the
// whole row.
static inline void BicubicClampedPixel3(
    const uint8_t* __restrict__ src_row, int sw, int dx, int p, int q,
    int16_t* __restrict__ out3) {
  const int num = (2*dx + 1)*p - q;
  const int twoq = 2*q;
  int ix = num / twoq;
  int rem = num - ix*twoq;
  if (rem < 0) { ix -= 1; rem += twoq; }   // true floor division (num can be
                                             // negative at dx=0)
  if (ix < 0) { ix = 0; rem = 0; }

  const float t = static_cast<float>(rem) / static_cast<float>(twoq);
  const float t2 = t*t, t3 = t2*t;
  const float fw0 = -0.5f*t3 +      t2 - 0.5f*t;
  const float fw1 =  1.5f*t3 - 2.5f*t2          + 1.0f;
  const float fw2 = -1.5f*t3 + 2.0f*t2 + 0.5f*t;
  const float fw3 =  0.5f*t3 - 0.5f*t2;
  const int16_t w0 = static_cast<int16_t>(std::lround(fw0 * 64.0f));
  const int16_t w1 = static_cast<int16_t>(std::lround(fw1 * 64.0f));
  const int16_t w2 = static_cast<int16_t>(std::lround(fw2 * 64.0f));
  const int16_t w3 = static_cast<int16_t>(std::lround(fw3 * 64.0f));

  const int ix_max = sw - 1;
  auto clampx = [ix_max](int v) { return v < 0 ? 0 : (v > ix_max ? ix_max : v); };
  const int i0 = clampx(ix - 1), i1 = clampx(ix), i2 = clampx(ix + 1), i3 = clampx(ix + 2);
  for (int ch = 0; ch < 3; ++ch) {
    const int acc = src_row[i0*3+ch]*w0 + src_row[i1*3+ch]*w1 +
                    src_row[i2*3+ch]*w2 + src_row[i3*3+ch]*w3;
    out3[ch] = static_cast<int16_t>((acc + 2) >> 2);
  }
}

static void BicubicHorizRowFast3_2to3(
    const uint8_t* __restrict__ src_row, int sw,
    int16_t* __restrict__ out, int dw) {
  out = static_cast<int16_t*>(__builtin_assume_aligned(out, 16));

  // Head: dx=0,1 both need left-edge clamping for this 4-tap kernel.
  BicubicClampedPixel3(src_row, sw, 0, 2, 3, out + 0*3);
  BicubicClampedPixel3(src_row, sw, 1, 2, 3, out + 1*3);

  int dx = 2;
  int chunk = 2;  // verified starting chunk for a 2-pixel head at ratio 2:3
  while (dx + 16 <= dw) {
    const int8_t* rel; const int16_t *w0t,*w1t,*w2t,*w3t;
    if (chunk == 0)      { rel=kCubicRelIx_2to3_c0; w0t=kCubicW0_2to3_c0; w1t=kCubicW1_2to3_c0; w2t=kCubicW2_2to3_c0; w3t=kCubicW3_2to3_c0; }
    else if (chunk == 1) { rel=kCubicRelIx_2to3_c1; w0t=kCubicW0_2to3_c1; w1t=kCubicW1_2to3_c1; w2t=kCubicW2_2to3_c1; w3t=kCubicW3_2to3_c1; }
    else                 { rel=kCubicRelIx_2to3_c2; w0t=kCubicW0_2to3_c2; w1t=kCubicW1_2to3_c2; w2t=kCubicW2_2to3_c2; w3t=kCubicW3_2to3_c2; }
    const int base_ix = (4*dx - 1) / 6;

    uint8x8_t t0A=vdup_n_u8(0),t0B=vdup_n_u8(0),t0C=vdup_n_u8(0);
    uint8x8_t t1A=vdup_n_u8(0),t1B=vdup_n_u8(0),t1C=vdup_n_u8(0);
    uint8x8_t t2A=vdup_n_u8(0),t2B=vdup_n_u8(0),t2C=vdup_n_u8(0);
    uint8x8_t t3A=vdup_n_u8(0),t3B=vdup_n_u8(0),t3C=vdup_n_u8(0);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      const uint8_t* P0 = src_row + (base_ix + rel[k] - 1) * 3;
      const uint8_t* P1 = P0 + 3, *P2 = P1 + 3, *P3 = P2 + 3;
      t0A=vld1_lane_u8(P0+0,t0A,k); t0B=vld1_lane_u8(P0+1,t0B,k); t0C=vld1_lane_u8(P0+2,t0C,k);
      t1A=vld1_lane_u8(P1+0,t1A,k); t1B=vld1_lane_u8(P1+1,t1B,k); t1C=vld1_lane_u8(P1+2,t1C,k);
      t2A=vld1_lane_u8(P2+0,t2A,k); t2B=vld1_lane_u8(P2+1,t2B,k); t2C=vld1_lane_u8(P2+2,t2C,k);
      t3A=vld1_lane_u8(P3+0,t3A,k); t3B=vld1_lane_u8(P3+1,t3B,k); t3C=vld1_lane_u8(P3+2,t3C,k);
    }
    int16x8_t w0=vld1q_s16(w0t), w1=vld1q_s16(w1t), w2=vld1q_s16(w2t), w3=vld1q_s16(w3t);

    #define BICUBIC_TAP(CH, OUTVAR) \
      do { \
        int16x8_t p0 = vreinterpretq_s16_u16(vmovl_u8(t0##CH)); \
        int16x8_t p1 = vreinterpretq_s16_u16(vmovl_u8(t1##CH)); \
        int16x8_t p2 = vreinterpretq_s16_u16(vmovl_u8(t2##CH)); \
        int16x8_t p3 = vreinterpretq_s16_u16(vmovl_u8(t3##CH)); \
        int16x8_t acc = vmulq_s16(p0, w0); \
        acc = vmlaq_s16(acc, p1, w1); \
        acc = vmlaq_s16(acc, p2, w2); \
        acc = vmlaq_s16(acc, p3, w3); \
        OUTVAR = vrshrq_n_s16(acc, 2); \
      } while (0)

    int16x8_t outA, outB, outC;
    BICUBIC_TAP(A, outA); BICUBIC_TAP(B, outB); BICUBIC_TAP(C, outC);
    #undef BICUBIC_TAP

    int16_t bufA[8], bufB[8], bufC[8];
    vst1q_s16(bufA, outA); vst1q_s16(bufB, outB); vst1q_s16(bufC, outC);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      out[(dx+k)*3+0] = bufA[k];
      out[(dx+k)*3+1] = bufB[k];
      out[(dx+k)*3+2] = bufC[k];
    }
    dx += 8; chunk = (chunk + 1) % 3;
  }
  for (; dx < dw; ++dx)
    BicubicClampedPixel3(src_row, sw, dx, 2, 3, out + dx*3);
}

static void BicubicHorizRowFast3_1to2(
    const uint8_t* __restrict__ src_row, int sw,
    int16_t* __restrict__ out, int dw) {
  out = static_cast<int16_t*>(__builtin_assume_aligned(out, 16));

  BicubicClampedPixel3(src_row, sw, 0, 1, 2, out + 0*3);
  BicubicClampedPixel3(src_row, sw, 1, 1, 2, out + 1*3);
  BicubicClampedPixel3(src_row, sw, 2, 1, 2, out + 2*3);

  int dx = 3;  // chunk 0 works for any head length at this ratio (both
               // chunks are identical -- see constants block)
  const int8_t* rel = kCubicRelIx_1to2_c0;
  const int16_t *w0t=kCubicW0_1to2_c0, *w1t=kCubicW1_1to2_c0, *w2t=kCubicW2_1to2_c0, *w3t=kCubicW3_1to2_c0;
  while (dx + 16 <= dw) {
    const int base_ix = (2*dx - 1) / 4;
    uint8x8_t t0A=vdup_n_u8(0),t0B=vdup_n_u8(0),t0C=vdup_n_u8(0);
    uint8x8_t t1A=vdup_n_u8(0),t1B=vdup_n_u8(0),t1C=vdup_n_u8(0);
    uint8x8_t t2A=vdup_n_u8(0),t2B=vdup_n_u8(0),t2C=vdup_n_u8(0);
    uint8x8_t t3A=vdup_n_u8(0),t3B=vdup_n_u8(0),t3C=vdup_n_u8(0);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      const uint8_t* P0 = src_row + (base_ix + rel[k] - 1) * 3;
      const uint8_t* P1 = P0 + 3, *P2 = P1 + 3, *P3 = P2 + 3;
      t0A=vld1_lane_u8(P0+0,t0A,k); t0B=vld1_lane_u8(P0+1,t0B,k); t0C=vld1_lane_u8(P0+2,t0C,k);
      t1A=vld1_lane_u8(P1+0,t1A,k); t1B=vld1_lane_u8(P1+1,t1B,k); t1C=vld1_lane_u8(P1+2,t1C,k);
      t2A=vld1_lane_u8(P2+0,t2A,k); t2B=vld1_lane_u8(P2+1,t2B,k); t2C=vld1_lane_u8(P2+2,t2C,k);
      t3A=vld1_lane_u8(P3+0,t3A,k); t3B=vld1_lane_u8(P3+1,t3B,k); t3C=vld1_lane_u8(P3+2,t3C,k);
    }
    int16x8_t w0=vld1q_s16(w0t), w1=vld1q_s16(w1t), w2=vld1q_s16(w2t), w3=vld1q_s16(w3t);
    #define BICUBIC_TAP(CH, OUTVAR) \
      do { \
        int16x8_t p0 = vreinterpretq_s16_u16(vmovl_u8(t0##CH)); \
        int16x8_t p1 = vreinterpretq_s16_u16(vmovl_u8(t1##CH)); \
        int16x8_t p2 = vreinterpretq_s16_u16(vmovl_u8(t2##CH)); \
        int16x8_t p3 = vreinterpretq_s16_u16(vmovl_u8(t3##CH)); \
        int16x8_t acc = vmulq_s16(p0, w0); \
        acc = vmlaq_s16(acc, p1, w1); \
        acc = vmlaq_s16(acc, p2, w2); \
        acc = vmlaq_s16(acc, p3, w3); \
        OUTVAR = vrshrq_n_s16(acc, 2); \
      } while (0)
    int16x8_t outA, outB, outC;
    BICUBIC_TAP(A, outA); BICUBIC_TAP(B, outB); BICUBIC_TAP(C, outC);
    #undef BICUBIC_TAP
    int16_t bufA[8], bufB[8], bufC[8];
    vst1q_s16(bufA, outA); vst1q_s16(bufB, outB); vst1q_s16(bufC, outC);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      out[(dx+k)*3+0] = bufA[k];
      out[(dx+k)*3+1] = bufB[k];
      out[(dx+k)*3+2] = bufC[k];
    }
    dx += 8;
  }
  for (; dx < dw; ++dx)
    BicubicClampedPixel3(src_row, sw, dx, 1, 2, out + dx*3);
}

// ---- Bicubic horizontal fast path, depth=4 ----
static inline void BicubicClampedPixel4(
    const uint8_t* __restrict__ src_row, int sw, int dx, int p, int q,
    int16_t* __restrict__ out4) {
  const int num = (2*dx + 1)*p - q;
  const int twoq = 2*q;
  int ix = num / twoq;
  int rem = num - ix*twoq;
  if (rem < 0) { ix -= 1; rem += twoq; }
  if (ix < 0) { ix = 0; rem = 0; }

  const float t = static_cast<float>(rem) / static_cast<float>(twoq);
  const float t2 = t*t, t3 = t2*t;
  const float fw0 = -0.5f*t3 +      t2 - 0.5f*t;
  const float fw1 =  1.5f*t3 - 2.5f*t2          + 1.0f;
  const float fw2 = -1.5f*t3 + 2.0f*t2 + 0.5f*t;
  const float fw3 =  0.5f*t3 - 0.5f*t2;
  const int16_t w0 = static_cast<int16_t>(std::lround(fw0 * 64.0f));
  const int16_t w1 = static_cast<int16_t>(std::lround(fw1 * 64.0f));
  const int16_t w2 = static_cast<int16_t>(std::lround(fw2 * 64.0f));
  const int16_t w3 = static_cast<int16_t>(std::lround(fw3 * 64.0f));

  const int ix_max = sw - 1;
  auto clampx = [ix_max](int v) { return v < 0 ? 0 : (v > ix_max ? ix_max : v); };
  const int i0 = clampx(ix - 1), i1 = clampx(ix), i2 = clampx(ix + 1), i3 = clampx(ix + 2);
  for (int ch = 0; ch < 4; ++ch) {
    const int acc = src_row[i0*4+ch]*w0 + src_row[i1*4+ch]*w1 +
                    src_row[i2*4+ch]*w2 + src_row[i3*4+ch]*w3;
    out4[ch] = static_cast<int16_t>((acc + 2) >> 2);
  }
}

static void BicubicHorizRowFast4_2to3(
    const uint8_t* __restrict__ src_row, int sw,
    int16_t* __restrict__ out, int dw) {
  out = static_cast<int16_t*>(__builtin_assume_aligned(out, 16));
  BicubicClampedPixel4(src_row, sw, 0, 2, 3, out + 0*4);
  BicubicClampedPixel4(src_row, sw, 1, 2, 3, out + 1*4);

  int dx = 2, chunk = 2;
  while (dx + 16 <= dw) {
    const int8_t* rel; const int16_t *w0t,*w1t,*w2t,*w3t;
    if (chunk == 0)      { rel=kCubicRelIx_2to3_c0; w0t=kCubicW0_2to3_c0; w1t=kCubicW1_2to3_c0; w2t=kCubicW2_2to3_c0; w3t=kCubicW3_2to3_c0; }
    else if (chunk == 1) { rel=kCubicRelIx_2to3_c1; w0t=kCubicW0_2to3_c1; w1t=kCubicW1_2to3_c1; w2t=kCubicW2_2to3_c1; w3t=kCubicW3_2to3_c1; }
    else                 { rel=kCubicRelIx_2to3_c2; w0t=kCubicW0_2to3_c2; w1t=kCubicW1_2to3_c2; w2t=kCubicW2_2to3_c2; w3t=kCubicW3_2to3_c2; }
    const int base_ix = (4*dx - 1) / 6;

    uint8x8_t t0A=vdup_n_u8(0),t0B=vdup_n_u8(0),t0C=vdup_n_u8(0),t0D=vdup_n_u8(0);
    uint8x8_t t1A=vdup_n_u8(0),t1B=vdup_n_u8(0),t1C=vdup_n_u8(0),t1D=vdup_n_u8(0);
    uint8x8_t t2A=vdup_n_u8(0),t2B=vdup_n_u8(0),t2C=vdup_n_u8(0),t2D=vdup_n_u8(0);
    uint8x8_t t3A=vdup_n_u8(0),t3B=vdup_n_u8(0),t3C=vdup_n_u8(0),t3D=vdup_n_u8(0);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      const uint8_t* P0 = src_row + (base_ix + rel[k] - 1) * 4;
      const uint8_t* P1 = P0 + 4, *P2 = P1 + 4, *P3 = P2 + 4;
      t0A=vld1_lane_u8(P0+0,t0A,k); t0B=vld1_lane_u8(P0+1,t0B,k); t0C=vld1_lane_u8(P0+2,t0C,k); t0D=vld1_lane_u8(P0+3,t0D,k);
      t1A=vld1_lane_u8(P1+0,t1A,k); t1B=vld1_lane_u8(P1+1,t1B,k); t1C=vld1_lane_u8(P1+2,t1C,k); t1D=vld1_lane_u8(P1+3,t1D,k);
      t2A=vld1_lane_u8(P2+0,t2A,k); t2B=vld1_lane_u8(P2+1,t2B,k); t2C=vld1_lane_u8(P2+2,t2C,k); t2D=vld1_lane_u8(P2+3,t2D,k);
      t3A=vld1_lane_u8(P3+0,t3A,k); t3B=vld1_lane_u8(P3+1,t3B,k); t3C=vld1_lane_u8(P3+2,t3C,k); t3D=vld1_lane_u8(P3+3,t3D,k);
    }
    int16x8_t w0=vld1q_s16(w0t), w1=vld1q_s16(w1t), w2=vld1q_s16(w2t), w3=vld1q_s16(w3t);
    #define BICUBIC_TAP4(CH, OUTVAR) \
      do { \
        int16x8_t p0 = vreinterpretq_s16_u16(vmovl_u8(t0##CH)); \
        int16x8_t p1 = vreinterpretq_s16_u16(vmovl_u8(t1##CH)); \
        int16x8_t p2 = vreinterpretq_s16_u16(vmovl_u8(t2##CH)); \
        int16x8_t p3 = vreinterpretq_s16_u16(vmovl_u8(t3##CH)); \
        int16x8_t acc = vmulq_s16(p0, w0); \
        acc = vmlaq_s16(acc, p1, w1); \
        acc = vmlaq_s16(acc, p2, w2); \
        acc = vmlaq_s16(acc, p3, w3); \
        OUTVAR = vrshrq_n_s16(acc, 2); \
      } while (0)
    int16x8_t outA, outB, outC, outD;
    BICUBIC_TAP4(A,outA); BICUBIC_TAP4(B,outB); BICUBIC_TAP4(C,outC); BICUBIC_TAP4(D,outD);
    #undef BICUBIC_TAP4
    int16_t bufA[8],bufB[8],bufC[8],bufD[8];
    vst1q_s16(bufA,outA); vst1q_s16(bufB,outB); vst1q_s16(bufC,outC); vst1q_s16(bufD,outD);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      out[(dx+k)*4+0]=bufA[k]; out[(dx+k)*4+1]=bufB[k];
      out[(dx+k)*4+2]=bufC[k]; out[(dx+k)*4+3]=bufD[k];
    }
    dx += 8; chunk = (chunk + 1) % 3;
  }
  for (; dx < dw; ++dx)
    BicubicClampedPixel4(src_row, sw, dx, 2, 3, out + dx*4);
}

static void BicubicHorizRowFast4_1to2(
    const uint8_t* __restrict__ src_row, int sw,
    int16_t* __restrict__ out, int dw) {
  out = static_cast<int16_t*>(__builtin_assume_aligned(out, 16));
  BicubicClampedPixel4(src_row, sw, 0, 1, 2, out + 0*4);
  BicubicClampedPixel4(src_row, sw, 1, 1, 2, out + 1*4);
  BicubicClampedPixel4(src_row, sw, 2, 1, 2, out + 2*4);

  int dx = 3;
  const int8_t* rel = kCubicRelIx_1to2_c0;
  const int16_t *w0t=kCubicW0_1to2_c0, *w1t=kCubicW1_1to2_c0, *w2t=kCubicW2_1to2_c0, *w3t=kCubicW3_1to2_c0;
  while (dx + 16 <= dw) {
    const int base_ix = (2*dx - 1) / 4;
    uint8x8_t t0A=vdup_n_u8(0),t0B=vdup_n_u8(0),t0C=vdup_n_u8(0),t0D=vdup_n_u8(0);
    uint8x8_t t1A=vdup_n_u8(0),t1B=vdup_n_u8(0),t1C=vdup_n_u8(0),t1D=vdup_n_u8(0);
    uint8x8_t t2A=vdup_n_u8(0),t2B=vdup_n_u8(0),t2C=vdup_n_u8(0),t2D=vdup_n_u8(0);
    uint8x8_t t3A=vdup_n_u8(0),t3B=vdup_n_u8(0),t3C=vdup_n_u8(0),t3D=vdup_n_u8(0);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      const uint8_t* P0 = src_row + (base_ix + rel[k] - 1) * 4;
      const uint8_t* P1 = P0 + 4, *P2 = P1 + 4, *P3 = P2 + 4;
      t0A=vld1_lane_u8(P0+0,t0A,k); t0B=vld1_lane_u8(P0+1,t0B,k); t0C=vld1_lane_u8(P0+2,t0C,k); t0D=vld1_lane_u8(P0+3,t0D,k);
      t1A=vld1_lane_u8(P1+0,t1A,k); t1B=vld1_lane_u8(P1+1,t1B,k); t1C=vld1_lane_u8(P1+2,t1C,k); t1D=vld1_lane_u8(P1+3,t1D,k);
      t2A=vld1_lane_u8(P2+0,t2A,k); t2B=vld1_lane_u8(P2+1,t2B,k); t2C=vld1_lane_u8(P2+2,t2C,k); t2D=vld1_lane_u8(P2+3,t2D,k);
      t3A=vld1_lane_u8(P3+0,t3A,k); t3B=vld1_lane_u8(P3+1,t3B,k); t3C=vld1_lane_u8(P3+2,t3C,k); t3D=vld1_lane_u8(P3+3,t3D,k);
    }
    int16x8_t w0=vld1q_s16(w0t), w1=vld1q_s16(w1t), w2=vld1q_s16(w2t), w3=vld1q_s16(w3t);
    #define BICUBIC_TAP4(CH, OUTVAR) \
      do { \
        int16x8_t p0 = vreinterpretq_s16_u16(vmovl_u8(t0##CH)); \
        int16x8_t p1 = vreinterpretq_s16_u16(vmovl_u8(t1##CH)); \
        int16x8_t p2 = vreinterpretq_s16_u16(vmovl_u8(t2##CH)); \
        int16x8_t p3 = vreinterpretq_s16_u16(vmovl_u8(t3##CH)); \
        int16x8_t acc = vmulq_s16(p0, w0); \
        acc = vmlaq_s16(acc, p1, w1); \
        acc = vmlaq_s16(acc, p2, w2); \
        acc = vmlaq_s16(acc, p3, w3); \
        OUTVAR = vrshrq_n_s16(acc, 2); \
      } while (0)
    int16x8_t outA, outB, outC, outD;
    BICUBIC_TAP4(A,outA); BICUBIC_TAP4(B,outB); BICUBIC_TAP4(C,outC); BICUBIC_TAP4(D,outD);
    #undef BICUBIC_TAP4
    int16_t bufA[8],bufB[8],bufC[8],bufD[8];
    vst1q_s16(bufA,outA); vst1q_s16(bufB,outB); vst1q_s16(bufC,outC); vst1q_s16(bufD,outD);
    #pragma GCC unroll 8
    for (int k = 0; k < 8; ++k) {
      out[(dx+k)*4+0]=bufA[k]; out[(dx+k)*4+1]=bufB[k];
      out[(dx+k)*4+2]=bufC[k]; out[(dx+k)*4+3]=bufD[k];
    }
    dx += 8;
  }
  for (; dx < dw; ++dx)
    BicubicClampedPixel4(src_row, sw, dx, 1, 2, out + dx*4);
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
static inline void BilinearVertRowFast(
    int dy, int sh, int p, int q, int* iy0, int* iy1, uint8_t* fy8) {
  const int num = (2*dy + 1)*p - q;
  const int twoq = 2*q;
  int iy = num / twoq;
  int rem = num - iy*twoq;
  if (rem < 0) { iy -= 1; rem += twoq; }
  if (iy < 0) { iy = 0; rem = 0; }
  const int iy_max = sh - 1;
  *iy0 = (iy < iy_max) ? iy : iy_max;
  *iy1 = (iy < iy_max) ? iy + 1 : iy_max;
  if (iy > iy_max) *iy0 = *iy1 = iy_max;
  const float t = static_cast<float>(rem) / static_cast<float>(twoq);
  *fy8 = static_cast<uint8_t>(t * 255.0f + 0.5f);
}

static inline void BicubicVertRowFast(
    int dy, int sh, int p, int q,
    int* iy0, int* iy1, int* iy2, int* iy3,
    int16_t* w0, int16_t* w1, int16_t* w2, int16_t* w3) {
  const int num = (2*dy + 1)*p - q;
  const int twoq = 2*q;
  int iy = num / twoq;
  int rem = num - iy*twoq;
  if (rem < 0) { iy -= 1; rem += twoq; }
  if (iy < 0) { iy = 0; rem = 0; }
  const float t = static_cast<float>(rem) / static_cast<float>(twoq);
  const float t2 = t*t, t3 = t2*t;
  const float fw0 = -0.5f*t3 +      t2 - 0.5f*t;
  const float fw1 =  1.5f*t3 - 2.5f*t2          + 1.0f;
  const float fw2 = -1.5f*t3 + 2.0f*t2 + 0.5f*t;
  const float fw3 =  0.5f*t3 - 0.5f*t2;
  *w0 = static_cast<int16_t>(std::lround(fw0 * 4096.0f));
  *w1 = static_cast<int16_t>(std::lround(fw1 * 4096.0f));
  *w2 = static_cast<int16_t>(std::lround(fw2 * 4096.0f));
  *w3 = static_cast<int16_t>(std::lround(fw3 * 4096.0f));
  const int iy_max = sh - 1;
  auto clampy = [iy_max](int v) { return v < 0 ? 0 : (v > iy_max ? iy_max : v); };
  *iy0 = clampy(iy - 1); *iy1 = clampy(iy); *iy2 = clampy(iy + 1); *iy3 = clampy(iy + 2);
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

// ---- Fast-path row-cache slot selector shared by bilinear and bicubic
// dispatch below: given a newly-needed source row index, either reuse an
// already-computed slot or recompute into the least-recently-replaced one.
// Mirrors the general path's ensure_hrow but simplified since the fast
// path's row advance pattern means at most 1 new row is ever needed per
// output row (the others are always carried over from the previous call).

// =============================================================================
// Fast-path Copy() dispatch for BILINEAR. Activates only when BOTH axes are
// an exact 2:3 or 1:2 ratio and no sharpening is requested (sharpening
// remains general-path only in this revision). Falls through to the
// existing general path otherwise, on either axis independently.
// =============================================================================
static bool CopyBilinearFastPath(
    const uint8_t* src_base, int src_stride,
    uint8_t* dest_base, int dest_stride, int depth,
    int sw, int sh, int dw, int dh,
    int dest_rect_x, int dest_rect_y) {
  const FastRatio hr = ClassifyRatio(sw, dw);
  const FastRatio vr = ClassifyRatio(sh, dh);
  if (hr == FastRatio::kGeneral || vr == FastRatio::kGeneral) return false;
  if (depth != 3 && depth != 4) return false;

  const int v_p = (vr == FastRatio::k1to2) ? 1 : 2;
  const int v_q = (vr == FastRatio::k1to2) ? 2 : 3;

  ExecuteInParallel([=](int num_threads, int i) {
    const int rows_per = dh / num_threads;
    const int height = (i == num_threads - 1) ? (dh - i * rows_per) : rows_per;
    const int dy0 = i * rows_per;
    const int row_bytes = dw * depth;

    const size_t row_stride = (static_cast<size_t>(row_bytes) + 15) & ~size_t(15);
    void* raw_ptr = nullptr;
    posix_memalign(&raw_ptr, 16, row_stride * 2);
    std::unique_ptr<uint8_t, void(*)(void*)> hbuf(static_cast<uint8_t*>(raw_ptr), std::free);
    uint8_t* hrow[2] = { hbuf.get(), hbuf.get() + row_stride };
    int cached_iy[2] = { -1, -1 };

    auto ensure_row = [&](int slot, int iy) {
      if (cached_iy[slot] == iy) return;
      const uint8_t* src_row = src_base + iy * src_stride;
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
      int iy0, iy1; uint8_t fy8;
      BilinearVertRowFast(dy, sh, v_p, v_q, &iy0, &iy1, &fy8);
      ensure_row(0, iy0);
      ensure_row(1, iy1);
      uint8_t* dest_row = dest_base + (dest_rect_y + dy) * dest_stride + dest_rect_x * depth;
      if (iy0 == iy1) {
        memcpy(dest_row, hrow[0], row_bytes);
      } else if (depth == 3) {
        BilinearVertBlend3(hrow[0], hrow[1], fy8, dest_row, dw);
      } else {
        BilinearVertBlend4(hrow[0], hrow[1], fy8, dest_row, dw);
      }
    }
  }, GetEffectiveNumThreads());
  return true;
}

// =============================================================================
// Fast-path Copy() dispatch for BICUBIC. Same activation conditions as the
// bilinear fast path above (both axes exact 2:3/1:2; sharpening is not a
// bicubic concept in this codebase so no extra guard is needed there).
// =============================================================================
static bool CopyBicubicFastPath(
    const uint8_t* src_base, int src_stride,
    uint8_t* dest_base, int dest_stride, int depth,
    int sw, int sh, int dw, int dh,
    int dest_rect_x, int dest_rect_y) {
  const FastRatio hr = ClassifyRatio(sw, dw);
  const FastRatio vr = ClassifyRatio(sh, dh);
  if (hr == FastRatio::kGeneral || vr == FastRatio::kGeneral) return false;
  if (depth != 3 && depth != 4) return false;

  const int v_p = (vr == FastRatio::k1to2) ? 1 : 2;
  const int v_q = (vr == FastRatio::k1to2) ? 2 : 3;

  ExecuteInParallel([=](int num_threads, int i) {
    const int rows_per = dh / num_threads;
    const int height = (i == num_threads - 1) ? (dh - i * rows_per) : rows_per;
    const int dy0 = i * rows_per;
    const int dw_elems = dw * depth;

    const size_t row_stride = (static_cast<size_t>(dw_elems) * sizeof(int16_t) + 15) & ~size_t(15);
    void* raw_ptr = nullptr;
    posix_memalign(&raw_ptr, 16, row_stride * 4);
    std::unique_ptr<uint8_t, void(*)(void*)> hbuf(static_cast<uint8_t*>(raw_ptr), std::free);
    int16_t* hrow[4] = {
      reinterpret_cast<int16_t*>(hbuf.get()),
      reinterpret_cast<int16_t*>(hbuf.get() + row_stride),
      reinterpret_cast<int16_t*>(hbuf.get() + row_stride * 2),
      reinterpret_cast<int16_t*>(hbuf.get() + row_stride * 3),
    };
    int cached_iy[4] = { -1, -1, -1, -1 };

    auto ensure_row = [&](int slot, int iy) {
      if (cached_iy[slot] == iy) return;
      const uint8_t* src_row = src_base + iy * src_stride;
      if (depth == 3) {
        if (hr == FastRatio::k2to3) BicubicHorizRowFast3_2to3(src_row, sw, hrow[slot], dw);
        else                        BicubicHorizRowFast3_1to2(src_row, sw, hrow[slot], dw);
      } else {
        if (hr == FastRatio::k2to3) BicubicHorizRowFast4_2to3(src_row, sw, hrow[slot], dw);
        else                        BicubicHorizRowFast4_1to2(src_row, sw, hrow[slot], dw);
      }
      cached_iy[slot] = iy;
    };

    for (int dy_local = 0; dy_local < height; ++dy_local) {
      const int dy = dy0 + dy_local;
      int iy0, iy1, iy2, iy3; int16_t w0, w1, w2, w3;
      BicubicVertRowFast(dy, sh, v_p, v_q, &iy0, &iy1, &iy2, &iy3, &w0, &w1, &w2, &w3);
      ensure_row(0, iy0); ensure_row(1, iy1); ensure_row(2, iy2); ensure_row(3, iy3);
      uint8_t* dest_row = dest_base + (dest_rect_y + dy) * dest_stride + dest_rect_x * depth;
      // No "rows match, just copy" shortcut here: unlike bilinear's 2-tap
      // kernel (where iy0==iy1 forces the blend to reduce to an exact
      // copy), bicubic's 4-tap weights can still assign real, nonzero
      // weight to iy0/iy3 even when iy1==iy2 -- e.g. at the very last
      // output row of a clamped bottom edge, iy1 and iy2 can both clamp to
      // the same iy_max while iy0 is still a distinct, legitimately-
      // weighted row. Always calling the real blend handles every case
      // correctly, including true single-source-row degeneracy (there all
      // four indices collapse to the same row, and the weighted sum
      // reduces to that row's value on its own, since the weights always
      // sum to exactly 1).
      BicubicVertBlend(hrow[0], hrow[1], hrow[2], hrow[3], w0, w1, w2, w3, dest_row, dw_elems);
    }
  }, GetEffectiveNumThreads());
  return true;
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
      }, GetEffectiveNumThreads());
      break;
    }

    case SCALE_BILINEAR: {
#ifdef __ARM_NEON
      if (valid_sharp_depth) {
        // Fast path: exact 2:3 or 1:2 ratio on both axes, no sharpening.
        // See CopyBilinearFastPath for the polyphase derivation. Falls
        // through to the general path below when either axis doesn't
        // match, or sharpening is requested.
        if (sharpen_strength == 0 &&
            CopyBilinearFastPath(src_base, src_stride, dest_base, dest_stride,
                                  depth, sw, sh, dw, dh,
                                  dest_rect.X, dest_rect.Y)) {
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

        const uint8_t sharp_strength = valid_sharp_depth && sharpen_strength > 0 ?
            16u << static_cast<uint8_t>(std::min(std::max(1, static_cast<int>(sharpen_strength)), 3)):
            0u;

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

    case SCALE_BICUBIC: {
#ifdef __ARM_NEON
      if (valid_sharp_depth) {
        // Fast path: exact 2:3 or 1:2 ratio on both axes. Falls through to
        // the general path below when either axis doesn't match.
        if (CopyBicubicFastPath(src_base, src_stride, dest_base, dest_stride,
                                 depth, sw, sh, dw, dh,
                                 dest_rect.X, dest_rect.Y)) {
          break;
        }

        std::vector<uint16_t> col_ix0(dw), col_ix1(dw), col_ix2(dw), col_ix3(dw);
        std::vector<int16_t>  col_w0(dw), col_w1(dw), col_w2(dw), col_w3(dw);
        for (int dx = 0; dx < dw; ++dx) {
          const float sx = src_rect.X +
              (dx + 0.5f) * static_cast<float>(sw) / static_cast<float>(dw)
              - 0.5f;
          const int ix = static_cast<int>(std::floor(sx));
          const float t = sx - ix;
          const float t2 = t * t, t3 = t2 * t;
          const float fw0 = -0.5f*t3 +      t2 - 0.5f*t;
          const float fw1 =  1.5f*t3 - 2.5f*t2          + 1.0f;
          const float fw2 = -1.5f*t3 + 2.0f*t2 + 0.5f*t;
          const float fw3 =  0.5f*t3 - 0.5f*t2;
          const int ix_min = src_rect.X;
          const int ix_max = src_rect.X + sw - 1;
          auto clampx = [&](int v) {
            return v < ix_min ? ix_min : (v > ix_max ? ix_max : v);
          };
          col_ix0[dx] = static_cast<uint16_t>(clampx(ix - 1) * depth);
          col_ix1[dx] = static_cast<uint16_t>(clampx(ix    ) * depth);
          col_ix2[dx] = static_cast<uint16_t>(clampx(ix + 1) * depth);
          col_ix3[dx] = static_cast<uint16_t>(clampx(ix + 2) * depth);

          // Opt #4: Horizontal Q6 Fixed Point (64.0f) scaling instead of Q12
          col_w0[dx] = static_cast<int16_t>(std::lround(fw0 * 64.0f));
          col_w1[dx] = static_cast<int16_t>(std::lround(fw1 * 64.0f));
          col_w2[dx] = static_cast<int16_t>(std::lround(fw2 * 64.0f));
          col_w3[dx] = static_cast<int16_t>(std::lround(fw3 * 64.0f));
        }

        const int ry_min = src_rect.Y;
        const int ry_max = src_rect.Y + sh - 1;
        auto clamp_ry = [&](int v) {
          return v < ry_min ? ry_min : (v > ry_max ? ry_max : v);
        };
        struct RowEntry {
          int r0, r1, r2, r3;
          int16_t w0, w1, w2, w3;
        };
        std::vector<RowEntry> row_table(dh);
        for (int dy = 0; dy < dh; ++dy) {
          const float sy = src_rect.Y +
              (dy + 0.5f) * static_cast<float>(sh) / static_cast<float>(dh)
              - 0.5f;
          const int iy = static_cast<int>(std::floor(sy));
          const float t = sy - static_cast<float>(iy);
          const float t2 = t * t, t3 = t2 * t;

          // Vertical layout remains Q12 for the 32-bit accumulators
          row_table[dy] = {
            clamp_ry(iy - 1), clamp_ry(iy),
            clamp_ry(iy + 1), clamp_ry(iy + 2),
            static_cast<int16_t>(std::lround((-0.5f*t3 +      t2 - 0.5f*t) * 4096.0f)),
            static_cast<int16_t>(std::lround(( 1.5f*t3 - 2.5f*t2 + 1.0f ) * 4096.0f)),
            static_cast<int16_t>(std::lround((-1.5f*t3 + 2.0f*t2 + 0.5f*t) * 4096.0f)),
            static_cast<int16_t>(std::lround(( 0.5f*t3 - 0.5f*t2          ) * 4096.0f)),
          };
        }

        // Opt #5: Pre-allocate globally aligned chunk for threads with perfectly aligned row strides
        const int max_workers = std::min((int)dh, GetEffectiveNumThreads());
        const int dw_elems = dw * depth;
        const size_t row_stride = (dw_elems * sizeof(int16_t) + 15) & ~15;
        const size_t buf_size = row_stride * 4;

        void* raw_ptr = nullptr;
        posix_memalign(&raw_ptr, 16, buf_size * max_workers);
        std::unique_ptr<uint8_t, void(*)(void*)> hbuf(
            static_cast<uint8_t*>(raw_ptr), std::free);

        ExecuteInParallel([=, &col_ix0, &col_ix1, &col_ix2, &col_ix3,
                              &col_w0, &col_w1, &col_w2, &col_w3,
                              &row_table](
                              int num_threads, int i) {
          const int rows_per = dh / num_threads;
          const int height   = (i == num_threads - 1)
                                   ? (dh - i * rows_per) : rows_per;
          const int dy0 = i * rows_per;

          uint8_t* thread_hbuf = static_cast<uint8_t*>(raw_ptr) + i * buf_size;
          int16_t* hrow[4] = {
            reinterpret_cast<int16_t*>(thread_hbuf),
            reinterpret_cast<int16_t*>(thread_hbuf + row_stride),
            reinterpret_cast<int16_t*>(thread_hbuf + row_stride * 2),
            reinterpret_cast<int16_t*>(thread_hbuf + row_stride * 3),
          };
          int cached_iy[4] = { -1, -1, -1, -1 };

          auto ensure_hrow = [&](int slot, int ry) {
            if (cached_iy[slot] != ry) {
              const uint8_t* src_row = src_base + ry * src_stride;
              if (depth == 3) {
                BicubicHorizRow3(src_row,
                    col_ix0.data(), col_ix1.data(), col_ix2.data(), col_ix3.data(),
                    col_w0.data(), col_w1.data(), col_w2.data(), col_w3.data(),
                    hrow[slot], dw);
              } else {
                BicubicHorizRow4(src_row,
                    col_ix0.data(), col_ix1.data(), col_ix2.data(), col_ix3.data(),
                    col_w0.data(), col_w1.data(), col_w2.data(), col_w3.data(),
                    hrow[slot], dw);
              }
              cached_iy[slot] = ry;
            }
          };

          for (int dy_local = 0; dy_local < height; ++dy_local) {
            const int dy = dy0 + dy_local;
            const auto& re = row_table[dy];
            ensure_hrow(0, re.r0);
            ensure_hrow(1, re.r1);
            ensure_hrow(2, re.r2);
            ensure_hrow(3, re.r3);

            uint8_t* dest_row = dest_base + (dest_rect.Y + dy) * dest_stride +
                                dest_rect.X * depth;
            BicubicVertBlend(hrow[0], hrow[1], hrow[2], hrow[3],
                              re.w0, re.w1, re.w2, re.w3, dest_row, dw_elems);
          }
        }, GetEffectiveNumThreads());
        break;
      }
#endif

      std::vector<int> sxi0(dw), sxi1(dw), sxi2(dw), sxi3(dw);
      std::vector<int> colw0(dw), colw1(dw), colw2(dw), colw3(dw);
      for (int dx = 0; dx < dw; ++dx) {
        const float sx = src_rect.X +
            (dx + 0.5f) * static_cast<float>(sw) / static_cast<float>(dw)
            - 0.5f;
        const int ix = static_cast<int>(std::floor(sx));
        const float t = sx - ix;
        const float t2 = t * t, t3 = t2 * t;
        const float fw0 = -0.5f*t3 +      t2 - 0.5f*t;
        const float fw1 =  1.5f*t3 - 2.5f*t2          + 1.0f;
        const float fw2 = -1.5f*t3 + 2.0f*t2 + 0.5f*t;
        const float fw3 =  0.5f*t3 - 0.5f*t2;
        const int ix_min = src_rect.X;
        const int ix_max = src_rect.X + sw - 1;
        auto clampx = [&](int v) { return v < ix_min ? ix_min : (v > ix_max ? ix_max : v); };
        sxi0[dx] = clampx(ix - 1) * depth;
        sxi1[dx] = clampx(ix    ) * depth;
        sxi2[dx] = clampx(ix + 1) * depth;
        sxi3[dx] = clampx(ix + 2) * depth;

        // Match Opt #4: Q6 scaling weights fallback
        colw0[dx] = static_cast<int>(std::lround(fw0 * 64.0f));
        colw1[dx] = static_cast<int>(std::lround(fw1 * 64.0f));
        colw2[dx] = static_cast<int>(std::lround(fw2 * 64.0f));
        colw3[dx] = static_cast<int>(std::lround(fw3 * 64.0f));
      }

      // Opt #5 applies heavily here - scalar originally initialized std::vector *inside*
      // the thread runner causing rapid-fire allocations per pool process
      const int max_workers = std::min((int)dh, GetEffectiveNumThreads());
      const int dw_elems = dw * depth;
      const size_t row_stride = (dw_elems * sizeof(int16_t) + 15) & ~15;
      const size_t buf_size = row_stride * 4;

      void* raw_ptr = nullptr;
      posix_memalign(&raw_ptr, 16, buf_size * max_workers);
      std::unique_ptr<uint8_t, void(*)(void*)> hbuf(
          static_cast<uint8_t*>(raw_ptr), std::free);

      ExecuteInParallel([=, &sxi0, &sxi1, &sxi2, &sxi3, &colw0, &colw1, &colw2, &colw3](int num_threads, int i) {
        const int rows_per = dh / num_threads;
        const int height   = (i == num_threads - 1)
                                 ? (dh - i * rows_per) : rows_per;
        const int dy0 = i * rows_per;

        uint8_t* thread_hbuf = static_cast<uint8_t*>(raw_ptr) + i * buf_size;
        int16_t* hrow[4] = {
          reinterpret_cast<int16_t*>(thread_hbuf),
          reinterpret_cast<int16_t*>(thread_hbuf + row_stride),
          reinterpret_cast<int16_t*>(thread_hbuf + row_stride * 2),
          reinterpret_cast<int16_t*>(thread_hbuf + row_stride * 3),
        };
        int cached_iy[4] = { -1, -1, -1, -1 };

        const int ry_min = src_rect.Y;
        const int ry_max = src_rect.Y + sh - 1;
        auto clamp_row = [&](int ry) {
          return ry < ry_min ? ry_min : (ry > ry_max ? ry_max : ry);
        };

        auto ensure_hrow = [&](int slot, int ry) {
          if (cached_iy[slot] == ry) return;
          const uint8_t* src_row = src_base + ry * src_stride;
          for (int dx = 0; dx < dw; ++dx) {
            const uint8_t* P0 = src_row + sxi0[dx];
            const uint8_t* P1 = src_row + sxi1[dx];
            const uint8_t* P2 = src_row + sxi2[dx];
            const uint8_t* P3 = src_row + sxi3[dx];
            for (int ch = 0; ch < depth; ++ch) {
              const int acc = P0[ch]*colw0[dx] + P1[ch]*colw1[dx] +
                               P2[ch]*colw2[dx] + P3[ch]*colw3[dx];
              hrow[slot][dx*depth + ch] = static_cast<int16_t>((acc + 2) >> 2); // Q6 Downshift
            }
          }
          cached_iy[slot] = ry;
        };

        for (int dy_local = 0; dy_local < height; ++dy_local) {
          const int dy = dy0 + dy_local;
          const float sy = src_rect.Y +
              (dy + 0.5f) * static_cast<float>(sh) / static_cast<float>(dh)
              - 0.5f;
          const int iy = static_cast<int>(std::floor(sy));
          const float t = sy - iy;
          const float t2 = t * t, t3 = t2 * t;
          const float fw0 = -0.5f*t3 +      t2 - 0.5f*t;
          const float fw1 =  1.5f*t3 - 2.5f*t2          + 1.0f;
          const float fw2 = -1.5f*t3 + 2.0f*t2 + 0.5f*t;
          const float fw3 =  0.5f*t3 - 0.5f*t2;
          const int w0 = static_cast<int>(std::lround(fw0 * 4096.0f));
          const int w1 = static_cast<int>(std::lround(fw1 * 4096.0f));
          const int w2 = static_cast<int>(std::lround(fw2 * 4096.0f));
          const int w3 = static_cast<int>(std::lround(fw3 * 4096.0f));

          const int r0 = clamp_row(iy - 1);
          const int r1 = clamp_row(iy);
          const int r2 = clamp_row(iy + 1);
          const int r3 = clamp_row(iy + 2);
          ensure_hrow(0, r0);
          ensure_hrow(1, r1);
          ensure_hrow(2, r2);
          ensure_hrow(3, r3);

          uint8_t* dest_row = dest_base + (dest_rect.Y + dy) * dest_stride +
                              dest_rect.X * depth;
          for (int e = 0; e < dw_elems; ++e) {
            const int acc = static_cast<int>(hrow[0][e]) * w0 +
                             static_cast<int>(hrow[1][e]) * w1 +
                             static_cast<int>(hrow[2][e]) * w2 +
                             static_cast<int>(hrow[3][e]) * w3;
            dest_row[e] = Clamp8((acc + 32768) >> 16);
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