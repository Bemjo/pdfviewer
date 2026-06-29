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
  _buffer = new uint8_t[GetBufferByteSize()];
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
    delete[] _buffer;
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

// Horizontal interpolation, depth=3.
static void BilinearHorizRow3(
    const uint8_t* __restrict__ src_row,
    const int* __restrict__ col_ix0,
    const int* __restrict__ col_ix1,
    const uint8_t* __restrict__ col_fx8,
    uint8_t* __restrict__ out, int dw) {
  int dx = 0;
  #pragma GCC ivdep
  for (; dx + 8 <= dw; dx += 8) {
    uint8_t stg_l[24], stg_r[24];
    for (int k = 0; k < 8; ++k) {
      const uint8_t* L = src_row + col_ix0[dx + k];
      const uint8_t* R = src_row + col_ix1[dx + k];
      stg_l[k*3+0]=L[0]; stg_l[k*3+1]=L[1]; stg_l[k*3+2]=L[2];
      stg_r[k*3+0]=R[0]; stg_r[k*3+1]=R[1]; stg_r[k*3+2]=R[2];
    }
    uint8x8_t vfx  = vld1_u8(col_fx8 + dx);
    uint8x8_t vfxc = vsub_u8(vdup_n_u8(255), vfx);
    uint8x8x3_t vl = vld3_u8(stg_l);
    uint8x8x3_t vr = vld3_u8(stg_r);

    uint16x8_t accA = vmull_u8(vl.val[0], vfxc);
    uint16x8_t accB = vmull_u8(vl.val[1], vfxc);
    uint16x8_t accC = vmull_u8(vl.val[2], vfxc);
    accA = vmlal_u8(accA, vr.val[0], vfx);
    accB = vmlal_u8(accB, vr.val[1], vfx);
    accC = vmlal_u8(accC, vr.val[2], vfx);

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

template <int ShiftVal>
static void BilinearVertBlendSharp3_Templated(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ prev,
    const uint8_t* __restrict__ next,
    uint8_t fy8, uint8_t sharp_strength,
    uint8_t* __restrict__ dest_row, int dw) {

  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const int nbytes = dw * 3;

  #pragma GCC ivdep
  for (int i = 0; i + 24 <= nbytes; i += 24) {
      uint8x8x3_t v0   = vld3_u8(row0 + i);
      uint8x8x3_t v1   = vld3_u8(row1 + i);
      uint8x8x3_t vp   = vld3_u8(prev + i);
      uint8x8x3_t vn   = vld3_u8(next + i);

      uint8x8x3_t vout;
      uint16x8_t bilA = vmull_u8(v0.val[0], vfyc);
      uint16x8_t bilB = vmull_u8(v0.val[1], vfyc);
      uint16x8_t bilC = vmull_u8(v0.val[2], vfyc);
      bilA = vmlal_u8(bilA, v1.val[0], vfy);
      bilB = vmlal_u8(bilB, v1.val[1], vfy);
      bilC = vmlal_u8(bilC, v1.val[2], vfy);

      uint8x8_t baseA = vrshrn_n_u16(bilA, 8);
      uint8x8_t baseB = vrshrn_n_u16(bilB, 8);
      uint8x8_t baseC = vrshrn_n_u16(bilC, 8);

      uint16x8_t shA = vmull_u8(baseA, vdup_n_u8(sharp_strength));
      uint16x8_t shB = vmull_u8(baseB, vdup_n_u8(sharp_strength));
      uint16x8_t shC = vmull_u8(baseC, vdup_n_u8(sharp_strength));

      int16x8_t sA = vreinterpretq_s16_u16(vaddw_u8(vshrq_n_u16(shA, 8), baseA));
      int16x8_t sB = vreinterpretq_s16_u16(vaddw_u8(vshrq_n_u16(shB, 8), baseB));
      int16x8_t sC = vreinterpretq_s16_u16(vaddw_u8(vshrq_n_u16(shC, 8), baseC));

      uint16x8_t outA = vaddl_u8(vp.val[0], vn.val[0]);
      uint16x8_t outB = vaddl_u8(vp.val[1], vn.val[1]);
      uint16x8_t outC = vaddl_u8(vp.val[2], vn.val[2]);

      sA = vsubq_s16(sA, vreinterpretq_s16_u16(vshrq_n_u16(outA, ShiftVal)));
      sB = vsubq_s16(sB, vreinterpretq_s16_u16(vshrq_n_u16(outB, ShiftVal)));
      sC = vsubq_s16(sC, vreinterpretq_s16_u16(vshrq_n_u16(outC, ShiftVal)));

      vout.val[0] = vqmovun_s16(sA);
      vout.val[1] = vqmovun_s16(sB);
      vout.val[2] = vqmovun_s16(sC);
      vst3_u8(dest_row + i, vout);
  }
}

static void BilinearVertBlendSharp3(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ prev,
    const uint8_t* __restrict__ next,
    uint8_t fy8, uint8_t sharp_strength,
    uint8_t* __restrict__ dest_row, int dw) {

  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const int nbytes = dw * 3;
  int i = 0;

  if (sharp_strength == 64) {
      BilinearVertBlendSharp3_Templated<3>(row0, row1, prev, next, fy8, sharp_strength, dest_row, dw);
      i = (nbytes / 24) * 24;
  } else if (sharp_strength == 128) {
      BilinearVertBlendSharp3_Templated<2>(row0, row1, prev, next, fy8, sharp_strength, dest_row, dw);
      i = (nbytes / 24) * 24;
  } else if (sharp_strength == 32) {
      BilinearVertBlendSharp3_Templated<4>(row0, row1, prev, next, fy8, sharp_strength, dest_row, dw);
      i = (nbytes / 24) * 24;
  } else {
    const uint8x8_t vsh = vdup_n_u8(sharp_strength);
    #pragma GCC ivdep
    for (; i + 24 <= nbytes; i += 24) {
      uint8x8x3_t v0 = vld3_u8(row0 + i);
      uint8x8x3_t v1 = vld3_u8(row1 + i);
      uint8x8x3_t vp = vld3_u8(prev + i);
      uint8x8x3_t vn = vld3_u8(next + i);
      uint8x8x3_t vout;
      for (int ch = 0; ch < 3; ++ch) {
        uint16x8_t bil = vmull_u8(v0.val[ch], vfyc);
        bil = vmlal_u8(bil, v1.val[ch], vfy);
        uint8x8_t base = vrshrn_n_u16(bil, 8);
        uint16x8_t sh_add = vmull_u8(base, vsh);
        int16x8_t s16 = vreinterpretq_s16_u16(
            vaddw_u8(vshrq_n_u16(sh_add, 8), base));
        uint16x8_t outer = vaddl_u8(vp.val[ch], vn.val[ch]);
        outer = vshrq_n_u16(vmulq_n_u16(outer, sharp_strength), 9);
        s16 = vsubq_s16(s16, vreinterpretq_s16_u16(outer));
        vout.val[ch] = vqmovun_s16(s16);
      }
      vst3_u8(dest_row + i, vout);
    }
  }

  // 8-byte tail.
  for (; i + 8 <= nbytes; i += 8) {
    uint8x8_t v0b = vld1_u8(row0 + i), v1b = vld1_u8(row1 + i);
    uint8x8_t vpb = vld1_u8(prev  + i), vnb = vld1_u8(next  + i);
    uint16x8_t bil = vmull_u8(v0b, vfyc);
    bil = vmlal_u8(bil, v1b, vfy);
    uint8x8_t base = vrshrn_n_u16(bil, 8);
    uint16x8_t sh_add = vmull_u8(base, vdup_n_u8(sharp_strength));
    int16x8_t s16 = vreinterpretq_s16_u16(
        vaddw_u8(vshrq_n_u16(sh_add, 8), base));
    uint16x8_t outer = vaddl_u8(vpb, vnb);

    if (sharp_strength == 64) outer = vshrq_n_u16(outer, 3);
    else if (sharp_strength == 128) outer = vshrq_n_u16(outer, 2);
    else if (sharp_strength == 32) outer = vshrq_n_u16(outer, 4);
    else outer = vshrq_n_u16(vmulq_n_u16(outer, sharp_strength), 9);

    s16 = vsubq_s16(s16, vreinterpretq_s16_u16(outer));
    vst1_u8(dest_row + i, vqmovun_s16(s16));
  }
  // Scalar tail.
  const int fy = fy8, fyc = 255 - fy, sh = sharp_strength;
  for (; i < nbytes; ++i) {
    const int bil = (row0[i]*fyc + row1[i]*fy + 128) >> 8;
    const int out = bil + ((bil*sh) >> 8) - (((int)prev[i] + next[i])*sh >> 9);
    dest_row[i] = static_cast<uint8_t>(out < 0 ? 0 : out > 255 ? 255 : out);
  }
}

template <int ShiftVal>
static void BilinearVertBlendSharp4_Templated(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ prev,
    const uint8_t* __restrict__ next,
    uint8_t fy8, uint8_t sharp_strength,
    uint8_t* __restrict__ dest_row, int dw) {

  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const uint8x8_t vsh  = vdup_n_u8(sharp_strength);
  const uint8x8_t alpha_mask = { 0, 0, 0, 255, 0, 0, 0, 255 };
  const int nbytes = dw * 4;

  #pragma GCC ivdep
  for (int i = 0; i + 16 <= nbytes; i += 16) {
    uint8x16_t v0 = vld1q_u8(row0 + i);
    uint8x16_t v1 = vld1q_u8(row1 + i);
    uint8x16_t vp = vld1q_u8(prev + i);
    uint8x16_t vn = vld1q_u8(next + i);

    uint16x8_t bilL = vmull_u8(vget_low_u8(v0), vfyc);
    uint16x8_t bilH = vmull_u8(vget_high_u8(v0), vfyc);
    bilL = vmlal_u8(bilL, vget_low_u8(v1), vfy);
    bilH = vmlal_u8(bilH, vget_high_u8(v1), vfy);

    uint8x8_t baseL = vrshrn_n_u16(bilL, 8);
    uint8x8_t baseH = vrshrn_n_u16(bilH, 8);

    uint16x8_t shL = vmull_u8(baseL, vsh);
    uint16x8_t shH = vmull_u8(baseH, vsh);
    int16x8_t sL = vreinterpretq_s16_u16(vaddw_u8(vshrq_n_u16(shL, 8), baseL));
    int16x8_t sH = vreinterpretq_s16_u16(vaddw_u8(vshrq_n_u16(shH, 8), baseH));

    uint16x8_t outL = vaddl_u8(vget_low_u8(vp), vget_low_u8(vn));
    uint16x8_t outH = vaddl_u8(vget_high_u8(vp), vget_high_u8(vn));
    sL = vsubq_s16(sL, vreinterpretq_s16_u16(vshrq_n_u16(outL, ShiftVal)));
    sH = vsubq_s16(sH, vreinterpretq_s16_u16(vshrq_n_u16(outH, ShiftVal)));

    uint8x8_t resL = vqmovun_s16(sL);
    uint8x8_t resH = vqmovun_s16(sH);

    resL = vbsl_u8(alpha_mask, vget_low_u8(v0), resL);
    resH = vbsl_u8(alpha_mask, vget_high_u8(v0), resH);

    vst1q_u8(dest_row + i, vcombine_u8(resL, resH));
  }
}

static void BilinearVertBlendSharp4(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ prev,
    const uint8_t* __restrict__ next,
    uint8_t fy8, uint8_t sharp_strength,
    uint8_t* __restrict__ dest_row, int dw) {

  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const int nbytes = dw * 4;
  int i = 0;

  if (sharp_strength == 64) {
      BilinearVertBlendSharp4_Templated<3>(row0, row1, prev, next, fy8, sharp_strength, dest_row, dw);
      i = (nbytes / 16) * 16;
  } else if (sharp_strength == 128) {
      BilinearVertBlendSharp4_Templated<2>(row0, row1, prev, next, fy8, sharp_strength, dest_row, dw);
      i = (nbytes / 16) * 16;
  } else if (sharp_strength == 32) {
      BilinearVertBlendSharp4_Templated<4>(row0, row1, prev, next, fy8, sharp_strength, dest_row, dw);
      i = (nbytes / 16) * 16;
  } else {
    const uint8x8_t vsh = vdup_n_u8(sharp_strength);
    const uint8x8_t alpha_mask = { 0, 0, 0, 255, 0, 0, 0, 255 };

    #pragma GCC ivdep
    for (; i + 16 <= nbytes; i += 16) {
      uint8x16_t v0 = vld1q_u8(row0 + i);
      uint8x16_t v1 = vld1q_u8(row1 + i);
      uint8x16_t vp = vld1q_u8(prev + i);
      uint8x16_t vn = vld1q_u8(next + i);

      uint16x8_t bilL = vmull_u8(vget_low_u8(v0), vfyc);
      uint16x8_t bilH = vmull_u8(vget_high_u8(v0), vfyc);
      bilL = vmlal_u8(bilL, vget_low_u8(v1), vfy);
      bilH = vmlal_u8(bilH, vget_high_u8(v1), vfy);

      uint8x8_t baseL = vrshrn_n_u16(bilL, 8);
      uint8x8_t baseH = vrshrn_n_u16(bilH, 8);

      uint16x8_t shL = vmull_u8(baseL, vsh);
      uint16x8_t shH = vmull_u8(baseH, vsh);
      int16x8_t sL = vreinterpretq_s16_u16(vaddw_u8(vshrq_n_u16(shL, 8), baseL));
      int16x8_t sH = vreinterpretq_s16_u16(vaddw_u8(vshrq_n_u16(shH, 8), baseH));

      uint16x8_t outL = vaddl_u8(vget_low_u8(vp), vget_low_u8(vn));
      uint16x8_t outH = vaddl_u8(vget_high_u8(vp), vget_high_u8(vn));
      sL = vsubq_s16(sL, vreinterpretq_s16_u16(vshrq_n_u16(vmulq_n_u16(outL, sharp_strength), 9)));
      sH = vsubq_s16(sH, vreinterpretq_s16_u16(vshrq_n_u16(vmulq_n_u16(outH, sharp_strength), 9)));

      uint8x8_t resL = vqmovun_s16(sL);
      uint8x8_t resH = vqmovun_s16(sH);

      resL = vbsl_u8(alpha_mask, vget_low_u8(v0), resL);
      resH = vbsl_u8(alpha_mask, vget_high_u8(v0), resH);
      vst1q_u8(dest_row + i, vcombine_u8(resL, resH));
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
  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const int nbytes = dw * 3;
  int i = 0;
  for (; i + 24 <= nbytes; i += 24) {
    uint8x8x3_t v0 = vld3_u8(row0 + i);
    uint8x8x3_t v1 = vld3_u8(row1 + i);
    uint16x8_t accA = vmull_u8(v0.val[0], vfyc);
    uint16x8_t accB = vmull_u8(v0.val[1], vfyc);
    uint16x8_t accC = vmull_u8(v0.val[2], vfyc);
    accA = vmlal_u8(accA, v1.val[0], vfy);
    accB = vmlal_u8(accB, v1.val[1], vfy);
    accC = vmlal_u8(accC, v1.val[2], vfy);
    uint8x8x3_t vout;
    vout.val[0] = vrshrn_n_u16(accA, 8);
    vout.val[1] = vrshrn_n_u16(accB, 8);
    vout.val[2] = vrshrn_n_u16(accC, 8);
    vst3_u8(dest_row + i, vout);
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

static void BilinearHorizRow4(
    const uint8_t* __restrict__ src_row,
    const int* __restrict__ col_ix0,
    const int* __restrict__ col_ix1,
    const uint8_t* __restrict__ col_fx8,
    uint8_t* __restrict__ out, int dw) {
  int dx = 0;
  for (; dx + 8 <= dw; dx += 8) {
    uint8_t stg_l[32], stg_r[32];
    for (int k = 0; k < 8; ++k) {
      const uint8_t* L = src_row + col_ix0[dx + k];
      const uint8_t* R = src_row + col_ix1[dx + k];
      stg_l[k*4+0]=L[0]; stg_l[k*4+1]=L[1]; stg_l[k*4+2]=L[2]; stg_l[k*4+3]=L[3];
      stg_r[k*4+0]=R[0]; stg_r[k*4+1]=R[1]; stg_r[k*4+2]=R[2]; stg_r[k*4+3]=R[3];
    }
    uint8x8_t vfx  = vld1_u8(col_fx8 + dx);
    uint8x8_t vfxc = vsub_u8(vdup_n_u8(255), vfx);
    uint8x8x4_t vl = vld4_u8(stg_l);
    uint8x8x4_t vr = vld4_u8(stg_r);
    uint16x8_t accA = vmull_u8(vl.val[0], vfxc);
    uint16x8_t accB = vmull_u8(vl.val[1], vfxc);
    uint16x8_t accC = vmull_u8(vl.val[2], vfxc);
    uint16x8_t accD = vmull_u8(vl.val[3], vfxc);
    accA = vmlal_u8(accA, vr.val[0], vfx);
    accB = vmlal_u8(accB, vr.val[1], vfx);
    accC = vmlal_u8(accC, vr.val[2], vfx);
    accD = vmlal_u8(accD, vr.val[3], vfx);
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
    PixelBuffer* dest, PixelBuffer::ScaleMode scale_mode) const {
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

    case SCALE_BILINEAR:
    case SCALE_BILINEAR_SHARP: {
#ifdef __ARM_NEON
      if (valid_sharp_depth) {
        std::vector<int>     col_ix0(dw), col_ix1(dw);
        std::vector<uint8_t> col_fx8(dw);
        for (int dx = 0; dx < dw; ++dx) {
          const float sx = src_rect.X +
              (dx + 0.5f) * static_cast<float>(sw) / static_cast<float>(dw)
              - 0.5f;
          int ix = static_cast<int>(sx);
          float fxf = sx - ix;
          if (ix < src_rect.X) { ix = src_rect.X; fxf = 0.0f; }
          const int ix_max = src_rect.X + sw - 1;
          const int ix1 = (ix < ix_max) ? ix + 1 : ix_max;
          col_ix0[dx] = ix  * depth;
          col_ix1[dx] = ix1 * depth;
          col_fx8[dx] = static_cast<uint8_t>(fxf * 255.0f + 0.5f);
        }

        const uint8_t sharp_strength = (valid_sharp_depth && scale_mode == SCALE_BILINEAR_SHARP) ? 64u : 0u;

        ExecuteInParallel([=, &col_ix0, &col_ix1, &col_fx8](
                              int num_threads, int i) {
          const int rows_per = dh / num_threads;
          const int height   = (i == num_threads - 1)
                                   ? (dh - i * rows_per) : rows_per;
          const int dy0 = i * rows_per;

          const int row_bytes = dw * depth;
          std::vector<uint8_t> hbuf(row_bytes * 4);
          uint8_t* hrow[4] = {
            hbuf.data(),
            hbuf.data() + row_bytes,
            hbuf.data() + row_bytes * 2,
            hbuf.data() + row_bytes * 3,
          };
          int cached_iy[4] = { -1, -1, -1, -1 };

          auto ensure_hrow = [&](int slot, int ry) {
            ry = std::max(src_rect.Y, std::min(src_rect.Y + sh - 1, ry));
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
              ensure_hrow(0, iy - 1);
              ensure_hrow(1, iy);
              ensure_hrow(2, iy1);
              ensure_hrow(3, iy1 + 1);

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
              ensure_hrow(1, iy);
              ensure_hrow(2, iy1);
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