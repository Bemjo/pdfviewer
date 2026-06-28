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

static void BilinearHorizRow3(
    const uint8_t* src_row,
    const int* col_ix0, const int* col_ix1, const uint8_t* col_fx8,
    uint8_t* out, int dw) {
  int dx = 0;
  for (; dx + 8 <= dw; dx += 8) {
    // Gather 8 left/right BGR pixels into staging buffers, then de-interleave
    // with vld3_u8.  The staging write-then-read stays hot in L1.
    uint8_t stg_l[24], stg_r[24];
    for (int k = 0; k < 8; ++k) {
      const uint8_t* L = src_row + col_ix0[dx + k];
      const uint8_t* R = src_row + col_ix1[dx + k];
      stg_l[k*3+0]=L[0]; stg_l[k*3+1]=L[1]; stg_l[k*3+2]=L[2];
      stg_r[k*3+0]=R[0]; stg_r[k*3+1]=R[1]; stg_r[k*3+2]=R[2];
    }
    uint8x8x3_t vl = vld3_u8(stg_l);
    uint8x8x3_t vr = vld3_u8(stg_r);
    uint8x8_t vfx  = vld1_u8(col_fx8 + dx);
    uint8x8_t vfxc = vsub_u8(vdup_n_u8(255), vfx);
    uint8x8x3_t vout;
    for (int ch = 0; ch < 3; ++ch) {
      uint16x8_t acc = vmull_u8(vl.val[ch], vfxc);
      acc = vmlal_u8(acc, vr.val[ch], vfx);
      vout.val[ch] = vrshrn_n_u16(acc, 8);
    }
    vst3_u8(out + dx * 3, vout);
  }
  for (; dx < dw; ++dx) {
    const uint8_t* L = src_row + col_ix0[dx];
    const uint8_t* R = src_row + col_ix1[dx];
    const uint32_t fx  = col_fx8[dx];
    const uint32_t fxc = 255u - fx;
    out[dx*3+0] = static_cast<uint8_t>((L[0]*fxc + R[0]*fx + 128u) >> 8);
    out[dx*3+1] = static_cast<uint8_t>((L[1]*fxc + R[1]*fx + 128u) >> 8);
    out[dx*3+2] = static_cast<uint8_t>((L[2]*fxc + R[2]*fx + 128u) >> 8);
  }
}

// Vertical blend with inline unsharp sharpening (depth=3).
//
// Standard bilinear vertical: out = row0*(1-fy) + row1*fy
//
// Sharpened vertical (same pass, no extra memory):
//   out = bilinear + k*(bilinear - (prev + next) / 2)
//       = (1+k)*bilinear - k*(prev+next)/2
//
// With k=1/4 in Q8 fixed-point:
//   bilinear_q8  = row0*(255-fy) + row1*fy           (u16, max 255*255)
//   out_s16      = bilinear*5/4 - (prev+next)*1/8
//                = bilinear*320/256 - (prev+next)*32/256
//
// prev = horizontally-filtered row above iy (or iy itself at top boundary)
// next = horizontally-filtered row below iy1 (or iy1 at bottom boundary)
//
// All arithmetic in int16 with vqmovun_s16 saturation to uint8 at the end.
// This gives visible sharpening with zero extra passes or bandwidth.
//
// sharp_strength: 0=no sharpening, 64=k~1/4 (recommended), 128=k~1/2 (strong)
static void BilinearVertBlendSharp3(
    const uint8_t* row0, const uint8_t* row1,
    const uint8_t* prev, const uint8_t* next,
    uint8_t fy8, uint8_t sharp_strength, uint8_t* dest_row, int dw) {
  // Weights in Q8:
  //   bilinear contribution = 256 + sharp_strength (written as two parts below)
  //   outer row contribution = -sharp_strength / 2  (negative, sharpening)
  // We compute:
  //   s16 = bilinear_u16 * (256 + sharp_strength) / 256
  //         - (prev_u8 + next_u8) * sharp_strength / 2 / 256
  // Using NEON:
  //   bilinear_u16 = vmull_u8(row0, fyc) + vmull_u8(row1, fy)   [0..65025]
  //   base_s16     = vrshrn_n_u16(bilinear, 8)                   [0..255]
  //   sharp_add    = base_s16 * sharp_strength >> 8               extra bilinear
  //   outer_u16    = vaddl_u8(prev8, next8)                       [0..510]
  //   outer_sub    = outer_u16 * sharp_strength >> 9 (÷2÷256)
  //   out_s16      = (int16)(base_s16) + sharp_add - outer_sub
  //   out_u8       = vqmovun_s16(out_s16)                        saturate+narrow

  const uint8x8_t vfy   = vdup_n_u8(fy8);
  const uint8x8_t vfyc  = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const uint8x8_t vsh   = vdup_n_u8(sharp_strength);

  const int nbytes = dw * 3;
  int i = 0;

  // Process 24 bytes = 8 BGR pixels per iteration.
  for (; i + 24 <= nbytes; i += 24) {
    // Load and de-interleave 8 pixels from each of the 4 rows.
    uint8x8x3_t v0   = vld3_u8(row0 + i);
    uint8x8x3_t v1   = vld3_u8(row1 + i);
    uint8x8x3_t vprev = vld3_u8(prev + i);
    uint8x8x3_t vnext = vld3_u8(next + i);

    uint8x8x3_t vout;
    for (int ch = 0; ch < 3; ++ch) {
      // Bilinear blend -> u16, then narrow to u8.
      uint16x8_t bil = vmull_u8(v0.val[ch], vfyc);
      bil = vmlal_u8(bil, v1.val[ch], vfy);
      uint8x8_t base = vrshrn_n_u16(bil, 8);

      // Sharpening: extra = base * sharp_strength >> 8
      uint16x8_t sh_add = vmull_u8(base, vsh);
      int16x8_t  s16    = vreinterpretq_s16_u16(vmovl_u8(base));
      s16 = vaddq_s16(s16, vreinterpretq_s16_u16(vshrq_n_u16(sh_add, 8)));

      // Outer rows: subtract (prev + next) * sharp_strength >> 9
      uint16x8_t outer = vaddl_u8(vprev.val[ch], vnext.val[ch]);
      outer = vmulq_n_u16(outer, sharp_strength);
      s16 = vsubq_s16(s16, vreinterpretq_s16_u16(vshrq_n_u16(outer, 9)));

      vout.val[ch] = vqmovun_s16(s16);
    }
    vst3_u8(dest_row + i, vout);
  }
  // 8-byte tail (processes up to 2 full BGR pixels and 2 spare bytes safely).
  for (; i + 8 <= nbytes; i += 8) {
    uint8x8_t v0b   = vld1_u8(row0 + i);
    uint8x8_t v1b   = vld1_u8(row1 + i);
    uint8x8_t vpb   = vld1_u8(prev + i);
    uint8x8_t vnb   = vld1_u8(next + i);

    uint16x8_t bil  = vmull_u8(v0b, vfyc);
    bil = vmlal_u8(bil, v1b, vfy);
    uint8x8_t base  = vrshrn_n_u16(bil, 8);

    uint16x8_t sh_add = vmull_u8(base, vsh);
    int16x8_t  s16    = vreinterpretq_s16_u16(vmovl_u8(base));
    s16 = vaddq_s16(s16, vreinterpretq_s16_u16(vshrq_n_u16(sh_add, 8)));

    uint16x8_t outer = vaddl_u8(vpb, vnb);
    outer = vmulq_n_u16(outer, sharp_strength);
    s16 = vsubq_s16(s16, vreinterpretq_s16_u16(vshrq_n_u16(outer, 9)));

    vst1_u8(dest_row + i, vqmovun_s16(s16));
  }
  // Scalar tail.
  const int fy  = fy8, fyc = 255 - fy;
  const int sh  = sharp_strength;
  for (; i < nbytes; ++i) {
    const int bil  = (row0[i]*fyc + row1[i]*fy + 128) >> 8;
    const int out  = bil + ((bil * sh) >> 8) - (((int)prev[i] + next[i]) * sh >> 9);
    dest_row[i] = static_cast<uint8_t>(out < 0 ? 0 : out > 255 ? 255 : out);
  }
}

// Non-sharpening vertical blend for depth=3 (used when sharp_strength==0).
static void BilinearVertBlend3(
    const uint8_t* row0, const uint8_t* row1,
    uint8_t fy8, uint8_t* dest_row, int dw) {
  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const int nbytes = dw * 3;
  int i = 0;
  for (; i + 24 <= nbytes; i += 24) {
    uint8x8x3_t v0 = vld3_u8(row0 + i);
    uint8x8x3_t v1 = vld3_u8(row1 + i);
    uint8x8x3_t vout;
    for (int ch = 0; ch < 3; ++ch) {
      uint16x8_t acc = vmull_u8(v0.val[ch], vfyc);
      acc = vmlal_u8(acc, v1.val[ch], vfy);
      vout.val[ch] = vrshrn_n_u16(acc, 8);
    }
    vst3_u8(dest_row + i, vout);
  }
  for (; i + 8 <= nbytes; i += 8) {
    uint8x8_t v0 = vld1_u8(row0 + i);
    uint8x8_t v1 = vld1_u8(row1 + i);
    uint16x8_t acc = vmull_u8(v0, vfyc);
    acc = vmlal_u8(acc, v1, vfy);
    vst1_u8(dest_row + i, vrshrn_n_u16(acc, 8));
  }
  const uint32_t fy  = fy8, fyc = 255u - fy;
  for (; i < nbytes; ++i)
    dest_row[i] = static_cast<uint8_t>((row0[i]*fyc + row1[i]*fy + 128u) >> 8);
}

static void BilinearHorizRow4(
    const uint8_t* src_row,
    const int* col_ix0, const int* col_ix1, const uint8_t* col_fx8,
    uint8_t* out, int dw) {
  int dx = 0;
  for (; dx + 8 <= dw; dx += 8) {
    uint8_t stg_l[32], stg_r[32];
    for (int k = 0; k < 8; ++k) {
      const uint8_t* L = src_row + col_ix0[dx + k];
      const uint8_t* R = src_row + col_ix1[dx + k];
      stg_l[k*4+0]=L[0]; stg_l[k*4+1]=L[1]; stg_l[k*4+2]=L[2]; stg_l[k*4+3]=L[3];
      stg_r[k*4+0]=R[0]; stg_r[k*4+1]=R[1]; stg_r[k*4+2]=R[2]; stg_r[k*4+3]=R[3];
    }
    uint8x8x4_t vl = vld4_u8(stg_l);
    uint8x8x4_t vr = vld4_u8(stg_r);
    uint8x8_t vfx  = vld1_u8(col_fx8 + dx);
    uint8x8_t vfxc = vsub_u8(vdup_n_u8(255), vfx);
    uint8x8x4_t vout;
    for (int ch = 0; ch < 4; ++ch) {
      uint16x8_t acc = vmull_u8(vl.val[ch], vfxc);
      acc = vmlal_u8(acc, vr.val[ch], vfx);
      vout.val[ch] = vrshrn_n_u16(acc, 8);
    }
    vst4_u8(out + dx * 4, vout);
  }
  for (; dx < dw; ++dx) {
    const uint8_t* L = src_row + col_ix0[dx];
    const uint8_t* R = src_row + col_ix1[dx];
    const uint32_t fx  = col_fx8[dx];
    const uint32_t fxc = 255u - fx;
    out[dx*4+0] = static_cast<uint8_t>((L[0]*fxc + R[0]*fx + 128u) >> 8);
    out[dx*4+1] = static_cast<uint8_t>((L[1]*fxc + R[1]*fx + 128u) >> 8);
    out[dx*4+2] = static_cast<uint8_t>((L[2]*fxc + R[2]*fx + 128u) >> 8);
    out[dx*4+3] = static_cast<uint8_t>((L[3]*fxc + R[3]*fx + 128u) >> 8);
  }
}

static void BilinearVertBlend4(
    const uint8_t* row0, const uint8_t* row1,
    uint8_t fy8, uint8_t* dest_row, int dw) {
  const uint8x8_t vfy  = vdup_n_u8(fy8);
  const uint8x8_t vfyc = vdup_n_u8(static_cast<uint8_t>(255u - fy8));
  const int nbytes = dw * 4;
  int i = 0;
  for (; i + 16 <= nbytes; i += 16) {
    uint8x16_t v0 = vld1q_u8(row0 + i);
    uint8x16_t v1 = vld1q_u8(row1 + i);
    uint16x8_t accl = vmull_u8(vget_low_u8(v0),  vfyc);
    accl = vmlal_u8(accl, vget_low_u8(v1),  vfy);
    uint16x8_t acch = vmull_u8(vget_high_u8(v0), vfyc);
    acch = vmlal_u8(acch, vget_high_u8(v1), vfy);
    vst1q_u8(dest_row + i,
             vcombine_u8(vrshrn_n_u16(accl, 8), vrshrn_n_u16(acch, 8)));
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

#endif // __ARM_NEON

} // namespace


// ---------------------------------------------------------------------------

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

  // Base pointers already adjusted for the buffer's own offset, so per-pixel
  // address = base + y * stride + x * depth  (no further offset arithmetic).
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

        // Gathered dest row buffer.  When the source row index doesn't change
        // between consecutive dest rows (common at upscale ratios ≥1×) we skip
        // re-gathering and NEON-memcpy the already-built row instead.
        const int row_bytes = dw * depth;
#ifdef __ARM_NEON
        std::vector<uint8_t> gathered(row_bytes);
#endif
        int cached_sy = -1;

        for (int dy_local = 0; dy_local < height; ++dy_local) {
          const int dy = dy0 + dy_local;
          const int sy = src_rect.Y + (dy * sh) / dh;
          uint8_t* dest_row = dest_base + (dest_rect.Y + dy) * dest_stride +
                              dest_rect.X * depth;

#ifdef __ARM_NEON
          if (sy == cached_sy) {
            // Same source row as last iteration — NEON-copy the cached result.
            memcpy(dest_row, gathered.data(), row_bytes);
            continue;
          }
          cached_sy = sy;
          const uint8_t* src_row = src_base + sy * src_stride;
          uint8_t* out = gathered.data();
#else
          (void)cached_sy;
          const uint8_t* src_row = src_base + sy * src_stride;
          uint8_t* out = dest_row;
#endif

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

#ifdef __ARM_NEON
          // First write: copy gathered buffer to dest row.
          memcpy(dest_row, gathered.data(), row_bytes);
#endif
        }
      });
      break;
    }

    case SCALE_BILINEAR: {
#ifdef __ARM_NEON
      if (depth == 3 || depth == 4) {
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

        // Sharpening strength: 0=off, 64=mild (k≈1/4), 128=strong (k≈1/2).
        // Applied only for depth==3; depth==4 uses plain blend.
        const uint8_t sharp_strength = (depth == 3) ? 64u : 0u;

        ExecuteInParallel([=, &col_ix0, &col_ix1, &col_fx8](
                              int num_threads, int i) {
          const int rows_per = dh / num_threads;
          const int height   = (i == num_threads - 1)
                                   ? (dh - i * rows_per) : rows_per;
          const int dy0 = i * rows_per;

          // 4-row horizontal-pass cache: hrow[0..3] correspond to source rows
          // iy-1, iy, iy+1, iy+2 (prev, top, bot, next for sharpening).
          const int row_bytes = dw * depth;
          std::vector<uint8_t> hbuf(row_bytes * 4);
          uint8_t* hrow[4] = {
            hbuf.data(),
            hbuf.data() + row_bytes,
            hbuf.data() + row_bytes * 2,
            hbuf.data() + row_bytes * 3,
          };
          int cached_iy[4] = { -1, -1, -1, -1 };

          // Helper: ensure hrow[slot] contains the horizontally-filtered result
          // for source row ry, clamped to [src_rect.Y, src_rect.Y+sh-1].
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

            if (depth == 3 && sharp_strength > 0) {
              // Slots: 0=prev(iy-1), 1=top(iy), 2=bot(iy1), 3=next(iy1+1)
              ensure_hrow(0, iy - 1);
              ensure_hrow(1, iy);
              ensure_hrow(2, iy1);
              ensure_hrow(3, iy1 + 1);

              if (iy == iy1) {
                // At boundary: top==bot, blend is just a copy.
                memcpy(dest_row, hrow[1], row_bytes);
              } else {
                BilinearVertBlendSharp3(hrow[1], hrow[2], hrow[0], hrow[3],
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
      // Fall through to scalar for other depths.
#endif // __ARM_NEON

      // Scalar bilinear (non-NEON or depth not 3/4).
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
