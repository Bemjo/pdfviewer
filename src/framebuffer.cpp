/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                           *
 *  Copyright (C) 2012-2020 Chuan Ji                                         *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *   http://www.apache.org/licenses/LICENSE-2.0                              *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// This file implements the framebuffer abstraction.

#include "framebuffer.hpp"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>

const char* const Framebuffer::DEFAULT_FRAMEBUFFER_DEVICE = "/dev/fb0";

Framebuffer* Framebuffer::Open(const std::string& device) {
  std::unique_ptr<Framebuffer> fb(new Framebuffer(device));

  if ((fb->_fd = open(device.c_str(), O_RDWR)) == -1) {
    goto error;
  }
  if ((ioctl(fb->_fd, FBIOGET_VSCREENINFO, &(fb->_vinfo)) == -1) ||
      (ioctl(fb->_fd, FBIOGET_FSCREENINFO, &(fb->_finfo)) == -1)) {
    goto error;
  }
  fb->_buffer = reinterpret_cast<uint8_t*>(mmap(
      nullptr, fb->GetBufferByteSize(), PROT_READ | PROT_WRITE, MAP_SHARED,
      fb->_fd, 0));
  if (fb->_buffer == MAP_FAILED) {
    goto error;
  }

  fb->_format.reset(new Format(fb->_vinfo));
  fb->_pixel_buffer.reset(new PixelBuffer(
      fb->GetSize(), fb->_format.get(), fb->_buffer, fb->GetAllocatedSize(),
      fb->GetOffset()));
  return fb.release();

error:
  perror(("Error initializing framebuffer device \"" + device + "\"").c_str());
  return nullptr;
}

bool Framebuffer::InitFreetype(const uint8_t* font_data, size_t font_size_bytes) {
  if (FT_Init_FreeType(&_ft)) {
    return false;
  }

  if (FT_New_Memory_Face(_ft, font_data, font_size_bytes, 0, &_face)) {
    FT_Done_FreeType(_ft);
    return false;
  }

  if (FT_Stroker_New(_ft, &_stroker)) {
    FT_Done_FreeType(_ft);
    FT_Done_Face(_face);
    return false;
  }

  return true;
}

Framebuffer::Framebuffer(const std::string& device)
    : _device(device),
      _buffer(nullptr),
      _format(nullptr),
      _pixel_buffer(nullptr),
      _ft(nullptr),
      _face(nullptr),
      _stroker(nullptr) {}

Framebuffer::~Framebuffer() {
  if (_buffer != nullptr && _buffer != MAP_FAILED) {
    memset(_buffer, 0, GetBufferByteSize());
    munmap(_buffer, GetBufferByteSize());
  }
  if (_fd != -1) {
    close(_fd);
  }

  if (_stroker != nullptr) {
    FT_Stroker_New(_ft, &_stroker);
  }

  if (_face != nullptr) {
    FT_Done_Face(_face);
  }

  if (_ft != nullptr) {
    FT_Done_FreeType(_ft);
  }
}

std::string Framebuffer::GetDebugInfoString() {
  std::ostringstream out;

  out << "Device:\t\t\t" << _device << std::endl;
  out << "Visible resolution:\t" << _vinfo.xres << " x " << _vinfo.yres
      << std::endl;
  out << "Virtual resolution:\t" << _vinfo.xres_virtual << " x "
      << _vinfo.yres_virtual << std::endl;
  out << "Offset:\t\t\t" << _vinfo.xoffset << ", " << _vinfo.yoffset
      << std::endl;
  out << "Buffer size:\t\t" << (_finfo.smem_len / _format->GetDepth()) << " ("
      << _finfo.smem_len << " bytes)" << std::endl;
  out << "Buffer width:\t\t" << (_finfo.line_length / _format->GetDepth())
      << " (" << _finfo.line_length << " bytes)" << std::endl;
  out << "Buffer height:\t\t" << (_finfo.smem_len / _finfo.line_length)
      << std::endl;
  out << "Bits per pixel:\t\t" << _vinfo.bits_per_pixel << std::endl;
  out << "Bit depth:\t\t" << _format->GetDepth() << std::endl;
  out << "Red:\t\t\t"
      << "length " << _vinfo.red.length << ", offset " << _vinfo.red.offset
      << std::endl;
  out << "Green:\t\t\t"
      << "length " << _vinfo.green.length << ", offset " << _vinfo.green.offset
      << std::endl;
  out << "Blue:\t\t\t"
      << "length " << _vinfo.blue.length << ", offset " << _vinfo.blue.offset
      << std::endl;
  out << "Non-std pixel format:\t" << _vinfo.nonstd << std::endl;

  return out.str();
}

PixelBuffer* Framebuffer::NewPixelBuffer(const PixelBuffer::Size& size) {
  return new PixelBuffer(size, _format.get());
}

int Framebuffer::GetBufferByteSize() const { return _finfo.smem_len; }

PixelBuffer::Size Framebuffer::GetSize() const {
  return PixelBuffer::Size(_vinfo.xres, _vinfo.yres);
}

PixelBuffer::Size Framebuffer::GetAllocatedSize() const {
  return PixelBuffer::Size(
      _finfo.line_length / _format->GetDepth(),
      _finfo.smem_len / _finfo.line_length);
}

PixelBuffer::Size Framebuffer::GetOffset() const {
  return PixelBuffer::Size(_vinfo.xoffset, _vinfo.yoffset);
}

void Framebuffer::Render(
    const PixelBuffer& src, const PixelBuffer::Rect& src_rect,
    const PixelBuffer::Rect& dest_rect) {
  const int depth = _format->GetDepth();
  const int fb_stride = _pixel_buffer->GetAllocatedStrideBytes();
  // fb_base: start of the visible area in the mmap'd buffer.
  uint8_t* fb_base = _buffer +
      (_vinfo.yoffset * (_finfo.line_length / depth) + _vinfo.xoffset) * depth;

  const int margin_left   = (dest_rect.Width  - src_rect.Width)  / 2;
  const int margin_right  =  dest_rect.Width  - margin_left - src_rect.Width;
  const int margin_top    = (dest_rect.Height - src_rect.Height) / 2;
  const int margin_bottom =  dest_rect.Height - margin_top - src_rect.Height;
  const int dest_row_bytes = dest_rect.Width * depth;
  const int src_row_bytes  = src_rect.Width  * depth;

  const uint8_t* src_base = src.GetRawBuffer();
  const int src_stride = src.GetAllocatedStrideBytes();

  for (int y = 0; y < margin_top; ++y)
    memset(fb_base + (dest_rect.Y + y) * fb_stride + dest_rect.X * depth,
           0, dest_row_bytes);

  for (int y = 0; y < src_rect.Height; ++y) {
    uint8_t* dest_row = fb_base +
        (dest_rect.Y + margin_top + y) * fb_stride + dest_rect.X * depth;
    if (margin_left)
      memset(dest_row, 0, margin_left * depth);
    memcpy(dest_row + margin_left * depth,
           src_base + (src_rect.Y + y) * src_stride + src_rect.X * depth,
           src_row_bytes);
    if (margin_right)
      memset(dest_row + (margin_left + src_rect.Width) * depth,
             0, margin_right * depth);
  }

  for (int y = 0; y < margin_bottom; ++y)
    memset(fb_base +
               (dest_rect.Y + margin_top + src_rect.Height + y) * fb_stride +
               dest_rect.X * depth,
           0, dest_row_bytes);
}

Framebuffer::Format::Format(const fb_var_screeninfo& vinfo) : _vinfo(vinfo) {}

int Framebuffer::Format::GetDepth() const {
  return (_vinfo.bits_per_pixel + 7) >> 3;
}

uint32_t Framebuffer::Format::Pack(uint8_t r, uint8_t g, uint8_t b) const {
  return ((static_cast<uint32_t>(r) >> (8 - _vinfo.red.length))
          << _vinfo.red.offset) |
         ((static_cast<uint32_t>(g) >> (8 - _vinfo.green.length))
          << _vinfo.green.offset) |
         ((static_cast<uint32_t>(b) >> (8 - _vinfo.blue.length))
          << _vinfo.blue.offset);
}

void Framebuffer::DrawText(int x, int y, const std::string& text, const TextRenderParams &params) {
  if (_ft == nullptr || _face == nullptr || _stroker == nullptr) {
    return;
  }

  FT_Set_Pixel_Sizes(_face, 0, std::max(8, params.FontSize));

  int max_width = 0;
  int current_line_width = 0;
  int lines = 1;

  for (char c : text) {
    if (c == '\n') {
      lines++;
      if (current_line_width > max_width) max_width = current_line_width;
      current_line_width = 0;
      continue;
    }
    // Load without rendering just to get the advance metrics
    if (FT_Load_Char(_face, c, FT_LOAD_DEFAULT)) continue;
    current_line_width += (_face->glyph->advance.x >> 6);
  }
  if (current_line_width > max_width) max_width = current_line_width;

  int line_height = _face->size->metrics.height >> 6;
  int total_height = lines * line_height;

  // Account for the outline width so flush-right/bottom text doesn't clip
  int padding = (params.OutlineWidth > 0) ? params.OutlineWidth : 0;

  // Calculate anchored starting coordinates
  int start_x = x;
  int start_y = y;

  // Horizontal shifts
  if (params.Anchor == Top || params.Anchor == Center || params.Anchor == Bottom) {
    start_x -= (max_width / 2);
  } else if (params.Anchor == TopRight || params.Anchor == Right || params.Anchor == BottomRight) {
    start_x -= (max_width + padding);
  }

  // Vertical shifts
  if (params.Anchor == Left || params.Anchor == Center || params.Anchor == Right) {
    start_y -= (total_height / 2);
  } else if (params.Anchor == BottomLeft || params.Anchor == Bottom || params.Anchor == BottomRight) {
    start_y -= (total_height + padding);
  }

  if (params.OutlineWidth > 0) {
    FT_Stroker_Set(_stroker, params.OutlineWidth * 64, FT_STROKER_LINECAP_ROUND, FT_STROKER_LINEJOIN_ROUND, 0);
  }

  const int depth = _format->GetDepth();
  const int screen_w = _vinfo.xres;
  const int screen_h = _vinfo.yres;

  uint8_t* fb_base = _buffer + (_vinfo.yoffset * (_finfo.line_length / depth) + _vinfo.xoffset) * depth;

  int current_x = start_x + padding; // offset initial padding so left edge doesn't clip
  int baseline_y = start_y + padding + (_face->size->metrics.ascender >> 6);

  const uint32_t rgb_mask = (((1 << _vinfo.red.length) - 1) << _vinfo.red.offset) |
                            (((1 << _vinfo.green.length) - 1) << _vinfo.green.offset) |
                            (((1 << _vinfo.blue.length) - 1) << _vinfo.blue.offset);

  // Updated lambda to accept the new Color struct
  auto blend_glyph_direct = [&](FT_BitmapGlyph bitmap_glyph, const Color& color) {
    FT_Bitmap* bitmap = &bitmap_glyph->bitmap;

    for (unsigned int row = 0; row < bitmap->rows; ++row) {
      int draw_y = baseline_y - bitmap_glyph->top + row;
      if (draw_y < 0 || draw_y >= screen_h) continue;

      for (unsigned int col = 0; col < bitmap->width; ++col) {
        int draw_x = current_x + bitmap_glyph->left + col;
        if (draw_x < 0 || draw_x >= screen_w) continue;

        uint8_t alpha = bitmap->buffer[row * bitmap->pitch + col];
        if (alpha == 0) continue;

        uint8_t* pixel_ptr = fb_base + draw_y * _finfo.line_length + draw_x * depth;
        uint32_t bg_pixel = 0;
        uint8_t inv_alpha = 255 - alpha;
        memcpy(&bg_pixel, pixel_ptr, depth);

        Color bg = {
          static_cast<uint8_t>(((bg_pixel >> _vinfo.red.offset) << (8 - _vinfo.red.length)) & 0xFF),
          static_cast<uint8_t>(((bg_pixel >> _vinfo.green.offset) << (8 - _vinfo.green.length)) & 0xFF),
          static_cast<uint8_t>(((bg_pixel >> _vinfo.blue.offset) << (8 - _vinfo.blue.length)) & 0xFF)
        };
        Color blend = {
          static_cast<uint8_t>((color.r * alpha + bg.r * inv_alpha) / 255),
          static_cast<uint8_t>((color.g * alpha + bg.g * inv_alpha) / 255),
          static_cast<uint8_t>((color.b * alpha + bg.b * inv_alpha) / 255)
        };

        uint32_t packed_pixel = _format->Pack(blend.r, blend.g, blend.b);
        if (depth == 4) {
          packed_pixel |= (bg_pixel & ~rgb_mask);
        }

        memcpy(pixel_ptr, &packed_pixel, depth);
      }
    }
  };

  for (char c : text) {
    if (c == '\n') {
      baseline_y += (_face->size->metrics.height >> 6);
      current_x = x;
      continue;
    }

    if (FT_Load_Char(_face, c, FT_LOAD_NO_BITMAP)) continue;

    FT_Glyph glyph;
    if (FT_Get_Glyph(_face->glyph, &glyph)) continue;

    if (params.OutlineWidth > 0) {
      FT_Glyph stroke_glyph = glyph;
      FT_Glyph_Stroke(&stroke_glyph, _stroker, 1);
      FT_Glyph_To_Bitmap(&stroke_glyph, FT_RENDER_MODE_NORMAL, 0, 1);

      // Pass the outline color
      blend_glyph_direct(reinterpret_cast<FT_BitmapGlyph>(stroke_glyph), params.Outline);
      FT_Done_Glyph(stroke_glyph);
    }

    FT_Glyph fill_glyph = glyph;
    FT_Glyph_To_Bitmap(&fill_glyph, FT_RENDER_MODE_NORMAL, 0, 1);

    // Pass the fill color
    blend_glyph_direct(reinterpret_cast<FT_BitmapGlyph>(fill_glyph), params.Fill);

    current_x += (_face->glyph->advance.x >> 6);
    FT_Done_Glyph(fill_glyph);
    FT_Done_Glyph(glyph);
  }
}