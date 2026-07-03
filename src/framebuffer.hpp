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

// This file declares the framebuffer abstraction.

#ifndef FRAMEBUFFER_HPP
#define FRAMEBUFFER_HPP

#include <linux/fb.h>

#include <cstdint>
#include <memory>
#include <string>

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_STROKER_H

#include "pixel_buffer.hpp"

// An abstraction for a framebuffer device.
class Framebuffer {
 public:
  struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;

    Color(uint8_t R = 0, uint8_t G = 0, uint8_t B = 0) :
          r(R), g(G), b(B) {}
  };

  enum AnchorPosition {
    TopLeft, Top, TopRight,
    Left, Center, Right,
    BottomLeft, Bottom, BottomRight
  };

  struct TextRenderParams {
      Color Fill;
      Color Outline;
      int OutlineWidth;
      int FontSize;
      AnchorPosition Anchor;

      // Explicit constructor to initialize everything
      TextRenderParams() :
          Fill(255, 255, 255),
          Outline(0, 0, 0),
          OutlineWidth(1),
          FontSize(18),
          Anchor(AnchorPosition::TopLeft) {}

      // Builder methods for inline chaining
      TextRenderParams& SetAnchor(AnchorPosition a) { Anchor = a; return *this; }
      TextRenderParams& SetFontSize(int s) { FontSize = s; return *this; }
      TextRenderParams& SetFill(const Color &c) { Fill = c; return *this; }
      TextRenderParams& SetOutline(const Color &c) { Outline = c; return *this; }
      TextRenderParams& SetWidth(int w) { OutlineWidth = w; return *this; }
  };

  static const char* const DEFAULT_FRAMEBUFFER_DEVICE;
  // Factory method to initialize a framebuffer device and returns an
  // abstraction object. Returns nullptr if the initialization failed. Caller
  // owns returned object.
  static Framebuffer* Open(
      const std::string& device = DEFAULT_FRAMEBUFFER_DEVICE);
  virtual ~Framebuffer();

  bool InitFreetype(const uint8_t* font_data, size_t font_size_bytes);

  // Creates a new pixel buffer with the given size. The pixel buffer will have
  // the same color settings as the screen. Caller owns returned value.
  PixelBuffer* NewPixelBuffer(const PixelBuffer::Size& size);

  // Retrieve the dimensions of the current display, in pixels.
  PixelBuffer::Size GetSize() const;
  // Retrieve the dimensions of the framebuffer device's allocated memory
  // buffer, a.k.a. its virtual resolution, in pixels.
  PixelBuffer::Size GetAllocatedSize() const;
  // Retrieve the offset of the visible area within the device's allocated
  // memory buffer, in pixels.
  PixelBuffer::Size GetOffset() const;

  // Renders a region of src onto the framebuffer. src_rect is the region
  // within src to read; dest_rect is the region of the framebuffer to write
  // into. src and dest must be the same size.
  void Render(const PixelBuffer& src,
              const PixelBuffer::Rect& src_rect,
              const PixelBuffer::Rect& dest_rect);

  void DrawText(int x, int y, const std::string& text,
                const TextRenderParams &params = TextRenderParams());

  // Return debugging information as a string.
  std::string GetDebugInfoString();

 private:
  // Color format of the framebuffer.
  class Format : public PixelBuffer::Format {
   public:
    // Grab settings from a fb_var_screeninfo.
    explicit Format(const fb_var_screeninfo& vinfo);
    // This is required to keep C++ happy.
    virtual ~Format() {}
    // See PixelBuffer::Format.
    int GetDepth() const override;
    // See PixelBuffer::Format.
    uint32_t Pack(uint8_t r, uint8_t g, uint8_t b) const override;

   private:
    fb_var_screeninfo _vinfo;
  };

  // The framebuffer device.
  const std::string _device;
  // File descriptor of the opened framebuffer device.
  int _fd;
  // Framebuffer info structures.
  fb_var_screeninfo _vinfo;
  fb_fix_screeninfo _finfo;
  // mmap'd buffer.
  uint8_t* _buffer;
  std::unique_ptr<Format> _format;
  // Pixel buffer object managing the mmap'ed buffer.
  std::unique_ptr<PixelBuffer> _pixel_buffer;

  FT_Library _ft;
  FT_Face _face;
  FT_Stroker _stroker;

  // Contructors are disallowed. Use factory method Open() instead.
  Framebuffer(const std::string& device);
  // No copying is allowed.
  Framebuffer(const Framebuffer&);
  Framebuffer& operator=(const Framebuffer&);

  // Returns the size of the mmap'd buffer in bytes.
  int GetBufferByteSize() const;
};

#endif
