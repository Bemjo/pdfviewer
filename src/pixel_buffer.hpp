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

// This file declares the PixelBuffer class, which represents a rectangular
// matrix of pixels.

#ifndef PIXEL_BUFFER_HPP
#define PIXEL_BUFFER_HPP

#include <cstdint>

// A class that represents a rectangular matrix of pixels.
class PixelBuffer {
 public:
  // Size in pixels.
  struct Size {
    int Width;
    int Height;

    Size(int width, int height) : Width(width), Height(height) {}
  };
  // Color format of a pixel buffer.
  class Format {
   public:
    // Length of a pixel, in bytes. Must be between 0 and 4.
    virtual int GetDepth() const = 0;
    // Method to pack an RGB tuple into a pixel value.
    virtual uint32_t Pack(uint8_t r, uint8_t g, uint8_t b) const = 0;
    // This is required to keep C++ happy.
    virtual ~Format() {}
  };
  // A rectangular area on this pixel buffer.
  struct Rect {
    // Coordinates of the top-left corner of the rect.
    int X, Y;
    // Size of the rect.
    int Width, Height;

    explicit Rect(int x = 0, int y = 0, int width = 0, int height = 0)
        : X(x), Y(y), Width(width), Height(height) {}
  };

  // Constructs a new PixelBuffer object, and allocate memory. Will take
  // ownership of allocated memory. Does NOT take ownership of format.
  PixelBuffer(const Size& size, const Format* format);
  // Constructs a new PixelBuffer object, using a pre-allocated buffer. Will NOT
  // take ownership of the buffer. Does NOT take ownership of format.
  PixelBuffer(
      const Size& size, const Format* format, uint8_t* buffer,
      const Size& allocated_size, const Size& offset);
  // Will free buffer if _has_ownship is true.
  ~PixelBuffer();

  // Returns the size of this buffer in pixels.
  Size GetSize() const;
  // Returns a rect covering the buffer exactly.
  Rect GetRect() const;

  // Copies a region in the current pixel buffer to another pixel buffer. The
  // destination region must be at least as large in both dimensions than the
  // source region. The source region is centered if the destination region is
  // larger, and the unaffected areas are set to black. This is multi-threaded.
  void Copy(
      const Rect& src_rect, const Rect& dest_rect, PixelBuffer* dest) const;

  uint8_t* GetRawBuffer() const;
  int GetAllocatedStrideBytes() const;
 private:
  // Size of the buffer.
  Size _size;
  // The allocated size of the buffer. If the buffer was allocated by this
  // class, _allocated_size is equal to _size. If the buffer was
  // allocated by the framebuffer device driver, _allocated_size may be
  // larger than _size.
  Size _allocated_size;
  // The offset of the buffer within the allocated buffer. If the buffer was
  // allocated by this class, this will always be (0, 0). If the buffer was
  // allocated by the framebuffer device driver, this may be non-zero.
  Size _offset;
  // Format of current buffer.
  const Format* _format;
  // Pointer to buffer memory. Whether we own this memory depends on the
  // _has_ownership flag.
  uint8_t* _buffer;
  // Whether we own _buffer.
  bool _has_ownership;

  // Returns the size of the buffer in bytes.
  int GetBufferByteSize() const;
  // Returns the address in memory corresponding to the pixel (x, y).
  uint8_t* GetPixelAddress(int x, int y) const;

  // Disable copy and assign.
  PixelBuffer(const PixelBuffer&);
  PixelBuffer& operator=(const PixelBuffer&);
};

#endif
