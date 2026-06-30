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

// This file defines a Viewer class, which maintains state for rendering a
// Document page on top of Framebuffer.

#ifndef VIEWER_HPP
#define VIEWER_HPP

#include <map>
#include <memory>
#include <mutex>

#include "cache.hpp"
#include "pixel_buffer.hpp"

class Document;
class Framebuffer;
class PixelBuffer;

class Viewer {
 public:
  // Default number of rendered pages to keep in cache.
  enum { DEFAULT_RENDER_CACHE_SIZE = 8 };

  // Default maximum render buffer width. Pages are rendered at most at this
  // width and upscaled to the framebuffer, preserving aspect ratio.
  static const int DEFAULT_MAX_RENDER_WIDTH  = 1280;
  // Minimum allowed render cap width.
  static const int MIN_RENDER_WIDTH  = 640;

  // Zoom modes.
  enum {
    // Automatically zoom to fit current page.
    ZOOM_TO_FIT = -3,
    // Automatically zoom to fit current page width.
    ZOOM_TO_WIDTH = -4,
  };

  // Maximum zoom ratio.
  static const float MAX_ZOOM;
  // Minimum zoom ratio.
  static const float MIN_ZOOM;

  // A structure representing zoom information.
  struct State {
    // The displayed page.
    int Page;
    // The total number of pages in the document. This is written by Render()
    // and is ignored by Render() itself.
    int NumPages;

    // The zoom ratio, or ZOOM_*.
    float Zoom;
    // If Zoom is ZOOM_*, this gives the actual zoom value. This is written by
    // Render() and is ignored by Render() itself.
    float ActualZoom;
    // Rotation of the document, in clockwise degrees.
    int Rotation;

    // Number of screen pixels from top of page to top of displayed view.
    int XOffset;
    // Number of screen pixels from left of page to left of displayed view.
    int YOffset;

    // Width of current page (after zoom and rotation). This is written by
    // Render(), and is ignored by Render() itself.
    int PageWidth;
    // Height of current page (after zoom and rotation). This is written by
    // Render(), and is ignored by Render() itself.
    int PageHeight;
    // Width of framebuffer. This is written by Render(), and is ignored by
    // Render() itself.
    int ScreenWidth;
    // Height of framebuffer. This is written by Render(), and is ignored by
    // Render() itself.
    int ScreenHeight;

    State(
        int page = 0, float zoom = ZOOM_TO_WIDTH, int rotation = 0,
        int x_offset = 0, int y_offset = 0)
        : Page(page),
          Zoom(zoom),
          Rotation(rotation),
          XOffset(x_offset),
          YOffset(y_offset) {}
  };

  struct PerPageState {
    float Zoom;
    int XOffset;
    int YOffset;
    int Rotation;
  };

  // Constructs a new Viewer object. Does not take ownership of the document or
  // the framebuffer object.
  Viewer(
      Document* doc, Framebuffer* fb, const State& state = State(),
      int render_cache_size = DEFAULT_RENDER_CACHE_SIZE,
      PixelBuffer::ScaleMode scale_mode = PixelBuffer::SCALE_BILINEAR,
      uint8_t sharpen_strength = 3,
      int max_render_width  = DEFAULT_MAX_RENDER_WIDTH);
  virtual ~Viewer();

  // Renders the present view to the framebuffer.
  void Render();

  // Stores the current state in the given pointer. Must be called AFTER at
  // least one call to Render().
  void GetState(State* state) const;
  // Sets the current settings. Will use minimum and maximum legal values to
  // replace illegal values. Has no effect until Render() is called.
  void SetState(const State& state);

  PerPageState GetPageState(int page = -1);

  // Changes the upscaling algorithm and flushes the render cache so the next
  // Render() call rebuilds with the new mode.
  void SetScaleMode(PixelBuffer::ScaleMode mode, uint8_t sharpen_strength);
  // Flushes the render cache, forcing a full re-render on next Render() call.
  void FlushCache();

 private:
  // The current document.
  Document* _doc;
  // The framebuffer device.
  Framebuffer* _fb;
  // Settings.
  State _state;
  // Total cache capacity.
  int _cache_size;
  // Upscaling algorithm when the render buffer is smaller than the framebuffer.
  PixelBuffer::ScaleMode _scale_mode;
  uint8_t _sharpen_strength;
  // Maximum render buffer width; content is upscaled to the framebuffer,
  // preserving aspect ratio.
  int _max_render_width;

  // Per-page zoom/pan/rotation state. Protected by _page_states_mutex so
  // background Load() threads can read safely while the main thread writes.
  std::map<int, PerPageState> _page_states;
  std::mutex _page_states_mutex;

  // Page rendered on the last Render() call. Used to detect navigation vs
  // in-page param changes.
  int _last_rendered_page;
  // Page for which adjacent Prepare() calls were last scheduled.
  int _last_preloaded_page;

  // Render cache: key is page number, value is the rendered viewport buffer.
  class RenderCache : public Cache<int, std::shared_ptr<PixelBuffer>> {
   public:
    RenderCache(Viewer* parent, int size);
    virtual ~RenderCache();

   protected:
    std::shared_ptr<PixelBuffer> Load(const int& page) override;
    void Discard(const int& page,
                 const std::shared_ptr<PixelBuffer>& value) override;
    bool EvictBefore(const int& a, const int& b, const int& center) const override;

   private:
    Viewer* _parent;
  };
  RenderCache _render_cache;

  // Resolves ZOOM_TO_FIT / ZOOM_TO_WIDTH for the given page and screen size.
  float ResolveZoom(int page, float zoom, int rotation,
                    int screen_w, int screen_h) const;
};

#endif
