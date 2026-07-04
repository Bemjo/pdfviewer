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

#include "viewer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>

#include "document.hpp"
#include "framebuffer.hpp"

const float Viewer::MAX_ZOOM = 20.0f;
const float Viewer::MIN_ZOOM = 0.1f;

struct ProfileTimer {
    std::chrono::time_point<std::chrono::steady_clock> start_time;

    // Starts the timer the moment it is created
    ProfileTimer() {
        Reset();
    }

    void Reset() {
        start_time = std::chrono::steady_clock::now();
    }

    // Returns formatted string in milliseconds
    std::string GetElapsedString() const {
        auto end_time = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        return "Time: " + std::to_string(diff.count()) + " ms";
    }
};

Viewer::Viewer(
    Document* doc, Framebuffer* fb, const Viewer::State& state,
    int render_cache_size, PixelBuffer::ScaleMode scale_mode,
    uint8_t sharpen_strength, int max_render_width)
    : _doc(doc),
      _fb(fb),
      _state(state),
      _cache_size(render_cache_size),
      _scale_mode(scale_mode),
      _sharpen_strength(sharpen_strength),
      _max_render_width(max_render_width),
      _last_rendered_page(-1),
      _last_preloaded_page(-1),
      _render_cache(this, render_cache_size) {
  assert(_doc != nullptr);
  assert(_fb != nullptr);
}

Viewer::~Viewer() {}

float Viewer::ResolveZoom(
    int page, float zoom, int rotation,
    int screen_w, int screen_h) const {
  if (zoom == ZOOM_TO_FIT || zoom == ZOOM_TO_WIDTH) {
    const Document::PageSize ps = _doc->GetPageSize(page, 1.0f, rotation);
    if (zoom == ZOOM_TO_WIDTH) {
      zoom = static_cast<float>(screen_w) / static_cast<float>(ps.Width);
    } else {
      zoom = std::min(
          static_cast<float>(screen_w) / static_cast<float>(ps.Width),
          static_cast<float>(screen_h) / static_cast<float>(ps.Height));
    }
  }
  return std::max(MIN_ZOOM, std::min(MAX_ZOOM, zoom));
}

void Viewer::Render() {
  // 1. Clamp page.
  const int n = _doc->GetNumPages();
  const int page = std::max(0, std::min(n - 1, _state.Page));
  const bool refresh = (page == _last_rendered_page);

  // early exit at trying to navigate past document boundaries.
  if (refresh && _state.Page != page) {
    _state.Page = page;
    return;
  }

  // 2. On navigation, manage the cache and restore per-page state.
  if (!refresh) {
    const int jump = (_last_rendered_page >= 0)
                     ? std::abs(page - _last_rendered_page) : 0;

    if (jump > _cache_size) {
      _render_cache.Flush();
      _last_preloaded_page = -1;
    } else {
      _render_cache.CancelOutsideRange(page - _cache_size, page + _cache_size);
    }

    {
      std::lock_guard<std::mutex> lock(_page_states_mutex);
      auto it = _page_states.find(page);
      if (it != _page_states.end()) {
        _state.Zoom = it->second.Zoom;
        _state.XOffset = it->second.XOffset;
        _state.YOffset = it->second.YOffset;
        _state.Rotation = it->second.Rotation;
      } else {
        _state.Zoom = ZOOM_TO_FIT;
        _state.XOffset = 0;
        _state.YOffset = 0;
        _state.Rotation = 0;
      }
    }
  }

  // 3. Compute viewport parameters.
  const PixelBuffer::Size screen_size = _fb->GetSize();
  const float zoom = ResolveZoom(
      page, _state.Zoom, _state.Rotation,
      screen_size.Width, screen_size.Height);

  const Document::PageSize full_ps =
      _doc->GetPageSize(page, zoom, _state.Rotation);

  const int max_x = std::max(0, full_ps.Width  - screen_size.Width);
  const int max_y = std::max(0, full_ps.Height - screen_size.Height);
  const int src_x = std::max(0, std::min(max_x, _state.XOffset));
  const int src_y = std::max(0, std::min(max_y, _state.YOffset));
  const int vp_w  = std::min(screen_size.Width,  full_ps.Width  - src_x);
  const int vp_h  = std::min(screen_size.Height, full_ps.Height - src_y);

  // 4. Write per-page state before touching the cache so the worker's
  //    Load() always sees the latest zoom/pan/rotation for this page.
  {
    std::lock_guard<std::mutex> lock(_page_states_mutex);
    _page_states[page] = PerPageState{_state.Zoom, src_x, src_y, _state.Rotation};
  }

  // 5. Retrieve from cache, or load if not present.
  _render_cache.SetCenter(page);
  ProfileTimer timer;
  std::shared_ptr<PixelBuffer> buffer = _render_cache.Get(page, refresh);
  const std::string cache_str = timer.GetElapsedString();
  timer.Reset();

  // 6. Blit to framebuffer. The buffer is already upscaled to screen_vp size.
  //    Pass the full framebuffer rect as dest so Copy() clears the margins
  //    in the same pass as the content — no separate clear needed.
  _fb->Render(*buffer,
              buffer->GetRect(),
              PixelBuffer::Rect(0, 0, screen_size.Width, screen_size.Height));

  const std::string fbrender_str = timer.GetElapsedString();

  const std::string filt = (
    _scale_mode == PixelBuffer::ScaleMode::SCALE_NONE ? "None" :
    _scale_mode == PixelBuffer::ScaleMode::SCALE_NEAREST ? "Nearest" :
    _scale_mode == PixelBuffer::ScaleMode::SCALE_BILINEAR ? "Bilinear" :
    "Bicubic"
  );
  const std::string page_str = "Page: " + std::to_string(page + 1) + " / " + std::to_string(n);
  const std::string filter_str = "Filter: " + filt + (_scale_mode == PixelBuffer::ScaleMode::SCALE_BILINEAR ? " - " + std::to_string(_sharpen_strength) : "");
  const std::string cache_time_str = "Cache Get: " + cache_str;
  const std::string fb_time_str = "FB Render: " + fbrender_str;

  _fb->DrawText(0, 0, page_str, Framebuffer::TextRenderParams().SetFontSize(32).SetWidth(3));
  _fb->DrawText(0, 40, filter_str, Framebuffer::TextRenderParams().SetFontSize(32).SetWidth(3));
  _fb->DrawText(0, 80, cache_time_str, Framebuffer::TextRenderParams().SetFontSize(32).SetWidth(3));
  _fb->DrawText(0, 120, fb_time_str, Framebuffer::TextRenderParams().SetFontSize(32).SetWidth(3));

  _last_rendered_page = page;

  // 7. Store corrected state.
  _state.Page = page;
  if ((_state.Zoom != ZOOM_TO_WIDTH) && (_state.Zoom != ZOOM_TO_FIT)) {
    _state.Zoom = zoom;
  }
  _state.ActualZoom = zoom;
  _state.XOffset = src_x;
  _state.YOffset = src_y;
  _state.PageWidth = full_ps.Width;
  _state.PageHeight = full_ps.Height;
  _state.ScreenWidth = screen_size.Width;
  _state.ScreenHeight = screen_size.Height;

  // 8. Schedule forward pre-caching once per page change (or after Flush).
  //    Cache layout with effective size S = _cache_size + floor((_cache_size-2)/2):
  //      1 slot  — current page
  //      n-1 slots — prefetched forward pages
  //      remaining slots — trailing pages retained naturally by FIFO eviction
  if (_last_preloaded_page != page) {
    _last_preloaded_page = page;

    for (int i = 1; i <= _cache_size; ++i) {
      const int fwd = page + i;
      if (fwd < n && !_render_cache.Contains(fwd)) {
        _render_cache.Prepare(fwd);
      }
      const int bwd = page - i;
      if (bwd >= 0 && !_render_cache.Contains(bwd)) {
        _render_cache.Prepare(bwd);
      }
    }
  }
}

void Viewer::GetState(Viewer::State* state) const {
  state->Page = _state.Page;
  state->NumPages = _state.NumPages;
  state->Zoom = _state.Zoom;
  state->ActualZoom = _state.ActualZoom;
  state->Rotation = _state.Rotation;
  state->XOffset = _state.XOffset;
  state->YOffset = _state.YOffset;
  state->PageWidth = _state.PageWidth;
  state->PageHeight = _state.PageHeight;
  state->ScreenWidth = _state.ScreenWidth;
  state->ScreenHeight = _state.ScreenHeight;
}

void Viewer::SetState(const State& state) { _state = state; }

Viewer::RenderCache::RenderCache(Viewer* parent, int size)
    : Cache<int, std::shared_ptr<PixelBuffer>>(size), _parent(parent) {}

Viewer::RenderCache::~RenderCache() { Clear(); }

bool Viewer::RenderCache::EvictBefore(
    const int& a, const int& b, const int& center) const {
  return std::abs(a - center) > std::abs(b - center);
}

std::shared_ptr<PixelBuffer> Viewer::RenderCache::Load(const int& page) {
  // Read per-page params under the mutex for as short a time as possible.
  Viewer::PerPageState params;
  bool found;
  {
    std::lock_guard<std::mutex> lock(_parent->_page_states_mutex);
    auto it = _parent->_page_states.find(page);
    found = (it != _parent->_page_states.end());
    if (found) {
      params = it->second;
    }
  }

  const PixelBuffer::Size screen = _parent->_fb->GetSize();
  const float zoom_raw = found ? params.Zoom     : Viewer::ZOOM_TO_FIT;
  const int   rotation = found ? params.Rotation : 0;
  const float zoom = _parent->ResolveZoom(
      page, zoom_raw, rotation, screen.Width, screen.Height);

  const Document::PageSize full_ps =
      _parent->_doc->GetPageSize(page, zoom, rotation);

  const int x_raw = found ? params.XOffset : 0;
  const int y_raw = found ? params.YOffset : 0;
  const int max_x = std::max(0, full_ps.Width  - screen.Width);
  const int max_y = std::max(0, full_ps.Height - screen.Height);
  const int src_x = std::max(0, std::min(max_x, x_raw));
  const int src_y = std::max(0, std::min(max_y, y_raw));
  // Screen-space viewport: the portion of the framebuffer this page covers.
  const int screen_vp_w = std::min(screen.Width,  full_ps.Width  - src_x);
  const int screen_vp_h = std::min(screen.Height, full_ps.Height - src_y);

  // Render-cap scale factor: derived from the FRAMEBUFFER's own dimensions,
  // not the viewport's. If the framebuffer is wider than _max_render_width,
  // we render at the resolution the framebuffer would have if its width
  // were _max_render_width and its height followed the framebuffer's own
  // aspect ratio (e.g. 1920x1080 @ cap=1280 -> render at 1280x720, still
  // 16:9). This fixed ratio is then applied uniformly to whatever portion
  // of the page is actually visible, so zoom/pan stay locked to the same
  // page location and content never stretches -- it only shrinks uniformly
  // before the upscale. Using the viewport's own size here (as opposed to
  // the framebuffer's) was the bug: a narrow viewport (e.g. a portrait page
  // on a landscape screen) could stay under the cap even though the
  // framebuffer itself exceeds it, silently disabling the cap.
  const bool needs_upscale = (screen.Width > _parent->_max_render_width) &&
      (_parent->_scale_mode != PixelBuffer::ScaleMode::SCALE_NONE);
  const double render_scale = needs_upscale
      ? static_cast<double>(_parent->_max_render_width) / static_cast<double>(screen.Width)
      : 1.0;

  // Allocate the full screen-sized output buffer that goes into the cache.
  std::shared_ptr<PixelBuffer> out_buffer(
      _parent->_fb->NewPixelBuffer(PixelBuffer::Size(screen_vp_w, screen_vp_h)));

  if (needs_upscale) {
    const double render_zoom = zoom * render_scale;

    // Pan offsets scale down to match the reduced zoom.
    const int render_src_x = static_cast<int>(src_x * render_scale);
    const int render_src_y = static_cast<int>(src_y * render_scale);

    // Render buffer dimensions follow the same framebuffer-derived scale
    // factor as the zoom, so the crop's own aspect ratio (which may differ
    // from the framebuffer's, e.g. a partial/portrait viewport) is
    // preserved -- only the overall resolution shrinks uniformly. Not
    // re-clamped against the page size at render_zoom -- doing so
    // previously shrank the crop near page edges and caused panning to
    // stretch/compress content instead of scrolling.
    const int actual_render_w = std::max(1, static_cast<int>(screen_vp_w * render_scale));
    const int actual_render_h = std::max(1, static_cast<int>(screen_vp_h * render_scale));

    // Render into a temporary capped buffer, then upscale into out_buffer.
    std::unique_ptr<PixelBuffer> tmp_buffer(
        _parent->_fb->NewPixelBuffer(PixelBuffer::Size(actual_render_w, actual_render_h)));
    _parent->_doc->Render(
        tmp_buffer->GetRawBuffer(), page, static_cast<float>(render_zoom), rotation,
        render_src_x, render_src_y, actual_render_w, actual_render_h);
    tmp_buffer->Copy(
        tmp_buffer->GetRect(), out_buffer->GetRect(),
        out_buffer.get(), _parent->_scale_mode, _parent->_sharpen_strength);
  } else {
    _parent->_doc->Render(
        out_buffer->GetRawBuffer(), page, zoom, rotation,
        src_x, src_y, screen_vp_w, screen_vp_h);
  }

  return out_buffer;
}

void Viewer::RenderCache::Discard(
    const int& page, const std::shared_ptr<PixelBuffer>& value) {
  (void)page;
  (void)value;
  // shared_ptr handles memory; nothing to do.
}

Viewer::PerPageState Viewer::GetPageState(int page) {
  std::lock_guard<std::mutex> lock(_page_states_mutex);
  if (page < 0) {
    return _page_states[_state.Page];
  }
  page = std::min(page, _state.NumPages - 1);
  return _page_states[page];
}

void Viewer::SetScaleMode(PixelBuffer::ScaleMode mode, uint8_t sharpen_strength) {
  _scale_mode = mode;
  _sharpen_strength = sharpen_strength;
  FlushCache();
}

void Viewer::FlushCache() {
  _render_cache.Flush();
  _last_preloaded_page = -1;
}