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

#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cmath>

#include "document.hpp"
#include "framebuffer.hpp"

const float Viewer::MAX_ZOOM = 20.0f;
const float Viewer::MIN_ZOOM = 0.1f;

Viewer::Viewer(
    Document* doc, Framebuffer* fb, const Viewer::State& state,
    int render_cache_size)
    : _doc(doc),
      _fb(fb),
      _state(state),
      _render_cache(this, render_cache_size),
      _last_page(-1) {
  assert(_doc != nullptr);
  assert(_fb != nullptr);
}

Viewer::~Viewer() {}

void Viewer::Render() {
  // 1. Process state.
  int page = std::max(0, std::min(_doc->GetNumPages() - 1, _state.Page));

  if (page != _last_page) {
    auto it = _page_states.find(page);
    if (it != _page_states.end()) {
      _state.Zoom     = it->second.Zoom;
      _state.XOffset  = it->second.XOffset;
      _state.YOffset  = it->second.YOffset;
      _state.Rotation = it->second.Rotation;
    } else {
      _state.Zoom     = ZOOM_TO_FIT;
      _state.XOffset  = 0;
      _state.YOffset  = 0;
      _state.Rotation = 0;
    }
    _last_page = page;
  }

  float zoom = _state.Zoom;
  const PixelBuffer::Size& screen_size = _fb->GetSize();
  const Document::PageSize& page_size = _doc->GetPageSize(page, 1.0f, _state.Rotation);

  if (zoom == ZOOM_TO_WIDTH) {
    zoom = static_cast<float>(screen_size.Width) / static_cast<float>(page_size.Width);
  } else if (zoom == ZOOM_TO_FIT) {
    zoom = std::min(
        static_cast<float>(screen_size.Width) / static_cast<float>(page_size.Width),
        static_cast<float>(screen_size.Height) / static_cast<float>(page_size.Height));
  }
  assert(zoom >= 0.0f);
  zoom = std::max(MIN_ZOOM, std::min(MAX_ZOOM, zoom));

  const Document::PageSize full_ps(
      static_cast<int>(page_size.Width * zoom),
      static_cast<int>(page_size.Height * zoom)
  );

  const int max_x = std::max(0, full_ps.Width  - screen_size.Width);
  const int max_y = std::max(0, full_ps.Height - screen_size.Height);
  const int src_x = std::max(0, std::min(max_x, _state.XOffset));
  const int src_y = std::max(0, std::min(max_y, _state.YOffset));
  const int vp_w  = std::min(screen_size.Width,  full_ps.Width  - src_x);
  const int vp_h  = std::min(screen_size.Height, full_ps.Height - src_y);

  _page_states[page]  = PerPageState{_state.Zoom, src_x, src_y, _state.Rotation};

  PixelBuffer::Rect src_rect(0, 0, vp_w, vp_h);

  if (_render_cache.GetSize() > 1) {
    // 2. Render page to buffer and cache.
    std::shared_ptr<PixelBuffer> buffer = _render_cache.Get(
        RenderCacheKey(page, zoom, _state.Rotation, src_x, src_y, vp_w, vp_h));
        _fb->Render(*buffer, src_rect);

    // Preload
    if (page < _doc->GetNumPages() - 1) {
      _render_cache.Prepare(
          RenderCacheKey(page + 1, zoom, _state.Rotation, src_x, src_y, vp_w, vp_h));
    }
  } else {
    // 2. Render directly into the framebuffer
    _doc->Render(_fb->GetRawBuffer(), _fb->GetFramebufferStride(), page, zoom, _state.Rotation, src_x, src_y, vp_w, vp_h);
  }

  // 4 . Store corrected state.
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

bool Viewer::RenderCacheKey::operator<(
    const Viewer::RenderCacheKey& other) const {
  if (Page != other.Page) {
    return Page < other.Page;
  }
  const int rotation_mod = Rotation % 360,
            other_rotation_mod = other.Rotation % 360;
  if (rotation_mod != other_rotation_mod) {
    return rotation_mod < other_rotation_mod;
  }
  if (fabs(Zoom / other.Zoom - 1.0f) >= 0.001f) {
    return Zoom < other.Zoom;
  }
  if (X != other.X) {
    return X < other.X;
  }
  if (Y != other.Y) {
    return Y < other.Y;
  }
  if (W != other.W) {
    return W < other.W;
  }
  if (H != other.H) {
    return H < other.H;
  }
  return false;
}

Viewer::RenderCache::RenderCache(Viewer* parent, int size)
    : Cache<RenderCacheKey, std::shared_ptr<PixelBuffer>>(size), _parent(parent) {}

Viewer::RenderCache::~RenderCache() { Clear(); }

std::shared_ptr<PixelBuffer> Viewer::RenderCache::Load(const RenderCacheKey& key) {
  std::shared_ptr<PixelBuffer> buffer(_parent->_fb->NewPixelBuffer(
      PixelBuffer::Size(key.W, key.H)));

  _parent->_doc->Render(
      buffer->GetRawBuffer(), buffer->GetAllocatedStrideBytes(),
      key.Page, key.Zoom, key.Rotation, key.X, key.Y, key.W, key.H);

  return buffer;
}

void Viewer::RenderCache::Discard(
    const RenderCacheKey& key, const std::shared_ptr<PixelBuffer>& value) {
}

Viewer::PerPageState Viewer::GetPageState(int page) {
  if (page < 0) {
    return _page_states[_state.Page];
  }

  page = std::min(page, _state.NumPages);
  return _page_states[page];
}