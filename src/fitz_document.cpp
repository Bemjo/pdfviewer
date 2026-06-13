/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                           *
 *  Copyright (C) 2020-2020 Chuan Ji                                         *
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

// This file defines FitzDocument, an implementation of the Document
// abstraction using Fitz.

#include "fitz_document.hpp"

#include <cassert>

#include "string_utils.hpp"

FitzDocument* FitzDocument::Open(const std::string& path, const std::string* password,
    size_t max_store_size_mb, size_t dl_cache_size) {
  size_t store_size = (max_store_size_mb == 0)
                        ? FZ_STORE_DEFAULT
                        : max_store_size_mb * 1024 * 1024;
  fz_context* fz_ctx = fz_new_context(nullptr, nullptr, store_size);
  fz_set_aa_level(fz_ctx, 2);
  fz_register_document_handlers(fz_ctx);
  // Disable warning messages in the console.
  fz_set_warning_callback(
      fz_ctx, [](void* user, const char* message) {}, nullptr);

  fz_document* fz_doc = nullptr;
  fz_try(fz_ctx) {
    fz_doc = fz_open_document(fz_ctx, path.c_str());
    if ((fz_doc == nullptr) || (!fz_count_pages(fz_ctx, fz_doc))) {
      fz_throw(
          fz_ctx, FZ_ERROR_GENERIC,
          const_cast<char*>("Cannot open document \"%s\""), path.c_str());
    }
    if (fz_needs_password(fz_ctx, fz_doc)) {
      if (password == nullptr) {
        fz_throw(
            fz_ctx, FZ_ERROR_GENERIC,
            const_cast<char*>(
                "Document \"%s\" is password protected.\n"
                "Please provide the password with \"-P <password>\"."),
            path.c_str());
      }
      if (!fz_authenticate_password(fz_ctx, fz_doc, password->c_str())) {
        fz_throw(
            fz_ctx, FZ_ERROR_GENERIC,
            const_cast<char*>("Incorrect password for document \"%s\"."),
            path.c_str());
      }
    }
  }
  fz_catch(fz_ctx) {
    if (fz_doc != nullptr) {
      fz_drop_document(fz_ctx, fz_doc);
    }
    fz_drop_context(fz_ctx);
    return nullptr;
  }

  FitzDocument* doc = new FitzDocument(fz_ctx, fz_doc);
  doc->SetDLCacheSize(dl_cache_size);
  return doc;
}

void FitzDocument::EvictFromDLCache() {
std::lock_guard<std::recursive_mutex> lock(_fz_mutex);
if (_display_list_cache.size() > 0 && _display_list_queue.size() > 0) {
    int p = _display_list_queue.front();
    _display_list_queue.pop_front();
    auto it = _display_list_cache.find(p);
    if (it != _display_list_cache.end()) {
      fz_drop_display_list(_fz_ctx, it->second);
      _display_list_cache.erase(it);
    }
  }
}

void FitzDocument::SetDLCacheSize(size_t size) {
  std::lock_guard<std::recursive_mutex> lock(_fz_mutex);
  while (_display_list_cache.size() > size) {
    EvictFromDLCache();
  }
  _display_list_cache.reserve(size);
  _max_dl_cache_size = size;
}

FitzDocument::FitzDocument(fz_context* fz_ctx, fz_document* fz_doc)
    : _fz_ctx(fz_ctx), _fz_doc(fz_doc) {
  assert(_fz_ctx != nullptr);
  assert(_fz_doc != nullptr);
}

FitzDocument::~FitzDocument() {
  std::lock_guard<std::recursive_mutex> lock(_fz_mutex);
  for (auto& kv : _display_list_cache) {
    fz_drop_display_list(_fz_ctx, kv.second);
  }
  fz_drop_document(_fz_ctx, _fz_doc);
  fz_drop_context(_fz_ctx);
}

int FitzDocument::GetNumPages() {
  std::lock_guard<std::recursive_mutex> lock(_fz_mutex);
  return fz_count_pages(_fz_ctx, _fz_doc);
}

const Document::PageSize FitzDocument::GetPageSize(
    int page, float zoom, int rotation) {
  std::lock_guard<std::recursive_mutex> lock(_fz_mutex);
  assert((page >= 0) && (page < GetNumPages()));
  const fz_matrix& m = ComputeTransformMatrix(zoom, rotation);
  fz_display_list* display_list = GetDisplayList(page);
  fz_rect bounds = fz_bound_display_list(_fz_ctx, display_list);
  fz_irect bbox = fz_round_rect(fz_transform_rect(bounds, m));
  return PageSize(bbox.x1 - bbox.x0, bbox.y1 - bbox.y0);
}

fz_display_list* FitzDocument::GetDisplayList(int page) {
  std::lock_guard<std::recursive_mutex> lock(_fz_mutex);
  auto it = _display_list_cache.find(page);
  if (it != _display_list_cache.end()) {
    return it->second;
  }

  // Evict display list from cache
  if (_display_list_cache.size() >= _max_dl_cache_size) {
    EvictFromDLCache();
  }

  FitzPageScopedPtr pp(_fz_ctx, fz_load_page(_fz_ctx, _fz_doc, page));
  fz_display_list* display_list = fz_new_display_list_from_page(_fz_ctx, pp.get());

  _display_list_cache[page] = display_list;
  _display_list_queue.push_back(page);
  return display_list;
}

bool FitzDocument::Render(
    uint8_t* buffer, int stride_bytes, int page, float zoom, int rotation,
    int clip_x, int clip_y, int clip_w, int clip_h) {
  std::lock_guard<std::recursive_mutex> lock(_fz_mutex);
  assert((page >= 0) && (page < GetNumPages()));

  // 1. Init MuPDF structures.
  const fz_matrix& m = ComputeTransformMatrix(zoom, rotation);
  fz_display_list* display_list = GetDisplayList(page);
  fz_rect bounds = fz_bound_display_list(_fz_ctx, display_list);
  fz_irect bbox, full_bbox = fz_round_rect(fz_transform_rect(bounds, m));

  bbox.x0 = std::max(full_bbox.x0, full_bbox.x0 + clip_x);
  bbox.y0 = std::max(full_bbox.y0, full_bbox.y0 + clip_y);
  bbox.x1 = std::min(full_bbox.x1, bbox.x0 + clip_w);
  bbox.y1 = std::min(full_bbox.y1, bbox.y0 + clip_h);

  if (stride_bytes != (bbox.x1 - bbox.x0) * 4) {
    return false;
  }

  FitzPixmapScopedPtr pixmap_ptr(
      _fz_ctx, fz_new_pixmap_with_bbox_and_data(
                   _fz_ctx, fz_device_bgr(_fz_ctx), bbox, nullptr, 1, buffer));
  FitzDeviceScopedPtr dev_ptr(
      _fz_ctx, fz_new_draw_device(_fz_ctx, fz_identity, pixmap_ptr.get()));

  // 2. Render page.
  fz_clear_pixmap_with_value(_fz_ctx, pixmap_ptr.get(), 0xff);
  fz_rect clip_rect = fz_rect_from_irect(bbox);
  fz_run_display_list(_fz_ctx, display_list, dev_ptr.get(), m, clip_rect, nullptr);
  // 4. Clean up.
  fz_close_device(_fz_ctx, dev_ptr.get());

  return true;
}

const Document::OutlineItem* FitzDocument::GetOutline() {
  std::lock_guard<std::recursive_mutex> lock(_fz_mutex);
  FitzOutlineScopedPtr outline_ptr(_fz_ctx, fz_load_outline(_fz_ctx, _fz_doc));
  if (outline_ptr.get() == nullptr) {
    return nullptr;
  }
  FitzOutlineItem* root = FitzOutlineItem::Build(_fz_ctx, _fz_doc, outline_ptr.get());
  return root;
}

int FitzDocument::Lookup(const OutlineItem* item) {
  return (dynamic_cast<const FitzOutlineItem*>(item))->GetDestPage();
}

std::string FitzDocument::GetPageText(int page, int line_sep) {
  std::lock_guard<std::recursive_mutex> lock(_fz_mutex);
  FitzPageScopedPtr page_ptr(_fz_ctx, fz_load_page(_fz_ctx, _fz_doc, page));
  return ::GetPageText(_fz_ctx, page_ptr.get(), line_sep);
}

std::vector<Document::SearchHit> FitzDocument::SearchOnPage(
    const std::string& search_string, int page, int context_length) {
  const size_t margin =
      context_length > static_cast<int>(search_string.length())
          ? (context_length - search_string.length() + 1) / 2
          : 0;

  const std::string page_text = GetPageText(page, ' ');
  std::vector<SearchHit> search_hits;
  for (size_t pos = 0;; ++pos) {
    if ((pos = CaseInsensitiveSearch(page_text, search_string, pos)) ==
        std::string::npos) {
      break;
    }
    const size_t context_start_pos = pos >= margin ? pos - margin : 0;
    search_hits.emplace_back(
        page, page_text.substr(context_start_pos, context_length),
        pos - context_start_pos);
  }
  return search_hits;
}
