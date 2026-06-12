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

// This file declares FitzDocument, an implementation of the Document
// abstraction using Fitz.

#ifndef FITZ_DOCUMENT_HPP
#define FITZ_DOCUMENT_HPP

#include <deque>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "document.hpp"
#include "fitz_utils.hpp"

// Document implementation using Fitz.
class FitzDocument : public Document {
 public:
 enum {
    DEFAULT_STORE_SIZE = 32,
    DEFAULT_DL_CACHE_SIZE = 24,
  };


  virtual ~FitzDocument();
  // Factory method to construct an instance of FitzDocument. path gives the
  // path to a file. password is the password to use to unlock the document;
  // specify nullptr if no password was provided. Does not take ownership of
  // password. Returns nullptr if the file cannot be opened.
  static FitzDocument* Open(
      const std::string& path, const std::string* password, size_t max_store_size_mb = DEFAULT_STORE_SIZE, size_t dl_cache_size = DEFAULT_DL_CACHE_SIZE);
  // See Document.
  int GetNumPages() override;
  // See Document.
  const PageSize GetPageSize(int page, float zoom, int rotation) override;
  // See Document. Thread-safe.
  virtual void Render(PixelWriter* pw,
      int page, float zoom, int rotation,
      int clip_x, int clip_y, int clip_w, int clip_h) override;
  // See Document.
  const OutlineItem* GetOutline() override;
  // See Document.
  int Lookup(const OutlineItem* item) override;
  // Returns the text content of a page, using line_sep to separate lines.
  std::string GetPageText(int page, int line_sep = '\n');

 protected:
  // See Document.
  std::vector<SearchHit> SearchOnPage(
      const std::string& search_string, int page, int context_length) override;

 private:
  // MuPDF structures.
  fz_context* _fz_ctx;
  fz_document* _fz_doc;
  // Mutex guarding MuPDF structures.
  std::recursive_mutex _fz_mutex;
  // Displaylist Caching mechanism
  int _max_dl_cache_size;
  std::deque<int> _display_list_queue;
  std::unordered_map<int, fz_display_list*> _display_list_cache;

  // We disallow the constructor; use the factory method Open() instead.
  FitzDocument(fz_context* _fz_context, fz_document* fz_document);
  // We disallow copying because we store lots of heap allocated state.
  explicit FitzDocument(const FitzDocument& other);
  FitzDocument& operator=(const FitzDocument& other);

  fz_display_list* GetDisplayList(int page, FitzPageScopedPtr& page_ptr);
  void SetDLCacheSize(size_t size);
  void EvictFromDLCache();
};

#endif
