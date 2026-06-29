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

// Main program file.

// Program name. May be overridden in the Makefile.
#ifndef JFBVIEW_PROGRAM_NAME
#define JFBVIEW_PROGRAM_NAME "jfbview"
#endif

// Binary program name. May be overridden in the Makefile.
#ifndef JFBVIEW_BINARY_NAME
#define JFBVIEW_BINARY_NAME "jfbview"
#endif

// Program version. May be overridden in the Makefile.
#ifndef JFBVIEW_VERSION
#define JFBVIEW_VERSION
#endif

#include <curses.h>
#include <fcntl.h>
#include <getopt.h>
#include <linux/vt.h>
#include <locale.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <time.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <atomic>
#include <thread>

#include "command.hpp"
#include "cpp_compat.hpp"
#include "fitz_document.hpp"
#include "framebuffer.hpp"
#include "image_document.hpp"
#include "outline_view.hpp"
#include "pdf_document.hpp"
#include "search_view.hpp"
#include "viewer.hpp"
#include "pdfmeta.hpp"

struct InputState {
  enum {DEFAULT_PAGE_EDGE_TIME = 100 };
  enum EdgeState
  {
      EDGE_IDLE,

      EDGE_WAIT_UP_RELEASE,
      EDGE_WAIT_DOWN_RELEASE,

      EDGE_READY_UP,
      EDGE_READY_DOWN,
  };

  int EdgeGuardTime;
  std::atomic<EdgeState> CurrEdgeState;
  std::atomic<bool> InputThreadExit;
  std::atomic<int> InputKey;
  std::atomic<int> InputRepeat;
  std::atomic<int> HeldKey;
  std::atomic<int> ClearHeldKey;

  InputState()
      : EdgeGuardTime(DEFAULT_PAGE_EDGE_TIME),
        CurrEdgeState(EDGE_IDLE),
        InputThreadExit(false),
        InputKey(ERR),
        InputRepeat(Command::NO_REPEAT),
        HeldKey(ERR),
        ClearHeldKey(ERR) {}
};

// Main program state.
struct State : public Viewer::State {
  // If true, just print debugging info and exit.
  bool PrintFBDebugInfoAndExit;
  // If true, exit main event loop.
  bool Exit;
  // If true (default), requires refresh after current command.
  bool Render;

  // The type of the displayed file.
  enum {
    AUTO_DETECT,
    PDF,
#ifndef JFBVIEW_NO_IMLIB2
    IMAGE,
#endif
  } DocumentType;
  // Viewer render cache size.
  int RenderCacheSize;
  // Maximum cache store size in bytes for MuPDF
  int MuPDFStoreSize;
  // Maximum render buffer size.
  int RenderCapWidth;
  int RenderCapHeight;
  // Upscaling algorithm (0=nearest, 1=bilinear, 2=bicubic).
  PixelBuffer::ScaleMode RenderScaleMode;
  uint8_t SharpenStrength;

  // Input file.
  std::string FilePath;
  // Password for the input file. If no password is provided, this will be
  // nullptr.
  std::unique_ptr<std::string> FilePassword;
  // Framebuffer device.
  std::string FramebufferDevice;
  // Output file to append to when rendering is complete.
  std::string StatusFile;

  std::string MetaRootDir;
  // Document instance.
  std::unique_ptr<Document> DocumentInst;
  // Outline view instance.
  std::unique_ptr<OutlineView> OutlineViewInst;
  // Search view instance.
  std::unique_ptr<SearchView> SearchViewInst;
  // Framebuffer instance.
  std::unique_ptr<Framebuffer> FramebufferInst;
  // Viewer instance.
  std::unique_ptr<Viewer> ViewerInst;

  InputState inputState;

  // Default state.
  State()
      : Viewer::State(),
        PrintFBDebugInfoAndExit(false),
        Exit(false),
        Render(true),
        DocumentType(AUTO_DETECT),
        RenderCacheSize(Viewer::DEFAULT_RENDER_CACHE_SIZE),
        MuPDFStoreSize(FitzDocument::DEFAULT_STORE_SIZE),
        RenderCapWidth(Viewer::DEFAULT_MAX_RENDER_WIDTH),
        RenderCapHeight(Viewer::DEFAULT_MAX_RENDER_HEIGHT),
        RenderScaleMode(PixelBuffer::SCALE_BILINEAR),
        FilePath(""),
        FilePassword(),
        FramebufferDevice(Framebuffer::DEFAULT_FRAMEBUFFER_DEVICE),
        StatusFile(""),
        MetaRootDir("docs/"),
        OutlineViewInst(nullptr),
        SearchViewInst(nullptr),
        FramebufferInst(nullptr),
        ViewerInst(nullptr) {}
};

// Returns the all lowercase version of a string.
static std::string ToLower(const std::string& s) {
  std::string r(s);
  std::transform(r.begin(), r.end(), r.begin(), &tolower);
  return r;
}

// Returns the file extension of a path, or the empty string. The extension is
// converted to lower case.
static std::string GetFileExtension(const std::string& path) {
  int path_len = path.length();
  if ((path_len >= 4) && (path[path_len - 4] == '.')) {
    return ToLower(path.substr(path_len - 3));
  }
  return std::string();
}

// Loads the file specified in a state. Returns true if the file has been
// loaded.
static bool LoadFile(State* state) {
#if !defined(JFBVIEW_ENABLE_LEGACY_PDF_IMPL) && \
    !defined(JFBVIEW_ENABLE_LEGACY_IMAGE_IMPL)
  Document* doc =
      FitzDocument::Open(state->FilePath, state->FilePassword.get(), state->MuPDFStoreSize);
#else
  if (state->DocumentType == State::AUTO_DETECT) {
    if (GetFileExtension(state->FilePath) == "pdf") {
      state->DocumentType = State::PDF;
    } else {
#ifndef JFBVIEW_NO_IMLIB2
      state->DocumentType = State::IMAGE;
#else
      fprintf(
          stderr,
          "Cannot detect file format. Plase specify a file format "
          "with --format. Try --help for help.\n");
      return false;
#endif
    }
  }
  Document* doc = nullptr;
  switch (state->DocumentType) {
    case State::PDF:
#ifdef JFBVIEW_ENABLE_LEGACY_PDF_IMPL
      doc = PDFDocument::Open(state->FilePath, state->FilePassword.get());
#else
      doc = FitzDocument::Open(state->FilePath, state->FilePassword.get());
#endif
      break;
#ifdef JFBVIEW_ENABLE_LEGACY_IMAGE_IMPL
#ifndef JFBVIEW_NO_IMLIB2
    case State::IMAGE:
      doc = ImageDocument::Open(state->FilePath);
      break;
#endif
#else
    case State::IMAGE:
      doc = FitzDocument::Open(state->FilePath, state->FilePassword.get());
      break;
#endif
    default:
      abort();
  }
#endif
  if (doc == nullptr) {
    fprintf(
        stderr, "Failed to open document \"%s\".\n", state->FilePath.c_str());
    return false;
  }
  state->DocumentInst.reset(doc);
  state->NumPages = state->DocumentInst->GetNumPages();
  return true;
}

static void LoadMetadata(State* state) {
  if (state->Page == 0) {
    const MetaData& data = PdfMetaLoad(state->FilePath, state->MetaRootDir);
    if (data.page > 0) {
      state->Page = std::max(0, std::min(static_cast<int>(data.page), state->NumPages - 1));
    }
  }
}

static void SaveMetadata(const State* state) {
  MetaData data = {
    static_cast<uint16_t>(state->Page)
  };
  PdfMetaSave(state->FilePath, data, state->MetaRootDir);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                 COMMANDS                                  *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

class ExitCommand : public Command {
 public:
  void Execute(int repeat, State* state) override { state->Exit = true; }
};

// Base class for move commands.
class MoveCommand : public Command {
 protected:
  // Returns how much to move by in a direction.
  int GetMoveSize(const State* state, bool horizontal) const {
    if (horizontal) {
      return state->ScreenWidth / 10;
    }
    return state->ScreenHeight / 10;
  }
};

class MoveDownCommand : public MoveCommand {
 public:
  void Execute(int repeat, State* state) override {
    state->YOffset += RepeatOrDefault(repeat, 1) * GetMoveSize(state, false);

    const bool at_bottom = (state->YOffset + state->ScreenHeight) >=
         (state->PageHeight - 1 + GetMoveSize(state, false));

    if (at_bottom) {
      if (state->Zoom > 0 && state->PageHeight > state->ScreenHeight) {
        if (state->inputState.CurrEdgeState.exchange(InputState::EDGE_IDLE, std::memory_order_relaxed) == InputState::EDGE_READY_DOWN) {
          if (++(state->Page) < state->NumPages) {
            state->YOffset = 0;
          }
        } else {
          state->inputState.CurrEdgeState.store(InputState::EDGE_WAIT_DOWN_RELEASE, std::memory_order_relaxed);
          state->Render = false;
        }
        return;
      } else if (++(state->Page) < state->NumPages) {
        state->YOffset = 0;
      }
    }
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }
};

class MoveUpCommand : public MoveCommand {
 public:
  void Execute(int repeat, State* state) override {
    state->YOffset -= RepeatOrDefault(repeat, 1) * GetMoveSize(state, false);

    const bool at_top = state->YOffset <= -GetMoveSize(state, false);

    if (at_top) {
      if (state->Zoom > 0 && state->PageHeight > state->ScreenHeight) {
        if (state->inputState.CurrEdgeState.exchange(InputState::EDGE_IDLE, std::memory_order_relaxed) == InputState::EDGE_READY_UP) {
          if (--(state->Page) >= 0) {
            state->YOffset = INT_MAX;
          }
        } else {
          state->inputState.CurrEdgeState.store(InputState::EDGE_WAIT_UP_RELEASE, std::memory_order_relaxed);
          state->Render = false;
        }
        return;
      } else if (--(state->Page) >= 0) {
        state->YOffset = INT_MAX;
      }
    }
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }
};

class MoveLeftCommand : public MoveCommand {
 public:
  void Execute(int repeat, State* state) override {
    state->XOffset -= RepeatOrDefault(repeat, 1) * GetMoveSize(state, true);
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }
};

class MoveRightCommand : public MoveCommand {
 public:
  void Execute(int repeat, State* state) override {
    state->XOffset += RepeatOrDefault(repeat, 1) * GetMoveSize(state, true);
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }
};

class ScreenDownCommand : public Command {
 public:
  void Execute(int repeat, State* state) override {
    state->YOffset += RepeatOrDefault(repeat, 1) * state->ScreenHeight;
    if (state->YOffset + state->ScreenHeight >=
        state->PageHeight - 1 + state->ScreenHeight) {
      if (++(state->Page) < state->NumPages) {
        state->YOffset = 0;
      }
    }
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }
};

class ScreenUpCommand : public Command {
 public:
  void Execute(int repeat, State* state) override {
    state->YOffset -= RepeatOrDefault(repeat, 1) * state->ScreenHeight;
    if (state->YOffset <= -state->ScreenHeight) {
      if (--(state->Page) >= 0) {
        state->YOffset = INT_MAX;
      }
    }
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }
};

class PageDownCommand : public Command {
 public:
  void Execute(int repeat, State* state) override {
    state->Page += RepeatOrDefault(repeat, 1);
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }
};

class PageUpCommand : public Command {
 public:
  void Execute(int repeat, State* state) override {
    state->Page -= RepeatOrDefault(repeat, 1);
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }
};

// Base class for zoom commands.
class ZoomCommand : public Command {
 protected:
  // How much to zoom in/out by each time.
  static const float ZOOM_COEFFICIENT;
  // Sets zoom, preserving original screen center.
  void SetZoom(float zoom, State* state) {
    // Position in page of screen center, as fraction of page size.
    // When the page is smaller than the screen it is centered, so the
    // visible page width/height is min(Screen,Page) in each axis.
    const int vis_w = std::min(state->ScreenWidth,  state->PageWidth);
    const int vis_h = std::min(state->ScreenHeight, state->PageHeight);
    const float center_ratio_x =
        static_cast<float>(state->XOffset + vis_w / 2) /
        static_cast<float>(state->PageWidth);
    const float center_ratio_y =
        static_cast<float>(state->YOffset + vis_h / 2) /
        static_cast<float>(state->PageHeight);
    // Bound zoom.
    zoom = std::max(Viewer::MIN_ZOOM, std::min(Viewer::MAX_ZOOM, zoom));
    // Quotient of new and old zoom ratios.
    const float q = zoom / state->ActualZoom;
    // New page size after zoom change.
    const float new_page_width = static_cast<float>(state->PageWidth) * q;
    const float new_page_height = static_cast<float>(state->PageHeight) * q;
    // New center position within page after zoom change.
    const float new_center_x = new_page_width * center_ratio_x;
    const float new_center_y = new_page_height * center_ratio_y;
    // New offsets.
    state->XOffset = static_cast<int>(new_center_x) - state->ScreenWidth / 2;
    state->YOffset = static_cast<int>(new_center_y) - state->ScreenHeight / 2;
    // New zoom.
    state->Render = std::abs(state->Zoom - zoom) > 0.00001f;
    state->Zoom = zoom;
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }
};
const float ZoomCommand::ZOOM_COEFFICIENT = 1.2f;

class ZoomInCommand : public ZoomCommand {
 public:
  void Execute(int repeat, State* state) override {
    SetZoom(
        state->ActualZoom * pow(ZOOM_COEFFICIENT, RepeatOrDefault(repeat, 1)),
        state);
  }
};

class ZoomOutCommand : public ZoomCommand {
 public:
  void Execute(int repeat, State* state) override {
    SetZoom(
        state->ActualZoom / pow(ZOOM_COEFFICIENT, RepeatOrDefault(repeat, 1)),
        state);
  }
};

class SetZoomCommand : public ZoomCommand {
 public:
  void Execute(int repeat, State* state) override {
    SetZoom(static_cast<float>(RepeatOrDefault(repeat, 100)) / 100.0f, state);
  }
};

class SetRotationCommand : public Command {
 public:
  void Execute(int repeat, State* state) override {
    state->Rotation = RepeatOrDefault(repeat, 0);
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }
};

class RotateCommand : public Command {
 public:
  explicit RotateCommand(int increment) : _increment(increment) {}

  void Execute(int repeat, State* state) override {
    state->Rotation += RepeatOrDefault(repeat, 1) * _increment;
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }

 private:
  int _increment;
};

class ZoomToFitCommand : public Command {
 public:
  void Execute(int repeat, State* state) override {
    state->Zoom = Viewer::ZOOM_TO_FIT;
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }
};

class ZoomToWidthCommand : public ZoomCommand {
 public:
  void Execute(int repeat, State* state) override {
    // Estimate page width at 100%.
    const float orig_page_width =
        static_cast<float>(state->PageWidth) / state->ActualZoom;
    // Estimate actual zoom ratio with zoom to width.
    const float actual_zoom =
        static_cast<float>(state->ScreenWidth) / orig_page_width;
    // Set center according to estimated actual zoom.
    SetZoom(actual_zoom, state);
    // Actually set zoom to width.
    state->Zoom = Viewer::ZOOM_TO_WIDTH;
  }
};

class GoToPageCommand : public Command {
 public:
  explicit GoToPageCommand(int default_page) : _default_page(default_page) {}

  void Execute(int repeat, State* state) override {
    int page =
        (std::max(
            1, std::min(
                   state->NumPages, RepeatOrDefault(repeat, _default_page)))) -
        1;
    if (page != state->Page) {
      state->Page = page;
      state->XOffset = 0;
      state->YOffset = 0;
    }
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }

 private:
  int _default_page;
};

class ShowOutlineViewCommand : public Command {
 public:
  void Execute(int repeat, State* state) override {
    const Document::OutlineItem* dest = state->OutlineViewInst->Run();
    if (dest == nullptr) {
      return;
    }
    const int dest_page = state->DocumentInst->Lookup(dest);
    if (dest_page >= 0) {
      GoToPageCommand c(0);
      c.Execute(dest_page + 1, state);
    }
  }
};

class ShowSearchViewCommand : public Command {
 public:
  void Execute(int repeat, State* state) override {
    const int dest_page = state->SearchViewInst->Run();
    if (dest_page >= 0) {
      GoToPageCommand c(0);
      c.Execute(dest_page + 1, state);
    }
  }
};

// Base class for SaveStateCommand and RestoreStateCommand.
class StateCommand : public Command {
 protected:
  // A global map from register number to saved state.
  static std::map<int, Viewer::State> _saved_states;
};
std::map<int, Viewer::State> StateCommand::_saved_states;

class SaveStateCommand : public StateCommand {
 public:
  void Execute(int repeat, State* state) override {
    state->ViewerInst->GetState(&(_saved_states[RepeatOrDefault(repeat, 0)]));
    state->Render = false;
  }
};

class RestoreStateCommand : public StateCommand {
 public:
  void Execute(int repeat, State* state) override {
    const int n = RepeatOrDefault(repeat, 0);
    if (_saved_states.count(n)) {
      state->ViewerInst->SetState(_saved_states[n]);
      state->ViewerInst->GetState(state);
    }
  }
};

class ReloadCommand : public StateCommand {
 public:
  void Execute(int repeat, State* state) override {
    if (LoadFile(state)) {
      state->ViewerInst = std::make_unique<Viewer>(
          state->DocumentInst.get(), state->FramebufferInst.get(), *state,
          state->RenderCacheSize, state->RenderScaleMode,
          state->RenderCapWidth, state->RenderCapHeight);
      state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
    } else {
      state->Exit = true;
    }
  }
};

class CycleScaleModeCommand : public Command {
 public:
  explicit CycleScaleModeCommand(int direction) : _direction(direction) {}

  void Execute(int repeat, State* state) override {
    const int n = static_cast<int>(PixelBuffer::ScaleMode::COUNT);
    const int mode = (static_cast<int>(state->RenderScaleMode) + _direction + n) % n;
    state->RenderScaleMode = static_cast<PixelBuffer::ScaleMode>(mode);
    state->ViewerInst->SetScaleMode(state->RenderScaleMode, state->SharpenStrength);
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }

 private:
  int _direction;
};

class CycleBilinearSharpen : public Command {
 public:
  explicit CycleScaleModeCommand(int direction) : _direction(direction) {}

  void Execute(int repeat, State* state) override {
    state->SharpenStrength = (state->SharpenStrength + _direction + 4) % 4;
    state->ViewerInst->SetScaleMode(state->RenderScaleMode, state->SharpenStrength);
    state->inputState.CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
  }

 private:
  int _direction;
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                               END COMMANDS                                *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Help text printed by --help or -h.
static const char* HELP_STRING =
    "\n" JFBVIEW_PROGRAM_NAME " " JFBVIEW_VERSION
    "\n"
    "\n"
    "Usage: " JFBVIEW_BINARY_NAME
    " [OPTIONS] FILE\n"
    "\n"
    "Options:\n"
    "\t--help, -h            Show this message.\n"
    "\t--fb=/path/to/dev     Specify output framebuffer device.\n"
    "\t--password=xx, -P xx  Unlock PDF document with the given password.\n"
    "\t--page=N, -p N        Open page N on start up.\n"
    "\t--zoom=N, -z N        Set initial zoom to N. E.g., -z 150 sets \n"
    "\t                      zoom level to 150%.\n"
    "\t--zoom_to_fit         Start in automatic zoom-to-fit mode.\n"
    "\t--zoom_to_width       Start in automatic zoom-to-width mode.\n"
    "\t--rotation=N, -r N    Set initial rotation to N degrees clockwise.\n"
#if defined(JFBVIEW_ENABLE_LEGACY_IMAGE_IMPL) && \
    defined(JFBVIEW_ENABLE_LEGACY_PDF_IMPL) && !defined(JFBVIEW_NO_IMLIB2)
    "\t--format=image, -f image\n"
    "\t                      Forces the program to treat the input file as an\n"
    "\t                      image.\n"
    "\t--format=pdf, -f pdf  Forces the program to treat the input file as a\n"
    "\t                      PDF document. Use this if your PDF file does not\n"
    "\t                      end in \".pdf\" (case is ignored).\n"
#endif
    "\t--cache_size=N        Cache at most N pages. If you have an older\n"
    "\t                      machine with limited RAM, or if you are loading\n"
    "\t                      huge documents, or if you just want to reduce\n"
    "\t                      memory usage, you might want to set this to a\n"
    "\t                      smaller number.\n"
    "\t--store_size=N        Set MuPDF store size limit to N MB (default 32).\n"
    "\t--guard_time=N        Time in milliseconds where input must be released\n"
    "\t                      to trigger a page change when moving up or down.\n"
    "\t                      Set to 0 to disable. Default 100.\n"
    "\t--render_cap=WxH      Cap the MuPDF render buffer to WxH pixels, e.g.\n"
    "\t                      1280x720 or 960x480. Frames larger than this are\n"
    "\t                      rendered at the cap resolution and upscaled to the\n"
    "\t                      framebuffer. Minimum 640x480. Default 1280x720.\n"
    "\t--scale_mode=N        Upscaling algorithm used when render_cap is smaller\n"
    "\t                      than the framebuffer. 0=nearest (fastest),\n"
    "\t                      1=bilinear (default), 2=bicubic (sharpest).\n"
    "\t--meta_dir=xx         Metadata relative root directory to store metadata file.\n"
    "\t                      Should only be set if the common directory of documents is changed.\n"
    "\t                      Default \"docs/\".'\n"
    "\n"
    "jfbview home page: https://github.com/jichu4n/jfbview\n"
    "Bug reports & suggestions: https://github.com/jichu4n/jfbview/issues\n"
    "\n";

// Parses the command line, and stores settings in state. Crashes the
// program if the commnad line contains errors.
static void ParseCommandLine(int argc, char* argv[], State* state) {
  // Tags for long options that don't have short option chars.
  enum {
    RENDER_CACHE_SIZE = 0x1000,
    STORE_SIZE,
    GUARD_TIME,
    ZOOM_TO_WIDTH,
    ZOOM_TO_FIT,
    FB,
    STATUS_FILE,
    PRINT_FB_DEBUG_INFO_AND_EXIT,
    META_DIR,
    RENDER_CAP,
    SCALE_MODE,
    SHARPEN
  };
  // Command line options.
  static const option LongFlags[] = {
      {"help", false, nullptr, 'h'},
      {"fb", true, nullptr, FB},
      {"status-file", true, nullptr, STATUS_FILE},
      {"password", true, nullptr, 'P'},
      {"page", true, nullptr, 'p'},
      {"zoom", true, nullptr, 'z'},
      {"zoom_to_width", false, nullptr, ZOOM_TO_WIDTH},
      {"zoom_to_fit", false, nullptr, ZOOM_TO_FIT},
      {"rotation", true, nullptr, 'r'},
      {"format", true, nullptr, 'f'},
      {"cache_size", true, nullptr, RENDER_CACHE_SIZE},
      {"meta_dir", true, nullptr, META_DIR},
      {"fb_debug_info", false, nullptr, PRINT_FB_DEBUG_INFO_AND_EXIT},
      {"store_size", true, nullptr, STORE_SIZE},
      {"guard_time", true,  nullptr, GUARD_TIME},
      {"render_cap", true, nullptr, RENDER_CAP},
      {"scale_mode", true, nullptr, SCALE_MODE},
      {"sharpen", true, nullptr, SHARPEN},
      {0, 0, 0, 0},
  };
  static const char* ShortFlags = "hP:p:z:r:c:f:";

  for (;;) {
    int opt_char = getopt_long(argc, argv, ShortFlags, LongFlags, nullptr);
    if (opt_char == -1) {
      break;
    }
    switch (opt_char) {
      case 'h':
        fprintf(stdout, "%s", HELP_STRING);
        exit(EXIT_FAILURE);
      case FB:
        state->FramebufferDevice = optarg;
        break;
      case STATUS_FILE:
        state->StatusFile = optarg;
        break;
      case 'f':
        if (ToLower(optarg) == "pdf") {
          state->DocumentType = State::PDF;
#ifndef JFBVIEW_NO_IMLIB2
        } else if (ToLower(optarg) == "image") {
          state->DocumentType = State::IMAGE;
#endif
        } else {
          fprintf(stderr, "Invalid file format \"%s\"\n", optarg);
          exit(EXIT_FAILURE);
        }
        break;
      case 'P':
        state->FilePassword = std::make_unique<std::string>(optarg);
        break;
      case RENDER_CACHE_SIZE:
        if (sscanf(optarg, "%d", &(state->RenderCacheSize)) < 1) {
          fprintf(stderr, "Invalid render cache size \"%s\"\n", optarg);
          exit(EXIT_FAILURE);
        }
        state->RenderCacheSize = std::max(1, state->RenderCacheSize);
        break;
      case STORE_SIZE:
        if (sscanf(optarg, "%d", &(state->MuPDFStoreSize)) < 0) {
          fprintf(stderr, "Invalid MuPDF store size \"%s\"\n", optarg);
          exit(EXIT_FAILURE);
        }
        state->MuPDFStoreSize = std::max(0, state->MuPDFStoreSize);
        break;
      case GUARD_TIME:
        if (sscanf(optarg, "%d", &(state->inputState.EdgeGuardTime)) < 0) {
          fprintf(stderr, "Invalid guard time \"%s\"\n", optarg);
          exit(EXIT_FAILURE);
        }
        state->inputState.EdgeGuardTime = std::max(0, state->inputState.EdgeGuardTime);
        break;
      case RENDER_CAP: {
        int w = 0, h = 0;
        if (sscanf(optarg, "%dx%d", &w, &h) < 2) {
          fprintf(
              stderr,
              "Invalid render cap \"%s\". Expected format: WxH, e.g. 1280x720.\n",
              optarg);
          exit(EXIT_FAILURE);
        }
        state->RenderCapWidth = std::max(w, Viewer::MIN_RENDER_WIDTH);
        state->RenderCapHeight = std::max(h, Viewer::MIN_RENDER_HEIGHT);
        break;
      }
      case SCALE_MODE: {
        int mode = 0;
        if (sscanf(optarg, "%d", &mode) < 1) {
          fprintf(
              stderr,
              "Invalid scale mode \"%s\".\n",
              optarg);
          exit(EXIT_FAILURE);
        }
        mode = std::min(std::max(0, mode), static_cast<int>(PixelBuffer::ScaleMode::COUNT)-1);
        state->RenderScaleMode = static_cast<PixelBuffer::ScaleMode>(mode);
        break;
      }
      case SHARPEN: {
        int str = 0;
        if (sscanf(optarg, "%d", &str) < 1) {
          fprintf(
              stderr,
              "Invalid sharpen strength \"%s\".\n",
              optarg);
          exit(EXIT_FAILURE);
        }
        state->SharpenStrength = static_cast<uint8_t>(std::min(std::max(0, str), 3));
        break;
      }
      case META_DIR:
        state->MetaRootDir = optarg;
        break;
      case 'p':
        if (sscanf(optarg, "%d", &(state->Page)) < 1) {
          fprintf(stderr, "Invalid page number \"%s\"\n", optarg);
          exit(EXIT_FAILURE);
        }
        --(state->Page);
        break;
      case 'z':
        if (sscanf(optarg, "%f", &(state->Zoom)) < 1) {
          fprintf(stderr, "Invalid zoom ratio \"%s\"\n", optarg);
          exit(EXIT_FAILURE);
        }
        state->Zoom /= 100.0f;
        break;
      case ZOOM_TO_WIDTH:
        state->Zoom = Viewer::ZOOM_TO_WIDTH;
        break;
      case ZOOM_TO_FIT:
        state->Zoom = Viewer::ZOOM_TO_FIT;
        break;
      case 'r':
        if (sscanf(optarg, "%d", &(state->Rotation)) < 1) {
          fprintf(stderr, "Invalid rotation degree \"%s\"\n", optarg);
          exit(EXIT_FAILURE);
        }
        break;
      case PRINT_FB_DEBUG_INFO_AND_EXIT:
        state->PrintFBDebugInfoAndExit = true;
        break;
      default:
        fprintf(stderr, "Try \"-h\" for help.\n");
        exit(EXIT_FAILURE);
    }
  }
  if (optind == argc) {
    if (!state->PrintFBDebugInfoAndExit) {
      fprintf(stderr, "No file specified. Try \"-h\" for help.\n");
      exit(EXIT_FAILURE);
    }
  } else if (optind < argc - 1) {
    fprintf(
        stderr,
        "Please specify exactly one input file. Try \"-h\" for "
        "help.\n");
    exit(EXIT_FAILURE);
  } else {
    state->FilePath = argv[optind];
  }
}

// Constructs the command registry.
std::unique_ptr<Registry> BuildRegistry() {
  std::unique_ptr<Registry> registry = std::make_unique<Registry>();

  registry->Register('q', std::move(std::make_unique<ExitCommand>()));

  registry->Register('h', std::move(std::make_unique<MoveLeftCommand>()));
  registry->Register(KEY_LEFT, std::move(std::make_unique<MoveLeftCommand>()));
  registry->Register('j', std::move(std::make_unique<MoveDownCommand>()));
  registry->Register(KEY_DOWN, std::move(std::make_unique<MoveDownCommand>()));
  registry->Register('k', std::move(std::make_unique<MoveUpCommand>()));
  registry->Register(KEY_UP, std::move(std::make_unique<MoveUpCommand>()));
  registry->Register('l', std::move(std::make_unique<MoveRightCommand>()));
  registry->Register(
      KEY_RIGHT, std::move(std::make_unique<MoveRightCommand>()));
  registry->Register(' ', std::move(std::make_unique<ScreenDownCommand>()));
  registry->Register(
      6 /* CTRL-F */, std::move(std::make_unique<ScreenDownCommand>()));  // ^F
  registry->Register(
      2 /* CTRL-B */, std::move(std::make_unique<ScreenUpCommand>()));  // ^B
  registry->Register('J', std::move(std::make_unique<PageDownCommand>()));
  registry->Register(KEY_NPAGE, std::move(std::make_unique<PageDownCommand>()));
  registry->Register('K', std::move(std::make_unique<PageUpCommand>()));
  registry->Register(KEY_PPAGE, std::move(std::make_unique<PageUpCommand>()));

  registry->Register('=', std::move(std::make_unique<ZoomInCommand>()));
  registry->Register('+', std::move(std::make_unique<ZoomInCommand>()));
  registry->Register('-', std::move(std::make_unique<ZoomOutCommand>()));
  registry->Register('z', std::move(std::make_unique<SetZoomCommand>()));
  registry->Register('s', std::move(std::make_unique<ZoomToWidthCommand>()));
  registry->Register('a', std::move(std::make_unique<ZoomToFitCommand>()));

  registry->Register('r', std::move(std::make_unique<SetRotationCommand>()));
  registry->Register('>', std::move(std::make_unique<RotateCommand>(90)));
  registry->Register('.', std::move(std::make_unique<RotateCommand>(90)));
  registry->Register('<', std::move(std::make_unique<RotateCommand>(-90)));
  registry->Register(',', std::move(std::make_unique<RotateCommand>(-90)));

  registry->Register('g', std::move(std::make_unique<GoToPageCommand>(0)));
  registry->Register(KEY_HOME, std::move(std::make_unique<GoToPageCommand>(0)));
  registry->Register(
      'G', std::move(std::make_unique<GoToPageCommand>(INT_MAX)));
  registry->Register(
      KEY_END, std::move(std::make_unique<GoToPageCommand>(INT_MAX)));

  registry->Register(
      '\t', std::move(std::make_unique<ShowOutlineViewCommand>()));
  registry->Register('/', std::move(std::make_unique<ShowSearchViewCommand>()));

  registry->Register('m', std::move(std::make_unique<SaveStateCommand>()));
  registry->Register('`', std::move(std::make_unique<RestoreStateCommand>()));

  registry->Register('e', std::move(std::make_unique<ReloadCommand>()));

  registry->Register('u', std::move(std::make_unique<CycleScaleModeCommand>(+1)));
  registry->Register('U', std::move(std::make_unique<CycleScaleModeCommand>(-1)));
  registry->Register('o', std::move(std::make_unique<CycleBilinearSharpen>(+1)));
  registry->Register('O', std::move(std::make_unique<CycleBilinearSharpen>(-1)));

  // MiSTer additions
  registry->Register(27 /* Escape */, std::move(std::make_unique<ExitCommand>()));
  registry->Register(KEY_ENTER, std::move(std::make_unique<PageDownCommand>()));
  registry->Register('\r', std::move(std::make_unique<PageDownCommand>()));
  registry->Register('\n', std::move(std::make_unique<PageDownCommand>()));

  return registry;
}

static void DetectVTChange(pid_t parent) {
  struct vt_event e;
  struct vt_stat s;

  int fd = open("/dev/tty", O_RDONLY);
  if (fd == -1) {
    return;
  }

  if (ioctl(fd, VT_GETSTATE, &s) == -1) {
    goto out;
  }
  for (;;) {
    if (ioctl(fd, VT_WAITEVENT, &e) == -1) {
      goto out;
    }
    if (e.newev == s.v_active) {
      if (ioctl(fd, VT_WAITACTIVE, static_cast<int>(s.v_active)) == -1) {
        goto out;
      }
      if (kill(parent, SIGWINCH)) {
        goto out;
        // I wanted to use SIGRTMIN, but getch was not interrupted.
        // So instead, I choiced SIGWINCH because getch already
        // recognises this (and returns KEY_RESIZE), and the program
        // should support SIGWINCH and perform the same action anyways.
      }
    }
  }

out:
  close(fd);
}

static long ElapsedMilliseconds(const timespec& start, const timespec& end) {
  return (end.tv_sec - start.tv_sec) * 1000 +
         (end.tv_nsec - start.tv_nsec) / 1000000;
}

static void ResetReleaseTimer(bool* timing_release, timespec* release_start) {
  *timing_release = false;
  *release_start = {0, 0};
}

static void ClearBufferedHeldInput(InputState* state, int key) {
  int expected = key;
  if (state->InputKey.compare_exchange_strong(
          expected, ERR, std::memory_order_relaxed)) {
    state->InputRepeat.store(Command::NO_REPEAT, std::memory_order_relaxed);
  }
  flushinp();
}

static void InputThread(InputState* state) {
  timespec now                = {0, 0};
  timespec edge_release_start = {0, 0};
  timespec hold_release_start = {0, 0};
  bool     timing_release     = false;
  bool     timing_hold_release = false;
  int      tracked_key        = ERR;
  bool     tracked_key_held   = false;
  int      repeat             = Command::NO_REPEAT;

  for (;;) {
    if (state->InputThreadExit.load(std::memory_order_relaxed)) {
      return;
    }

    const int clear_key = state->ClearHeldKey.exchange(ERR, std::memory_order_relaxed);
    if (clear_key != ERR &&
        state->HeldKey.load(std::memory_order_relaxed) == clear_key) {
      ClearBufferedHeldInput(state, clear_key);
    }

    int key = getch();

    // Handle page edge guard timing
    InputState::EdgeState currState = state->CurrEdgeState.load(std::memory_order_relaxed);

    if (currState == InputState::EDGE_WAIT_UP_RELEASE || currState == InputState::EDGE_WAIT_DOWN_RELEASE) {
      const bool match_down = currState == InputState::EDGE_WAIT_DOWN_RELEASE && (key == KEY_DOWN || key == 'j');
      const bool match_up = currState == InputState::EDGE_WAIT_UP_RELEASE && (key == KEY_UP || key == 'k');

      if (key != ERR && !(match_down || match_up)) {
        state->CurrEdgeState.store(InputState::EDGE_IDLE, std::memory_order_relaxed);
        timing_release     = false;
        edge_release_start = {0, 0};
      } else {
        if (key != ERR) {
          timing_release     = false;
          edge_release_start = {0, 0};
        } else if (!timing_release) {
          clock_gettime(CLOCK_MONOTONIC, &edge_release_start);
          timing_release = true;
        } else {
          clock_gettime(CLOCK_MONOTONIC, &now);
          const long elapsed_ms =
              ElapsedMilliseconds(edge_release_start, now);
          if (elapsed_ms >= state->EdgeGuardTime) {
            if (currState == InputState::EDGE_WAIT_UP_RELEASE) {
              state->CurrEdgeState.store(InputState::EDGE_READY_UP, std::memory_order_relaxed);
            } else {
              state->CurrEdgeState.store(InputState::EDGE_READY_DOWN, std::memory_order_relaxed);
            }
            timing_release     = false;
            edge_release_start = {0, 0};
          }
        }
        usleep(2000);
        continue;
      }
    } else {
      ResetReleaseTimer(&timing_release, &edge_release_start);
    }

    if (key != ERR) {
      if (isdigit(key)) {
        nodelay(stdscr, false);
        do {
          if (repeat == Command::NO_REPEAT) {
            repeat = key - '0';
          } else {
            repeat = repeat * 10 + key - '0';
          }
        } while (isdigit(key = getch()));
        nodelay(stdscr, true);
        state->InputRepeat.store(repeat, std::memory_order_relaxed);
      }

      if (key == tracked_key) {
        tracked_key_held = true;
        state->HeldKey.store(key, std::memory_order_relaxed);
      } else {
        tracked_key = key;
        tracked_key_held = false;
        state->HeldKey.store(ERR, std::memory_order_relaxed);
      }
      ResetReleaseTimer(&timing_hold_release, &hold_release_start);

      repeat = Command::NO_REPEAT;
      int current = state->InputKey.load(std::memory_order_relaxed);
      if (key != current) {
        state->InputKey.store(key, std::memory_order_relaxed);
      }
    } else if (tracked_key != ERR) {
      if (!timing_hold_release) {
        clock_gettime(CLOCK_MONOTONIC, &hold_release_start);
        timing_hold_release = true;
      } else {
        clock_gettime(CLOCK_MONOTONIC, &now);
        const long elapsed_ms =
            ElapsedMilliseconds(hold_release_start, now);
        if (elapsed_ms >= state->EdgeGuardTime) {
          if (tracked_key_held) {
            ClearBufferedHeldInput(state, tracked_key);
          }
          state->HeldKey.store(ERR, std::memory_order_relaxed);
          tracked_key = ERR;
          tracked_key_held = false;
          ResetReleaseTimer(&timing_hold_release, &hold_release_start);
        }
      }
    }
    usleep(2000);
  }
}

void PrintFBDebugInfo(Framebuffer* fb) {
  assert(fb != nullptr);
  fprintf(stdout, "%s", fb->GetDebugInfoString().c_str());
}

static const char* FRAMEBUFFER_ERROR_HELP_STR = R"(
Troubleshooting tips:

1. Try adding yourself to the "video" group, e.g.:

       sudo usermod -a -G video $USER

   You will typically need to log out and back in for this to take effect.

2. Alternatively, try running this command as root, e.g.:

       sudo jfbview <file>

3. Verify that the framebuffer device exists. If not, please supply the correct
   device with "--fb=<path to device>".
)";

extern int JpdfgrepMain(int argc, char* argv[]);
extern int JpdfcatMain(int argc, char* argv[]);

int main(int argc, char* argv[]) {
  // Dispatch to jpdfgrep and jpdfcat.
  const std::string argv0 = argv[0];
  const std::string basename = argv0.substr(argv0.find_last_of('/') + 1);
  if (basename == "jpdfgrep") {
    return JpdfgrepMain(argc, argv);
  } else if (basename == "jpdfcat") {
    return JpdfcatMain(argc, argv);
  }

  // Main program state.
  State state;

  // 1. Initialization.
  ParseCommandLine(argc, argv, &state);
  state.FramebufferInst.reset(Framebuffer::Open(state.FramebufferDevice));
  if (state.FramebufferInst == nullptr) {
    fprintf(stderr, "%s", FRAMEBUFFER_ERROR_HELP_STR);
    exit(EXIT_FAILURE);
  }

  if (state.PrintFBDebugInfoAndExit) {
    PrintFBDebugInfo(state.FramebufferInst.get());
    exit(EXIT_SUCCESS);
  }

  if (!LoadFile(&state)) {
    exit(EXIT_FAILURE);
  }

  LoadMetadata(&state);

  setlocale(LC_ALL, "");
  initscr();
  start_color();
  keypad(stdscr, true);
  nonl();
  cbreak();
  noecho();
  curs_set(false);
  set_escdelay(25);
  nodelay(stdscr, true);
  // This is necessary to prevent curses erasing the framebuffer on first call
  // to getch().
  refresh();

  state.ViewerInst = std::make_unique<Viewer>(
      state.DocumentInst.get(), state.FramebufferInst.get(), state,
      state.RenderCacheSize, state.RenderScaleMode,
      state.RenderCapWidth, state.RenderCapHeight);
  std::unique_ptr<Registry> registry(BuildRegistry());

  state.OutlineViewInst = std::make_unique<OutlineView>(
      state.DocumentInst->GetOutline(), state.StatusFile);
  state.SearchViewInst =
      std::make_unique<SearchView>(state.DocumentInst.get(), state.StatusFile);

  pid_t parent = getpid();
  if (!fork()) {
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) == -1) {
      exit(EXIT_FAILURE);
    }
    // Possible race condition. Cannot be fixed by doing before
    // fork, because this is cleared at fork. Instead, we now
    // check that we have not been reparented. This should
    // nullify the race condition.
    if (getppid() != parent) {
      exit(EXIT_SUCCESS);
    }
    DetectVTChange(parent);
    exit(EXIT_FAILURE);
  }

  std::thread input_thread(InputThread, &state.inputState);

  // 2. Main event loop.
  state.Render = true;
  while (!state.Exit) {
    // 2.1 Render.
    if (state.Render) {
      state.ViewerInst->SetState(state);
      state.ViewerInst->Render();
      state.ViewerInst->GetState(&state);
      state.Render = false;

      if (!state.StatusFile.empty()) {
        FILE* status_file = fopen(state.StatusFile.c_str(), "a");
        if (status_file) {
          fprintf(status_file, "render_complete\n");
          fclose(status_file);
        }
      }
    }

    // 2.2. Grab input.
    const int last = state.inputState.InputKey.exchange(ERR, std::memory_order_relaxed);

    // 2.3. Run command.
    if (last != ERR) {
      if (last != KEY_RESIZE) {
        const int repeat = state.inputState.InputRepeat.exchange(Command::NO_REPEAT, std::memory_order_relaxed);
        if (state.inputState.HeldKey.load(std::memory_order_relaxed) == last) {
          state.inputState.ClearHeldKey.store(last, std::memory_order_relaxed);
        }
        state.Render = true;
        registry->Dispatch(last, repeat, &state);
      }
    } else {
      usleep(2000);
    }
  }

  SaveMetadata(&state);

  // 3. Clean up.
  state.inputState.InputThreadExit.store(true, std::memory_order_relaxed);
  input_thread.join();
  state.ViewerInst.reset();
  state.SearchViewInst.reset();
  state.OutlineViewInst.reset();
  // Hack alert: Calling endwin() immediately after the framebuffer destructor
  // (which clears the screen) appears to cause a race condition where the next
  // shell prompt after this program exits would also get erased. Adding a
  // short sleep appears to fix the issue.
  state.DocumentInst.reset();
  state.FramebufferInst.reset();
  usleep(100 * 1000);
  endwin();

  return EXIT_SUCCESS;
}
