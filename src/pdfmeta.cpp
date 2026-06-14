#include "pdfmeta.hpp"

#include <algorithm>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

static const uint32_t META_MAGIC    = 0x4D455441u;
static const uint16_t META_VERSION  = 2;
static const int      SAMPLE_SIZE   = 4096;
static const int      SMALL_FILE    = SAMPLE_SIZE * 5;
static const int      PATH_SIZE     = 256;
static const int      MAX_PROBE     = 50;

#pragma pack(push, 1)
struct MetaHeader {
  uint32_t magic;
  uint16_t version;
  uint16_t count;
};

struct MetaSlot {
  char     path[PATH_SIZE];
  MetaData data;
};
#pragma pack(pop)

static_assert(sizeof(MetaHeader) == 8,   "MetaHeader must be 8 bytes");
static_assert(sizeof(MetaSlot)   == 258, "MetaSlot must be 258 bytes");

static constexpr int META_FILE_SIZE = 16 * 1024 * 1024;
static constexpr int HEADER_SIZE    = static_cast<int>(sizeof(MetaHeader));
static constexpr int NUM_SLOTS      = (META_FILE_SIZE - HEADER_SIZE) /
                                       static_cast<int>(sizeof(MetaSlot));
static constexpr int FILE_SIZE      = HEADER_SIZE +
                                       NUM_SLOTS * static_cast<int>(sizeof(MetaSlot));

static_assert(FILE_SIZE <= META_FILE_SIZE, "File must not exceed META_FILE_SIZE");

static constexpr int CeilLog2(int n, int bits = 0) {
  return (1 << bits) >= n ? bits : CeilLog2(n, bits + 1);
}
static constexpr int HOME_SLOT_BITS = CeilLog2(NUM_SLOTS);

static uint64_t Fnv1a64(
    const uint8_t* data, size_t len,
    uint64_t hash = 14695981039346656037ULL) {
  for (size_t i = 0; i < len; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

static std::string GetMetaPath(
    const std::string& doc_path,
    const std::string& root_dir) {
  if (root_dir.empty()) {
    return "";
  }
  const size_t pos = doc_path.find(root_dir);
  if (pos == std::string::npos) {
    return "";
  }
  return doc_path.substr(0, pos + root_dir.size()) + "pdfviewer.meta";
}

static std::string RelativePath(
    const std::string& doc_path,
    const std::string& root_dir) {
  if (root_dir.empty()) {
    return "";
  }
  const size_t pos = doc_path.find(root_dir);
  if (pos == std::string::npos) {
    return "";
  }
  const std::string after = doc_path.substr(pos + root_dir.size());
  if (!after.empty() && after[0] == '/') {
    return after.substr(1);
  }
  return after;
}

static bool GetSystemAndBasename(
    const std::string& rel_path,
    std::string* system,
    std::string* basename) {
  const size_t slash = rel_path.find('/');
  if (slash == std::string::npos) {
    return false;
  }
  const size_t base_slash = rel_path.rfind('/');
  if (base_slash == std::string::npos) {
    return false;
  }
  *system   = rel_path.substr(0, slash);
  *basename = rel_path.substr(base_slash + 1);
  return !system->empty() && !basename->empty();
}

static off_t SlotOffset(int index) {
  return static_cast<off_t>(HEADER_SIZE) +
         static_cast<off_t>(index) * static_cast<off_t>(sizeof(MetaSlot));
}

static int HomeSlot(uint64_t key) {
  return static_cast<int>(
      (key * 11400714819323198485ULL) >> (64 - HOME_SLOT_BITS)) % NUM_SLOTS;
}

static bool ReadHeader(int fd, MetaHeader* hdr) {
  return pread(fd, hdr, sizeof(MetaHeader), 0) ==
         static_cast<ssize_t>(sizeof(MetaHeader));
}

static bool ReadSlot(int fd, int index, MetaSlot* slot) {
  return pread(fd, slot, sizeof(MetaSlot), SlotOffset(index)) ==
         static_cast<ssize_t>(sizeof(MetaSlot));
}

static bool WriteSlot(int fd, int index, const MetaSlot& slot) {
  return pwrite(fd, &slot, sizeof(MetaSlot), SlotOffset(index)) ==
         static_cast<ssize_t>(sizeof(MetaSlot));
}

static bool Preallocate(int fd) {
  if (ftruncate(fd, FILE_SIZE) != 0) {
    return false;
  }
  MetaHeader hdr = {META_MAGIC, META_VERSION, 0};
  return pwrite(fd, &hdr, sizeof(hdr), 0) ==
         static_cast<ssize_t>(sizeof(hdr));
}

static int OpenOrCreate(const std::string& path) {
  int fd = open(path.c_str(), O_RDWR);
  if (fd >= 0) {
    MetaHeader hdr;
    if (ReadHeader(fd, &hdr) && hdr.magic == META_MAGIC) {
      if (hdr.version == META_VERSION) {
        return fd;
      }
      if (lseek(fd, 0, SEEK_SET) < 0 || !Preallocate(fd)) {
        close(fd);
        return -1;
      }
      return fd;
    }
    close(fd);
  }

  fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    return -1;
  }
  if (!Preallocate(fd)) {
    close(fd);
    return -1;
  }
  return fd;
}

// Linear probe from home slot, up to max_probe steps.
// Returns slot index on match, -1 if not found, -2 on I/O error.

static int Lookup(
    int fd, uint64_t key, const std::string& rel_path,
    int max_probe = MAX_PROBE) {
  int i = HomeSlot(key);
  for (int dist = 0; dist < max_probe; ++dist) {
    MetaSlot slot;
    if (!ReadSlot(fd, i, &slot)) {
      return -2;
    }
    if (slot.path[0] == '\0') {
      return -1;
    }
    if (std::string(slot.path) == rel_path) {
      return i;
    }
    i = (i + 1) % NUM_SLOTS;
  }
  return -1;
}

// Single linear probe from home slot, up to max_probe steps.
// Updates data in place if path already exists, otherwise inserts into
// the first empty slot found. Returns slot index on success, -1 if no
// empty slot found within max_probe steps, -2 on I/O error.
static int Insert(
    int fd, MetaHeader* hdr,
    uint64_t key, const std::string& rel_path, const MetaData& data,
    int max_probe = MAX_PROBE) {
  int i         = HomeSlot(key);
  int empty_idx = -1;

  for (int dist = 0; dist < max_probe; ++dist) {
    MetaSlot slot;
    if (!ReadSlot(fd, i, &slot)) {
      return -2;
    }
    if (slot.path[0] == '\0') {
      if (empty_idx < 0) {
        empty_idx = i;
      }
      break;
    }
    if (std::string(slot.path) == rel_path) {
      slot.data = data;
      if (!WriteSlot(fd, i, slot)) {
        return -2;
      }
      return i;
    }
    i = (i + 1) % NUM_SLOTS;
  }

  if (empty_idx < 0) {
    return -1;
  }

  if (rel_path.size() >= static_cast<size_t>(PATH_SIZE)) {
    return -1;
  }
  MetaSlot entry;
  std::copy(rel_path.begin(), rel_path.end(), entry.path);
  entry.path[rel_path.size()] = '\0';
  entry.data = data;
  if (!WriteSlot(fd, empty_idx, entry)) {
    return -2;
  }
  ++hdr->count;
  return empty_idx;
}

}  // namespace

uint64_t PdfMetaHashFile(
    const std::string& path, const std::string& root_dir) {
  if (root_dir.empty() || path.find(root_dir) == std::string::npos) {
    return 0;
  }

  const std::string rel = RelativePath(path, root_dir);
  std::string system, basename;
  if (!GetSystemAndBasename(rel, &system, &basename)) {
    return 0;
  }

  struct stat st;
  if (stat(path.c_str(), &st) != 0 || st.st_size == 0) {
    return 0;
  }

  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return 0;
  }

  uint8_t buf[SAMPLE_SIZE];
  uint64_t h = 14695981039346656037ULL;
  const off_t file_size = st.st_size;

  if (file_size <= SMALL_FILE) {
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
      h = Fnv1a64(buf, static_cast<size_t>(n), h);
    }
  } else {
    const off_t offsets[5] = {
      file_size / 20,
      file_size / 4,
      file_size * 9  / 20,
      file_size * 13 / 20,
      file_size * 17 / 20,
    };
    for (int i = 0; i < 5; ++i) {
      if (lseek(fd, offsets[i], SEEK_SET) < 0) {
        continue;
      }
      const ssize_t n = read(fd, buf, sizeof(buf));
      if (n > 0) {
        h = Fnv1a64(buf, static_cast<size_t>(n), h);
      }
    }
  }
  close(fd);

  const std::string path_parts = system + basename;
  h = Fnv1a64(reinterpret_cast<const uint8_t*>(path_parts.c_str()),
               path_parts.size(), h);

  if (h == 0) {
    h = 2ULL;
  }
  return h;
}

void PdfMetaSave(
    const std::string& doc_path, const MetaData& data,
    const std::string& root_dir) {
  const std::string meta_path = GetMetaPath(doc_path, root_dir);
  if (meta_path.empty()) {
    return;
  }

  const uint64_t key = PdfMetaHashFile(doc_path, root_dir);
  if (key == 0) {
    return;
  }

  const std::string rel = RelativePath(doc_path, root_dir);
  if (rel.empty() || rel.size() >= static_cast<size_t>(PATH_SIZE)) {
    return;
  }

  const int fd = OpenOrCreate(meta_path);
  if (fd < 0) {
    return;
  }

  MetaHeader hdr;
  if (!ReadHeader(fd, &hdr)) {
    close(fd);
    return;
  }

  const int slot_idx = Insert(fd, &hdr, key, rel, data);
  if (slot_idx >= 0) {
    pwrite(fd, &hdr, sizeof(MetaHeader), 0);
  }

  close(fd);
}

MetaData PdfMetaLoad(
    const std::string& doc_path, const std::string& root_dir) {
  MetaData result = {0};

  const std::string meta_path = GetMetaPath(doc_path, root_dir);
  if (meta_path.empty()) {
    return result;
  }

  const uint64_t key = PdfMetaHashFile(doc_path, root_dir);
  if (key == 0) {
    return result;
  }

  const std::string rel = RelativePath(doc_path, root_dir);
  if (rel.empty() || rel.size() >= static_cast<size_t>(PATH_SIZE)) {
    return result;
  }

  const int fd = open(meta_path.c_str(), O_RDONLY);
  if (fd < 0) {
    return result;
  }

  MetaHeader hdr;
  if (!ReadHeader(fd, &hdr) ||
      hdr.magic   != META_MAGIC ||
      hdr.version != META_VERSION) {
    close(fd);
    return result;
  }

  const int i = Lookup(fd, key, rel);
  if (i >= 0) {
    MetaSlot slot;
    if (ReadSlot(fd, i, &slot)) {
      result = slot.data;
    }
  }

  close(fd);
  return result;
}
