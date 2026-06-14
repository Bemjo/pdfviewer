// PDF viewer metadata: saves the last-visited page number per document.
// Uses a fixed-size open-addressing hash table with Robin Hood hashing.
// The metadata file (pdfviewer.meta) lives in the root_dir directory
// (e.g. /media/fat/docs/) and covers all documents beneath it.

#ifndef PDFMETA_HPP
#define PDFMETA_HPP

#include <cstdint>
#include <string>

struct MetaData {
  uint16_t page;
};

// Computes a 64-bit FNV-1a hash from the SYSTEM directory name and the
// document basename, ignoring any intermediate subdirectory components.
// root_dir is the needle prefix (e.g. "/docs/"). Returns 0 on error.
uint64_t PdfMetaHashFile(const std::string& path, const std::string& root_dir);

// Saves data to the global metadata file for the document at doc_path.
// Does nothing if root_dir is empty or doc_path does not contain root_dir.
void PdfMetaSave(const std::string& doc_path, const MetaData& data, const std::string& root_dir);

// Loads saved data for the document at doc_path.
// Returns zero-initialised MetaData if not found or on error.
MetaData PdfMetaLoad(const std::string& doc_path, const std::string& root_dir);

#endif
