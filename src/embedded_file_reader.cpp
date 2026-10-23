#include "embedded_file_reader.h"

#include <cstring>

namespace {

constexpr uint32_t kEndOfCentralDirectorySignature = 0x06054B50;
constexpr uint32_t kCentralDirectorySignature = 0x02014B50;
constexpr uint32_t kLocalFileHeaderSignature = 0x04034B50;
constexpr uint16_t kStoredMethod = 0;
constexpr size_t kEndOfCentralDirectorySize = 22;
constexpr size_t kCentralDirectoryHeaderSize = 46;
constexpr size_t kLocalFileHeaderSize = 30;

uint16_t ReadU16(const uint8_t* data)
{
    return static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8);
}

uint32_t ReadU32(const uint8_t* data)
{
    return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) | (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

bool RangeValid(size_t offset, size_t length, size_t total)
{
    return offset <= total && length <= total - offset;
}

uint32_t Crc32(const uint8_t* data, size_t size)
{
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (size_t bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
    }
    return ~crc;
}
}  // namespace

EmbeddedFileReader::EmbeddedFileReader(const uint8_t* data, size_t size)
    : data_(data), size_(size)
{
}

EmbeddedFileReader::Status EmbeddedFileReader::Read(const std::string& name, std::vector<uint8_t>* output) const
{
    if (data_ == nullptr || size_ < kEndOfCentralDirectorySize || output == nullptr)
        return Status::INVALID_ARCHIVE;
    output->clear();
    constexpr size_t kMaxZipCommentSize = 65535;
    const size_t search_size = std::min(size_, kEndOfCentralDirectorySize + kMaxZipCommentSize);
    const size_t search_start = size_ - search_size;
    size_t eocd = size_;
    for (size_t pos = size_ - kEndOfCentralDirectorySize + 1; pos > search_start; --pos) {
        const size_t candidate = pos - 1;
        if (!RangeValid(candidate, kEndOfCentralDirectorySize, size_))
            continue;
        if (ReadU32(data_ + candidate) != kEndOfCentralDirectorySignature)
            continue;
        const uint16_t comment_length = ReadU16(data_ + candidate + 20);
        if (candidate + kEndOfCentralDirectorySize + static_cast<size_t>(comment_length) != size_)
            continue;
        eocd = candidate;
        break;
    }
    if (eocd == size_)
        return Status::INVALID_ARCHIVE;
    const uint16_t disk_number = ReadU16(data_ + eocd + 4);
    const uint16_t central_directory_disk = ReadU16(data_ + eocd + 6);
    const uint16_t entries_on_disk = ReadU16(data_ + eocd + 8);
    const uint16_t entries = ReadU16(data_ + eocd + 10);
    const uint32_t central_directory_size = ReadU32(data_ + eocd + 12);
    const uint32_t central_directory_offset = ReadU32(data_ + eocd + 16);
    const uint16_t comment_length = ReadU16(data_ + eocd + 20);
    if (disk_number != 0 || central_directory_disk != 0 || entries_on_disk != entries)
        return Status::INVALID_ARCHIVE;
    if (eocd + kEndOfCentralDirectorySize + static_cast<size_t>(comment_length) != size_)
        return Status::INVALID_ARCHIVE;
    if (entries == 0xFFFF || central_directory_size == 0xFFFFFFFF || central_directory_offset == 0xFFFFFFFF)
        return Status::INVALID_ARCHIVE;
    const size_t central_directory_start = static_cast<size_t>(central_directory_offset);
    const size_t central_directory_length = static_cast<size_t>(central_directory_size);
    if (!RangeValid(central_directory_start, central_directory_length, size_))
        return Status::INVALID_ARCHIVE;
    const size_t central_directory_end =
        central_directory_start + central_directory_length;
    if (central_directory_end != eocd)
        return Status::INVALID_ARCHIVE;
    size_t entry_offset = central_directory_start;
    for (uint16_t i = 0; i < entries; ++i) {
        if (!RangeValid(entry_offset, kCentralDirectoryHeaderSize, central_directory_end))
            return Status::INVALID_ARCHIVE;
        const uint8_t* entry = data_ + entry_offset;
        if (ReadU32(entry) != kCentralDirectorySignature)
            return Status::INVALID_ARCHIVE;
        const uint16_t flags = ReadU16(entry + 8);
        const uint16_t method = ReadU16(entry + 10);
        const uint32_t crc = ReadU32(entry + 16);
        const uint32_t compressed_size = ReadU32(entry + 20);
        const uint32_t uncompressed_size = ReadU32(entry + 24);
        const uint16_t name_length = ReadU16(entry + 28);
        const uint16_t extra_length = ReadU16(entry + 30);
        const uint16_t entry_comment_length = ReadU16(entry + 32);
        const uint32_t local_offset = ReadU32(entry + 42);
        if (compressed_size == 0xFFFFFFFF || uncompressed_size == 0xFFFFFFFF || local_offset == 0xFFFFFFFF)
            return Status::INVALID_ARCHIVE;
        const size_t variable_length = static_cast<size_t>(name_length) + static_cast<size_t>(extra_length) + static_cast<size_t>(entry_comment_length);
        if (!RangeValid(entry_offset, kCentralDirectoryHeaderSize + variable_length, central_directory_end))
            return Status::INVALID_ARCHIVE;
        const char* entry_name = reinterpret_cast<const char*>(entry + kCentralDirectoryHeaderSize);
        if (name.size() != static_cast<size_t>(name_length) || memcmp(entry_name, name.data(), name_length) != 0) {
            entry_offset += kCentralDirectoryHeaderSize + variable_length;
            continue;
        }
        if (flags != 0)
            return Status::INVALID_FILE;
        if (method != kStoredMethod)
            return Status::UNSUPPORTED_COMPRESSION;
        if (compressed_size != uncompressed_size)
            return Status::INVALID_FILE;
        const size_t local_header_offset = static_cast<size_t>(local_offset);
        if (local_header_offset >= central_directory_start || !RangeValid(local_header_offset, kLocalFileHeaderSize, central_directory_start))
            return Status::INVALID_FILE;
        const uint8_t* local = data_ + local_header_offset;
        if (ReadU32(local) != kLocalFileHeaderSignature)
            return Status::INVALID_FILE;
        const uint16_t local_flags = ReadU16(local + 6);
        const uint16_t local_method = ReadU16(local + 8);
        const uint16_t local_name_length = ReadU16(local + 26);
        const uint16_t local_extra_length = ReadU16(local + 28);
        if (local_flags != flags || local_method != method)
            return Status::INVALID_FILE;
        if (local_name_length != name_length)
            return Status::INVALID_FILE;
        const size_t local_header_variable_length = static_cast<size_t>(local_name_length) + static_cast<size_t>(local_extra_length);
        if (!RangeValid(local_header_offset, kLocalFileHeaderSize + local_header_variable_length, central_directory_start))
            return Status::INVALID_FILE;
        const char* local_name = reinterpret_cast<const char*>(local + kLocalFileHeaderSize);
        if (memcmp(local_name, name.data(), name_length) != 0)
            return Status::INVALID_FILE;
        const size_t data_offset = local_header_offset + kLocalFileHeaderSize + local_header_variable_length;
        const size_t file_size = static_cast<size_t>(uncompressed_size);
        if (!RangeValid(data_offset, file_size, size_))
            return Status::INVALID_FILE;
        if (data_offset + file_size > central_directory_start)
            return Status::INVALID_FILE;
        const uint8_t* file_data = data_ + data_offset;
        if (Crc32(file_data, file_size) != crc)
            return Status::INVALID_FILE;
        output->assign(file_data, file_data + file_size);
        return Status::OK;
    }
    return Status::FILE_NOT_FOUND;
}

const char* EmbeddedFileReader::StatusDescription(Status status)
{
    switch (status) {
    case Status::OK:
        return "OK";
    case Status::INVALID_ARCHIVE:
        return "Invalid embedded ZIP archive";
    case Status::FILE_NOT_FOUND:
        return "File not found in embedded ZIP archive";
    case Status::UNSUPPORTED_COMPRESSION:
        return "Unsupported ZIP compression method";
    case Status::INVALID_FILE:
        return "Invalid embedded file";
    default:
        return "Unknown error";
    }
}