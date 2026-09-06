#ifndef EMBEDDED_FILE_READER_H
#define EMBEDDED_FILE_READER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class EmbeddedFileReader {
public:
    enum class Status {
        OK,
        INVALID_ARCHIVE,
        FILE_NOT_FOUND,
        UNSUPPORTED_COMPRESSION,
        INVALID_FILE
    };

    EmbeddedFileReader(const uint8_t* data, size_t size);
    Status Read(const std::string& name, std::vector<uint8_t>* output) const;
    static const char* StatusDescription(Status status);

private:
    const uint8_t* data_;
    size_t size_;
};

#endif