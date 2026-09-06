#ifndef METADATA_DECODER_H
#define METADATA_DECODER_H

#include <cstddef>
#include <cstdint>

class ConsolePrinter;

namespace tflite {
class Buffer;
class Metadata;
}

class MetadataDecoder {
public:
    static void Print(ConsolePrinter& printer, const tflite::Metadata* metadata, const tflite::Buffer* buffer, const uint8_t* model_data, size_t model_size);

private:
    static void PrintConversionMetadata(ConsolePrinter& printer, const uint8_t* data, size_t size);
    static void PrintTfliteMetadata(ConsolePrinter& printer, const uint8_t* data, size_t size, const uint8_t* model_data, size_t model_size);
    static void PrintUnknownMetadata(ConsolePrinter& printer, const uint8_t* data, size_t size);
    static void PrintDataPreview(ConsolePrinter& printer, const uint8_t* data, size_t size);
};

#endif