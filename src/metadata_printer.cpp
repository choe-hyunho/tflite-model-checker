#include "metadata_printer.h"

#include <algorithm>
#include <cstdio>

#include "console_printer.h"
#include "model_document.h"
#include "metadata_decoder.h"

MetadataPrinter::MetadataPrinter(ConsolePrinter& printer, const ModelDocument& document)
    : printer_(printer), document_(document) {}

void MetadataPrinter::Print() const
{
    printer_.Section("Metadata");
    if (!document_.IsValid()) {
        printer_.Line("(invalid model)");
        return;
    }
    const auto* metadata = document_.Metadata();
    const auto* buffers = document_.Buffers();
    if (metadata == nullptr || metadata->size() == 0) {
        printer_.Line("(none)");
        return;
    }
    for (size_t i = 0; i < metadata->size(); ++i) {
        const auto* entry = metadata->Get(i);
        if (entry == nullptr) {
            continue;
        }
        char title[128];
        snprintf(title, sizeof(title), "[%zu] %s", i, entry->name() != nullptr ? entry->name()->c_str() : "");
        printer_.SubSection(title);
        const uint32_t buffer_index = entry->buffer();
        printer_.Key("Buffer", static_cast<size_t>(buffer_index));
        const tflite::Buffer* buffer = nullptr;
        if (buffers != nullptr &&
            static_cast<size_t>(buffer_index) < buffers->size()) {
            buffer = buffers->Get(static_cast<size_t>(buffer_index));
        }
        MetadataDecoder::Print(printer_, entry, buffer, document_.Data(), document_.Size());
    }
}