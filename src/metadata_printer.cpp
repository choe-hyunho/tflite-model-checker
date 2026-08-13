#include "metadata_printer.h"

#include "console_printer.h"
#include "model_document.h"

MetadataPrinter::MetadataPrinter(
    ConsolePrinter& printer,
    const ModelDocument& document)
    : printer_(printer),
      document_(document)
{
}

void MetadataPrinter::Print() const
{
    printer_.Section("Metadata");

    if (!document_.IsValid()) {
        printer_.Line("(invalid model)");
        return;
    }

    const auto* metadata = document_.Metadata();

    if (metadata == nullptr || metadata->size() == 0) {
        printer_.Line("(none)");
        return;
    }

    printer_.TableHeader(
        "Index",
        "Name",
        "Buffer");

    for (size_t i = 0; i < metadata->size(); ++i) {

        const auto* entry = metadata->Get(i);

        char index[16];
        char buffer[16];

        snprintf(index,
                 sizeof(index),
                 "%zu",
                 i);

        snprintf(buffer,
                 sizeof(buffer),
                 "%u",
                 entry->buffer());

        printer_.TableRow(
            index,
            entry->name()
                ? entry->name()->c_str()
                : "",
            buffer);
    }
}