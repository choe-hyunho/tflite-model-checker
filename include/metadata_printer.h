#ifndef METADATA_PRINTER_H
#define METADATA_PRINTER_H

class ConsolePrinter;
class ModelDocument;

class MetadataPrinter {
public:
    MetadataPrinter(
        ConsolePrinter& printer,
        const ModelDocument& document);

    void Print() const;

private:
    ConsolePrinter& printer_;
    const ModelDocument& document_;
};

#endif