#ifndef RESHAPE_PRINTER_H
#define RESHAPE_PRINTER_H

#include "console_printer.h"

class ModelDocument;

class ReshapePrinter {
public:
    ReshapePrinter(ConsolePrinter& printer, const ModelDocument& document);
    void Print() const;

private:
    ConsolePrinter& printer_;
    const ModelDocument& document_;
};

#endif