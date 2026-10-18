#ifndef OPERATOR_PRINTER_H
#define OPERATOR_PRINTER_H

#include "console_printer.h"
#include "model_document.h"

class OperatorPrinter {
public:
    OperatorPrinter(ConsolePrinter& printer, const ModelDocument& document, bool verbose);
    void Print() const;

private:
    void PrintOperator(size_t index, const tflite::Operator* op, const tflite::OperatorCode* opcode, const tflite::SubGraph* subgraph) const;

private:
    ConsolePrinter& printer_;
    const ModelDocument& document_;
    bool verbose_;
};

#endif