#ifndef MODEL_PRINTER_H
#define MODEL_PRINTER_H

#include <cstdint>

#include "flatbuffers/vector.h"

class ConsolePrinter;
class ModelDocument;

class ModelPrinter {
public:
    ModelPrinter(ConsolePrinter& printer, const ModelDocument& document);
    void Print() const;
    void PrintMetadata() const;
    void PrintOperators() const;

private:
    void PrintInputsOutputs() const;
    void PrintTensorInterface(const char* title, const flatbuffers::Vector<int32_t>* indices) const;

private:
    ConsolePrinter& printer_;
    const ModelDocument& document_;
};

#endif