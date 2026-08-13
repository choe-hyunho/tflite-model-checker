#ifndef VALIDATION_PRINTER_H
#define VALIDATION_PRINTER_H

#include <cstddef>

class ConsolePrinter;
class ModelDocument;

class ValidationPrinter {
public:
    ValidationPrinter(ConsolePrinter& printer, const ModelDocument& document);
    void Print() const;

private:
    bool CheckTensorIndices() const;
    bool CheckOperatorIndices() const;
    bool CheckBuffers() const;
    bool CheckGraph() const;
    bool CheckTensorIndex(size_t tensor_index, size_t tensor_count) const;

private:
    ConsolePrinter& printer_;
    const ModelDocument& document_;
};

#endif