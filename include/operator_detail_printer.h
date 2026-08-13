#ifndef OPERATOR_DETAIL_PRINTER_H
#define OPERATOR_DETAIL_PRINTER_H

#include <cstdint>
#include <string>

#include "flatbuffers/vector.h"

class ConsolePrinter;
class ModelDocument;

namespace tflite {
class Operator;
}

class OperatorDetailPrinter {
public:
    OperatorDetailPrinter(
        ConsolePrinter& printer,
        const ModelDocument& document);

    void Print() const;

private:
    std::string TensorDescription(
        int32_t tensor_index) const;

    std::string TensorListDescription(
        const flatbuffers::Vector<int32_t>* tensors) const;

    void PrintFullyConnectedOptions(
        const tflite::Operator* op) const;

    void PrintFullyConnectedParameters(
        const tflite::Operator* op) const;

    std::string CustomOptionsDescription(
        const tflite::Operator* op) const;

    size_t TensorElementCount(
        size_t tensor_index) const;

    size_t TensorBufferBytes(
        size_t tensor_index) const;

    std::string TensorShapeDescription(
        size_t tensor_index) const;

private:
    ConsolePrinter& printer_;
    const ModelDocument& document_;
};

#endif