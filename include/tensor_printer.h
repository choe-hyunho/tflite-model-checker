#ifndef TENSOR_PRINTER_H
#define TENSOR_PRINTER_H

#include <cstddef>
#include <string>
#include <sstream>

class ConsolePrinter;
class ModelDocument;
class TensorGraph;

namespace tflite {
class Tensor;
}

class TensorPrinter {
public:
    TensorPrinter(ConsolePrinter& printer, const ModelDocument& document, const TensorGraph& graph);

    void Print() const;

private:
    void PrintTensor(size_t index, const tflite::Tensor* tensor) const;
    std::string OperatorDescription(size_t operator_index) const;
    bool IsModelInput(size_t tensor_index) const;
    bool IsModelOutput(size_t tensor_index) const;
    std::string TensorType(const tflite::Tensor* tensor) const;
    std::string TensorShape(const tflite::Tensor* tensor) const;
    std::string TensorRole(size_t tensor_index) const;
    size_t TensorDataSize(const tflite::Tensor* tensor) const;
    size_t ExpectedTensorBytes(const tflite::Tensor* tensor) const;
    std::string TensorDataPreview(const tflite::Tensor* tensor, size_t max_elements = 8) const;
    std::string TensorProducer(size_t tensor_index) const;

private:
    ConsolePrinter& printer_;
    const ModelDocument& document_;
    const TensorGraph& graph_;
};

#endif