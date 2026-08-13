#pragma once

#include <cstddef>

class ConsolePrinter;
class ModelDocument;
class TensorGraph;

class ModelSummaryPrinter {
public:
    ModelSummaryPrinter(
        ConsolePrinter& printer,
        const ModelDocument& document,
        const TensorGraph& graph);

    void Print() const;

private:
    size_t InputCount() const;
    size_t OutputCount() const;
    size_t TensorCount() const;
    size_t ConstantTensorCount() const;
    size_t RuntimeTensorCount() const;
    size_t OperatorCount() const;
    size_t OperatorCodeCount() const;
    size_t ConstantDataBytes() const;

    size_t ParameterCount() const;
    size_t ParameterBytes() const;
    size_t TensorElementCount(
        size_t tensor_index) const;
    size_t TensorBufferBytes(
        size_t tensor_index) const;

    ConsolePrinter& printer_;
    const ModelDocument& document_;
    const TensorGraph& graph_;
};