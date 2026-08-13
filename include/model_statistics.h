#ifndef MODEL_STATISTICS_H
#define MODEL_STATISTICS_H

#include <cstddef>

class ModelDocument;

class ModelStatistics
{
public:
    bool Build(const ModelDocument& document);
    void Clear();
    size_t TensorCount() const;
    size_t OperatorCount() const;
    size_t ConstantTensorCount() const;
    size_t VariableTensorCount() const;
    size_t DynamicTensorCount() const;
    size_t ConstantBufferBytes() const;

private:
    size_t tensor_count_ = 0;
    size_t operator_count_ = 0;
    size_t constant_tensor_count_ = 0;
    size_t variable_tensor_count_ = 0;
    size_t dynamic_tensor_count_ = 0;
    size_t constant_buffer_bytes_ = 0;
};

#endif