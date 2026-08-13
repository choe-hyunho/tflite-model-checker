#ifndef MODEL_VALIDATOR_H
#define MODEL_VALIDATOR_H

#include <cstddef>

#include "model_document.h"

class ModelValidator {
public:
    explicit ModelValidator(const ModelDocument& document);
    bool Validate() const;

private:
    bool ValidateModelInterface() const;
    bool ValidateTensorIndexList(const flatbuffers::Vector<int32_t>* indices, size_t tensor_count) const;
    bool ValidateOperatorCodes() const;
    bool ValidateTensors() const;
    bool ValidateTensor(size_t tensor_index, const tflite::Tensor* tensor) const;
    bool ValidateQuantization(const tflite::Tensor* tensor) const;
    bool ValidateGraph() const;
    bool ValidateTensorGraph(size_t tensor_index, const tflite::Tensor* tensor) const;
    bool ValidateOperators() const;
    bool ValidateOperatorTensorReferences(size_t operator_index, const tflite::Operator* op) const;
    bool ValidateFullyConnected(size_t operator_index, const tflite::Operator* op) const;
    bool ValidateConv2D(size_t operator_index, const tflite::Operator* op) const;
    bool ValidateDepthwiseConv2D(size_t operator_index, const tflite::Operator* op) const;
    bool ValidateMean(size_t operator_index, const tflite::Operator* op) const;
    bool ValidateReshape(size_t operator_index, const tflite::Operator* op) const;
    bool ValidateSoftmax(size_t operator_index, const tflite::Operator* op) const;
    bool ValidateFullyConnectedOptions(size_t operator_index, const tflite::Operator* op) const;
    bool ValidateConv2DOptions(size_t operator_index, const tflite::Operator* op) const;
    bool ValidateDepthwiseConv2DOptions(size_t operator_index, const tflite::Operator* op) const;
    bool ValidateMeanOptions(size_t operator_index, const tflite::Operator* op) const;
    void SetError(size_t operator_index, const char* message) const;
    void SetError(const char* message) const;

private:
    const ModelDocument& document_;
    mutable std::string last_error_;
};

#endif