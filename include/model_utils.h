#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H

#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

/******************************************************************************
 * Tensor
 *****************************************************************************/
std::string TensorShapeToString(const flatbuffers::Vector<int32_t>* dims);
const char* TensorTypeName(TfLiteType type);

/******************************************************************************
 * Operators
 *****************************************************************************/
tflite::BuiltinOperator BuiltinOperatorCode(const tflite::OperatorCode* opcode);
const char* BuiltinOperatorName(tflite::BuiltinOperator op);
const char* OperatorName(const tflite::OperatorCode* opcode);
const char* BuiltinOptionsName(tflite::BuiltinOptions type);
const char* ActivationFunctionName(tflite::ActivationFunctionType type);
const char* FullyConnectedOptionsWeightsFormatName(tflite::FullyConnectedOptionsWeightsFormat format);
const char* CustomOptionsFormatName(tflite::CustomOptionsFormat type);

/******************************************************************************
 * Quantization
 *****************************************************************************/
std::string QuantizationToString(const tflite::QuantizationParameters* quantization);

/******************************************************************************
 * Buffer
 *****************************************************************************/
std::string BufferSizeToString(const tflite::Buffer* buffer);

/******************************************************************************
 * Misc
 *****************************************************************************/
std::string BoolToString(bool value);

#endif