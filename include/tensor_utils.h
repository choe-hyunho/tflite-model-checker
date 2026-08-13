#ifndef TENSOR_UTILS_H_
#define TENSOR_UTILS_H_

#include <cstddef>
#include <string>

#include "tensorflow/lite/schema/schema_generated.h"

namespace TensorUtils {

size_t ElementCount(const tflite::Tensor* tensor);
size_t BufferBytes(const tflite::Tensor* tensor, const tflite::Model* model);
std::string Shape(const tflite::Tensor* tensor);

}

#endif  // TENSOR_UTILS_H_