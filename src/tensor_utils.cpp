#include "tensor_utils.h"

#include <sstream>

namespace TensorUtils {

size_t ElementCount(
    const tflite::Tensor* tensor)
{
    if (tensor == nullptr)
        return 0;

    const auto* shape =
        tensor->shape();

    if (shape == nullptr)
        return 0;

    size_t count = 1;

    for (size_t i = 0;
         i < shape->size();
         ++i) {

        const int32_t dimension =
            shape->Get(i);

        if (dimension < 0)
            return 0;

        count *=
            static_cast<size_t>(
                dimension);
    }

    return count;
}

std::string Shape(
    const tflite::Tensor* tensor)
{
    if (tensor == nullptr)
        return "?";

    const auto* shape =
        tensor->shape();

    if (shape == nullptr)
        return "?";

    if (shape->size() == 0)
        return "scalar";

    std::ostringstream out;

    for (size_t i = 0;
         i < shape->size();
         ++i) {

        if (i != 0)
            out << "x";

        out << shape->Get(i);
    }

    return out.str();
}

size_t BufferBytes(
    const tflite::Tensor* tensor,
    const tflite::Model* model)
{
    if (tensor == nullptr ||
        model == nullptr) {

        return 0;
    }

    const auto* buffers =
        model->buffers();

    if (buffers == nullptr)
        return 0;

    const size_t buffer_index =
        static_cast<size_t>(
            tensor->buffer());

    if (buffer_index >= buffers->size())
        return 0;

    const auto* buffer =
        buffers->Get(buffer_index);

    if (buffer == nullptr ||
        buffer->data() == nullptr) {

        return 0;
    }

    return buffer->data()->size();
}
}  // namespace TensorUtils