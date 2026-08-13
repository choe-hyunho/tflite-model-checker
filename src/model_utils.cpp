#include "model_utils.h"

#include <iomanip>
#include <sstream>

/******************************************************************************
 * Tensor
 *****************************************************************************/

std::string TensorShapeToString(const flatbuffers::Vector<int32_t>* dims)
{
    if (dims == nullptr)
        return "(null)";
    if (dims->size() == 0)
        return "(scalar)";

    std::ostringstream ss;

    for (size_t i = 0; i < dims->size(); ++i) {
        if (i != 0)
            ss << 'x';
        ss << dims->Get(i);
    }
    return ss.str();
}

const char* TensorTypeName(TfLiteType type)
{
    switch (type) {
    case kTfLiteNoType:
        return "NONE";
    case kTfLiteFloat32:
        return "FLOAT32";
    case kTfLiteInt32:
        return "INT32";
    case kTfLiteUInt8:
        return "UINT8";
    case kTfLiteInt64:
        return "INT64";
    case kTfLiteString:
        return "STRING";
    case kTfLiteBool:
        return "BOOL";
    case kTfLiteInt16:
        return "INT16";
    case kTfLiteComplex64:
        return "COMPLEX64";
    case kTfLiteInt8:
        return "INT8";
    case kTfLiteFloat16:
        return "FLOAT16";
    case kTfLiteFloat64:
        return "FLOAT64";
    case kTfLiteComplex128:
        return "COMPLEX128";
    case kTfLiteUInt64:
        return "UINT64";
    case kTfLiteResource:
        return "RESOURCE";
    case kTfLiteVariant:
        return "VARIANT";
    case kTfLiteUInt32:
        return "UINT32";
    default:
        return "(unknown)";
    }
}

/******************************************************************************
 * Operators
 *****************************************************************************/
tflite::BuiltinOperator BuiltinOperatorCode(const tflite::OperatorCode* opcode)
{
    if (opcode == nullptr)
        return tflite::BuiltinOperator_ADD;
    if (opcode->builtin_code() != tflite::BuiltinOperator_ADD)
        return opcode->builtin_code();
    return static_cast<tflite::BuiltinOperator>(
        opcode->deprecated_builtin_code());
}

const char* BuiltinOperatorName(tflite::BuiltinOperator op)
{
    const char* name = tflite::EnumNameBuiltinOperator(op);
    return name ? name : "(unknown)";
}

const char* OperatorName(const tflite::OperatorCode* opcode)
{
    if (opcode == nullptr)
        return "(null)";
    if (opcode->builtin_code() == tflite::BuiltinOperator_CUSTOM) {
        if (opcode->custom_code())
            return opcode->custom_code()->c_str();
        return "CUSTOM";
    }

    return BuiltinOperatorName(BuiltinOperatorCode(opcode));
}

const char* BuiltinOptionsName(tflite::BuiltinOptions type)
{
    const char* name = tflite::EnumNameBuiltinOptions(type);

    return name ? name : "(unknown)";
}

const char* ActivationFunctionName(tflite::ActivationFunctionType type)
{
    switch (type) {
    case tflite::ActivationFunctionType_NONE:
        return "NONE";
    case tflite::ActivationFunctionType_RELU:
        return "RELU";
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
        return "RELU_N1_TO_1";
    case tflite::ActivationFunctionType_RELU6:
        return "RELU6";
    case tflite::ActivationFunctionType_TANH:
        return "TANH";
    case tflite::ActivationFunctionType_SIGN_BIT:
        return "SIGN_BIT";
    default:
        return "UNKNOWN";
    }
}

const char* FullyConnectedOptionsWeightsFormatName(tflite::FullyConnectedOptionsWeightsFormat format)
{
    switch (format) {
    case tflite::FullyConnectedOptionsWeightsFormat_DEFAULT:
        return "DEFAULT";
    case tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8:
        return "SHUFFLED4x16INT8";
    default:
        return "UNKNOWN";
    }
}

const char* CustomOptionsFormatName(tflite::CustomOptionsFormat type)
{
    const char* name = tflite::EnumNameCustomOptionsFormat(type);
    return name ? name : "(unknown)";
}

/******************************************************************************
 * Quantization
 *****************************************************************************/
std::string QuantizationToString(const tflite::QuantizationParameters* quantization)
{
    if (quantization == nullptr)
        return "None";

    const auto* scales = quantization->scale();
    const auto* zero_points = quantization->zero_point();
    const bool has_scales = scales != nullptr && scales->size() > 0;
    const bool has_zero_points = zero_points != nullptr && zero_points->size() > 0;

    if (!has_scales && !has_zero_points)
        return "None";

    std::ostringstream out;

    if (has_scales) {
        out << "scale=[";
        for (size_t i = 0; i < scales->size(); ++i) {
            if (i != 0)
                out << ", ";
            out << scales->Get(i);
        }
        out << "]";
    }
    if (has_zero_points) {
        if (has_scales)
            out << ", ";
        out << "zero_point=[";
        for (size_t i = 0; i < zero_points->size(); ++i) {
            if (i != 0)
                out << ", ";
            out << zero_points->Get(i);
        }
        out << "]";
    }
    out << ", dimension=" << quantization->quantized_dimension();
    return out.str();
}

/******************************************************************************
 * Buffer
 *****************************************************************************/
std::string BufferSizeToString(const tflite::Buffer* buffer)
{
    if (buffer == nullptr)
        return "None";

    const auto* data = buffer->data();

    if (data == nullptr)
        return "None";

    std::ostringstream ss;

    ss << data->size() << " bytes";
    return ss.str();
}

/******************************************************************************
 * Misc
 *****************************************************************************/
std::string BoolToString(bool value)
{
    return value ? "Yes" : "No";
}