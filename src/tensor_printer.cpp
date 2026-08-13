#include "tensor_printer.h"

#include <cstdio>
#include <sstream>
#include <string>
#include <algorithm>

#include "console_printer.h"
#include "model_document.h"
#include "model_utils.h"
#include "tensor_graph.h"

TensorPrinter::TensorPrinter(ConsolePrinter& printer, const ModelDocument& document, const TensorGraph& graph)
    : printer_(printer), document_(document), graph_(graph) {}

void TensorPrinter::Print() const
{
    printer_.Section("Tensors");
    if (!document_.IsValid()) {
        printer_.Line("(invalid model)");
        return;
    }

    const auto* subgraph = document_.PrimarySubgraph();

    if (subgraph == nullptr) {
        printer_.Line("(no subgraph)");
        return;
    }

    const auto* tensors = subgraph->tensors();

    if (tensors == nullptr) {
        printer_.Line("(no tensors)");
        return;
    }
    for (size_t i = 0; i < tensors->size(); ++i) {
        PrintTensor(i, tensors->Get(i));
        printer_.Blank();
    }
}

void TensorPrinter::PrintTensor(size_t index, const tflite::Tensor* tensor) const
{
    char title[32];

    snprintf(title, sizeof(title), "Tensor %zu", index);
    printer_.SubSection(title);
    printer_.Key("Name", tensor->name() ? tensor->name()->c_str() : "");
    printer_.Key("Role", TensorRole(index));

    const char* type_name = tflite::EnumNameTensorType(tensor->type());

    printer_.Key("Type", type_name != nullptr ? type_name : "INVALID");
    printer_.Key("Shape", TensorShapeToString(tensor->shape()));
    printer_.Key("Buffer", static_cast<size_t>(tensor->buffer()));

    const size_t actual_size = TensorDataSize(tensor);
    const size_t expected_size = ExpectedTensorBytes(tensor);

    printer_.Key("Expected Bytes", expected_size);
    printer_.Key("Buffer Bytes", actual_size);

    const auto* info = graph_.Tensor(index);

    if (info != nullptr && info->producer >= 0) {
        printer_.Key("Buffer Check", "N/A (runtime tensor)");
    } else if (expected_size != 0) {
        printer_.Key("Buffer Check", actual_size == expected_size ? "OK" : "MISMATCH");
    }

    const std::string preview = TensorDataPreview(tensor);

    if (!preview.empty()) {
        printer_.Key("Data", preview);
    }
    printer_.Key("Variable", BoolToString(tensor->is_variable()));
    printer_.Key("Quantization", QuantizationToString(tensor->quantization()));
    if (info == nullptr) {
        return;
    }
    printer_.Key("Producer", TensorProducer(index));

    std::ostringstream consumers;

    for (size_t i = 0; i < info->consumers.size(); ++i) {
        if (i != 0) {
            consumers << ", ";
        }
        consumers << OperatorDescription(info->consumers[i]);
    }
    if (info->consumers.empty()) {
        printer_.Key("Consumers", "NONE");
    } else {
        printer_.Key("Consumers", consumers.str());
    }
}

std::string TensorPrinter::OperatorDescription(size_t operator_index) const
{
    const auto* subgraph = document_.PrimarySubgraph();

    if (subgraph == nullptr) {
        return "?";
    }

    const auto* operators = subgraph->operators();

    if (operators == nullptr || operator_index >= operators->size()) {
        return "?";
    }

    const auto* op = operators->Get(operator_index);

    if (op == nullptr) {
        return "?";
    }

    const auto* opcode = document_.OperatorCode(op->opcode_index());

    if (opcode == nullptr) {
        return "?";
    }

    const auto builtin = opcode->builtin_code();
    std::ostringstream out;

    out << "Operator " << operator_index << " (" << BuiltinOperatorName(builtin) << ")";
    return out.str();
}

bool TensorPrinter::IsModelInput(size_t tensor_index) const
{
    const auto* subgraph = document_.PrimarySubgraph();

    if (subgraph == nullptr) {
        return false;
    }

    const auto* inputs = subgraph->inputs();

    if (inputs == nullptr) {
        return false;
    }
    for (size_t i = 0; i < inputs->size(); ++i) {
        if (inputs->Get(i) == static_cast<int32_t>(tensor_index)) {
            return true;
        }
    }
    return false;
}

bool TensorPrinter::IsModelOutput(size_t tensor_index) const
{
    const auto* subgraph = document_.PrimarySubgraph();

    if (subgraph == nullptr) {
        return false;
    }

    const auto* outputs = subgraph->outputs();

    if (outputs == nullptr) {
        return false;
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
        if (outputs->Get(i) == static_cast<int32_t>(tensor_index)) {
            return true;
        }
    }
    return false;
}

std::string TensorPrinter::TensorRole(size_t tensor_index) const
{
    const auto* subgraph = document_.PrimarySubgraph();

    if (subgraph == nullptr) {
        return "UNKNOWN";
    }

    const auto* inputs = subgraph->inputs();

    if (inputs != nullptr) {
        for (size_t i = 0; i < inputs->size(); ++i) {
            if (inputs->Get(i) == static_cast<int32_t>(tensor_index)) {
                return "INPUT";
            }
        }
    }

    const auto* outputs = subgraph->outputs();

    if (outputs != nullptr) {
        for (size_t i = 0; i < outputs->size(); ++i) {
            if (outputs->Get(i) == static_cast<int32_t>(tensor_index)) {
                return "OUTPUT";
            }
        }
    }
    return "INTERMEDIATE";
}

size_t TensorPrinter::TensorDataSize(const tflite::Tensor* tensor) const
{
    if (tensor == nullptr) {
        return 0;
    }

    const auto* buffers = document_.Buffers();

    if (buffers == nullptr) {
        return 0;
    }

    const size_t buffer_index = static_cast<size_t>(tensor->buffer());

    if (buffer_index >= buffers->size()) {
        return 0;
    }

    const auto* buffer = buffers->Get(buffer_index);

    if (buffer == nullptr || buffer->data() == nullptr) {
        return 0;
    }
    return buffer->data()->size();
}

size_t TensorPrinter::ExpectedTensorBytes(const tflite::Tensor* tensor) const
{
    if (tensor == nullptr) {
        return 0;
    }

    const auto* shape = tensor->shape();

    if (shape == nullptr) {
        return 0;
    }

    size_t element_count = 1;

    for (size_t i = 0; i < shape->size(); ++i) {
        const int32_t dimension = shape->Get(i);

        if (dimension < 0) {
            return 0;
        }
        element_count *= static_cast<size_t>(dimension);
    }

    switch (tensor->type()) {
    case tflite::TensorType_FLOAT32:
        return element_count * 4;
    case tflite::TensorType_FLOAT16:
        return element_count * 2;
    case tflite::TensorType_INT32:
    case tflite::TensorType_UINT32:
        return element_count * 4;
    case tflite::TensorType_UINT8:
    case tflite::TensorType_BOOL:
    case tflite::TensorType_INT8:
        return element_count;
    case tflite::TensorType_INT64:
    case tflite::TensorType_UINT64:
        return element_count * 8;
    case tflite::TensorType_INT16:
    case tflite::TensorType_UINT16:
        return element_count * 2;
    case tflite::TensorType_INT4:
        return (element_count + 1) / 2;
    default:
        return 0;
    }
}

std::string TensorPrinter::TensorDataPreview(const tflite::Tensor* tensor, size_t max_elements) const
{
    if (tensor == nullptr) {
        return "";
    }

    const auto* buffers = document_.Buffers();

    if (buffers == nullptr) {
        return "";
    }

    const size_t buffer_index = static_cast<size_t>(tensor->buffer());

    if (buffer_index >= buffers->size()) {
        return "";
    }

    const auto* buffer = buffers->Get(buffer_index);

    if (buffer == nullptr || buffer->data() == nullptr || buffer->data()->size() == 0) {
        return "";
    }

    const uint8_t* data = buffer->data()->data();
    const size_t data_size = buffer->data()->size();
    const size_t expected_size = ExpectedTensorBytes(tensor);

    if (expected_size == 0 || data_size < expected_size) {
        return "";
    }

    const auto* shape = tensor->shape();

    if (shape == nullptr) {
        return "";
    }

    size_t element_count = 1;

    for (size_t i = 0; i < shape->size(); ++i) {
        const int32_t dimension = shape->Get(i);

        if (dimension < 0) {
            return "";
        }
        element_count *= static_cast<size_t>(dimension);
    }

    const size_t count = std::min(element_count, max_elements);
    std::ostringstream out;

    out << "[";
    for (size_t i = 0; i < count; ++i) {
        if (i != 0) {
            out << ", ";
        }
        switch (tensor->type()) {
        case tflite::TensorType_FLOAT32: {
            float value;
            memcpy(&value, data + i * sizeof(float), sizeof(float));
            out << value;
            break;
        }
        case tflite::TensorType_INT8: {
            int8_t value;
            memcpy(&value, data + i * sizeof(value), sizeof(value));
            out << static_cast<int>(value);
            break;
        }
        case tflite::TensorType_UINT8: {
            uint8_t value;
            memcpy(&value, data + i * sizeof(value), sizeof(value));
            out << static_cast<unsigned>(value);
            break;
        }
        case tflite::TensorType_INT16: {
            int16_t value;
            memcpy(&value, data + i * sizeof(value), sizeof(value));
            out << value;
            break;
        }
        case tflite::TensorType_INT32: {
            int32_t value;
            memcpy(&value, data + i * sizeof(value), sizeof(value));
            out << value;
            break;
        }
        case tflite::TensorType_INT64: {
            int64_t value;
            memcpy(&value, data + i * sizeof(value), sizeof(value));
            out << value;
            break;
        }
        default:
            return "";
        }
    }
    if (count < element_count) {
        out << ", ...";
    }
    out << "]";
    return out.str();
}

std::string TensorPrinter::TensorProducer(size_t tensor_index) const
{
    const TensorKind kind = graph_.Kind(tensor_index);

    switch (kind) {
    case TensorKind::INPUT:
        return "MODEL INPUT";
    case TensorKind::CONSTANT:
        return "CONSTANT";
    case TensorKind::RUNTIME:
        break;
    }

    const auto* info = graph_.Tensor(tensor_index);

    if (info == nullptr || info->producer < 0) {
        return "UNKNOWN";
    }
    return OperatorDescription(static_cast<size_t>(info->producer));
}