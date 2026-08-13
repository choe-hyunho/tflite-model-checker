#include "operator_detail_printer.h"

#include <sstream>

#include "builtin_options_printer.h"
#include "console_printer.h"
#include "model_document.h"
#include "model_utils.h"
#include "tensor_utils.h"

#include "tensorflow/lite/schema/schema_generated.h"

OperatorDetailPrinter::OperatorDetailPrinter(ConsolePrinter& printer, const ModelDocument& document)
    : printer_(printer), document_(document) {}

void OperatorDetailPrinter::Print() const
{
    printer_.Section("Operator Details");
    if (!document_.IsValid()) {
        printer_.Line("(invalid model)");
        return;
    }

    const auto* subgraph = document_.PrimarySubgraph();

    if (subgraph == nullptr) {
        printer_.Line("(no subgraph)");
        return;
    }

    const auto* operators = subgraph->operators();
    const auto* opcodes = document_.OperatorCodes();

    if (operators == nullptr || opcodes == nullptr) {
        printer_.Line("(no operators)");
        return;
    }
    for (size_t i = 0; i < operators->size(); ++i) {
        const auto* op = operators->Get(i);
        const auto* opcode = opcodes->Get(op->opcode_index());

        printer_.SubSection(("Operator " + std::to_string(i)).c_str());
        printer_.Key("Name", OperatorName(opcode));

        const auto builtin = BuiltinOperatorCode(opcode);

        printer_.Key("Builtin", BuiltinOperatorName(builtin));

        if (opcode->builtin_code() == tflite::BuiltinOperator_FULLY_CONNECTED) {
            PrintFullyConnectedOptions(op);
            PrintFullyConnectedParameters(op);
        }

        if (builtin == tflite::BuiltinOperator_CUSTOM) {
            const char* custom_code = opcode->custom_code() != nullptr ? opcode->custom_code()->c_str() : "";
            printer_.Key("Custom Code", custom_code);
        }

        printer_.Key("Version", static_cast<size_t>(opcode->version()));
        printer_.Key("Builtin Options", BuiltinOptionsName( op->builtin_options_type()));

        if (builtin != tflite::BuiltinOperator_CUSTOM) {
            BuiltinOptionsPrinter options(printer_);
            options.Print(op);
        }
        printer_.Key("Custom Options", CustomOptionsDescription(op));
        printer_.Key("Inputs", TensorListDescription(op->inputs()));
        printer_.Key("Outputs", TensorListDescription(op->outputs()));
        if (op->inputs())
            printer_.Blank();
    }
}

std::string OperatorDetailPrinter::TensorDescription(int32_t tensor_index) const
{
    if (tensor_index < 0)
        return "NONE";

    const auto* subgraph = document_.PrimarySubgraph();

    if (subgraph == nullptr)
        return "?";

    const auto* tensors = subgraph->tensors();

    if (tensors == nullptr || static_cast<size_t>(tensor_index) >= tensors->size()) {
        return "?";
    }

    const auto* tensor = tensors->Get(static_cast<size_t>(tensor_index));

    if (tensor == nullptr)
        return "?";

    std::ostringstream out;

    out << tensor_index;
    if (tensor->name() != nullptr && tensor->name()->size() != 0) {
        out << " (" << tensor->name()->c_str() << ")";
    }
    return out.str();
}

std::string OperatorDetailPrinter::TensorListDescription(const flatbuffers::Vector<int32_t>* tensors) const
{
    if (tensors == nullptr || tensors->size() == 0) {
        return "NONE";
    }

    std::ostringstream out;

    for (size_t i = 0; i < tensors->size(); ++i) {
        if (i != 0)
            out << ", ";
        out << TensorDescription(tensors->Get(i));
    }
    return out.str();
}

void OperatorDetailPrinter::PrintFullyConnectedOptions(const tflite::Operator* op) const
{
    if (op == nullptr)
        return;
    if (op->builtin_options_type() != tflite::BuiltinOptions_FullyConnectedOptions) {
        return;
    }

    const auto* options = op->builtin_options_as_FullyConnectedOptions();

    if (options == nullptr) {
        printer_.Key("FC Options", "(missing)");
        return;
    }

    printer_.Key("Fused Activation", ActivationFunctionName(options->fused_activation_function()));
    printer_.Key("Weights Format", FullyConnectedOptionsWeightsFormatName(options->weights_format()));
    printer_.Key("Keep Num Dimensions", BoolToString(options->keep_num_dims()));
    printer_.Key("Asymmetric Quantize", BoolToString(options->asymmetric_quantize_inputs()));
}

void OperatorDetailPrinter::PrintFullyConnectedParameters(const tflite::Operator* op) const
{
    if (op == nullptr)
        return;

    const auto* subgraph = document_.PrimarySubgraph();

    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        return;
    }

    const auto* inputs = op->inputs();

    if (inputs == nullptr || inputs->size() < 3) {
        return;
    }

    const int32_t weights_index = inputs->Get(1);
    const int32_t bias_index = inputs->Get(2);

    if (weights_index < 0 || bias_index < 0) {
        return;
    }

    const size_t weights = static_cast<size_t>(weights_index);
    const size_t bias = static_cast<size_t>(bias_index);
    const auto* weights_tensor = subgraph->tensors()->Get(weights);
    const auto* bias_tensor = subgraph->tensors()->Get(bias);

    if (weights_tensor == nullptr || bias_tensor == nullptr) {
        return;
    }

    printer_.Key("Weights Tensor", weights);
    printer_.Key("Weights Shape", TensorShapeDescription(weights));
    printer_.Key("Weights Parameters", TensorElementCount(weights));
    printer_.Key("Weights Bytes", TensorBufferBytes(weights));
    printer_.Key("Bias Tensor", bias);
    printer_.Key("Bias Shape", TensorShapeDescription(bias));
    printer_.Key("Bias Parameters", TensorElementCount(bias));
    printer_.Key("Bias Bytes", TensorBufferBytes(bias));
    printer_.Key("Parameters", TensorElementCount(weights) + TensorElementCount(bias));
}

std::string OperatorDetailPrinter::CustomOptionsDescription(const tflite::Operator* op) const
{
    if (op == nullptr)
        return "None";

    const auto* options = op->custom_options();

    if (options == nullptr || options->size() == 0) {
        return "None";
    }

    std::ostringstream out;

    out << CustomOptionsFormatName(op->custom_options_format());

    out << " (" << options->size() << " bytes)";
    return out.str();
}

size_t OperatorDetailPrinter::TensorElementCount(size_t tensor_index) const
{
    const auto* subgraph = document_.PrimarySubgraph();

    if (subgraph == nullptr || subgraph->tensors() == nullptr || tensor_index >= subgraph->tensors()->size()) {
        return 0;
    }

    const auto* tensor = subgraph->tensors()->Get(tensor_index);

    if (tensor == nullptr || tensor->shape() == nullptr) {
        return 0;
    }

    size_t count = 1;
    const auto* shape = tensor->shape();

    for (size_t i = 0; i < shape->size(); ++i) {
        const int32_t dimension = shape->Get(i);
        if (dimension < 0)
            return 0;
        count *= static_cast<size_t>(dimension);
    }
    return count;
}

size_t OperatorDetailPrinter::TensorBufferBytes(size_t tensor_index) const
{
    const auto* subgraph = document_.PrimarySubgraph();

    const auto* buffers = document_.Buffers();

    if (subgraph == nullptr || subgraph->tensors() == nullptr || buffers == nullptr || tensor_index >= subgraph->tensors()->size()) {
        return 0;
    }

    const auto* tensor = subgraph->tensors()->Get(tensor_index);
    if (tensor == nullptr)
        return 0;
    const size_t buffer_index = static_cast<size_t>(tensor->buffer());
    if (buffer_index >= buffers->size())
        return 0;
    const auto* buffer = buffers->Get(buffer_index);
    if (buffer == nullptr || buffer->data() == nullptr) {
        return 0;
    }
    return buffer->data()->size();
}

std::string OperatorDetailPrinter::TensorShapeDescription(size_t tensor_index) const
{
    const auto* subgraph = document_.PrimarySubgraph();

    if (subgraph == nullptr || subgraph->tensors() == nullptr || tensor_index >= subgraph->tensors()->size()) {
        return "?";
    }

    const auto* tensor = subgraph->tensors()->Get(tensor_index);

    if (tensor == nullptr || tensor->shape() == nullptr) {
        return "?";
    }

    const auto* shape = tensor->shape();

    if (shape->size() == 0)
        return "scalar";

    std::ostringstream out;

    for (size_t i = 0; i < shape->size(); ++i) {
        if (i != 0)
            out << "x";
        out << shape->Get(i);
    }
    return out.str();
}