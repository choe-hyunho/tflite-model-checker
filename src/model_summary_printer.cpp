#include "model_summary_printer.h"

#include "console_printer.h"
#include "model_document.h"
#include "tensor_graph.h"

#include "tensorflow/lite/schema/schema_generated.h"

ModelSummaryPrinter::ModelSummaryPrinter(
    ConsolePrinter& printer,
    const ModelDocument& document,
    const TensorGraph& graph)
    : printer_(printer),
      document_(document),
      graph_(graph)
{
}

void ModelSummaryPrinter::Print() const
{
    printer_.Section("Model Summary");

    if (!document_.IsValid()) {
        printer_.Line("(invalid model)");
        return;
    }

    printer_.Key(
        "Inputs",
        InputCount());

    printer_.Key(
        "Outputs",
        OutputCount());

    printer_.Key(
        "Tensors",
        TensorCount());

    printer_.Key(
        "Constant Tensors",
        ConstantTensorCount());

    printer_.Key(
        "Runtime Tensors",
        RuntimeTensorCount());

    printer_.Key(
        "Operators",
        OperatorCount());

    printer_.Key(
        "Operator Codes",
        OperatorCodeCount());

    printer_.Key(
        "Constant Data",
        ConstantDataBytes());

    printer_.Key(
        "Parameters",
        ParameterCount());

    printer_.Key(
        "Parameter Bytes",
        ParameterBytes());
}

size_t ModelSummaryPrinter::InputCount() const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr ||
        subgraph->inputs() == nullptr) {

        return 0;
    }

    return subgraph->inputs()->size();
}

size_t ModelSummaryPrinter::OutputCount() const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr ||
        subgraph->outputs() == nullptr) {

        return 0;
    }

    return subgraph->outputs()->size();
}

size_t ModelSummaryPrinter::TensorCount() const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr ||
        subgraph->tensors() == nullptr) {

        return 0;
    }

    return subgraph->tensors()->size();
}

size_t ModelSummaryPrinter::OperatorCount() const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr ||
        subgraph->operators() == nullptr) {

        return 0;
    }

    return subgraph->operators()->size();
}

size_t ModelSummaryPrinter::OperatorCodeCount() const
{
    const auto* model =
        document_.Model();

    if (model == nullptr ||
        model->operator_codes() == nullptr) {

        return 0;
    }

    return model->operator_codes()->size();
}

size_t ModelSummaryPrinter::ConstantTensorCount() const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr ||
        subgraph->tensors() == nullptr) {

        return 0;
    }

    const auto* tensors =
        subgraph->tensors();

    size_t count = 0;

    for (size_t i = 0;
         i < tensors->size();
         ++i) {

        if (graph_.Kind(i) ==
            TensorKind::CONSTANT) {

            ++count;
        }
    }

    return count;
}

size_t ModelSummaryPrinter::RuntimeTensorCount() const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr ||
        subgraph->tensors() == nullptr) {

        return 0;
    }

    const auto* tensors =
        subgraph->tensors();

    size_t count = 0;

    for (size_t i = 0;
         i < tensors->size();
         ++i) {

        if (graph_.Kind(i) ==
            TensorKind::RUNTIME) {

            ++count;
        }
    }

    return count;
}

size_t ModelSummaryPrinter::ConstantDataBytes() const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr ||
        subgraph->tensors() == nullptr) {

        return 0;
    }

    const auto* tensors =
        subgraph->tensors();

    const auto* buffers =
        document_.Buffers();

    if (buffers == nullptr)
        return 0;

    size_t total = 0;

    for (size_t i = 0;
         i < tensors->size();
         ++i) {

        const auto* tensor =
            tensors->Get(i);

        if (tensor == nullptr)
            continue;

        const size_t buffer_index =
            static_cast<size_t>(
                tensor->buffer());

        if (buffer_index >= buffers->size())
            continue;

        const auto* buffer =
            buffers->Get(buffer_index);

        if (buffer == nullptr ||
            buffer->data() == nullptr) {

            continue;
        }

        total += buffer->data()->size();
    }

    return total;
}

size_t ModelSummaryPrinter::ParameterCount() const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr ||
        subgraph->operators() == nullptr) {

        return 0;
    }

    const auto* operators =
        subgraph->operators();

    size_t total = 0;

    for (size_t i = 0;
         i < operators->size();
         ++i) {

        const auto* op =
            operators->Get(i);

        if (op == nullptr)
            continue;

        const auto* opcode =
            document_.OperatorCode(
                op->opcode_index());

        if (opcode == nullptr)
            continue;

        if (opcode->builtin_code() !=
            tflite::BuiltinOperator_FULLY_CONNECTED) {

            continue;
        }

        const auto* inputs =
            op->inputs();

        if (inputs == nullptr ||
            inputs->size() < 3) {

            continue;
        }

        const int32_t weights_index =
            inputs->Get(1);

        const int32_t bias_index =
            inputs->Get(2);

        if (weights_index >= 0)
            total += TensorElementCount(
                static_cast<size_t>(
                    weights_index));

        if (bias_index >= 0)
            total += TensorElementCount(
                static_cast<size_t>(
                    bias_index));
    }

    return total;
}

size_t ModelSummaryPrinter::TensorElementCount(
    size_t tensor_index) const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr ||
        subgraph->tensors() == nullptr ||
        tensor_index >= subgraph->tensors()->size()) {

        return 0;
    }

    const auto* tensor =
        subgraph->tensors()->Get(
            tensor_index);

    if (tensor == nullptr ||
        tensor->shape() == nullptr) {

        return 0;
    }

    size_t count = 1;

    const auto* shape =
        tensor->shape();

    for (size_t i = 0;
         i < shape->size();
         ++i) {

        const int32_t dimension =
            shape->Get(i);

        if (dimension < 0)
            return 0;

        count *= static_cast<size_t>(
            dimension);
    }

    return count;
}

size_t ModelSummaryPrinter::ParameterBytes() const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr ||
        subgraph->operators() == nullptr) {

        return 0;
    }

    const auto* operators =
        subgraph->operators();

    size_t total = 0;

    for (size_t i = 0;
         i < operators->size();
         ++i) {

        const auto* op =
            operators->Get(i);

        if (op == nullptr)
            continue;

        const auto* opcode =
            document_.OperatorCode(
                op->opcode_index());

        if (opcode == nullptr ||
            opcode->builtin_code() !=
                tflite::BuiltinOperator_FULLY_CONNECTED) {

            continue;
        }

        const auto* inputs =
            op->inputs();

        if (inputs == nullptr ||
            inputs->size() < 3) {

            continue;
        }

        if (inputs->Get(1) >= 0)
            total += TensorBufferBytes(
                static_cast<size_t>(
                    inputs->Get(1)));

        if (inputs->Get(2) >= 0)
            total += TensorBufferBytes(
                static_cast<size_t>(
                    inputs->Get(2)));
    }

    return total;
}

size_t ModelSummaryPrinter::TensorBufferBytes(
    size_t tensor_index) const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    const auto* buffers =
        document_.Buffers();

    if (subgraph == nullptr ||
        subgraph->tensors() == nullptr ||
        buffers == nullptr ||
        tensor_index >= subgraph->tensors()->size()) {

        return 0;
    }

    const auto* tensor =
        subgraph->tensors()->Get(
            tensor_index);

    if (tensor == nullptr)
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