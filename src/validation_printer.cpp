#include "validation_printer.h"

#include <vector>

#include "console_printer.h"
#include "model_document.h"

#include "tensorflow/lite/schema/schema_generated.h"

ValidationPrinter::ValidationPrinter(
    ConsolePrinter& printer,
    const ModelDocument& document)
    : printer_(printer),
      document_(document)
{
}

void ValidationPrinter::Print() const
{
    printer_.Section("Validation");

    if (!document_.IsValid()) {
        printer_.Status("Model", false);
        return;
    }

    printer_.Status(
        "Tensor Indices",
        CheckTensorIndices());

    printer_.Status(
        "Operator Indices",
        CheckOperatorIndices());

    printer_.Status(
        "Buffers",
        CheckBuffers());

    printer_.Status(
        "Graph",
        CheckGraph());
}

bool ValidationPrinter::CheckTensorIndices() const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr)
        return false;

    const auto* tensors =
        subgraph->tensors();

    if (tensors == nullptr)
        return false;

    const size_t tensor_count =
        tensors->size();

    const auto* inputs =
        subgraph->inputs();

    if (inputs != nullptr) {

        for (size_t i = 0; i < inputs->size(); ++i) {

            if (!CheckTensorIndex(
                    static_cast<size_t>(inputs->Get(i)),
                    tensor_count)) {

                return false;
            }
        }
    }

    const auto* outputs =
        subgraph->outputs();

    if (outputs != nullptr) {

        for (size_t i = 0; i < outputs->size(); ++i) {

            if (!CheckTensorIndex(
                    static_cast<size_t>(outputs->Get(i)),
                    tensor_count)) {

                return false;
            }
        }
    }

    const auto* operators =
        subgraph->operators();

    if (operators == nullptr)
        return true;

    for (size_t i = 0; i < operators->size(); ++i) {

        const auto* op =
            operators->Get(i);

        if (op == nullptr)
            return false;

        const auto* op_inputs =
            op->inputs();

        if (op_inputs != nullptr) {

            for (size_t j = 0;
                 j < op_inputs->size();
                 ++j) {

                const int32_t index =
                    op_inputs->Get(j);

                // -1 means an optional/unconnected input.
                if (index < 0)
                    continue;

                if (!CheckTensorIndex(
                        static_cast<size_t>(index),
                        tensor_count)) {

                    return false;
                }
            }
        }

        const auto* op_outputs =
            op->outputs();

        if (op_outputs != nullptr) {

            for (size_t j = 0;
                 j < op_outputs->size();
                 ++j) {

                const int32_t index =
                    op_outputs->Get(j);

                if (index < 0)
                    continue;

                if (!CheckTensorIndex(
                        static_cast<size_t>(index),
                        tensor_count)) {

                    return false;
                }
            }
        }
    }

    return true;
}

bool ValidationPrinter::CheckTensorIndex(
    size_t tensor_index,
    size_t tensor_count) const
{
    return tensor_index < tensor_count;
}

bool ValidationPrinter::CheckOperatorIndices() const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr)
        return false;

    const auto* operators =
        subgraph->operators();

    if (operators == nullptr)
        return true;

    const size_t operator_count =
        operators->size();

    const size_t opcode_count =
        document_.OperatorCodeCount();

    for (size_t i = 0;
         i < operator_count;
         ++i) {

        const auto* op =
            operators->Get(i);

        if (op == nullptr)
            return false;

        const uint32_t opcode_index =
            op->opcode_index();

        if (opcode_index >= opcode_count)
            return false;
    }

    return true;
}

bool ValidationPrinter::CheckBuffers() const
{
    const size_t buffer_count =
        document_.BufferCount();

    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr)
        return false;

    const auto* tensors =
        subgraph->tensors();

    if (tensors == nullptr)
        return false;

    for (size_t i = 0;
         i < tensors->size();
         ++i) {

        const auto* tensor =
            tensors->Get(i);

        if (tensor == nullptr)
            return false;

        if (tensor->buffer() >= buffer_count)
            return false;
    }

    return true;
}

bool ValidationPrinter::CheckGraph() const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr)
        return false;

    const auto* tensors =
        subgraph->tensors();

    const auto* operators =
        subgraph->operators();

    if (tensors == nullptr)
        return false;

    if (operators == nullptr)
        return true;

    const size_t tensor_count =
        tensors->size();

    /*
     * producer[i] is the operator producing tensor i.
     *
     * -1 means that the tensor has no operator producer.
     */
    std::vector<int> producer(
        tensor_count,
        -1);

    /*
     * First pass:
     *
     * Determine the producer of every tensor and
     * reject multiple producers.
     */
    for (size_t op_index = 0;
         op_index < operators->size();
         ++op_index) {

        const auto* op =
            operators->Get(op_index);

        if (op == nullptr)
            return false;

        const auto* outputs =
            op->outputs();

        if (outputs == nullptr)
            return false;

        for (size_t i = 0;
             i < outputs->size();
             ++i) {

            const int32_t tensor_index =
                outputs->Get(i);

            if (tensor_index < 0)
                continue;

            if (static_cast<size_t>(tensor_index)
                >= tensor_count) {

                return false;
            }

            int& previous =
                producer[tensor_index];

            if (previous != -1) {
                /*
                 * Two operators produce the same tensor.
                 */
                return false;
            }

            previous =
                static_cast<int>(op_index);
        }
    }

    /*
     * Second pass:
     *
     * Verify that an operator doesn't consume a tensor
     * before that tensor has been produced.
     */
    for (size_t op_index = 0;
         op_index < operators->size();
         ++op_index) {

        const auto* op =
            operators->Get(op_index);

        const auto* inputs =
            op->inputs();

        if (inputs == nullptr)
            return false;

        for (size_t i = 0;
             i < inputs->size();
             ++i) {

            const int32_t tensor_index =
                inputs->Get(i);

            /*
             * -1 is an optional input.
             */
            if (tensor_index < 0)
                continue;

            if (static_cast<size_t>(tensor_index)
                >= tensor_count) {

                return false;
            }

            const int tensor_producer =
                producer[tensor_index];

            /*
             * No producer is valid for:
             *
             * - model inputs
             * - constant tensors
             * - tensors supplied externally
             */
            if (tensor_producer == -1)
                continue;

            /*
             * The producer must occur before the
             * consuming operator.
             */
            if (tensor_producer >=
                static_cast<int>(op_index)) {

                return false;
            }
        }
    }

    /*
     * Third pass:
     *
     * Verify that model inputs and outputs refer to
     * valid tensors.
     */
    const auto* inputs =
        subgraph->inputs();

    if (inputs != nullptr) {

        for (size_t i = 0;
             i < inputs->size();
             ++i) {

            const int32_t tensor_index =
                inputs->Get(i);

            if (tensor_index < 0 ||
                static_cast<size_t>(tensor_index)
                    >= tensor_count) {

                return false;
            }
        }
    }

    const auto* outputs =
        subgraph->outputs();

    if (outputs != nullptr) {

        for (size_t i = 0;
             i < outputs->size();
             ++i) {

            const int32_t tensor_index =
                outputs->Get(i);

            if (tensor_index < 0 ||
                static_cast<size_t>(tensor_index)
                    >= tensor_count) {

                return false;
            }
        }
    }

    return true;
}