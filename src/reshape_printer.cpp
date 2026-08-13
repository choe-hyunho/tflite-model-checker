#include "reshape_printer.h"

#include "model_document.h"
#include "model_utils.h"

ReshapePrinter::ReshapePrinter(
    ConsolePrinter& printer,
    const ModelDocument& document)
    : printer_(printer),
      document_(document)
{
}

void ReshapePrinter::Print() const
{
    printer_.Section("Reshape Operators");

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

    bool found = false;

    for (size_t i = 0; i < operators->size(); ++i) {

        const auto* op = operators->Get(i);
        const auto* opcode = opcodes->Get(op->opcode_index());

        if (opcode->builtin_code() !=
            tflite::BuiltinOperator_RESHAPE)
            continue;

        found = true;

        printer_.SubSection(
            ("Reshape " + std::to_string(i)).c_str());

        if (op->inputs() && op->inputs()->size() > 0)
            printer_.Key(
                "Input Tensor",
                static_cast<size_t>(op->inputs()->Get(0)));

        if (op->outputs() && op->outputs()->size() > 0)
            printer_.Key(
                "Output Tensor",
                static_cast<size_t>(op->outputs()->Get(0)));

        const auto* options =
            op->builtin_options_as_ReshapeOptions();

        if (options && options->new_shape()) {

            printer_.Key(
                "New Shape",
                TensorShapeToString(options->new_shape()));
        }

        printer_.Blank();
    }

    if (!found)
        printer_.Line("(no reshape operators)");
}