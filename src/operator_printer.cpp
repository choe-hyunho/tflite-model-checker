#include "operator_printer.h"

#include <sstream>

#include "model_utils.h"

OperatorPrinter::OperatorPrinter(ConsolePrinter& printer, const ModelDocument& document)
    : printer_(printer), document_(document) {}

void OperatorPrinter::Print() const
{
    printer_.Section("Operators");

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

    if (operators == nullptr || operators->size() == 0) {
        printer_.Line("(no operators)");
        return;
    }

    const auto* opcodes = document_.OperatorCodes();

    printer_.TableHeader("Index", "Operator", "Inputs", "Outputs");
    for (size_t i = 0; i < operators->size(); ++i) {
        const auto* op = operators->Get(i);
        const auto* opcode = opcodes->Get(op->opcode_index());

        PrintOperator(i, op, opcode, subgraph);
    }
}

void OperatorPrinter::PrintOperator(size_t index, const tflite::Operator* op, const tflite::OperatorCode* opcode, const tflite::SubGraph*) const
{
    std::ostringstream inputs;
    std::ostringstream outputs;
    const auto* in = op->inputs();

    if (in) {
        for (size_t i = 0; i < in->size(); ++i) {
            if (i)
                inputs << ",";
            inputs << in->Get(i);
        }
    }

    const auto* out = op->outputs();

    if (out) {
        for (size_t i = 0; i < out->size(); ++i) {
            if (i)
                outputs << ",";
            outputs << out->Get(i);
        }
    }

    printer_.TableRow(std::to_string(index).c_str(), OperatorName(opcode), inputs.str().c_str(), outputs.str().c_str());
}