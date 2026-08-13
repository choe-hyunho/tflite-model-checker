#include "model_printer.h"

#include <sstream>
#include <string>

#include "model_utils.h"
#include "console_printer.h"
#include "model_document.h"
#include "tensorflow/lite/schema/schema_generated.h"

ModelPrinter::ModelPrinter(
    ConsolePrinter& printer,
    const ModelDocument& document)
    : printer_(printer),
      document_(document)
{
}

void ModelPrinter::Print() const
{
    printer_.Banner(
        "TensorFlow Lite Model Checker",
        "1.0");

    printer_.Section("Model");

    if (!document_.IsValid()) {
        printer_.Status("Model", false);
        return;
    }

    printer_.Status("Model", true);

    printer_.Key(
        "Schema Version",
        static_cast<size_t>(document_.Version()));

    printer_.Key(
        "Subgraphs",
        document_.SubgraphCount());

    printer_.Key(
        "Operator Codes",
        document_.OperatorCodeCount());

    printer_.Key(
        "Buffers",
        document_.BufferCount());

    printer_.Key(
        "Metadata",
        document_.MetadataCount());

    PrintInputsOutputs();
}

void ModelPrinter::PrintInputsOutputs() const
{
    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr) {
        printer_.Section("Interface");
        printer_.Line("(no subgraph)");
        return;
    }

    printer_.Section("Interface");

    PrintTensorInterface(
        "Inputs",
        subgraph->inputs());

    PrintTensorInterface(
        "Outputs",
        subgraph->outputs());
}

void ModelPrinter::PrintTensorInterface(
    const char* title,
    const flatbuffers::Vector<int32_t>* indices) const
{
    printer_.SubSection(title);

    if (indices == nullptr ||
        indices->size() == 0) {

        printer_.Line("(none)");
        return;
    }

    const auto* subgraph =
        document_.PrimarySubgraph();

    if (subgraph == nullptr) {
        printer_.Line("(no subgraph)");
        return;
    }

    const auto* tensors =
        subgraph->tensors();

    if (tensors == nullptr) {
        printer_.Line("(no tensors)");
        return;
    }

    for (size_t i = 0;
         i < indices->size();
         ++i) {

        const int32_t tensor_index =
            indices->Get(i);

        if (tensor_index < 0 ||
            static_cast<size_t>(tensor_index) >=
                tensors->size()) {

            printer_.Key(
                "Tensor",
                "INVALID");

            continue;
        }

        const auto* tensor =
            tensors->Get(
                static_cast<size_t>(tensor_index));

        if (tensor == nullptr) {
            printer_.Key(
                "Tensor",
                "INVALID");

            continue;
        }

        std::ostringstream value;

        value << tensor_index;

        if (tensor->name() != nullptr &&
            tensor->name()->size() != 0) {

            value << " ("
                  << tensor->name()->c_str()
                  << ")";
        }

        printer_.Key(
            "Tensor",
            value.str());

        printer_.Key(
            "Type",
            TensorTypeName(
                static_cast<TfLiteType>(
                    tensor->type())));

        printer_.Key(
            "Shape",
            TensorShapeToString(
                tensor->shape()));
    }
}