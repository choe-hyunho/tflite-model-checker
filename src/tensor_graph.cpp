#include "tensor_graph.h"

#include "model_document.h"

TensorGraph::TensorGraph(const ModelDocument& document)
    : document_(document) {}

bool TensorGraph::Build()
{
    tensors_.clear();
    if (!document_.IsValid()) {
        return false;
    }

    const auto* subgraph = document_.PrimarySubgraph();

    if (subgraph == nullptr) {
        return false;
    }

    const auto* tensors = subgraph->tensors();

    if (tensors == nullptr) {
        return false;
    }
    tensors_.resize(tensors->size());

    const auto* operators = subgraph->operators();

    if (operators == nullptr) {
        return true;
    }
    for (size_t op_index = 0; op_index < operators->size(); ++op_index) {
        const auto* op = operators->Get(op_index);
        const auto* outputs = op->outputs();

        if (outputs) {
            for (size_t i = 0; i < outputs->size(); ++i) {
                int tensor = outputs->Get(i);

                if (tensor >= 0 && static_cast<size_t>(tensor) < tensors_.size()) {
                    tensors_[tensor].producer = static_cast<int>(op_index);
                }
            }
        }

        const auto* inputs = op->inputs();

        if (inputs) {
            for (size_t i = 0; i < inputs->size(); ++i) {
                int tensor = inputs->Get(i);

                if (tensor >= 0 && static_cast<size_t>(tensor) < tensors_.size()) {
                    tensors_[tensor].consumers.push_back(static_cast<int>(op_index));
                }
            }
        }
    }
    return true;
}

const TensorGraph::TensorInfo* TensorGraph::Tensor(size_t index) const
{
    if (index >= tensors_.size()) {
        return nullptr;
    }
    return &tensors_[index];
}

size_t TensorGraph::TensorCount() const
{
    return tensors_.size();
}

TensorKind TensorGraph::Kind(size_t tensor_index) const
{
    const auto* subgraph = document_.PrimarySubgraph();

    if (subgraph == nullptr) {
        return TensorKind::RUNTIME;
    }

    const auto* inputs = subgraph->inputs();

    if (inputs != nullptr) {
        for (size_t i = 0; i < inputs->size(); ++i) {
            if (inputs->Get(i) == static_cast<int32_t>(tensor_index)) {
                return TensorKind::INPUT;
            }
        }
    }

    const auto* info = Tensor(tensor_index);

    if (info == nullptr) {
        return TensorKind::RUNTIME;
    }
    if (info->producer < 0) {
        return TensorKind::CONSTANT;
    }
    return TensorKind::RUNTIME;
}