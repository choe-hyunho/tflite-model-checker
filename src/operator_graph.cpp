#include "operator_graph.h"

#include <algorithm>

#include "model_document.h"
#include "tensor_graph.h"

OperatorGraph::OperatorGraph(
    const ModelDocument& document,
    const TensorGraph& tensor_graph)
    : document_(document),
      tensor_graph_(tensor_graph)
{
}

bool OperatorGraph::Build()
{
    operators_.clear();

    if (!document_.IsValid())
        return false;

    const auto* subgraph = document_.PrimarySubgraph();
    if (subgraph == nullptr)
        return false;

    const auto* ops = subgraph->operators();
    if (ops == nullptr)
        return false;

    operators_.resize(ops->size());

    for (size_t op_index = 0; op_index < ops->size(); ++op_index) {

        const auto* op = ops->Get(op_index);
        const auto* inputs = op->inputs();

        if (inputs == nullptr)
            continue;

        for (size_t i = 0; i < inputs->size(); ++i) {

            int tensor = inputs->Get(i);
            if (tensor < 0)
                continue;

            const auto* info =
                tensor_graph_.Tensor(static_cast<size_t>(tensor));

            if (info == nullptr)
                continue;

            if (info->producer < 0)
                continue;

            int pred = info->producer;

            auto& preds = operators_[op_index].predecessors;
            if (std::find(preds.begin(), preds.end(), pred) == preds.end())
                preds.push_back(pred);

            auto& succs = operators_[pred].successors;
            if (std::find(succs.begin(), succs.end(),
                          static_cast<int>(op_index)) == succs.end()) {
                succs.push_back(static_cast<int>(op_index));
            }
        }
    }

    return true;
}

const OperatorGraph::OperatorInfo*
OperatorGraph::Operator(size_t index) const
{
    if (index >= operators_.size())
        return nullptr;

    return &operators_[index];
}

size_t OperatorGraph::OperatorCount() const
{
    return operators_.size();
}