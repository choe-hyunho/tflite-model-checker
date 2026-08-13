#ifndef OPERATOR_GRAPH_H
#define OPERATOR_GRAPH_H

#include <vector>

class ModelDocument;
class TensorGraph;

class OperatorGraph {
public:
    struct OperatorInfo {
        std::vector<int> predecessors;
        std::vector<int> successors;
    };

    OperatorGraph(const ModelDocument& document, const TensorGraph& tensor_graph);
    bool Build();
    const OperatorInfo* Operator(size_t index) const;
    size_t OperatorCount() const;

private:
    const ModelDocument& document_;
    const TensorGraph& tensor_graph_;
    std::vector<OperatorInfo> operators_;
};

#endif