#ifndef TENSOR_GRAPH_H
#define TENSOR_GRAPH_H

#include <vector>

class ModelDocument;

enum class TensorKind {
    INPUT,
    CONSTANT,
    RUNTIME
};

class TensorGraph {
public:
    struct TensorInfo {
        int producer = -1;
        std::vector<int> consumers;
    };

    explicit TensorGraph(const ModelDocument& document);

    bool Build();

    const TensorInfo* Tensor(size_t index) const;

    size_t TensorCount() const;

    TensorKind Kind(size_t tensor_index) const;

private:
    const ModelDocument& document_;
    std::vector<TensorInfo> tensors_;
};

#endif