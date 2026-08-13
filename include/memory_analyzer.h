#ifndef MEMORY_ANALYZER_H
#define MEMORY_ANALYZER_H

#include <vector>

class ModelDocument;

struct TensorLifetime
{
    int tensor;

    int first_op;
    int last_op;

    size_t bytes;

    bool constant;
};

class MemoryAnalyzer
{
public:

    bool Build(const ModelDocument& document);

    const TensorLifetime&
    Lifetime(int tensor) const;

    size_t PeakMemory() const;

    size_t TensorCount() const;

private:

    std::vector<TensorLifetime> lifetimes_;

    size_t peak_memory_ = 0;
};

#endif