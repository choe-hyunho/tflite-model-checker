#include "model_document.h"

ModelDocument::ModelDocument(const tflite::Model* model, const uint8_t* data, size_t size)
    : model_(model), data_(data), size_(size)
{
}

bool ModelDocument::IsValid() const
{
    return model_ != nullptr;
}

const tflite::Model* ModelDocument::Model() const
{
    return model_;
}

const uint8_t* ModelDocument::Data() const
{
    return data_;
}

size_t ModelDocument::Size() const
{
    return size_;
}

const tflite::SubGraph* ModelDocument::PrimarySubgraph() const
{
    if (model_ == nullptr) {
        return nullptr;
    }

    const auto* subgraphs = model_->subgraphs();

    if (subgraphs == nullptr) {
        return nullptr;
    }

    if (subgraphs->size() == 0) {
        return nullptr;
    }

    return subgraphs->Get(0);
}

const flatbuffers::Vector<flatbuffers::Offset<tflite::SubGraph>>* ModelDocument::Subgraphs() const
{
    return model_ ? model_->subgraphs() : nullptr;
}

const tflite::OperatorCode* ModelDocument::OperatorCode(size_t index) const
{
    const auto* codes = OperatorCodes();

    if (codes == nullptr) {
        return nullptr;
    }

    if (index >= codes->size()) {
        return nullptr;
    }

    return codes->Get(index);
}

const flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>>* ModelDocument::OperatorCodes() const
{
    return model_ ? model_->operator_codes() : nullptr;
}

const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* ModelDocument::Buffers() const
{
    return model_ ? model_->buffers() : nullptr;
}

const flatbuffers::Vector<flatbuffers::Offset<tflite::Metadata>>* ModelDocument::Metadata() const
{
    return model_ ? model_->metadata() : nullptr;
}

size_t ModelDocument::SubgraphCount() const
{
    const auto* v = Subgraphs();
    return v ? v->size() : 0;
}

size_t ModelDocument::BufferCount() const
{
    const auto* v = Buffers();
    return v ? v->size() : 0;
}

size_t ModelDocument::OperatorCodeCount() const
{
    const auto* v = OperatorCodes();
    return v ? v->size() : 0;
}

uint32_t ModelDocument::Version() const
{
    return model_ ? model_->version() : 0;
}

size_t ModelDocument::MetadataCount() const
{
    const auto* metadata = Metadata();

    return metadata ? metadata->size() : 0;
}