#ifndef MODEL_DOCUMENT_H
#define MODEL_DOCUMENT_H

#include <cstddef>

#include "tensorflow/lite/schema/schema_generated.h"

class ModelDocument {
public:
    explicit ModelDocument(const tflite::Model* model = nullptr);
    bool IsValid() const;
    const tflite::Model* Model() const;
    const tflite::SubGraph* PrimarySubgraph() const;
    const flatbuffers::Vector<flatbuffers::Offset<tflite::SubGraph>>* Subgraphs() const;
    const tflite::OperatorCode* OperatorCode(size_t index) const;
    const flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>>* OperatorCodes() const;
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* Buffers() const;
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Metadata>>* Metadata() const;
    size_t SubgraphCount() const;
    size_t BufferCount() const;
    size_t OperatorCodeCount() const;
    uint32_t Version() const;
    size_t MetadataCount() const;

private:
    const tflite::Model* model_;
};

#endif