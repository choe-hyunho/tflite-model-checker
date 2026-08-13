#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/lite/schema/schema_generated.h"

class ModelLoader {
public:
    ModelLoader();
    bool Load(const std::string& filename);
    bool IsLoaded() const;
    const uint8_t* Data() const;
    size_t Size() const;
    const tflite::Model* GetModel() const;
    const std::string& Filename() const;

private:
    std::string filename_;
    std::vector<uint8_t> data_;
};

#endif