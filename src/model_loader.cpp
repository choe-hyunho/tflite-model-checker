#include "model_loader.h"

#include <fstream>

ModelLoader::ModelLoader() {}

bool ModelLoader::Load(const std::string& filename)
{
    filename_ = filename;
    data_.clear();

    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file) {
        return false;
    }

    const std::streamsize size = file.tellg();

    if (size <= 0) {
        return false;
    }

    data_.resize(static_cast<size_t>(size));

    file.seekg(0, std::ios::beg);

    if (!file.read(reinterpret_cast<char*>(data_.data()), size)) {
        data_.clear();
        return false;
    }

    return true;
}

bool ModelLoader::IsLoaded() const
{
    return !data_.empty();
}

const uint8_t* ModelLoader::Data() const
{
    return data_.empty() ? nullptr : data_.data();
}

size_t ModelLoader::Size() const
{
    return data_.size();
}

const tflite::Model* ModelLoader::GetModel() const
{
    if (data_.empty()) {
        return nullptr;
    }

    return tflite::GetModel(data_.data());
}

const std::string& ModelLoader::Filename() const
{
    return filename_;
}