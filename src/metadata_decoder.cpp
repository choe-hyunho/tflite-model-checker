#include "metadata_decoder.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>

#include "console_printer.h"
#include "embedded_file_reader.h"
#include "model_document.h"
#include "conversion_metadata_generated.h"
#include "metadata_schema_generated.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr size_t kPreviewBytes = 32;
constexpr unsigned int kMetadataSchemaMajor = 1;
constexpr unsigned int kMetadataSchemaMinor = 5;
constexpr unsigned int kMetadataSchemaPatch = 0;

std::string StringValue(const flatbuffers::String* value)
{
    return value != nullptr ? value->str() : "";
}

std::string BoolValue(bool value)
{
    return value ? "true" : "false";
}

std::string HexValue(uint64_t value)
{
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "0x%llX", static_cast<unsigned long long>(value));
    return buffer;
}

static std::string ByteVectorToHex(const flatbuffers::Vector<uint8_t>* data)
{
    if (data == nullptr || data->size() == 0)
        return "";
    constexpr size_t kPreviewBytes = 64;
    const size_t count = std::min(static_cast<size_t>(data->size()), kPreviewBytes);
    std::ostringstream out;
    for (size_t i = 0; i < count; ++i) {
        if (i != 0)
            out << " ";
        char byte[4];
        snprintf(byte, sizeof(byte), "%02X", data->Get(i));
        out << byte;
    }
    if (count < data->size())
        out << " ...";
    return out.str();
}

std::string VectorFloatToString(const flatbuffers::Vector<float>* values)
{
    if (values == nullptr || values->size() == 0)
        return "[]";
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values->size(); ++i) {
        if (i != 0)
            out << ", ";
        out << values->Get(i);
    }
    out << "]";
    return out.str();
}

std::string VectorStringToString(const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>* values)
{
    if (values == nullptr || values->size() == 0)
        return "[]";
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values->size(); ++i) {
        if (i != 0)
            out << ", ";
        const auto* value = values->Get(i);
        out << (value != nullptr ? value->str() : "");
    }
    out << "]";
    return out.str();
}

static std::string VectorInt32ToString(const flatbuffers::Vector<int32_t>* values)
{
    if (values == nullptr || values->size() == 0)
        return "[]";
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values->size(); ++i) {
        if (i != 0)
            out << ", ";
        out << values->Get(i);
    }
    out << "]";
    return out.str();
}

static std::string VectorUInt32ToString(const flatbuffers::Vector<uint32_t>* values)
{
    if (values == nullptr || values->size() == 0)
        return "[]";
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values->size(); ++i) {
        if (i != 0)
            out << ", ";
        out << values->Get(i);
    }
    out << "]";
    return out.str();
}

static std::string VectorInt64ToString(const flatbuffers::Vector<int64_t>* values)
{
    if (values == nullptr || values->size() == 0)
        return "[]";
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values->size(); ++i) {
        if (i != 0)
            out << ", ";
        out << values->Get(i);
    }
    out << "]";
    return out.str();
}

static bool GetSparseIndexVectorValues(tflite::SparseIndexVector type, const void* data, std::vector<int64_t>* values)
{
    if (values == nullptr)
        return false;
    values->clear();
    if (data == nullptr)
        return false;
    switch (type) {
    case tflite::SparseIndexVector_Int32Vector: {
        const auto* vector = static_cast<const tflite::Int32Vector*>(data);
        const auto* source = vector != nullptr ? vector->values() : nullptr;
        if (source == nullptr)
            return false;
        values->reserve(source->size());
        for (size_t i = 0; i < source->size(); ++i)
            values->push_back(static_cast<int64_t>(source->Get(i)));
        return true;
    }
    case tflite::SparseIndexVector_Uint16Vector: {
        const auto* vector = static_cast<const tflite::Uint16Vector*>(data);
        const auto* source = vector != nullptr ? vector->values() : nullptr;
        if (source == nullptr)
            return false;
        values->reserve(source->size());
        for (size_t i = 0; i < source->size(); ++i)
            values->push_back(static_cast<int64_t>(source->Get(i)));
        return true;
    }
    case tflite::SparseIndexVector_Uint8Vector: {
        const auto* vector = static_cast<const tflite::Uint8Vector*>(data);
        const auto* source = vector != nullptr ? vector->values() : nullptr;
        if (source == nullptr)
            return false;
        values->reserve(source->size());
        for (size_t i = 0; i < source->size(); ++i)
            values->push_back(static_cast<int64_t>(source->Get(i)));
        return true;
    }
    default:
        return false;
    }
}

static std::string SparseIndexVectorToString(tflite::SparseIndexVector type, const void* data)
{
    if (data == nullptr)
        return "[]";
    std::ostringstream out;
    out << "[";
    bool first = true;
    switch (type) {
    case tflite::SparseIndexVector_Int32Vector: {
        const auto* vector = static_cast<const tflite::Int32Vector*>(data);
        const auto* values = vector != nullptr ? vector->values() : nullptr;
        if (values != nullptr) {
            for (size_t i = 0; i < values->size(); ++i) {
                if (!first)
                    out << ", ";
                out << values->Get(i);
                first = false;
            }
        }
        break;
    }
    case tflite::SparseIndexVector_Uint16Vector: {
        const auto* vector = static_cast<const tflite::Uint16Vector*>(data);
        const auto* values = vector != nullptr ? vector->values() : nullptr;
        if (values != nullptr) {
            for (size_t i = 0; i < values->size(); ++i) {
                if (!first)
                    out << ", ";
                out << values->Get(i);
                first = false;
            }
        }
        break;
    }
    case tflite::SparseIndexVector_Uint8Vector: {
        const auto* vector = static_cast<const tflite::Uint8Vector*>(data);
        const auto* values = vector != nullptr ? vector->values() : nullptr;
        if (values != nullptr) {
            for (size_t i = 0; i < values->size(); ++i) {
                if (!first)
                    out << ", ";
                out << static_cast<unsigned int>(values->Get(i));
                first = false;
            }
        }
        break;
    }
    default:
        break;
    }
    out << "]";
    return out.str();
}

static std::string TensorShapeToString(const tflite::Tensor* tensor)
{
    if (tensor == nullptr || tensor->shape() == nullptr)
        return "[]";
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < tensor->shape()->size(); ++i) {
        if (i != 0)
            out << ", ";
        out << tensor->shape()->Get(i);
    }
    out << "]";
    return out.str();
}

static std::string ReadMetadataString(const uint8_t* data, size_t size)
{
    if (data == nullptr || size == 0)
        return "";
    size_t length = 0;
    while (length < size && data[length] != '\0')
        ++length;
    while (length > 0 && (data[length - 1] == '\r' || data[length - 1] == '\n'))
        --length;
    return std::string(reinterpret_cast<const char*>(data), length);
}

static bool ParseSemanticVersion(const char* version, unsigned int* major, unsigned int* minor, unsigned int* patch)
{
    if (version == nullptr || major == nullptr || minor == nullptr || patch == nullptr)
        return false;
    if (version[0] == '\0')
        return false;
    const char* p = version;
    unsigned int* values[] = {major, minor, patch};
    for (int component = 0; component < 3; ++component) {
        unsigned int value = 0;
        size_t digits = 0;
        while (*p >= '0' && *p <= '9') {
            value = value * 10U + static_cast<unsigned int>(*p - '0');
            ++digits;
            if (digits > 9)
                return false;
            ++p;
        }
        if (digits == 0)
            return false;
        *values[component] = value;
        if (component < 2) {
            if (*p != '.')
                return false;
            ++p;
        }
    }
    return *p == '\0';
}

static void PrintContent(ConsolePrinter& printer, const tflite::Content* content, const tflite::Tensor* model_tensor)
{
    if (content == nullptr)
        return;
    const auto properties_type = content->content_properties_type();
    if (properties_type != tflite::ContentProperties_NONE) {
        const char* properties_name = tflite::EnumNameContentProperties(properties_type);
        printer.SubSection("Content Properties");
        printer.Key("Type", properties_name != nullptr && properties_name[0] != '\0' ? properties_name : "UNKNOWN");
        switch (properties_type) {
        case tflite::ContentProperties_FeatureProperties:
            if (content->content_properties_as_FeatureProperties() == nullptr)
                printer.Key("Status", "Invalid FeatureProperties");
            break;
        case tflite::ContentProperties_ImageProperties: {
            const auto* properties = content->content_properties_as_ImageProperties();
            if (properties == nullptr) {
                printer.Key("Status", "Invalid ImageProperties");
                break;
            }
            const char* color_space = tflite::EnumNameColorSpaceType(properties->color_space());
            if (color_space != nullptr && color_space[0] != '\0') {
                printer.Key("Color Space", color_space);
            } else {
                printer.Key("Color Space", "UNKNOWN");
                printer.Key("Color Space Status", "Invalid enum value");
            }
            const auto* default_size = properties->default_size();
            if (default_size != nullptr) {
                printer.Key("Default Width", default_size->width());
                printer.Key("Default Height", default_size->height());
                if (default_size->width() == 0 || default_size->height() == 0)
                    printer.Key("Default Size Status", "Invalid image size");
            }
            break;
        }
        case tflite::ContentProperties_BoundingBoxProperties: {
            const auto* properties = content->content_properties_as_BoundingBoxProperties();
            if (properties == nullptr) {
                printer.Key("Status", "Invalid BoundingBoxProperties");
                break;
            }
            const char* type = tflite::EnumNameBoundingBoxType(properties->type());
            const char* coordinate_type = tflite::EnumNameCoordinateType(properties->coordinate_type());
            if (type != nullptr && type[0] != '\0') {
                printer.Key("Bounding Box Type", type);
            } else {
                printer.Key("Bounding Box Type", "UNKNOWN");
                printer.Key("Bounding Box Type Status", "Invalid enum value");
            }
            if (coordinate_type != nullptr && coordinate_type[0] != '\0')
                printer.Key("Coordinate Type", coordinate_type);
            else {
                printer.Key("Coordinate Type", "UNKNOWN");
                printer.Key("Coordinate Type Status", "Invalid enum value");
            }
            const auto* index = properties->index();
            if (index != nullptr && index->size() != 0) {
                printer.Key("Index", VectorUInt32ToString(index));
                if (index->size() != 4) {
                    printer.Key("Index Status", "Expected 4 indices");
                } else {
                    bool invalid = false;
                    bool seen[4] = {false, false, false, false};
                    for (size_t i = 0; i < index->size(); ++i) {
                        const uint32_t value = index->Get(i);
                        if (value >= 4 || seen[value]) {
                            invalid = true;
                            break;
                        }
                        seen[value] = true;
                    }
                    if (invalid)
                        printer.Key("Index Status", "Invalid bounding box index");
                }
            }
            break;
        }
        case tflite::ContentProperties_AudioProperties: {
            const auto* properties = content->content_properties_as_AudioProperties();
            if (properties == nullptr) {
                printer.Key("Status", "Invalid AudioProperties");
                break;
            }
            printer.Key("Sample Rate", properties->sample_rate());
            printer.Key("Channels", properties->channels());
            if (properties->sample_rate() == 0 || properties->channels() == 0)
                printer.Key("Status", "Invalid audio properties");
            break;
        }
        default:
            printer.Key("Status", "Unknown content properties type");
            break;
        }
    }
    const auto* range = content->range();
    if (range != nullptr) {
        printer.SubSection("Value Range");
        printer.Key("Minimum", range->min());
        printer.Key("Maximum", range->max());
        if (range->min() > range->max()) {
            printer.Key("Status", "Invalid range");
        } else if (model_tensor != nullptr && model_tensor->shape() != nullptr) {
            const int32_t dimension_count = static_cast<int32_t>(model_tensor->shape()->size());
            if (range->min() < 0 || range->max() < 0 ||
                range->min() >= dimension_count || range->max() >= dimension_count) {
                printer.Key("Status", "Range exceeds tensor dimensions");
            }
        }
    }
}

static void PrintStats(ConsolePrinter& printer, const tflite::Stats* stats)
{
    if (stats == nullptr)
        return;
    printer.SubSection("Statistics");
    const auto* minimum = stats->min();
    const auto* maximum = stats->max();
    if (minimum != nullptr && minimum->size() != 0)
        printer.Key("Minimum", VectorFloatToString(minimum));
    if (maximum != nullptr && maximum->size() != 0)
        printer.Key("Maximum", VectorFloatToString(maximum));
    if (minimum != nullptr && maximum != nullptr) {
        if (minimum->size() != maximum->size()) {
            char status[128];
            snprintf(status, sizeof(status), "Minimum/maximum count mismatch (%zu vs %zu)",
                     static_cast<size_t>(minimum->size()),
                     static_cast<size_t>(maximum->size()));
            printer.Key("Status", status);
        } else {
            bool invalid = false;
            for (size_t i = 0; i < minimum->size(); ++i) {
                if (!std::isfinite(minimum->Get(i)) ||
                    !std::isfinite(maximum->Get(i)) ||
                    minimum->Get(i) > maximum->Get(i)) {
                    invalid = true;
                    break;
                }
            }
            if (invalid)
                printer.Key("Status", "Invalid statistics range");
        }
    }
}

static const char* AssociatedFileTypeDescription(tflite::AssociatedFileType type)
{
    switch (type) {
    case tflite::AssociatedFileType_DESCRIPTIONS:
        return "Descriptions";
    case tflite::AssociatedFileType_TENSOR_AXIS_LABELS:
        return "Tensor axis labels";
    case tflite::AssociatedFileType_TENSOR_VALUE_LABELS:
        return "Tensor value labels";
    case tflite::AssociatedFileType_TENSOR_AXIS_SCORE_CALIBRATION:
        return "Tensor axis score calibration";
    case tflite::AssociatedFileType_VOCABULARY:
        return "Vocabulary";
    case tflite::AssociatedFileType_SCANN_INDEX_FILE:
        return "ScaNN index";
    default:
        return "Unknown";
    }
}

static void PrintLabels(ConsolePrinter& printer, const std::vector<uint8_t>& data)
{
    if (data.empty())
        return;
    printer.SubSection("Contents");
    const char* begin = reinterpret_cast<const char*>(data.data());
    const char* end = begin + data.size();
    const char* line = begin;
    size_t index = 0;
    while (line < end) {
        const char* next = static_cast<const char*>(memchr(line, '\n', static_cast<size_t>(end - line)));
        const char* line_end = next != nullptr ? next : end;
        while (line_end > line && (line_end[-1] == '\r' || line_end[-1] == '\n'))
            --line_end;
        if (line_end > line) {
            if (memchr(line, '\0', static_cast<size_t>(line_end - line)) != nullptr) {
                printer.Key("Status", "Label contains embedded NUL byte");
                ++index;
            } else {
                std::string label(line, line_end);
                char key[32];
                snprintf(key, sizeof(key), "Label %zu", index++);
                printer.Key(key, label);
            }
        }
        if (next == nullptr)
            break;
        line = next + 1;
    }
    if (index == 0)
        printer.Key("Status", "No labels found");
}

static void PrintTextFile(ConsolePrinter& printer, const std::vector<uint8_t>& data)
{
    if (data.empty())
        return;
    printer.SubSection("Contents");
    std::string text(reinterpret_cast<const char*>(data.data()), data.size());
    while (!text.empty() && (text.back() == '\0' || text.back() == '\r' || text.back() == '\n'))
        text.pop_back();
    if (text.find('\0') != std::string::npos) {
        printer.Key("Status", "Text contains embedded NUL byte");
        return;
    }
    if (!text.empty())
        printer.Key("Text", text);
    else
        printer.Key("Status", "No text found");
}

static void PrintFileData(ConsolePrinter& printer, const std::vector<uint8_t>& data)
{
    if (data.empty())
        return;
    constexpr size_t kPreviewBytes = 64;
    const size_t count = std::min(data.size(), kPreviewBytes);
    std::ostringstream out;
    for (size_t i = 0; i < count; ++i) {
        if (i != 0)
            out << " ";
        char byte[4];
        snprintf(byte, sizeof(byte), "%02X", data[i]);
        out << byte;
    }
    if (count < data.size())
        out << " ...";
    printer.SubSection("Contents");
    printer.Key("Data", out.str());
}

static void PrintScoreCalibration(ConsolePrinter& printer, const std::vector<uint8_t>& data)
{
    if (data.empty())
        return;
    printer.SubSection("Contents");
    const char* begin = reinterpret_cast<const char*>(data.data());
    const char* end = begin + data.size();
    size_t line_number = 1;
    while (begin < end) {
        const char* line_end = begin;
        while (line_end < end && *line_end != '\n' && *line_end != '\r')
            ++line_end;
        if (line_end != begin) {
            std::string line(begin, line_end);
            if (line.find('\0') != std::string::npos) {
                char status[128];
                snprintf(status, sizeof(status), "Line %zu: Embedded NUL byte",
                         line_number);
                printer.Key("Status", status);
            } else {
                std::vector<const char*> fields;
                fields.push_back(line.c_str());
                for (char* p = line.data(); *p != '\0'; ++p) {
                    if (*p == ',') {
                        *p = '\0';
                        fields.push_back(p + 1);
                    }
                }
                if (fields.size() != 3 && fields.size() != 4) {
                    char status[128];
                    snprintf(status, sizeof(status), "Line %zu: Expected 3 or 4 fields, found %zu",
                             line_number, fields.size());
                    printer.Key("Status", status);
                } else {
                    bool invalid = false;
                    for (size_t i = 0; i < fields.size(); ++i) {
                        char* parse_end = nullptr;
                        errno = 0;
                        const float value = strtof(fields[i], &parse_end);
                        if (parse_end == fields[i] || *parse_end != '\0' || errno == ERANGE || !std::isfinite(value)) {
                            invalid = true;
                            break;
                        }
                        if (i == 0 && value < 0.0f) {
                            invalid = true;
                            break;
                        }
                    }
                    if (invalid) {
                        char status[128];
                        snprintf(status, sizeof(status), "Line %zu: Invalid score calibration values",
                                 line_number);
                        printer.Key("Status", status);
                    }
                }
            }
        }
        while (line_end < end && (*line_end == '\n' || *line_end == '\r'))
            ++line_end;
        begin = line_end;
        ++line_number;
    }
}

static void PrintAssociatedFiles(ConsolePrinter& printer, const flatbuffers::Vector<flatbuffers::Offset<tflite::AssociatedFile>>* files, const uint8_t* model_data, size_t model_size)
{
    if (files == nullptr || files->size() == 0)
        return;
    printer.Key("Associated Files", static_cast<size_t>(files->size()));
    for (size_t i = 0; i < files->size(); ++i) {
        const auto* file = files->Get(i);
        if (file == nullptr) {
            printer.Key("Status", "Invalid");
            continue;
        }
        char title[32];
        snprintf(title, sizeof(title), "Associated File %zu", i);
        printer.SubSection(title);
        if (file->name() == nullptr || file->name()->size() == 0) {
            printer.Key("Status", "Missing name");
        } else if (memchr(file->name()->data(), '\0', file->name()->size()) != nullptr) {
            printer.Key("Status", "Name contains embedded NUL byte");
            continue;
        } else {
            printer.Key("Name", file->name()->c_str());
        }
        if (file->description() != nullptr) {
            if (memchr(file->description()->data(), '\0', file->description()->size()) != nullptr)
                printer.Key("Description Status", "Contains embedded NUL byte");
            else
                printer.Key("Description", file->description()->c_str());
        }
        const char* type_name = tflite::EnumNameAssociatedFileType(file->type());
        printer.Key("Type", AssociatedFileTypeDescription(file->type()));
        if (type_name != nullptr && type_name[0] != '\0') {
            printer.Key("Type ID", type_name);
        } else {
            printer.Key("Type Status", "Unknown associated file type");
        }
        if (file->locale() != nullptr) {
            if (memchr(file->locale()->data(), '\0', file->locale()->size()) != nullptr)
                printer.Key("Locale Status", "Contains embedded NUL byte");
            else
                printer.Key("Locale", file->locale()->c_str());
        }
        if (file->version() != nullptr) {
            if (memchr(file->version()->data(), '\0', file->version()->size()) != nullptr)
                printer.Key("Version Status", "Contains embedded NUL byte");
            else
                printer.Key("Version", file->version()->c_str());
        }
        if (file->name() == nullptr || file->name()->size() == 0 || model_data == nullptr || model_size == 0)
            continue;
        EmbeddedFileReader reader(model_data, model_size);
        std::vector<uint8_t> file_data;
        const EmbeddedFileReader::Status status = reader.Read(file->name()->str(), &file_data);
        if (status != EmbeddedFileReader::Status::OK) {
            printer.Key("File", EmbeddedFileReader::StatusDescription(status));
            continue;
        }
        printer.KeyBytes("File Size", file_data.size());
        if (file_data.empty())
            printer.Key("File Status", "Empty embedded file");
        switch (file->type()) {
        case tflite::AssociatedFileType_TENSOR_AXIS_LABELS:
        case tflite::AssociatedFileType_TENSOR_VALUE_LABELS:
            PrintLabels(printer, file_data);
            break;
        case tflite::AssociatedFileType_TENSOR_AXIS_SCORE_CALIBRATION:
            PrintScoreCalibration(printer, file_data);
            break;
        case tflite::AssociatedFileType_DESCRIPTIONS:
        case tflite::AssociatedFileType_VOCABULARY:
            PrintTextFile(printer, file_data);
            break;
        default:
            PrintFileData(printer, file_data);
            break;
        }
    }
}

static void PrintProcessUnits(ConsolePrinter& printer, const char* name, const flatbuffers::Vector<flatbuffers::Offset<tflite::ProcessUnit>>* units, const uint8_t* model_data, size_t model_size)
{
    if (units == nullptr || units->size() == 0)
        return;
    printer.Key(name, static_cast<size_t>(units->size()));
    for (size_t i = 0; i < units->size(); ++i) {
        const auto* unit = units->Get(i);
        if (unit == nullptr)
            continue;
        char title[64];
        snprintf(title, sizeof(title), "%s %zu", name, i);
        printer.SubSection(title);
        switch (unit->options_type()) {
        case tflite::ProcessUnitOptions_NONE:
            printer.Key("Type", "None");
            printer.Key("Status", "No options");
            break;
        case tflite::ProcessUnitOptions_NormalizationOptions: {
            const auto* options = unit->options_as_NormalizationOptions();
            printer.Key("Type", "Normalization");
            if (options == nullptr) {
                printer.Key("Status", "Invalid options");
                break;
            }
            const auto* mean = options->mean();
            const auto* std = options->std();
            printer.Key("Mean", VectorFloatToString(mean));
            printer.Key("Std", VectorFloatToString(std));
            printer.Key("Formula", "(x - mean) / std");
            const size_t mean_count = mean != nullptr ? static_cast<size_t>(mean->size()) : 0;
            const size_t std_count = std != nullptr ? static_cast<size_t>(std->size()) : 0;
            if (mean_count == 0 || std_count == 0) {
                printer.Key("Status", "Missing normalization values");
            } else if (mean_count != 1 && std_count != 1 && mean_count != std_count) {
                char status[128];
                snprintf(status, sizeof(status), "Mean/std count mismatch (%zu vs %zu)", mean_count, std_count);
                printer.Key("Status", status);
            } else {
                bool invalid = false;
                for (size_t j = 0; j < mean_count; ++j) {
                    if (!std::isfinite(mean->Get(j))) {
                        invalid = true;
                        break;
                    }
                }
                if (!invalid) {
                    for (size_t j = 0; j < std_count; ++j) {
                        if (!std::isfinite(std->Get(j)) || std->Get(j) <= 0.0f) {
                            invalid = true;
                            break;
                        }
                    }
                }
                if (invalid)
                    printer.Key("Status", "Invalid normalization values");
            }
            break;
        }
        case tflite::ProcessUnitOptions_ScoreCalibrationOptions: {
            const auto* options = unit->options_as_ScoreCalibrationOptions();
            printer.Key("Type", "Score Calibration");
            if (options == nullptr) {
                printer.Key("Status", "Invalid options");
                break;
            }
            const char* transformation = tflite::EnumNameScoreTransformationType(options->score_transformation());
            if (transformation != nullptr && transformation[0] != '\0') {
                printer.Key("Transformation", transformation);
            } else {
                printer.Key("Transformation", "UNKNOWN");
                printer.Key("Transformation Status", "Invalid enum value");
            }
            printer.Key("Default Score", options->default_score());
            if (!std::isfinite(options->default_score()))
                printer.Key("Status", "Invalid default score");
            break;
        }
        case tflite::ProcessUnitOptions_ScoreThresholdingOptions: {
            const auto* options = unit->options_as_ScoreThresholdingOptions();
            printer.Key("Type", "Score Thresholding");
            if (options == nullptr) {
                printer.Key("Status", "Invalid options");
                break;
            }
            printer.Key("Global Score Threshold", options->global_score_threshold());
            if (!std::isfinite(options->global_score_threshold()))
                printer.Key("Status", "Invalid score threshold");
            break;
        }
        case tflite::ProcessUnitOptions_BertTokenizerOptions: {
            const auto* options = unit->options_as_BertTokenizerOptions();
            printer.Key("Type", "BERT Tokenizer");
            if (options == nullptr) {
                printer.Key("Status", "Invalid options");
                break;
            }
            const auto* vocab_file = options->vocab_file();
            if (vocab_file == nullptr || vocab_file->size() == 0)
                printer.Key("Status", "Missing vocabulary file");
            PrintAssociatedFiles(printer, vocab_file, model_data, model_size);
            break;
        }
        case tflite::ProcessUnitOptions_SentencePieceTokenizerOptions: {
            const auto* options = unit->options_as_SentencePieceTokenizerOptions();
            printer.Key("Type", "SentencePiece Tokenizer");
            if (options == nullptr) {
                printer.Key("Status", "Invalid options");
                break;
            }
            const auto* sentencePiece_model = options->sentencePiece_model();
            const auto* vocab_file = options->vocab_file();
            if ((sentencePiece_model == nullptr || sentencePiece_model->size() == 0) && (vocab_file == nullptr || vocab_file->size() == 0))
                printer.Key("Status", "Missing tokenizer files");
            PrintAssociatedFiles(printer, sentencePiece_model, model_data, model_size);
            PrintAssociatedFiles(printer, vocab_file, model_data, model_size);
            break;
        }
        case tflite::ProcessUnitOptions_RegexTokenizerOptions: {
            const auto* options = unit->options_as_RegexTokenizerOptions();
            printer.Key("Type", "Regex Tokenizer");
            if (options == nullptr) {
                printer.Key("Status", "Invalid options");
                break;
            }
            const auto* pattern = options->delim_regex_pattern();
            if (pattern == nullptr || pattern->size() == 0) {
                printer.Key("Status", "Missing delimiter regex");
            } else {
                printer.Key("Delimiter Regex", pattern->c_str());
            }
            PrintAssociatedFiles(printer, options->vocab_file(), model_data, model_size);
            break;
        }
        default:
            printer.Key("Type", "Unknown");
            printer.Key("Status", "Unknown options type");
            break;
        }
    }
}

static bool ResolveModelTensor(const tflite::SubGraph* model_subgraph, bool input, size_t index, const tflite::Tensor** tensor, uint32_t* tensor_index)
{
    if (tensor == nullptr || tensor_index == nullptr)
        return false;
    *tensor = nullptr;
    *tensor_index = 0;
    if (model_subgraph == nullptr)
        return false;
    const auto* indices = input ? model_subgraph->inputs() : model_subgraph->outputs();
    if (indices == nullptr || index >= indices->size())
        return false;
    const int32_t index_value = indices->Get(index);
    if (index_value < 0)
        return false;
    *tensor_index = static_cast<uint32_t>(index_value);
    if (model_subgraph->tensors() == nullptr ||
        static_cast<size_t>(*tensor_index) >= model_subgraph->tensors()->size())
        return false;
    *tensor = model_subgraph->tensors()->Get(*tensor_index);
    return *tensor != nullptr;
}

static bool TensorNamesMatch(const tflite::TensorMetadata* metadata, const tflite::Tensor* model_tensor)
{
    if (metadata == nullptr || model_tensor == nullptr)
        return false;
    if (metadata->name() == nullptr || model_tensor->name() == nullptr)
        return true;
    return strcmp(metadata->name()->c_str(), model_tensor->name()->c_str()) == 0;
}

static bool FindModelTensorByName(const tflite::SubGraph* model_subgraph, const char* name)
{
    if (model_subgraph == nullptr || name == nullptr || model_subgraph->tensors() == nullptr)
        return false;
    const auto* tensors = model_subgraph->tensors();
    for (size_t i = 0; i < tensors->size(); ++i) {
        const auto* tensor = tensors->Get(i);
        if (tensor != nullptr && tensor->name() != nullptr &&
            strcmp(tensor->name()->c_str(), name) == 0)
            return true;
    }
    return false;
}

static void PrintTensorGroups(ConsolePrinter& printer, const char* title, const flatbuffers::Vector<flatbuffers::Offset<tflite::TensorGroup>>* groups, const tflite::SubGraph* model_subgraph)
{
    if (groups == nullptr || groups->size() == 0)
        return;
    printer.SubSection(title);

    for (size_t i = 0; i < groups->size(); ++i) {
        const auto* group = groups->Get(i);
        if (group == nullptr) {
            printer.Key("Group", "Invalid");
            continue;
        }
        printer.Key("Group", i);
        if (group->name() == nullptr || group->name()->size() == 0)
            printer.Key("Status", "Missing name");
        else
            printer.Key("Name", group->name()->c_str());
        const auto* tensor_names = group->tensor_names();
        if (tensor_names == nullptr || tensor_names->size() == 0) {
            printer.Key("Tensor Count", static_cast<size_t>(0));
            printer.Key("Tensor Status", "No tensors");
            continue;
        }
        printer.Key("Tensor Count", static_cast<size_t>(tensor_names->size()));
        for (size_t j = 0; j < tensor_names->size(); ++j) {
            const auto* name = tensor_names->Get(j);
            printer.Key("Tensor Index", j);
            if (name == nullptr) {
                printer.Key("Tensor Status", "Missing tensor name");
                continue;
            }
            if (name->size() == 0) {
                printer.Key("Tensor Status", "Empty tensor name");
                continue;
            }
            bool duplicate = false;
            for (size_t k = 0; k < j; ++k) {
                const auto* previous_name = tensor_names->Get(k);
                if (previous_name != nullptr && previous_name->size() > 0 &&
                    previous_name->str() == name->str()) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate)
                printer.Key("Tensor Status", "Duplicate tensor name");
            printer.Key("Tensor Name", name->c_str());
            if (!FindModelTensorByName(model_subgraph, name->c_str()))
                printer.Key("Tensor Status", "Not found in model");
        }
    }
}

static void PrintTensorMetadataSummary(ConsolePrinter& printer, const char* name, size_t metadata_count, size_t model_count)
{
    printer.Key(name, metadata_count);
    if (metadata_count == 0 && model_count != 0) {
        printer.Key("Status", "Metadata not provided");
    } else if (metadata_count != model_count) {
        char status[128];
        snprintf(status, sizeof(status), "Metadata/model count mismatch (%zu vs %zu)", metadata_count, model_count);
        printer.Key("Status", status);
    }
}

static std::string FileIdentifier(const uint8_t* data, size_t size)
{
    if (data == nullptr || size < 8)
        return "";
    const uint8_t* identifier = data + 4;
    for (size_t i = 0; i < 4; ++i) {
        if (identifier[i] < 0x20 || identifier[i] > 0x7E)
            return "";
    }
    return std::string(reinterpret_cast<const char*>(identifier), 4);
}

static void PrintQuantizationTensor(ConsolePrinter& printer, const char* name, int32_t tensor_index, const tflite::SubGraph* model_subgraph, const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers)
{
    printer.SubSection(name);
    printer.Key("Tensor Index", static_cast<int>(tensor_index));
    if (tensor_index < 0) {
        printer.Key("Status", "Invalid tensor index");
        return;
    }
    if (model_subgraph == nullptr || model_subgraph->tensors() == nullptr) {
        printer.Key("Status", "No model subgraph");
        return;
    }
    if (static_cast<size_t>(tensor_index) >= model_subgraph->tensors()->size()) {
        printer.Key("Status", "Invalid tensor index");
        return;
    }
    const auto* tensor = model_subgraph->tensors()->Get(static_cast<size_t>(tensor_index));
    if (tensor == nullptr) {
        printer.Key("Status", "Null tensor");
        return;
    }
    if (tensor->name() != nullptr)
        printer.Key("Tensor Name", tensor->name()->c_str());
    const char* type = tflite::EnumNameTensorType(tensor->type());
    printer.Key("Tensor Type", type != nullptr && type[0] != '\0' ? type : "UNKNOWN");
    printer.Key("Tensor Shape", TensorShapeToString(tensor));
    const uint32_t buffer_index = tensor->buffer();
    printer.Key("Buffer Index", static_cast<size_t>(buffer_index));
    if (buffers == nullptr || static_cast<size_t>(buffer_index) >= buffers->size()) {
        printer.Key("Buffer Status", "Invalid buffer index");
        return;
    }
    const auto* buffer = buffers->Get(buffer_index);
    if (buffer == nullptr) {
        printer.Key("Buffer Status", "Null buffer");
        return;
    }
    printer.KeyBytes("Buffer Data Size", buffer->data() != nullptr ? static_cast<size_t>(buffer->data()->size()) : 0);
    printer.Key("Buffer Offset", buffer->offset());
    printer.Key("Buffer Size", buffer->size());
}

static void PrintQuantizedDimensions(ConsolePrinter& printer, const tflite::Tensor* tensor, const flatbuffers::Vector<int32_t>* dimensions)
{
    if (dimensions == nullptr) {
        printer.Key("Quantized Dimensions Status", "Missing quantized dimensions");
        return;
    }
    if (dimensions->size() == 0) {
        printer.Key("Quantized Dimensions", static_cast<size_t>(0));
        printer.Key("Quantized Dimensions Status", "No quantized dimensions");
        return;
    }
    printer.Key("Quantized Dimensions", VectorInt32ToString(dimensions));
    bool duplicate = false;
    for (size_t i = 0; i < dimensions->size() && !duplicate; ++i) {
        for (size_t j = i + 1; j < dimensions->size(); ++j) {
            if (dimensions->Get(i) == dimensions->Get(j)) {
                duplicate = true;
                break;
            }
        }
    }
    if (duplicate)
        printer.Key("Quantized Dimensions Status", "Duplicate dimension");
    if (tensor == nullptr || tensor->shape() == nullptr)
        return;
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < dimensions->size(); ++i) {
        if (i != 0)
            out << ", ";
        const int32_t dimension = dimensions->Get(i);
        if (dimension < 0 ||
            static_cast<size_t>(dimension) >= static_cast<size_t>(tensor->shape()->size())) {
            out << "?";
            continue;
        }
        out << tensor->shape()->Get(static_cast<size_t>(dimension));
    }
    out << "]";
    printer.Key("Quantized Dimension Sizes", out.str());
}

static void PrintTensorQuantization(ConsolePrinter& printer, const tflite::Tensor* tensor, const tflite::SubGraph* model_subgraph, const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers)
{
    if (tensor == nullptr || tensor->quantization() == nullptr)
        return;
    const auto* quantization = tensor->quantization();
    const auto details_type = quantization->details_type();
    printer.SubSection("Quantization");
    if (details_type == tflite::QuantizationDetails_NONE) {
        const auto* minimum = quantization->min();
        const auto* maximum = quantization->max();
        const auto* scale = quantization->scale();
        const auto* zero_point = quantization->zero_point();
        const size_t minimum_count = minimum != nullptr ? static_cast<size_t>(minimum->size()) : 0;
        const size_t maximum_count = maximum != nullptr ? static_cast<size_t>(maximum->size()) : 0;
        const size_t scale_count = scale != nullptr ? static_cast<size_t>(scale->size()) : 0;
        const size_t zero_point_count = zero_point != nullptr ? static_cast<size_t>(zero_point->size()) : 0;
        if (scale_count != 0 && zero_point_count == 0)
            printer.Key("Quantization Status", "Scale present without zero point");
        if (minimum_count != 0)
            printer.Key("Minimum", VectorFloatToString(minimum));
        if (maximum_count != 0)
            printer.Key("Maximum", VectorFloatToString(maximum));
        if (scale_count != 0)
            printer.Key("Scale", VectorFloatToString(scale));
        if (zero_point_count != 0)
            printer.Key("Zero Point", VectorInt64ToString(zero_point));
        if (minimum_count != 0 && maximum_count != 0) {
            if (minimum_count != maximum_count) {
                printer.Key("Range Status", "Minimum/maximum count mismatch");
            } else {
                bool invalid = false;
                for (size_t i = 0; i < minimum_count; ++i) {
                    if (!std::isfinite(minimum->Get(i)) || !std::isfinite(maximum->Get(i)) || minimum->Get(i) > maximum->Get(i)) {
                        invalid = true;
                        break;
                    }
                }
                if (invalid)
                    printer.Key("Range Status", "Invalid quantization range");
            }
        }
        if (scale_count > 1 && minimum_count != 0 && maximum_count != 0 && (minimum_count != scale_count || maximum_count != scale_count)) {
            char status[128];
            snprintf(status, sizeof(status), "Range count mismatch (%zu/%zu vs %zu)", minimum_count, maximum_count, scale_count);
            printer.Key("Quantization Status", status);
        }
        if (scale_count != 0) {
            printer.Key("Quantization Mode", scale_count == 1 ? "Per-tensor" : "Per-axis");
            const int32_t dimension = quantization->quantized_dimension();
            printer.Key("Quantized Dimension", static_cast<int>(dimension));
            if (tensor->shape() != nullptr && dimension >= 0 &&
                static_cast<size_t>(dimension) < tensor->shape()->size()) {
                const int32_t dimension_size = tensor->shape()->Get(static_cast<size_t>(dimension));
                printer.Key("Quantized Dimension Size", dimension_size);
                if (scale_count > 1 && dimension_size >= 0 && scale_count != static_cast<size_t>(dimension_size)) {
                    char status[128];
                    snprintf(status, sizeof(status), "Scale count mismatch (%zu vs %d)", scale_count, dimension_size);
                    printer.Key("Quantization Status", status);
                }
            } else {
                printer.Key("Quantized Dimension Status", "Outside tensor rank");
            }
        }
        if (scale_count != 0 && zero_point_count != 0 && scale_count != 1 && zero_point_count != 1 && scale_count != zero_point_count) {
            printer.Key("Quantization Status", "Scale/zero point count mismatch");
        }
        if (scale_count != 0) {
            bool invalid = false;
            for (size_t i = 0; i < scale_count; ++i) {
                if (!std::isfinite(scale->Get(i)) || scale->Get(i) <= 0.0f) {
                    invalid = true;
                    break;
                }
            }
            if (invalid)
                printer.Key("Quantization Status", "Invalid scale");
        }
        return;
    }
    const char* details_name = tflite::EnumNameQuantizationDetails(details_type);
    printer.Key("Details Type", details_name != nullptr && details_name[0] != '\0' ? details_name : "UNKNOWN");
    switch (details_type) {
    case tflite::QuantizationDetails_CustomQuantization: {
        const auto* details = quantization->details_as_CustomQuantization();
        if (details == nullptr) {
            printer.Key("Details Status", "Invalid CustomQuantization");
            break;
        }
        const auto* custom = details->custom();
        if (custom == nullptr) {
            printer.KeyBytes("Custom Data Size", 0);
            break;
        }
        const size_t custom_size = static_cast<size_t>(custom->size());
        printer.KeyBytes("Custom Data Size", custom_size);
        const size_t count = std::min(custom_size, kPreviewBytes);
        std::ostringstream out;
        for (size_t i = 0; i < count; ++i) {
            if (i != 0)
                out << " ";
            char byte[4];
            snprintf(byte, sizeof(byte), "%02X", custom->Get(i));
            out << byte;
        }
        if (count < custom_size)
            out << " ...";
        printer.Key("Custom Data", out.str());
        break;
    }
    case tflite::QuantizationDetails_BlockwiseQuantization: {
        const auto* details = quantization->details_as_BlockwiseQuantization();
        if (details == nullptr) {
            printer.Key("Details Status", "Invalid BlockwiseQuantization");
            break;
        }
        if (details->scales() < 0)
            printer.Key("Scales Tensor Status", "Invalid tensor index");
        else
            PrintQuantizationTensor(printer, "Scales Tensor", details->scales(), model_subgraph, buffers);
        if (details->zero_points() == -1) {
            printer.Key("Zero Points Tensor", "-1 (implicit zero)");
        } else if (details->zero_points() < -1) {
            printer.Key("Zero Points Tensor Status", "Invalid tensor index");
        } else {
            PrintQuantizationTensor(printer, "Zero Points Tensor", details->zero_points(), model_subgraph, buffers);
        }
        const int32_t block_size = details->block_size();
        printer.Key("Block Size", static_cast<int>(block_size));
        if (block_size < 0) {
            printer.Key("Block Size Status", "Invalid block size");
        } else if (block_size == 0) {
            printer.Key("Quantization Mode", "Per-channel");
        } else {
            printer.Key("Quantization Mode", "Blockwise");
        }
        break;
    }
    case tflite::QuantizationDetails_MultiAxisQuantization: {
        const auto* details = quantization->details_as_MultiAxisQuantization();
        if (details == nullptr) {
            printer.Key("Details Status", "Invalid MultiAxisQuantization");
            break;
        }
        if (details->scales() < 0)
            printer.Key("Scales Tensor Status", "Invalid tensor index");
        else
            PrintQuantizationTensor(printer, "Scales Tensor", details->scales(), model_subgraph, buffers);
        if (details->zero_points() == -1) {
            printer.Key("Zero Points Tensor", "-1 (implicit zero)");
        } else if (details->zero_points() < -1) {
            printer.Key("Zero Points Tensor Status", "Invalid tensor index");
        } else {
            PrintQuantizationTensor(printer, "Zero Points Tensor", details->zero_points(), model_subgraph, buffers);
        }
        const int32_t block_size = details->block_size();
        printer.Key("Block Size", static_cast<int>(block_size));
        if (block_size < 0) {
            printer.Key("Block Size Status", "Invalid block size");
        } else if (block_size == 0) {
            printer.Key("Quantization Mode", "Per-channel");
        } else {
            printer.Key("Quantization Mode", "Blockwise");
        }
        const auto* dimensions = details->quantized_dimensions();
        PrintQuantizedDimensions(printer, tensor, dimensions);
        break;
    }
    default:
        printer.Key("Details Status", "Unknown quantization details type");
        break;
    }
}

static void PrintTensorSparsity(ConsolePrinter& printer, const tflite::Tensor* tensor)
{
    if (tensor == nullptr || tensor->sparsity() == nullptr)
        return;
    const auto* sparsity = tensor->sparsity();
    printer.SubSection("Sparsity");
    const auto* traversal_order = sparsity->traversal_order();
    if (traversal_order == nullptr || traversal_order->size() == 0) {
        printer.Key("Traversal Order", static_cast<size_t>(0));
        printer.Key("Traversal Order Status", "Missing traversal order");
    } else {
        printer.Key("Traversal Order", VectorInt32ToString(traversal_order));
    }
    if (traversal_order != nullptr && traversal_order->size() != 0 && tensor != nullptr && tensor->shape() != nullptr) {
        const size_t rank = static_cast<size_t>(tensor->shape()->size());
        if (traversal_order->size() < rank) {
            printer.Key("Traversal Order Status", "Insufficient dimensions");
        } else {
            bool invalid = false;
            std::vector<bool> seen(rank, false);
            for (size_t i = 0; i < rank; ++i) {
                const int32_t dimension = traversal_order->Get(i);
                if (dimension < 0 || static_cast<size_t>(dimension) >= rank || seen[static_cast<size_t>(dimension)]) {
                    invalid = true;
                    break;
                }
                seen[static_cast<size_t>(dimension)] = true;
            }
            if (invalid)
                printer.Key("Traversal Order Status", "Invalid dimension permutation");
        }
    }
    const auto* block_map = sparsity->block_map();
    const auto* dimensions = sparsity->dim_metadata();
    if (traversal_order != nullptr && dimensions != nullptr && traversal_order->size() != dimensions->size()) {
        char status[128];
        snprintf(status, sizeof(status), "Traversal/dimension metadata count mismatch (%zu vs %zu)", static_cast<size_t>(traversal_order->size()), static_cast<size_t>(dimensions->size()));
        printer.Key("Traversal Order Status", status);
    }
    if (block_map != nullptr && block_map->size() != 0)
        printer.Key("Block Map", VectorInt32ToString(block_map));
    if (block_map != nullptr && block_map->size() != 0 && tensor->shape() != nullptr && traversal_order != nullptr && traversal_order->size() == tensor->shape()->size()) {
        printer.Key("Block Map Status", "Unexpected block map");
    }
    if (block_map != nullptr && block_map->size() != 0 && tensor->shape() != nullptr && traversal_order != nullptr && traversal_order->size() > tensor->shape()->size()) {
        const size_t rank = static_cast<size_t>(tensor->shape()->size());
        const size_t block_dimension_count = traversal_order->size() - rank;
        if (block_map->size() != block_dimension_count) {
            char status[128];
            snprintf(status, sizeof(status), "Block map count mismatch (%zu vs %zu)", static_cast<size_t>(block_map->size()), block_dimension_count);
            printer.Key("Block Map Status", status);
        }
    }
    if (block_map != nullptr && block_map->size() != 0 &&
        tensor->shape() != nullptr) {
        const size_t rank = static_cast<size_t>(tensor->shape()->size());
        bool invalid = false;
        for (size_t i = 0; i < block_map->size(); ++i) {
            const int32_t dimension = block_map->Get(i);
            if (dimension < 0 || static_cast<size_t>(dimension) >= rank) {
                invalid = true;
                break;
            }
        }
        if (invalid)
            printer.Key("Block Map Status", "Invalid original dimension");
    }
    if (dimensions == nullptr || dimensions->size() == 0)
        return;
    printer.Key("Dimensions", static_cast<size_t>(dimensions->size()));
    for (size_t i = 0; i < dimensions->size(); ++i) {
        const auto* dimension = dimensions->Get(i);
        char title[32];
        snprintf(title, sizeof(title), "Dimension %zu", i);
        printer.SubSection(title);
        if (dimension == nullptr) {
            printer.Key("Status", "Invalid dimension metadata");
            continue;
        }
        const char* format = tflite::EnumNameDimensionType(dimension->format());
        if (format != nullptr && format[0] != '\0') {
            printer.Key("Format", format);
        } else {
            printer.Key("Format", "UNKNOWN");
            printer.Key("Format Status", "Invalid enum value");
        }
        const int32_t dense_size = dimension->dense_size();
        printer.Key("Dense Size", dense_size);
        if (dimension->format() == tflite::DimensionType_DENSE) {
            if (dense_size < 0) {
                printer.Key("Dense Size Status", "Invalid dense size");
            } else if (traversal_order != nullptr && i < traversal_order->size() && tensor->shape() != nullptr) {
                const int32_t tensor_dimension = traversal_order->Get(i);
                const size_t rank = static_cast<size_t>(tensor->shape()->size());
                if (tensor_dimension >= 0 && static_cast<size_t>(tensor_dimension) < rank) {
                    const int32_t actual_size = tensor->shape()->Get(static_cast<size_t>(tensor_dimension));
                    if (actual_size >= 0 && dense_size != actual_size) {
                        char status[128];
                        snprintf(status, sizeof(status), "Dense size mismatch (%d vs %d)", dense_size, actual_size);
                        printer.Key("Dense Size Status", status);
                    }
                }
            }
            if (dimension->array_segments() != nullptr || dimension->array_indices() != nullptr)
                printer.Key("Sparse Array Status", "Unexpected CSR arrays");
            continue;
        }
        if (dimension->format() != tflite::DimensionType_SPARSE_CSR)
            continue;
        const char* segments_type = tflite::EnumNameSparseIndexVector(dimension->array_segments_type());
        if (segments_type != nullptr && segments_type[0] != '\0') {
            printer.Key("Array Segments Type", segments_type);
        } else if (dimension->array_segments() != nullptr) {
            printer.Key("Array Segments Type", "UNKNOWN");
            printer.Key("Array Segments Type Status", "Invalid enum value");
        }
        if (dimension->array_segments() != nullptr) {
            printer.Key("Array Segments", SparseIndexVectorToString(dimension->array_segments_type(), dimension->array_segments()));
        } else {
            printer.Key("Array Segments Status", "Missing array segments");
        }
        if (dimension->array_segments() != nullptr && dimension->array_indices() != nullptr) {
            std::vector<int64_t> segments;
            std::vector<int64_t> indices;
            const bool segments_valid = GetSparseIndexVectorValues(dimension->array_segments_type(), dimension->array_segments(), &segments);
            const bool indices_valid = GetSparseIndexVectorValues(dimension->array_indices_type(), dimension->array_indices(), &indices);
            if (!segments_valid || !indices_valid) {
                printer.Key("CSR Status", "Invalid sparse index vector");
            } else if (segments.empty()) {
                printer.Key("CSR Status", "Empty array segments");
            } else {
                bool invalid_segments = segments.front() != 0;
                for (size_t j = 1; j < segments.size() && !invalid_segments; ++j) {
                    if (segments[j] < segments[j - 1])
                        invalid_segments = true;
                }
                if (!invalid_segments && segments.back() != static_cast<int64_t>(indices.size()))
                    invalid_segments = true;
                if (invalid_segments)
                    printer.Key("CSR Status", "Invalid array segments");
            }
        }
        if (dimension->array_indices() != nullptr && tensor->shape() != nullptr && traversal_order != nullptr && i < traversal_order->size()) {
            std::vector<int64_t> indices;
            if (GetSparseIndexVectorValues(dimension->array_indices_type(), dimension->array_indices(), &indices)) {
                const int32_t tensor_dimension = traversal_order->Get(i);
                const size_t rank = static_cast<size_t>(tensor->shape()->size());
                if (tensor_dimension >= 0 &&
                    static_cast<size_t>(tensor_dimension) < rank) {
                    const int32_t dimension_size = tensor->shape()->Get(static_cast<size_t>(tensor_dimension));
                    if (dimension_size >= 0) {
                        bool invalid_indices = false;
                        for (size_t j = 0; j < indices.size(); ++j) {
                            if (indices[j] < 0 || indices[j] >= static_cast<int64_t>(dimension_size)) {
                                invalid_indices = true;
                                break;
                            }
                        }
                        if (invalid_indices)
                            printer.Key("CSR Status", "Array index outside dimension");
                    }
                }
            }
        }
        const char* indices_type = tflite::EnumNameSparseIndexVector(dimension->array_indices_type());
        if (indices_type != nullptr && indices_type[0] != '\0') {
            printer.Key("Array Indices Type", indices_type);
        } else if (dimension->array_indices() != nullptr) {
            printer.Key("Array Indices Type", "UNKNOWN");
            printer.Key("Array Indices Type Status", "Invalid enum value");
        }
        if (dimension->array_indices() != nullptr) {
            printer.Key("Array Indices", SparseIndexVectorToString(dimension->array_indices_type(), dimension->array_indices()));
        } else {
            printer.Key("Array Indices Status", "Missing array indices");
        }
    }
}

static void PrintTensorDetails(ConsolePrinter& printer, const tflite::Tensor* tensor)
{
    if (tensor == nullptr)
        return;
    if (tensor->is_variable())
        printer.Key("Variable", "true");
    if (tensor->has_rank())
        printer.Key("Has Rank", "true");
    const auto* shape_signature = tensor->shape_signature();
    if (shape_signature != nullptr && shape_signature->size() != 0) {
        printer.Key("Shape Signature", VectorInt32ToString(shape_signature));
        for (size_t i = 0; i < shape_signature->size(); ++i) {
            if (shape_signature->Get(i) < -1) {
                printer.Key("Shape Signature Status", "Invalid dimension");
                break;
            }
        }
    }
    const auto* shape = tensor->shape();
    if (shape != nullptr && shape->size() != 0) {
        for (size_t i = 0; i < shape->size(); ++i) {
            if (shape->Get(i) < 0) {
                printer.Key("Shape Status", "Invalid dimension");
                break;
            }
        }
    }
    if (shape != nullptr && shape_signature != nullptr &&
        shape->size() != 0 && shape_signature->size() != 0) {
        if (shape->size() != shape_signature->size()) {
            printer.Key("Shape Signature Status", "Shape/signature rank mismatch");
        } else {
            for (size_t i = 0; i < shape->size(); ++i) {
                const int32_t shape_dimension = shape->Get(i);
                const int32_t signature_dimension = shape_signature->Get(i);
                if (signature_dimension >= 0 && signature_dimension != shape_dimension) {
                    printer.Key("Shape Signature Status", "Shape/signature dimension mismatch");
                    break;
                }
            }
        }
    }
    const auto* variant_tensors = tensor->variant_tensors();
    if (variant_tensors != nullptr && variant_tensors->size() > 1)
        printer.Key("Variant Tensors Status", "Multiple variant subtypes");
    if (tensor->type() != tflite::TensorType_VARIANT && variant_tensors != nullptr && variant_tensors->size() != 0)
        printer.Key("Variant Tensors Status", "Unexpected variant tensors");
    if (variant_tensors != nullptr && variant_tensors->size() != 0) {
        printer.Key("Variant Tensors", static_cast<size_t>(variant_tensors->size()));
        for (size_t i = 0; i < variant_tensors->size(); ++i) {
            const auto* variant = variant_tensors->Get(i);
            if (variant == nullptr)
                continue;
            char title[32];
            snprintf(title, sizeof(title), "Variant Tensor %zu", i);
            printer.SubSection(title);
            const char* type = tflite::EnumNameTensorType(variant->type());
            if (type != nullptr && type[0] != '\0') {
                printer.Key("Type", type);
            } else {
                printer.Key("Type", "UNKNOWN");
                printer.Key("Type Status", "Invalid enum value");
            }
            printer.Key("Shape", VectorInt32ToString(variant->shape()));
            printer.Key("Has Rank", variant->has_rank() ? "true" : "false");
            const auto* shape = variant->shape();
            const size_t shape_count = shape != nullptr ? static_cast<size_t>(shape->size()) : 0;
            if (!variant->has_rank() && shape_count != 0)
                printer.Key("Shape Status", "Shape must be empty when rank is unknown");
        }
    }
    if (tensor->external_buffer() != 0)
        printer.Key("External Buffer", static_cast<size_t>(tensor->external_buffer()));
}

static void PrintModelTensor(ConsolePrinter& printer, const char* title, const tflite::Tensor* tensor, uint32_t tensor_index, const tflite::SubGraph* model_subgraph, const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers)
{
    if (tensor == nullptr)
        return;
    printer.SubSection(title);
    printer.Key("Tensor Index", tensor_index);
    if (tensor->name() != nullptr)
        printer.Key("Model Tensor Name", tensor->name()->c_str());
    const char* type = tflite::EnumNameTensorType(tensor->type());
    if (type != nullptr && type[0] != '\0') {
        printer.Key("Tensor Type", type);
    } else {
        printer.Key("Tensor Type", "UNKNOWN");
        printer.Key("Tensor Type Status", "Invalid enum value");
    }
    printer.Key("Tensor Shape", TensorShapeToString(tensor));
    PrintTensorDetails(printer, tensor);
    PrintTensorQuantization(printer, tensor, model_subgraph, buffers);
    const uint32_t buffer_index = tensor->buffer();
    printer.Key("Buffer Index", static_cast<size_t>(buffer_index));
    if (buffers == nullptr || static_cast<size_t>(buffer_index) >= buffers->size()) {
        printer.Key("Buffer Status", "Invalid buffer index");
    } else {
        const auto* buffer = buffers->Get(buffer_index);
        if (buffer == nullptr) {
            printer.Key("Buffer Status", "Null buffer");
        } else {
            printer.KeyBytes("Buffer Data Size", buffer->data() != nullptr ? static_cast<size_t>(buffer->data()->size()) : 0);
            printer.Key("Buffer Offset", buffer->offset());
            printer.Key("Buffer Size", buffer->size());
        }
    }
}

static void PrintTensorMetadata(ConsolePrinter& printer, const char* title, const tflite::TensorMetadata* tensor, const tflite::Tensor* model_tensor, bool resolved, uint32_t tensor_index, const tflite::SubGraph* model_subgraph, const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers, const uint8_t* model_data, size_t model_size)
{
    if (tensor == nullptr)
        return;
    printer.SubSection(title);
    if (!resolved) {
        printer.Key("Tensor Status", "No matching model tensor");
    } else {
        if (tensor->name() == nullptr || tensor->name()->size() == 0)
            printer.Key("Tensor Status", "Metadata tensor name not provided");
        else if (!TensorNamesMatch(tensor, model_tensor))
            printer.Key("Tensor Status", "Metadata/model tensor name mismatch");
        printer.Key("Tensor Index", static_cast<size_t>(tensor_index));
        if (model_tensor->name() != nullptr)
            printer.Key("Model Tensor Name", model_tensor->name()->c_str());
        const char* type = tflite::EnumNameTensorType(model_tensor->type());
        if (type != nullptr && type[0] != '\0') {
            printer.Key("Tensor Type", type);
        } else {
            printer.Key("Tensor Type", "UNKNOWN");
            printer.Key("Tensor Type Status", "Invalid enum value");
        }
        printer.Key("Tensor Shape", TensorShapeToString(model_tensor));
        PrintTensorDetails(printer, model_tensor);
        PrintTensorSparsity(printer, model_tensor);
        PrintTensorQuantization(printer, model_tensor, model_subgraph, buffers);
        const uint32_t buffer_index = model_tensor->buffer();
        printer.Key("Buffer Index", static_cast<size_t>(buffer_index));
        if (buffers == nullptr || static_cast<size_t>(buffer_index) >= buffers->size()) {
            printer.Key("Buffer Status", "Invalid buffer index");
        } else {
            const auto* buffer = buffers->Get(buffer_index);
            if (buffer == nullptr) {
                printer.Key("Buffer Status", "Null buffer");
            } else {
                printer.KeyBytes("Buffer Data Size", buffer->data() != nullptr ? static_cast<size_t>(buffer->data()->size()) : 0);
                printer.Key("Buffer Offset", buffer->offset());
                printer.Key("Buffer Size", buffer->size());
            }
        }
    }
    if (tensor->name() != nullptr)
        printer.Key("Metadata Name", tensor->name()->c_str());
    if (tensor->description() != nullptr)
        printer.Key("Description", tensor->description()->c_str());
    const auto* dimension_names = tensor->dimension_names();
    if (dimension_names != nullptr && dimension_names->size() != 0) {
        printer.Key("Dimension Names", VectorStringToString(dimension_names));
        bool invalid_dimension_name = false;
        for (size_t i = 0; i < dimension_names->size(); ++i) {
            const auto* name = dimension_names->Get(i);
            if (name == nullptr || name->size() == 0) {
                invalid_dimension_name = true;
                break;
            }
        }
        if (invalid_dimension_name)
            printer.Key("Dimension Names Status", "Missing or empty dimension name");
        if (model_tensor != nullptr && model_tensor->shape() != nullptr && dimension_names->size() != model_tensor->shape()->size()) {
            char status[128];
            snprintf(status, sizeof(status), "Dimension name count mismatch (%zu vs %zu)", static_cast<size_t>(dimension_names->size()), static_cast<size_t>(model_tensor->shape()->size()));
            printer.Key("Dimension Names Status", status);
        }
    }
    PrintContent(printer, tensor->content(), model_tensor);
    PrintStats(printer, tensor->stats());
    PrintProcessUnits(printer, "Process Units", tensor->process_units(), model_data, model_size);
    PrintAssociatedFiles(printer, tensor->associated_files(), model_data, model_size);
}

static void PrintCustomMetadata(ConsolePrinter& printer, const flatbuffers::Vector<flatbuffers::Offset<tflite::CustomMetadata>>* metadata)
{
    if (metadata == nullptr || metadata->size() == 0)
        return;
    printer.SubSection("Custom Metadata");
    for (size_t i = 0; i < metadata->size(); ++i) {
        const auto* item = metadata->Get(i);
        printer.Key("Metadata Index", i);
        if (item == nullptr) {
            printer.Key("Status", "Invalid");
            continue;
        }
        if (item->name() == nullptr || item->name()->size() == 0)
            printer.Key("Status", "Missing name");
        else
            printer.Key("Name", item->name()->c_str());
        const auto* data = item->data();
        if (data == nullptr) {
            printer.Key("Data Size", static_cast<size_t>(0));
            continue;
        }
        printer.Key("Data Size", static_cast<size_t>(data->size()));
        printer.Key("Data", ByteVectorToHex(data));
    }
}
}  // namespace

void MetadataDecoder::Print(ConsolePrinter& printer, const tflite::Metadata* metadata, const tflite::Buffer* buffer, const uint8_t* model_data, size_t model_size)
{
    if (metadata == nullptr || buffer == nullptr) {
        printer.Line("(invalid metadata)");
        return;
    }
    const char* name = metadata->name() != nullptr ? metadata->name()->c_str() : "";
    const auto* data_vector = buffer->data();
    if (data_vector == nullptr || data_vector->size() == 0) {
        printer.Line("Buffer is empty");
        return;
    }
    const uint8_t* data = data_vector->data();
    const size_t size = static_cast<size_t>(data_vector->size());
    printer.KeyBytes("Size", size);
    if (strcmp(name, "min_runtime_version") == 0) {
    const std::string version = ReadMetadataString(data, size);
    printer.Key("Runtime Version", version);
    return;
    }
    if (strcmp(name, "CONVERSION_METADATA") == 0) {
        PrintConversionMetadata(printer, data, size);
        return;
    }
    if (strcmp(name, "TFLITE_METADATA") == 0 &&
        FileIdentifier(data, size) == tflite::ModelMetadataIdentifier()) {
        PrintTfliteMetadata(printer, data, size, model_data, model_size);
        return;
    }
    if (FileIdentifier(data, size) == tflite::ModelMetadataIdentifier()) {
        PrintTfliteMetadata(printer, data, size, model_data, model_size);
        return;
    }
    printer.Key("Format", "Unknown");
    PrintUnknownMetadata(printer, data, size);
}

void MetadataDecoder::PrintConversionMetadata(ConsolePrinter& printer, const uint8_t* data, size_t size)
{
    flatbuffers::Verifier verifier(data, size);
    if (!tflite::VerifyConversionMetadataBuffer(verifier)) {
        printer.Key("Format", "Invalid ConversionMetadata FlatBuffer");
        PrintUnknownMetadata(printer, data, size);
        return;
    }
    const auto* metadata = tflite::GetConversionMetadata(data);
    if (metadata == nullptr) {
        printer.Key("Format", "Invalid ConversionMetadata");
        return;
    }
    printer.Key("Format", "Conversion Metadata");
    const auto* environment = metadata->environment();
    if (environment != nullptr) {
        printer.SubSection("Environment");
        printer.Key("TensorFlow Version", StringValue(environment->tensorflow_version()).c_str());
        printer.Key("API Version", static_cast<size_t>(environment->api_version()));
        printer.Key("Model Type", tflite::EnumNameModelType(environment->model_type()));
        printer.Key("Model Hash", HexValue(environment->model_hash()).c_str());
    }
    const auto* options = metadata->options();
    if (options != nullptr) {
        printer.SubSection("Options");
        printer.Key("Allow Custom Ops", BoolValue(options->allow_custom_ops()).c_str());
        printer.Key("Enable Select TF Ops", BoolValue(options->enable_select_tf_ops()).c_str());
        printer.Key("Force Select TF Ops", BoolValue(options->force_select_tf_ops()).c_str());
        const auto* modes = options->model_optimization_modes();
        if (modes != nullptr && modes->size() != 0) {
            std::ostringstream optimization;
            for (size_t i = 0; i < modes->size(); ++i) {
                if (i != 0)
                    optimization << ", ";
                optimization << tflite::EnumNameModelOptimizationMode(static_cast<tflite::ModelOptimizationMode>(modes->Get(i)));
            }
            printer.Key("Optimization Modes", optimization.str());
        }
        const auto* block_sizes = options->sparsity_block_sizes();
        if (block_sizes != nullptr && block_sizes->size() != 0) {
            for (size_t i = 0; i < block_sizes->size(); ++i) {
                const auto* block_size = block_sizes->Get(i);
                if (block_size == nullptr || block_size->values() == nullptr)
                    continue;
                std::ostringstream values;
                values << "[";
                for (size_t j = 0; j < block_size->values()->size(); ++j) {
                    if (j != 0)
                        values << ", ";
                    values << block_size->values()->Get(j);
                }
                values << "]";
                char key[64];
                snprintf(key, sizeof(key), "Sparsity Block Size %zu", i);
                printer.Key(key, values.str());
            }
        }
    }
}

void MetadataDecoder::PrintTfliteMetadata(ConsolePrinter& printer, const uint8_t* data, size_t size, const uint8_t* model_data, size_t model_size)
{
    flatbuffers::Verifier verifier(data, size);
    if (!tflite::VerifyModelMetadataBuffer(verifier)) {
        printer.Key("Format", "Invalid TFLite Metadata FlatBuffer");
        PrintUnknownMetadata(printer, data, size);
        return;
    }
    const auto* metadata = tflite::GetModelMetadata(data);
    if (metadata == nullptr) {
        printer.Key("Format", "Invalid TFLite Metadata");
        return;
    }
    printer.Key("Format", "TFLite Model Metadata");
    printer.Key("Identifier", tflite::ModelMetadataIdentifier());
    printer.SubSection("Model");
    if (metadata->name() != nullptr)
        printer.Key("Name", metadata->name()->c_str());
    if (metadata->description() != nullptr)
        printer.Key("Description", metadata->description()->c_str());
    if (metadata->version() != nullptr)
        printer.Key("Version", metadata->version()->c_str());
    if (metadata->author() != nullptr)
        printer.Key("Author", metadata->author()->c_str());
    if (metadata->license() != nullptr)
        printer.Key("License", metadata->license()->c_str());
    const auto* min_parser_version = metadata->min_parser_version();
    if (min_parser_version != nullptr) {
        printer.Key("Minimum Parser Version", min_parser_version->c_str());
        if (min_parser_version->size() == 0) {
            printer.Key("Parser Version Status", "Empty minimum parser version");
        } else {
            unsigned int major = 0;
            unsigned int minor = 0;
            unsigned int patch = 0;
            if (!ParseSemanticVersion(min_parser_version->c_str(), &major, &minor, &patch)) {
                printer.Key("Parser Version Status", "Invalid semantic version");
            } else if (major > kMetadataSchemaMajor ||
                       (major == kMetadataSchemaMajor && minor > kMetadataSchemaMinor) ||
                       (major == kMetadataSchemaMajor && minor == kMetadataSchemaMinor &&
                        patch > kMetadataSchemaPatch)) {
                printer.Key("Parser Version Status", "Requires newer metadata parser");
            }
        }
    }
    PrintAssociatedFiles(printer, metadata->associated_files(), model_data, model_size);
    const auto* subgraphs = metadata->subgraph_metadata();
    const auto* model = model_data != nullptr ? tflite::GetModel(model_data) : nullptr;
    const size_t metadata_count = subgraphs != nullptr ? static_cast<size_t>(subgraphs->size()) : 0;
    printer.Key("Subgraphs", metadata_count);
    if (model == nullptr || model->subgraphs() == nullptr) {
        if (metadata_count != 0)
            printer.Key("Model Status", "No model subgraphs available");
        return;
    }
    const size_t model_count = static_cast<size_t>(model->subgraphs()->size());
    if (metadata_count == 0 && model_count != 0) {
        printer.Key("Subgraph Status", "Metadata not provided");
        return;
    }
    if (metadata_count != model_count)
        printer.Key("Subgraph Status", "Metadata/model count mismatch");
    for (size_t i = 0; i < subgraphs->size(); ++i) {
        const auto* subgraph = subgraphs->Get(i);
        char title[32];
        snprintf(title, sizeof(title), "Subgraph %zu", i);
        printer.SubSection(title);
        if (subgraph == nullptr) {
            printer.Key("Status", "Invalid metadata");
            continue;
        }
        const tflite::SubGraph* model_subgraph = nullptr;
        if (model != nullptr && model->subgraphs() != nullptr && i < model->subgraphs()->size())
            model_subgraph = model->subgraphs()->Get(i);
        if (model_subgraph == nullptr)
            printer.Key("Status", "No matching model subgraph");
        if (model_subgraph != nullptr && subgraph->name() != nullptr && model_subgraph->name() != nullptr && strcmp(subgraph->name()->c_str(), model_subgraph->name()->c_str()) != 0) {
            printer.Key("Name Status", "Metadata/model name mismatch");
        }
        if (subgraph->name() != nullptr)
            printer.Key("Name", subgraph->name()->c_str());
        if (subgraph->description() != nullptr)
            printer.Key("Description", subgraph->description()->c_str());
        const auto* inputs = subgraph->input_tensor_metadata();
        {
            const size_t metadata_count = inputs != nullptr ? static_cast<size_t>(inputs->size()) : 0;
            const size_t model_count = model_subgraph != nullptr && model_subgraph->inputs() != nullptr ? static_cast<size_t>(model_subgraph->inputs()->size()) : 0;
            PrintTensorMetadataSummary(printer, "Input Tensors", metadata_count, model_count);
            if (inputs != nullptr) {
                for (size_t j = 0; j < inputs->size(); ++j) {
                    const auto* tensor = inputs->Get(j);
                    char tensor_title[32];
                    snprintf(tensor_title, sizeof(tensor_title), "Input Tensor %zu", j);
                    if (tensor == nullptr) {
                        printer.SubSection(tensor_title);
                        printer.Key("Tensor Status", "Invalid metadata");
                        continue;
                    }
                    const tflite::Tensor* model_tensor = nullptr;
                    uint32_t tensor_index = 0;
                    const bool resolved = ResolveModelTensor(model_subgraph, true, j, &model_tensor, &tensor_index);
                    PrintTensorMetadata(printer, tensor_title, tensor, model_tensor, resolved, tensor_index, model_subgraph, model != nullptr ? model->buffers() : nullptr, model_data, model_size);
                }
            }
            if (inputs == nullptr || inputs->size() == 0) {
                if (model_subgraph != nullptr && model_subgraph->inputs() != nullptr) {
                    for (size_t j = 0; j < model_subgraph->inputs()->size(); ++j) {
                        const int32_t tensor_index_value = model_subgraph->inputs()->Get(j);
                        if (tensor_index_value < 0)
                            continue;
                        const uint32_t tensor_index = static_cast<uint32_t>(tensor_index_value);
                        const auto* tensors = model_subgraph->tensors();
                        if (tensors == nullptr || static_cast<size_t>(tensor_index) >= tensors->size())
                            continue;
                        const auto* tensor = tensors->Get(tensor_index);
                        char tensor_title[32];
                        snprintf(tensor_title, sizeof(tensor_title), "Model Input Tensor %zu", j);
                        PrintModelTensor(printer, tensor_title, tensor, tensor_index, model_subgraph, model != nullptr ? model->buffers() : nullptr);
                    }
                }
            }
        }
        const auto* outputs = subgraph->output_tensor_metadata();
        {
            const size_t metadata_count = outputs != nullptr ? static_cast<size_t>(outputs->size()) : 0;
            const size_t model_count = model_subgraph != nullptr && model_subgraph->outputs() != nullptr ? static_cast<size_t>(model_subgraph->outputs()->size()) : 0;
            PrintTensorMetadataSummary(printer, "Output Tensors", metadata_count, model_count);
            if (outputs != nullptr) {
                for (size_t j = 0; j < outputs->size(); ++j) {
                    const auto* tensor = outputs->Get(j);
                    char tensor_title[32];
                    snprintf(tensor_title, sizeof(tensor_title), "Output Tensor %zu", j);
                    if (tensor == nullptr) {
                        printer.SubSection(tensor_title);
                        printer.Key("Tensor Status", "Invalid metadata");
                        continue;
                    }
                    const tflite::Tensor* model_tensor = nullptr;
                    uint32_t tensor_index = 0;
                    const bool resolved = ResolveModelTensor(model_subgraph, false, j, &model_tensor, &tensor_index);
                    PrintTensorMetadata(printer, tensor_title, tensor, model_tensor, resolved, tensor_index, model_subgraph, model != nullptr ? model->buffers() : nullptr, model_data, model_size);
                }
            }
            if (outputs == nullptr || outputs->size() == 0) {
                if (model_subgraph != nullptr && model_subgraph->outputs() != nullptr) {
                    for (size_t j = 0; j < model_subgraph->outputs()->size(); ++j) {
                        const int32_t tensor_index_value = model_subgraph->outputs()->Get(j);
                        if (tensor_index_value < 0)
                            continue;
                        const uint32_t tensor_index = static_cast<uint32_t>(tensor_index_value);
                        const auto* tensors = model_subgraph->tensors();
                        if (tensors == nullptr || static_cast<size_t>(tensor_index) >= tensors->size())
                            continue;
                        const auto* tensor = tensors->Get(tensor_index);
                        char tensor_title[32];
                        snprintf(tensor_title, sizeof(tensor_title), "Model Output Tensor %zu", j);
                        PrintModelTensor(printer, tensor_title, tensor, tensor_index, model_subgraph, model != nullptr ? model->buffers() : nullptr);
                    }
                }
            }
        }
        PrintAssociatedFiles(printer, subgraph->associated_files(), model_data, model_size);
        PrintProcessUnits(printer, "Input Process Units", subgraph->input_process_units(), model_data, model_size);
        PrintProcessUnits(printer, "Output Process Units", subgraph->output_process_units(), model_data, model_size);
        PrintTensorGroups(printer, "Input Tensor Groups", subgraph->input_tensor_groups(), model_subgraph);
        PrintTensorGroups(printer, "Output Tensor Groups", subgraph->output_tensor_groups(), model_subgraph);
        PrintCustomMetadata(printer, subgraph->custom_metadata());
    }
}

void MetadataDecoder::PrintUnknownMetadata(ConsolePrinter& printer, const uint8_t* data, size_t size)
{
    if (data == nullptr || size == 0)
        return;
    const std::string identifier = FileIdentifier(data, size);
    if (!identifier.empty())
        printer.Key("Identifier", identifier);
    constexpr size_t kPreviewBytes = 64;
    const size_t count = std::min(size, kPreviewBytes);
    std::ostringstream out;
    for (size_t i = 0; i < count; ++i) {
        if (i != 0)
            out << " ";
        char byte[4];
        snprintf(byte, sizeof(byte), "%02X", data[i]);
        out << byte;
    }
    if (count < size)
        out << " ...";
    printer.Key("Data", out.str());
}

void MetadataDecoder::PrintDataPreview(ConsolePrinter& printer, const uint8_t* data, size_t size)
{
    PrintUnknownMetadata(printer, data, size);
}