#include "model_validator.h"

#include <cmath>
#include <sstream>

#include "flatbuffers/flexbuffers.h"

#include "model_utils.h"

ModelValidator::ModelValidator(const ModelDocument& document)
    : document_(document), current_subgraph_(nullptr)
{
}

bool ModelValidator::Validate() const
{
    last_error_.clear();
    if (!document_.IsValid()) {
        last_error_ = "invalid model";
        return false;
    }
    if (!ValidateModelInterface()) {
        if (last_error_.empty())
            last_error_ = "model interface validation failed";
        return false;
    }
    if (!ValidateOperatorCodes()) {
        if (last_error_.empty())
            last_error_ = "operator code validation failed";
        return false;
    }
    if (!ValidateTensors()) {
        if (last_error_.empty())
            last_error_ = "tensor validation failed";
        return false;
    }
    if (!ValidateGraph()) {
        if (last_error_.empty())
            last_error_ = "graph validation failed";
        return false;
    }
    if (!ValidateOperators()) {
        if (last_error_.empty())
            last_error_ = "operator validation failed";
        return false;
    }
    return true;
}

bool ModelValidator::ValidateTensorIndexList(const flatbuffers::Vector<int32_t>* indices, size_t tensor_count) const
{
    if (indices == nullptr)
        return false;
    for (size_t i = 0; i < indices->size(); ++i) {
        const int32_t index = indices->Get(i);
        if (index < 0)
            return false;
        if (static_cast<size_t>(index) >= tensor_count) {
            return false;
        }
        // No duplicate tensor indices.
        for (size_t j = 0; j < i; ++j) {
            if (indices->Get(j) == index)
                return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateModelInterface() const
{
    const auto* model = document_.Model();
    if (model == nullptr) {
        SetError("missing model");
        return false;
    }
    const auto* subgraphs = model->subgraphs();
    if (subgraphs == nullptr || subgraphs->size() == 0) {
        SetError("missing subgraphs");
        return false;
    }
    for (size_t i = 0; i < subgraphs->size(); ++i) {
        const auto* subgraph = subgraphs->Get(i);
        if (subgraph == nullptr) {
            SetError("missing subgraph");
            return false;
        }
        const auto* tensors = subgraph->tensors();
        const auto* inputs = subgraph->inputs();
        const auto* outputs = subgraph->outputs();
        if (tensors == nullptr) {
            SetError("missing tensor list");
            return false;
        }
        if (inputs == nullptr || outputs == nullptr) {
            SetError("missing subgraph inputs or outputs");
            return false;
        }
        if (!ValidateTensorIndexList(inputs, tensors->size())) {
            SetError("invalid subgraph input tensor indices");
            return false;
        }
        if (!ValidateTensorIndexList(outputs, tensors->size())) {
            SetError("invalid subgraph output tensor indices");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateOperatorCodes() const
{
    const auto* model = document_.Model();
    if (model == nullptr)
        return false;
    const auto* operator_codes = model->operator_codes();
    if (operator_codes == nullptr) {
        SetError("missing operator codes");
        return false;
    }
    const auto* subgraphs = model->subgraphs();
    if (subgraphs == nullptr || subgraphs->size() == 0) {
        SetError("missing subgraphs");
        return false;
    }
    bool valid = true;
    for (size_t subgraph_index = 0; subgraph_index < subgraphs->size(); ++subgraph_index) {
        const auto* subgraph = subgraphs->Get(subgraph_index);
        if (subgraph == nullptr) {
            SetError("missing subgraph");
            valid = false;
            continue;
        }
        const auto* operators = subgraph->operators();
        if (operators == nullptr) {
            SetError("missing operator list");
            valid = false;
            continue;
        }
        for (size_t operator_index = 0; operator_index < operators->size(); ++operator_index) {
            const auto* op = operators->Get(operator_index);
            if (op == nullptr) {
                SetError(operator_index, "null operator");
                valid = false;
                continue;
            }
            const uint32_t opcode_index = op->opcode_index();
            if (opcode_index >= operator_codes->size()) {
                SetError(operator_index, "operator code index out of range");
                valid = false;
                continue;
            }
            const auto* opcode = operator_codes->Get(opcode_index);
            if (opcode == nullptr) {
                SetError(operator_index, "operator code is null");
                valid = false;
                continue;
            }
            const int builtin_code = BuiltinOperatorCode(opcode);
            if (builtin_code < static_cast<int>(tflite::BuiltinOperator_MIN) || builtin_code > static_cast<int>(tflite::BuiltinOperator_MAX)) {
                SetError(operator_index, "builtin operator code out of range");
                valid = false;
            }
        }
    }
    return valid;
}

bool ModelValidator::ValidateTensors() const
{
    const auto* model = document_.Model();
    if (model == nullptr || model->subgraphs() == nullptr || model->subgraphs()->size() == 0) {
        SetError("missing subgraphs");
        return false;
    }
    bool valid = true;
    for (size_t subgraph_index = 0; subgraph_index < model->subgraphs()->size(); ++subgraph_index) {
        const auto* subgraph = model->subgraphs()->Get(subgraph_index);
        if (subgraph == nullptr) {
            SetError("missing subgraph");
            valid = false;
            continue;
        }
        const auto* tensors = subgraph->tensors();
        if (tensors == nullptr) {
            SetError("missing tensor list");
            valid = false;
            continue;
        }
        for (size_t tensor_index = 0; tensor_index < tensors->size(); ++tensor_index) {
            const auto* tensor = tensors->Get(tensor_index);
            if (tensor == nullptr) {
                SetError(tensor_index, "null tensor");
                valid = false;
                continue;
            }
            if (!ValidateTensor(tensor_index, tensor))
                valid = false;
        }
    }
    return valid;
}

bool ModelValidator::ValidateTensor(size_t tensor_index, const tflite::Tensor* tensor) const
{
    if (tensor == nullptr) {
        SetError("null tensor");
        return false;
    }
    if (!ValidateQuantization(tensor)) {
        SetError("tensor has invalid quantization");
        return false;
    }
    const int type = static_cast<int>(tensor->type());
    if (type < tflite::TensorType_MIN || type > tflite::TensorType_MAX) {
        SetError("tensor has invalid type");
        return false;
    }
    const auto* shape = tensor->shape();
    if (shape == nullptr) {
        SetError("tensor has no shape");
        return false;
    }
    for (size_t i = 0; i < shape->size(); ++i) {
        if (shape->Get(i) < 0) {
            SetError("tensor has a negative dimension");
            return false;
        }
    }
    const auto* buffers = document_.Buffers();
    if (buffers == nullptr) {
        SetError("missing buffers");
        return false;
    }
    const size_t buffer_index = static_cast<size_t>(tensor->buffer());
    if (buffer_index >= buffers->size()) {
        SetError("tensor buffer index out of range");
        return false;
    }
    const auto* buffer = buffers->Get(buffer_index);
    if (buffer == nullptr) {
        SetError("tensor buffer is null");
        return false;
    }
    if (buffer->data() == nullptr || buffer->data()->size() == 0)
        return true;
    size_t element_count = 1;
    for (size_t i = 0; i < shape->size(); ++i) {
        const size_t dimension = static_cast<size_t>(shape->Get(i));
        if (dimension != 0 &&
            element_count > static_cast<size_t>(-1) / dimension) {
            SetError("tensor element count overflows");
            return false;
        }
        element_count *= dimension;
    }
    size_t element_size = 0;
    switch (tensor->type()) {
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_INT32:
    case tflite::TensorType_UINT32:
        element_size = 4;
        break;
    case tflite::TensorType_FLOAT16:
    case tflite::TensorType_INT16:
    case tflite::TensorType_UINT16:
        element_size = 2;
        break;
    case tflite::TensorType_UINT8:
    case tflite::TensorType_INT8:
    case tflite::TensorType_BOOL:
        element_size = 1;
        break;
    case tflite::TensorType_INT64:
    case tflite::TensorType_UINT64:
        element_size = 8;
        break;
    case tflite::TensorType_INT4:
        element_size = 0;
        break;
    default:
        return true;
    }
    size_t expected_bytes;
    if (tensor->type() == tflite::TensorType_INT4) {
        expected_bytes = (element_count + 1) / 2;
    } else {
        if (element_size != 0 &&
            element_count > static_cast<size_t>(-1) / element_size) {
            SetError("tensor byte count overflows");
            return false;
        }
        expected_bytes = element_count * element_size;
    }
    const size_t actual_bytes = buffer->data()->size();
    if (actual_bytes < expected_bytes) {
        SetError("tensor buffer is too small");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateQuantization(const tflite::Tensor* tensor) const
{
    if (tensor == nullptr)
        return false;
    const auto* quantization = tensor->quantization();
    // No quantization parameters is valid.
    if (quantization == nullptr)
        return true;
    const auto* scales = quantization->scale();
    const auto* zero_points = quantization->zero_point();
    // Both must exist together.
    if (scales == nullptr && zero_points == nullptr) {
        return true;
    }
    if (scales == nullptr || zero_points == nullptr) {
        return false;
    }
    const size_t scale_count = scales->size();
    const size_t zero_point_count = zero_points->size();
    // TFLite quantization requires at least
    // one scale when quantization is specified.
    if (scale_count == 0)
        return false;
    // Per-tensor quantization:
    //
    // scale       = 1
    // zero_point  = 1
    //
    // Per-axis quantization:
    //
    // scale and zero_point may contain multiple
    // entries, but they must have matching sizes.
    if (scale_count != zero_point_count)
        return false;
    // Every scale must be finite and positive.
    for (size_t i = 0; i < scale_count; ++i) {
        const float scale = scales->Get(i);
        if (!std::isfinite(scale))
            return false;
        if (scale <= 0.0f)
            return false;
    }
    return true;
}

bool ModelValidator::ValidateBinaryBroadcast(size_t operator_index, const tflite::Tensor* input1, const tflite::Tensor* input2, const tflite::Tensor* output) const
{
    if (input1 == nullptr || input2 == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input1_shape = input1->shape();
    const auto* input2_shape = input2->shape();
    const auto* output_shape = output->shape();
    if (input1_shape == nullptr || input2_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    const size_t rank1 = input1_shape->size();
    const size_t rank2 = input2_shape->size();
    const size_t output_rank = output_shape->size();
    const size_t rank = rank1 > rank2 ? rank1 : rank2;
    if (output_rank != rank) {
        SetError(operator_index, "output rank does not match broadcast rank");
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        const int32_t dimension1 = i < rank - rank1 ? 1 : input1_shape->Get(i - (rank - rank1));
        const int32_t dimension2 = i < rank - rank2 ? 1 : input2_shape->Get(i - (rank - rank2));
        const int32_t output_dimension = output_shape->Get(i);
        if (dimension1 < 0 || dimension2 < 0 || output_dimension < 0) {
            SetError(operator_index, "contains a negative tensor dimension");
            return false;
        }
        if (dimension1 != dimension2 && dimension1 != 1 && dimension2 != 1) {
            SetError(operator_index, "input tensor shapes are not broadcast compatible");
            return false;
        }
        const int32_t expected_dimension = dimension1 > dimension2 ? dimension1 : dimension2;
        if (output_dimension != expected_dimension) {
            SetError(operator_index, "output shape does not match broadcast shape");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateUnaryElementwise(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (input_shape->size() != output_shape->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    for (size_t i = 0; i < input_shape->size(); ++i) {
        if (input_shape->Get(i) < 0 || output_shape->Get(i) < 0) {
            SetError(operator_index, "contains a negative tensor dimension");
            return false;
        }
        if (input_shape->Get(i) != output_shape->Get(i)) {
            SetError(operator_index, "input and output shapes do not match");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateComparison(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input1_index = inputs->Get(0);
    const int32_t input2_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input1_index < 0 || input2_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input1_index) >= tensors->size() ||
        static_cast<size_t>(input2_index) >= tensors->size() ||
        static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input1 = tensors->Get(static_cast<size_t>(input1_index));
    const auto* input2 = tensors->Get(static_cast<size_t>(input2_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input1 == nullptr || input2 == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input1->type() != input2->type()) {
        SetError(operator_index, "comparison input tensor types do not match");
        return false;
    }
    if (output->type() != tflite::TensorType_BOOL) {
        SetError(operator_index, "comparison output tensor must be BOOL");
        return false;
    }
    if (!ValidateBinaryBroadcast(operator_index, input1, input2, output))
        return false;
    return true;
}

bool ModelValidator::ValidateLogicalBinary(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input1_index = inputs->Get(0);
    const int32_t input2_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input1_index < 0 || input2_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input1_index) >= tensors->size() ||
        static_cast<size_t>(input2_index) >= tensors->size() ||
        static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input1 = tensors->Get(static_cast<size_t>(input1_index));
    const auto* input2 = tensors->Get(static_cast<size_t>(input2_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input1 == nullptr || input2 == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input1->type() != tflite::TensorType_BOOL ||
        input2->type() != tflite::TensorType_BOOL ||
        output->type() != tflite::TensorType_BOOL) {
        SetError(operator_index, "logical operator tensors must be BOOL");
        return false;
    }
    if (!ValidateBinaryBroadcast(operator_index, input1, input2, output))
        return false;
    return true;
}

bool ModelValidator::ValidateGraph() const
{
    const auto* model = document_.Model();
    if (model == nullptr || model->subgraphs() == nullptr || model->subgraphs()->size() == 0) {
        SetError("missing subgraphs");
        return false;
    }
    for (size_t i = 0; i < model->subgraphs()->size(); ++i) {
        if (!ValidateSubgraphGraph(i, model->subgraphs()->Get(i)))
            return false;
    }
    return true;
}

bool ModelValidator::ValidateSubgraph(size_t subgraph_index, const tflite::SubGraph* subgraph) const
{
    if (subgraph == nullptr) {
        SetError("null subgraph");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const auto* inputs = subgraph->inputs();
    const auto* outputs = subgraph->outputs();
    const auto* operators = subgraph->operators();
    if (tensors == nullptr) {
        SetError("subgraph has no tensor list");
        return false;
    }
    if (inputs == nullptr || outputs == nullptr || operators == nullptr) {
        SetError("subgraph is missing inputs, outputs or operators");
        return false;
    }
    if (!ValidateTensorIndexList(inputs, tensors->size())) {
        SetError("invalid subgraph input tensor indices");
        return false;
    }
    if (!ValidateTensorIndexList(outputs, tensors->size())) {
        SetError("invalid subgraph output tensor indices");
        return false;
    }
    for (size_t i = 0; i < operators->size(); ++i) {
        const auto* op = operators->Get(i);
        if (op == nullptr) {
            SetError("subgraph contains null operator");
            return false;
        }
        const auto* op_inputs = op->inputs();
        const auto* op_outputs = op->outputs();
        if (op_inputs == nullptr || op_outputs == nullptr) {
            SetError("operator is missing inputs or outputs");
            return false;
        }
        for (size_t j = 0; j < op_inputs->size(); ++j) {
            const int32_t index = op_inputs->Get(j);
            if (index < 0 || static_cast<size_t>(index) >= tensors->size()) {
                SetError("operator input tensor index is out of range");
                return false;
            }
        }
        for (size_t j = 0; j < op_outputs->size(); ++j) {
            const int32_t index = op_outputs->Get(j);
            if (index < 0 || static_cast<size_t>(index) >= tensors->size()) {
                SetError("operator output tensor index is out of range");
                return false;
            }
        }
    }
    return true;
}

bool ModelValidator::ValidateSubgraphGraph(size_t subgraph_index, const tflite::SubGraph* subgraph) const
{
    (void)subgraph_index;
    if (subgraph == nullptr) {
        SetError("missing subgraph");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (tensors == nullptr)
        return false;
    bool valid = true;
    for (size_t i = 0; i < tensors->size(); ++i) {
        const auto* tensor = tensors->Get(i);
        if (tensor == nullptr) {
            valid = false;
            continue;
        }
        if (!ValidateTensorGraph(i, tensor, subgraph))
            valid = false;
    }
    return valid;
}

const tflite::SubGraph* ModelValidator::CurrentSubgraph() const
{
    if (current_subgraph_ != nullptr)
        return current_subgraph_;
    return document_.PrimarySubgraph();
}

bool ModelValidator::ValidateTensorGraph(size_t tensor_index, const tflite::Tensor* tensor, const tflite::SubGraph* subgraph) const
{
    if (tensor == nullptr || subgraph == nullptr)
        return false;
    const auto* operators = subgraph->operators();
    if (operators == nullptr)
        return false;
    const auto* inputs = subgraph->inputs();
    const auto* outputs = subgraph->outputs();
    bool is_input = false;
    bool is_output = false;
    if (inputs != nullptr) {
        for (size_t i = 0; i < inputs->size(); ++i) {
            if (inputs->Get(i) == static_cast<int32_t>(tensor_index)) {
                is_input = true;
                break;
            }
        }
    }
    if (outputs != nullptr) {
        for (size_t i = 0; i < outputs->size(); ++i) {
            if (outputs->Get(i) == static_cast<int32_t>(tensor_index)) {
                is_output = true;
                break;
            }
        }
    }
    size_t producer_count = 0;
    size_t consumer_count = 0;
    bool consumed_by_custom_operator = false;
    for (size_t op_index = 0; op_index < operators->size(); ++op_index) {
        const auto* op = operators->Get(op_index);
        if (op == nullptr)
            return false;
        const auto* op_inputs = op->inputs();
        if (op_inputs != nullptr) {
            for (size_t i = 0; i < op_inputs->size(); ++i) {
                if (op_inputs->Get(i) == static_cast<int32_t>(tensor_index)) {
                    ++consumer_count;
                    const auto* opcode = document_.OperatorCode(op->opcode_index());
                    if (opcode != nullptr && BuiltinOperatorCode(opcode) == tflite::BuiltinOperator_CUSTOM)
                        consumed_by_custom_operator = true;
                }
            }
        }
        const auto* op_outputs = op->outputs();
        if (op_outputs != nullptr) {
            for (size_t i = 0; i < op_outputs->size(); ++i) {
                if (op_outputs->Get(i) == static_cast<int32_t>(tensor_index))
                    ++producer_count;
            }
        }
    }
    if (is_input && producer_count != 0) {
        SetError("model input tensor has an operator producer");
        return false;
    }
    if (is_output && !is_input && producer_count == 0) {
        SetError("model output tensor has no producer");
        return false;
    }
    if (producer_count > 1) {
        SetError("tensor has multiple operator producers");
        return false;
    }
    if (is_input && producer_count != 0)
        return false;
    if (!is_input && !is_output && !tensor->is_variable() && producer_count == 0 && consumer_count == 0)
        return true;
    if (!is_input && !tensor->is_variable() && producer_count == 0) {
        const auto* buffers = document_.Buffers();
        if (buffers == nullptr) {
            SetError("missing buffers for constant tensor");
            return false;
        }
        const size_t buffer_index = static_cast<size_t>(tensor->buffer());
        if (buffer_index >= buffers->size()) {
            SetError("constant tensor buffer index out of range");
            return false;
        }
        const auto* buffer = buffers->Get(buffer_index);
        const bool has_buffer_data = buffer != nullptr && buffer->data() != nullptr && buffer->data()->size() != 0;
        if (!has_buffer_data && !consumed_by_custom_operator) {
            SetError("constant tensor has no buffer data");
            return false;
        }
    }
    if (is_output && !is_input && producer_count == 0)
        return false;
    if (!is_output && producer_count > 0 && consumer_count == 0) {
        SetError("intermediate tensor has no consumer");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateOperators() const
{
    const auto* model = document_.Model();
    if (model == nullptr || model->subgraphs() == nullptr || model->subgraphs()->size() == 0) {
        SetError("missing subgraphs");
        return false;
    }
    bool valid = true;
    for (size_t subgraph_index = 0; subgraph_index < model->subgraphs()->size(); ++subgraph_index) {
        current_subgraph_ = model->subgraphs()->Get(subgraph_index);
        if (current_subgraph_ == nullptr) {
            SetError("missing subgraph");
            valid = false;
            continue;
        }
        const auto* operators = current_subgraph_->operators();
        const auto* tensors = current_subgraph_->tensors();
        if (operators == nullptr || tensors == nullptr) {
            SetError("missing operators or tensors");
            valid = false;
            continue;
        }
        for (size_t op_index = 0; op_index < operators->size(); ++op_index) {
            const auto* op = operators->Get(op_index);
            if (op == nullptr) {
                SetError(op_index, "null operator");
                valid = false;
                continue;
            }
            if (!ValidateOperatorTensorReferences(op_index, op)) {
                valid = false;
                continue;
            }
            const auto* opcode = document_.OperatorCode(op->opcode_index());
            if (opcode == nullptr) {
                SetError(op_index, "missing operator code");
                valid = false;
                continue;
            }
            switch (BuiltinOperatorCode(opcode)) {
            case tflite::BuiltinOperator_ABS:
                if (!ValidateAbs(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_ADD:
                if (!ValidateAdd(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateAddOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_ADD_N:
                if (!ValidateAddN(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_ARG_MAX:
                if (!ValidateArgMax(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_ARG_MIN:
                if (!ValidateArgMin(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_ASSIGN_VARIABLE:
                if (!ValidateAssignVariable(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateAssignVariableOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_AVERAGE_POOL_2D:
                if (!ValidateAveragePool2D(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateAveragePool2DOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_BATCH_MATMUL:
                if (!ValidateBatchMatMul(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_BATCH_TO_SPACE_ND:
                if (!ValidateBatchToSpaceNd(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateBatchToSpaceNdOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_BROADCAST_ARGS:
                if (!ValidateBroadcastArgs(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_BROADCAST_TO:
                if (!ValidateBroadcastTo(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_CALL_ONCE:
                if (!ValidateCallOnce(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateCallOnceOptions(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateCallOnceSubgraph(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_CAST:
                if (!ValidateCast(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_CEIL:
                if (!ValidateCeil(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_CONCATENATION:
                if (!ValidateConcatenation(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateConcatenationOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_CONV_2D:
                if (!ValidateConv2D(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateConv2DOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_COS:
                if (!ValidateCos(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_CUMSUM:
                if (!ValidateCumsum(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateCumsumOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_DEPTH_TO_SPACE:
                if (!ValidateDepthToSpace(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateDepthToSpaceOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
                if (!ValidateDepthwiseConv2D(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateDepthwiseConv2DOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_DEQUANTIZE:
                if (!ValidateDequantize(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_DIV:
                if (!ValidateDiv(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateDivOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE:
                if (!ValidateDynamicUpdateSlice(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateDynamicUpdateSliceOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_ELU:
                if (!ValidateElu(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_EMBEDDING_LOOKUP:
                if (!ValidateEmbeddingLookup(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_EQUAL:
                if (!ValidateEqual(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_EXP:
                if (!ValidateExp(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_EXPAND_DIMS:
                if (!ValidateExpandDims(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_FILL:
                if (!ValidateFill(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_FLOOR:
                if (!ValidateFloor(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_FLOOR_DIV:
                if (!ValidateFloorDiv(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_FLOOR_MOD:
                if (!ValidateFloorMod(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_FULLY_CONNECTED:
                if (!ValidateFullyConnected(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateFullyConnectedOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_GATHER:
                if (!ValidateGather(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateGatherOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_GATHER_ND:
                if (!ValidateGatherNd(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_GREATER:
                if (!ValidateGreater(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_GREATER_EQUAL:
                if (!ValidateGreaterEqual(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_HARD_SWISH:
                if (!ValidateHardSwish(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_IF:
                if (!ValidateIf(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateIfOptions(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateIfSubgraphs(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_L2_NORMALIZATION:
                if (!ValidateL2Normalization(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_L2_POOL_2D:
                if (!ValidateL2Pool2D(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateL2Pool2DOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_LEAKY_RELU:
                if (!ValidateLeakyRelu(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateLeakyReluOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_LESS:
                if (!ValidateLess(op_index, op)) {
                    valid = false;
                }
                break;

            case tflite::BuiltinOperator_LESS_EQUAL:
                if (!ValidateLessEqual(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_LOG:
                if (!ValidateLog(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_LOG_SOFTMAX:
                if (!ValidateLogSoftmax(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateLogSoftmaxOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_LOGICAL_AND:
                if (!ValidateLogicalAnd(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_LOGICAL_NOT:
                if (!ValidateLogicalNot(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_LOGICAL_OR:
                if (!ValidateLogicalOr(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_LOGISTIC:
                if (!ValidateLogistic(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_MAXIMUM:
                if (!ValidateMaximum(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateMaximumOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_MAX_POOL_2D:
                if (!ValidateMaxPool2D(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateMaxPool2DOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_MEAN:
                if (!ValidateMean(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateMeanOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_MINIMUM:
                if (!ValidateMinimum(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateMinimumOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_MIRROR_PAD:
                if (!ValidateMirrorPad(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateMirrorPadOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_MUL:
                if (!ValidateMul(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateMulOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_NEG:
                if (!ValidateNeg(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_NOT_EQUAL:
                if (!ValidateNotEqual(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_PACK:
                if (!ValidatePack(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidatePackOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_PAD:
                if (!ValidatePad(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidatePadOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_PADV2:
                if (!ValidatePadV2(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidatePadV2Options(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_PRELU:
                if (!ValidatePrelu(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_QUANTIZE:
                if (!ValidateQuantize(op_index, op)) {
                    valid = false;
                    continue;
                }
                break;
            case tflite::BuiltinOperator_READ_VARIABLE:
                if (!ValidateReadVariable(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_REDUCE_ALL:
                if (!ValidateReduceAll(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateReduceAllOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_REDUCE_MAX:
                if (!ValidateReduceMax(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateReduceMaxOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_REDUCE_MIN:
                if (!ValidateReduceMin(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateReduceMinOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_RELU:
                if (!ValidateRelu(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_RELU6:
                if (!ValidateRelu6(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_RESHAPE:
                if (!ValidateReshape(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_RESIZE_BILINEAR:
                if (!ValidateResizeBilinear(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateResizeBilinearOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR:
                if (!ValidateResizeNearestNeighbor(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateResizeNearestNeighborOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_REVERSE_V2:
                if (!ValidateReverseV2(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_RFFT2D:
                if (!ValidateRfft2D(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_ROUND:
                if (!ValidateRound(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_RSQRT:
                if (!ValidateRsqrt(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SELECT_V2:
                if (!ValidateSelectV2(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SHAPE:
                if (!ValidateShape(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateShapeOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SIN:
                if (!ValidateSin(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SLICE:
                if (!ValidateSlice(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SOFTMAX:
                if (!ValidateSoftmax(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SPACE_TO_BATCH_ND:
                if (!ValidateSpaceToBatchNd(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateSpaceToBatchNdOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SPACE_TO_DEPTH:
                if (!ValidateSpaceToDepth(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateSpaceToDepthOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SPLIT:
                if (!ValidateSplit(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateSplitOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SPLIT_V:
                if (!ValidateSplitV(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateSplitVOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SQRT:
                if (!ValidateSqrt(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SQUARE:
                if (!ValidateSquare(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SQUARED_DIFFERENCE:
                if (!ValidateSquaredDifference(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SQUEEZE:
                if (!ValidateSqueeze(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateSqueezeOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_STRIDED_SLICE:
                if (!ValidateStridedSlice(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateStridedSliceOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SUB:
                if (!ValidateSub(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateSubOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SUM:
                if (!ValidateSum(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_SVDF:
                if (!ValidateSvdf(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateSvdfOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_TANH:
                if (!ValidateTanh(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_TRANSPOSE:
                if (!ValidateTranspose(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateTransposeOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_TRANSPOSE_CONV:
                if (!ValidateTransposeConv(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateTransposeConvOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
                if (!ValidateUnidirectionalSequenceLSTM(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateUnidirectionalSequenceLSTMOptions(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_UNPACK:
                if (!ValidateUnpack(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_VAR_HANDLE:
                if (!ValidateVarHandle(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_WHILE:
                if (!ValidateWhile(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateWhileOptions(op_index, op)) {
                    valid = false;
                    continue;
                }
                if (!ValidateWhileSubgraphs(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_ZEROS_LIKE:
                if (!ValidateZerosLike(op_index, op)) {
                    valid = false;
                }
                break;
            case tflite::BuiltinOperator_CUSTOM:
                if (!ValidateCustom(op_index, op)) {
                    valid = false;
                }
                break;
            default:
                SetError(op_index, "unsupported operator");
                valid = false;
                break;
            }
        }
    }
    current_subgraph_ = nullptr;
    return valid;
}

bool ModelValidator::ValidateOperatorTensorReferences(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr) {
        SetError(operator_index, "missing subgraph");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (tensors == nullptr) {
        SetError(operator_index, "missing tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing input/output list");
        return false;
    }
    for (size_t i = 0; i < inputs->size(); ++i) {
        const int32_t tensor_index = inputs->Get(i);
        if (tensor_index < 0)
            continue;
        if (static_cast<size_t>(tensor_index) >= tensors->size()) {
            SetError(operator_index, "input tensor index out of range");
            return false;
        }
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
        const int32_t tensor_index = outputs->Get(i);
        if (tensor_index < 0)
            continue;
        if (static_cast<size_t>(tensor_index) >= tensors->size()) {
            SetError(operator_index, "output tensor index out of range");
            return false;
        }
    }
    return true;
}

void ModelValidator::SetError(size_t operator_index, const char* message) const
{
    std::ostringstream out;
    out << "Operator " << operator_index << ": " << message;
    last_error_ = out.str();
}

void ModelValidator::SetError(const char* message) const
{
    last_error_ = message ? message : "";
}

const std::string& ModelValidator::LastError() const
{
    return last_error_;
}

bool ModelValidator::ValidateAbs(size_t operator_index, const tflite::Operator* op) const
{
    if (!ValidateUnaryElementwise(operator_index, op))
        return false;
    const auto* options = op->builtin_options_as_AbsOptions();
    if (op->builtin_options_type() != tflite::BuiltinOptions_AbsOptions || options == nullptr) {
        SetError(operator_index, "invalid ABS options type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateAdd(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input1_index = inputs->Get(0);
    const int32_t input2_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input1_index < 0 || input2_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input1_index) >= tensors->size() || static_cast<size_t>(input2_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input1 = tensors->Get(static_cast<size_t>(input1_index));
    const auto* input2 = tensors->Get(static_cast<size_t>(input2_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input1 == nullptr || input2 == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input1->type() != input2->type() || input1->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (!ValidateBinaryBroadcast(operator_index, input1, input2, output)) {
        return false;
    }
    return true;
}

bool ModelValidator::ValidateAddOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_AddOptions) {
        SetError(operator_index, "invalid ADD options type");
        return false;
    }
    const auto* options = op->builtin_options_as_AddOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing ADD options");
        return false;
    }
    switch (options->fused_activation_function()) {
    case tflite::ActivationFunctionType_NONE:
    case tflite::ActivationFunctionType_RELU:
    case tflite::ActivationFunctionType_RELU6:
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
        break;
    default:
        SetError(operator_index, "invalid fused activation function");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateAddN(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() < 2) {
        SetError(operator_index, "requires at least two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t output_index = outputs->Get(0);
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "output tensor has no shape");
        return false;
    }
    const auto* first_input = static_cast<const tflite::Tensor*>(nullptr);
    for (size_t i = 0; i < inputs->size(); ++i) {
        const int32_t input_index = inputs->Get(i);
        if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
            SetError(operator_index, "contains an invalid input tensor index");
            return false;
        }
        const auto* input = tensors->Get(static_cast<size_t>(input_index));
        if (input == nullptr || input->shape() == nullptr) {
            SetError(operator_index, "input tensor has no shape");
            return false;
        }
        if (i == 0) {
            first_input = input;
            continue;
        }
        if (input->type() != first_input->type()) {
            SetError(operator_index, "input tensor types do not match");
            return false;
        }
        if (input->shape()->size() != first_input->shape()->size()) {
            SetError(operator_index, "input tensor ranks do not match");
            return false;
        }
        for (size_t j = 0; j < first_input->shape()->size(); ++j) {
            if (input->shape()->Get(j) != first_input->shape()->Get(j)) {
                SetError(operator_index, "input tensor shapes do not match");
                return false;
            }
        }
    }
    if (first_input == nullptr) {
        SetError(operator_index, "missing input tensor");
        return false;
    }
    if (output->type() != first_input->type()) {
        SetError(operator_index, "output tensor type does not match inputs");
        return false;
    }
    if (output->shape()->size() != first_input->shape()->size()) {
        SetError(operator_index, "output tensor rank does not match inputs");
        return false;
    }
    for (size_t i = 0; i < first_input->shape()->size(); ++i) {
        if (output->shape()->Get(i) != first_input->shape()->Get(i)) {
            SetError(operator_index, "output tensor shape does not match inputs");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateArgMax(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and axis tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t axis_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || axis_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(axis_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* axis = tensors->Get(static_cast<size_t>(axis_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || axis == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input->shape() == nullptr || axis->shape() == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (axis->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "axis tensor must be INT32");
        return false;
    }
    if (axis->shape()->size() > 1 || (axis->shape()->size() == 1 && axis->shape()->Get(0) != 1)) {
        SetError(operator_index, "axis tensor must be scalar");
        return false;
    }
    const size_t input_rank = input->shape()->size();
    if (input_rank == 0) {
        SetError(operator_index, "input tensor must have rank greater than zero");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_ArgMaxOptions) {
        SetError(operator_index, "invalid ARG_MAX options type");
        return false;
    }
    const auto* options = op->builtin_options_as_ArgMaxOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing ARG_MAX options");
        return false;
    }
    if (output->type() != options->output_type()) {
        SetError(operator_index, "output tensor type does not match ARG_MAX output type");
        return false;
    }
    if (output->type() != tflite::TensorType_INT32 && output->type() != tflite::TensorType_INT64) {
        SetError(operator_index, "unsupported ARG_MAX output type");
        return false;
    }
    if (output->shape()->size() != input_rank - 1) {
        SetError(operator_index, "output rank does not match ARG_MAX reduction");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateArgMin(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and axis tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t axis_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || axis_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(axis_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* axis = tensors->Get(static_cast<size_t>(axis_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || axis == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input->shape() == nullptr || axis->shape() == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (axis->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "axis tensor must be INT32");
        return false;
    }
    if (axis->shape()->size() > 1 || (axis->shape()->size() == 1 && axis->shape()->Get(0) != 1)) {
        SetError(operator_index, "axis tensor must be scalar");
        return false;
    }
    const size_t input_rank = input->shape()->size();
    if (input_rank == 0) {
        SetError(operator_index, "input tensor must have rank greater than zero");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_ArgMinOptions) {
        SetError(operator_index, "invalid ARG_MIN options type");
        return false;
    }
    const auto* options = op->builtin_options_as_ArgMinOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing ARG_MIN options");
        return false;
    }
    if (output->type() != options->output_type()) {
        SetError(operator_index, "output tensor type does not match ARG_MIN output type");
        return false;
    }
    if (output->type() != tflite::TensorType_INT32 && output->type() != tflite::TensorType_INT64) {
        SetError(operator_index, "unsupported ARG_MIN output type");
        return false;
    }
    if (output->shape()->size() != input_rank - 1) {
        SetError(operator_index, "output rank does not match ARG_MIN reduction");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateAssignVariable(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires resource handle and value tensors");
        return false;
    }
    if (outputs->size() != 0) {
        SetError(operator_index, "must not have output tensors");
        return false;
    }
    const int32_t resource_index = inputs->Get(0);
    const int32_t value_index = inputs->Get(1);
    if (resource_index < 0 || value_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(resource_index) >= tensors->size() || static_cast<size_t>(value_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* resource = tensors->Get(static_cast<size_t>(resource_index));
    const auto* value = tensors->Get(static_cast<size_t>(value_index));
    if (resource == nullptr || value == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (resource->type() != tflite::TensorType_RESOURCE) {
        SetError(operator_index, "resource handle tensor must be RESOURCE");
        return false;
    }
    if (value->shape() == nullptr) {
        SetError(operator_index, "value tensor has no shape");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateAssignVariableOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_AssignVariableOptions) {
        SetError(operator_index, "invalid ASSIGN_VARIABLE options type");
        return false;
    }
    const auto* options = op->builtin_options_as_AssignVariableOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing ASSIGN_VARIABLE options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateAveragePool2D(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input->shape() == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "input or output tensor has no shape");
        return false;
    }
    if (input->shape()->size() != 4 || output->shape()->size() != 4) {
        SetError(operator_index, "input and output tensors must be rank 4");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    for (size_t i = 0; i < 4; ++i) {
        if (input->shape()->Get(i) <= 0 || output->shape()->Get(i) <= 0) {
            SetError(operator_index, "contains invalid tensor dimension");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateAveragePool2DOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_Pool2DOptions) {
        SetError(operator_index, "invalid AVERAGE_POOL_2D options type");
        return false;
    }
    const auto* options = op->builtin_options_as_Pool2DOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing AVERAGE_POOL_2D options");
        return false;
    }
    if (options->stride_w() <= 0 || options->stride_h() <= 0) {
        SetError(operator_index, "pool stride must be positive");
        return false;
    }
    if (options->filter_width() <= 0 || options->filter_height() <= 0) {
        SetError(operator_index, "pool filter size must be positive");
        return false;
    }
    switch (options->padding()) {
    case tflite::Padding_VALID:
    case tflite::Padding_SAME:
        break;
    default:
        SetError(operator_index, "invalid pool padding");
        return false;
    }
    switch (options->fused_activation_function()) {
    case tflite::ActivationFunctionType_NONE:
    case tflite::ActivationFunctionType_RELU:
    case tflite::ActivationFunctionType_RELU6:
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
        break;
    default:
        SetError(operator_index, "invalid fused activation function");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateBatchMatMul(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input1_index = inputs->Get(0);
    const int32_t input2_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input1_index < 0 || input2_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input1_index) >= tensors->size() || static_cast<size_t>(input2_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input1 = tensors->Get(static_cast<size_t>(input1_index));
    const auto* input2 = tensors->Get(static_cast<size_t>(input2_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input1 == nullptr || input2 == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* shape1 = input1->shape();
    const auto* shape2 = input2->shape();
    const auto* output_shape = output->shape();
    if (shape1 == nullptr || shape2 == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (shape1->size() < 2 || shape2->size() < 2) {
        SetError(operator_index, "BATCH_MATMUL inputs must have rank at least two");
        return false;
    }
    if (output_shape->size() < 2) {
        SetError(operator_index, "BATCH_MATMUL output must have rank at least two");
        return false;
    }
    if (input1->type() != input2->type() || input1->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (shape1->size() != shape2->size() || shape1->size() != output_shape->size()) {
        SetError(operator_index, "BATCH_MATMUL tensor ranks do not match");
        return false;
    }
    const size_t rank = shape1->size();
    for (size_t i = 0; i + 2 < rank; ++i) {
        const int32_t dimension1 = shape1->Get(i);
        const int32_t dimension2 = shape2->Get(i);
        const int32_t output_dimension = output_shape->Get(i);
        if (dimension1 < 0 || dimension2 < 0 || output_dimension < 0) {
            SetError(operator_index, "contains a negative tensor dimension");
            return false;
        }
        if (dimension1 != dimension2 && dimension1 != 1 && dimension2 != 1) {
            SetError(operator_index, "batch dimensions are not broadcast compatible");
            return false;
        }
        const int32_t expected_dimension = dimension1 > dimension2 ? dimension1 : dimension2;
        if (output_dimension != expected_dimension) {
            SetError(operator_index, "output batch shape does not match broadcast shape");
            return false;
        }
    }
    const int32_t input1_rows = shape1->Get(rank - 2);
    const int32_t input1_columns = shape1->Get(rank - 1);
    const int32_t input2_rows = shape2->Get(rank - 2);
    const int32_t input2_columns = shape2->Get(rank - 1);
    const int32_t output_rows = output_shape->Get(rank - 2);
    const int32_t output_columns = output_shape->Get(rank - 1);
    if (input1_rows <= 0 || input1_columns <= 0 || input2_rows <= 0 || input2_columns <= 0 || output_rows <= 0 || output_columns <= 0) {
        SetError(operator_index, "contains an invalid matrix dimension");
        return false;
    }
    if (input1_columns != input2_rows) {
        SetError(operator_index, "matrix dimensions are incompatible");
        return false;
    }
    if (output_rows != input1_rows || output_columns != input2_columns) {
        SetError(operator_index, "output matrix shape does not match inputs");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateBatchToSpaceNd(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 3) {
        SetError(operator_index, "requires input, block shape and crops tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t block_shape_index = inputs->Get(1);
    const int32_t crops_index = inputs->Get(2);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || block_shape_index < 0 || crops_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(block_shape_index) >= tensors->size() || static_cast<size_t>(crops_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* block_shape = tensors->Get(static_cast<size_t>(block_shape_index));
    const auto* crops = tensors->Get(static_cast<size_t>(crops_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || block_shape == nullptr || crops == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* block_shape_shape = block_shape->shape();
    const auto* crops_shape = crops->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || block_shape_shape == nullptr || crops_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (block_shape->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "block shape tensor must be INT32");
        return false;
    }
    if (crops->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "crops tensor must be INT32");
        return false;
    }
    if (block_shape_shape->size() != 1) {
        SetError(operator_index, "block shape tensor must be one-dimensional");
        return false;
    }
    const int32_t block_rank = block_shape_shape->Get(0);
    if (block_rank <= 0) {
        SetError(operator_index, "block shape tensor must contain at least one dimension");
        return false;
    }
    if (crops_shape->size() != 2 || crops_shape->Get(0) != block_rank || crops_shape->Get(1) != 2) {
        SetError(operator_index, "crops tensor shape is invalid");
        return false;
    }
    if (input_shape->size() != static_cast<size_t>(block_rank) + 1) {
        SetError(operator_index, "input rank does not match block shape rank");
        return false;
    }
    if (output_shape->size() != input_shape->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateBatchToSpaceNdOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() == tflite::BuiltinOptions_NONE)
        return true;
    if (op->builtin_options_type() != tflite::BuiltinOptions_BatchToSpaceNDOptions) {
        SetError(operator_index, "invalid BATCH_TO_SPACE_ND options type");
        return false;
    }
    const auto* options = op->builtin_options_as_BatchToSpaceNDOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing BATCH_TO_SPACE_ND options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateBroadcastArgs(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input1_index = inputs->Get(0);
    const int32_t input2_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input1_index < 0 || input2_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input1_index) >= tensors->size() || static_cast<size_t>(input2_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input1 = tensors->Get(static_cast<size_t>(input1_index));
    const auto* input2 = tensors->Get(static_cast<size_t>(input2_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input1 == nullptr || input2 == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* shape1 = input1->shape();
    const auto* shape2 = input2->shape();
    const auto* output_shape = output->shape();
    if (shape1 == nullptr || shape2 == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input1->type() != tflite::TensorType_INT32 || input2->type() != tflite::TensorType_INT32 || output->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "BROADCAST_ARGS tensors must be INT32");
        return false;
    }
    if (shape1->size() != 1 || shape2->size() != 1 || output_shape->size() != 1) {
        SetError(operator_index, "BROADCAST_ARGS tensors must be one-dimensional");
        return false;
    }
    if (shape1->Get(0) <= 0 || shape2->Get(0) <= 0 || output_shape->Get(0) <= 0) {
        SetError(operator_index, "BROADCAST_ARGS tensors must have positive lengths");
        return false;
    }
    if (output_shape->Get(0) != std::max(shape1->Get(0), shape2->Get(0))) {
        SetError(operator_index, "BROADCAST_ARGS output shape is invalid");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateBroadcastTo(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and shape tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t shape_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || shape_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() ||static_cast<size_t>(shape_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* shape = tensors->Get(static_cast<size_t>(shape_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || shape == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* shape_shape = shape->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || shape_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (shape->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "shape tensor must be INT32");
        return false;
    }
    if (shape_shape->size() != 1) {
        SetError(operator_index, "shape tensor must be one-dimensional");
        return false;
    }
    if (shape_shape->Get(0) <= 0) {
        SetError(operator_index, "shape tensor must contain at least one dimension");
        return false;
    }
    if (output->type() != input->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (output_shape->size() != static_cast<size_t>(shape_shape->Get(0))) {
        SetError(operator_index, "output rank does not match shape tensor");
        return false;
    }
    const size_t input_rank = input_shape->size();
    const size_t output_rank = output_shape->size();
    if (input_rank > output_rank) {
        SetError(operator_index, "input rank exceeds output rank");
        return false;
    }
    const size_t rank_offset = output_rank - input_rank;
    for (size_t i = 0; i < input_rank; ++i) {
        const int32_t input_dimension = input_shape->Get(i);
        const int32_t output_dimension = output_shape->Get(rank_offset + i);
        if (input_dimension <= 0 || output_dimension <= 0) {
            SetError(operator_index, "contains an invalid tensor dimension");
            return false;
        }
        if (input_dimension != output_dimension && input_dimension != 1) {
            SetError(operator_index, "input shape is not broadcast compatible with output shape");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateCallOnce(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* model = document_.Model();
    if (model == nullptr || model->subgraphs() == nullptr) {
        SetError(operator_index, "missing model or subgraphs");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 0) {
        SetError(operator_index, "CALL_ONCE must not have input tensors");
        return false;
    }
    if (outputs->size() != 0) {
        SetError(operator_index, "CALL_ONCE must not have output tensors");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_CallOnceOptions) {
        SetError(operator_index, "invalid CALL_ONCE options type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateCallOnceOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_CallOnceOptions) {
        SetError(operator_index, "invalid CALL_ONCE options type");
        return false;
    }
    const auto* options = op->builtin_options_as_CallOnceOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing CALL_ONCE options");
        return false;
    }
    const auto* model = document_.Model();
    if (model == nullptr || model->subgraphs() == nullptr) {
        SetError(operator_index, "missing model or subgraphs");
        return false;
    }
    const int32_t subgraph_index = options->init_subgraph_index();
    if (subgraph_index < 0 || static_cast<size_t>(subgraph_index) >= model->subgraphs()->size()) {
        SetError(operator_index, "CALL_ONCE init subgraph index is out of range");
        return false;
    }
    if (model->subgraphs()->Get(static_cast<size_t>(subgraph_index)) == nullptr) {
        SetError(operator_index, "CALL_ONCE init subgraph is null");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateCallOnceSubgraph(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* model = document_.Model();
    if (model == nullptr || model->subgraphs() == nullptr) {
        SetError(operator_index, "missing model or subgraphs");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_CallOnceOptions) {
        SetError(operator_index, "invalid CALL_ONCE options type");
        return false;
    }
    const auto* options = op->builtin_options_as_CallOnceOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing CALL_ONCE options");
        return false;
    }
    const int32_t init_index = options->init_subgraph_index();
    if (init_index < 0 || static_cast<size_t>(init_index) >= model->subgraphs()->size()) {
        SetError(operator_index, "CALL_ONCE init subgraph index is out of range");
        return false;
    }
    const auto* init_subgraph = model->subgraphs()->Get(static_cast<size_t>(init_index));
    if (init_subgraph == nullptr) {
        SetError(operator_index, "CALL_ONCE init subgraph is null");
        return false;
    }
    const auto* inputs = init_subgraph->inputs();
    const auto* outputs = init_subgraph->outputs();
    const auto* operators = init_subgraph->operators();
    if (inputs == nullptr || outputs == nullptr || operators == nullptr) {
        SetError(operator_index, "CALL_ONCE init subgraph is missing tensors or operators");
        return false;
    }
    if (inputs->size() != 0 || outputs->size() != 0) {
        SetError(operator_index, "CALL_ONCE init subgraph must not have inputs or outputs");
        return false;
    }
    for (size_t i = 0; i < operators->size(); ++i) {
        const auto* init_op = operators->Get(i);
        if (init_op == nullptr) {
            SetError(operator_index, "CALL_ONCE init subgraph contains a null operator");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateCast(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input->shape() == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "input or output tensor has no shape");
        return false;
    }
    if (input->shape()->size() != output->shape()->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    for (size_t i = 0; i < input->shape()->size(); ++i) {
        if (input->shape()->Get(i) != output->shape()->Get(i)) {
            SetError(operator_index, "input and output shapes do not match");
            return false;
        }
    }
    if (input->type() == output->type()) {
        SetError(operator_index, "input and output types are identical");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_NONE && op->builtin_options_type() != tflite::BuiltinOptions_CastOptions) {
        SetError(operator_index, "invalid CAST options type");
        return false;
    }
    if (op->builtin_options_type() == tflite::BuiltinOptions_CastOptions) {
        const auto* options = op->builtin_options_as_CastOptions();
        if (options == nullptr) {
            SetError(operator_index, "missing CAST options");
            return false;
        }
        if (options->in_data_type() != input->type()) {
            SetError(operator_index, "CAST input type does not match tensor type");
            return false;
        }
        if (options->out_data_type() != output->type()) {
            SetError(operator_index, "CAST output type does not match tensor type");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateCeil(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateUnaryElementwise(operator_index, op);
}

bool ModelValidator::ValidateCircularBuffer(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "CIRCULAR_BUFFER requires exactly one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "CIRCULAR_BUFFER requires exactly one output tensor");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid input tensor index");
        return false;
    }
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || input->shape() == nullptr) {
        SetError(operator_index, "contains an invalid input tensor");
        return false;
    }
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (input->shape()->size() < 4 || output->shape()->size() < 4) {
        SetError(operator_index, "CIRCULAR_BUFFER tensors must have rank at least four");
        return false;
    }
    if (input->shape()->Get(0) != output->shape()->Get(0)) {
        SetError(operator_index, "CIRCULAR_BUFFER dimension 0 does not match");
        return false;
    }
    if (input->shape()->Get(1) != 1) {
        SetError(operator_index, "CIRCULAR_BUFFER input dimension 1 must be 1");
        return false;
    }
    if (input->shape()->Get(2) != output->shape()->Get(2)) {
        SetError(operator_index, "CIRCULAR_BUFFER dimension 2 does not match");
        return false;
    }
    if (input->shape()->Get(3) != output->shape()->Get(3)) {
        SetError(operator_index, "CIRCULAR_BUFFER dimension 3 does not match");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "CIRCULAR_BUFFER input and output tensor types do not match");
        return false;
    }
    if (input->type() != tflite::TensorType_INT8) {
        SetError(operator_index, "CIRCULAR_BUFFER tensors must be INT8");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateConcatenation(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() < 2) {
        SetError(operator_index, "requires at least two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t output_index = outputs->Get(0);
    if (output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range output tensor index");
        return false;
    }
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (output == nullptr) {
        SetError(operator_index, "contains null output tensor");
        return false;
    }
    if (output->shape() == nullptr || output->shape()->size() == 0) {
        SetError(operator_index, "output tensor has no shape");
        return false;
    }
    const auto* first_input = tensors->Get(static_cast<size_t>(inputs->Get(0)));
    if (inputs->Get(0) < 0 || static_cast<size_t>(inputs->Get(0)) >= tensors->size() || first_input == nullptr) {
        SetError(operator_index, "contains invalid input tensor");
        return false;
    }
    if (first_input->shape() == nullptr || first_input->shape()->size() == 0) {
        SetError(operator_index, "input tensor has no shape");
        return false;
    }
    const size_t rank = first_input->shape()->size();
    if (output->shape()->size() != rank) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    const auto* options = op->builtin_options_as_ConcatenationOptions();
    if (op->builtin_options_type() != tflite::BuiltinOptions_ConcatenationOptions || options == nullptr) {
        SetError(operator_index, "invalid CONCATENATION options");
        return false;
    }
    int32_t axis = options->axis();
    if (axis < 0)
        axis += static_cast<int32_t>(rank);
    if (axis < 0 || static_cast<size_t>(axis) >= rank) {
        SetError(operator_index, "CONCATENATION axis is out of range");
        return false;
    }
    int64_t expected_axis_size = 0;
    for (size_t i = 0; i < inputs->size(); ++i) {
        const int32_t input_index = inputs->Get(i);
        if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
            SetError(operator_index, "contains an invalid input tensor index");
            return false;
        }
        const auto* input = tensors->Get(static_cast<size_t>(input_index));
        if (input == nullptr || input->shape() == nullptr) {
            SetError(operator_index, "contains an invalid input tensor");
            return false;
        }
        if (input->type() != first_input->type()) {
            SetError(operator_index, "input tensor types do not match");
            return false;
        }
        if (input->shape()->size() != rank) {
            SetError(operator_index, "input tensor ranks do not match");
            return false;
        }
        for (size_t j = 0; j < rank; ++j) {
            const int32_t dimension = input->shape()->Get(j);
            if (dimension <= 0) {
                SetError(operator_index, "contains invalid input tensor dimension");
                return false;
            }
            if (j == static_cast<size_t>(axis)) {
                expected_axis_size += dimension;
            } else if (dimension != first_input->shape()->Get(j)) {
                SetError(operator_index, "input tensor dimensions do not match");
                return false;
            }
        }
    }
    if (output->type() != first_input->type()) {
        SetError(operator_index, "output tensor type does not match inputs");
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        const int32_t output_dimension = output->shape()->Get(i);
        if (output_dimension <= 0) {
            SetError(operator_index, "contains invalid output tensor dimension");
            return false;
        }
        const int32_t expected_dimension = i == static_cast<size_t>(axis) ? static_cast<int32_t>(expected_axis_size) : first_input->shape()->Get(i);
        if (output_dimension != expected_dimension) {
            SetError(operator_index, "output shape does not match concatenated input shapes");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateConcatenationOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_ConcatenationOptions) {
        SetError(operator_index, "invalid CONCATENATION options type");
        return false;
    }
    const auto* options = op->builtin_options_as_ConcatenationOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing CONCATENATION options");
        return false;
    }
    switch (options->fused_activation_function()) {
    case tflite::ActivationFunctionType_NONE:
    case tflite::ActivationFunctionType_RELU:
    case tflite::ActivationFunctionType_RELU6:
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
        break;
    default:
        SetError(operator_index, "invalid fused activation function");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateConv2D(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() < 3) {
        SetError(operator_index, "requires input, weights and bias tensors");
        return false;
    }
    if (outputs->size() < 1) {
        SetError(operator_index, "has no output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t weights_index = inputs->Get(1);
    const int32_t bias_index = inputs->Get(2);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || weights_index < 0 || bias_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(weights_index) >= tensors->size() || static_cast<size_t>(bias_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* weights = tensors->Get(static_cast<size_t>(weights_index));
    const auto* bias = tensors->Get(static_cast<size_t>(bias_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || weights == nullptr || bias == nullptr || output == nullptr) {
        SetError(operator_index, "contains a null tensor");
        return false;
    }
    if (input->shape() == nullptr || weights->shape() == nullptr || bias->shape() == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "missing tensor shape");
        return false;
    }
    if (input->shape()->size() != 4) {
        SetError(operator_index, "input tensor must be 4-dimensional");
        return false;
    }
    if (weights->shape()->size() != 4) {
        SetError(operator_index, "weights tensor must be 4-dimensional");
        return false;
    }
    if (bias->shape()->size() != 1) {
        SetError(operator_index, "bias tensor must be 1-dimensional");
        return false;
    }
    if (output->shape()->size() != 4) {
        SetError(operator_index, "output tensor must be 4-dimensional");
        return false;
    }
    if (output->type() != input->type()) {
        SetError(operator_index, "input and output types do not match");
        return false;
    }
    if (input->type() == tflite::TensorType_INT16) {
        if (weights->type() != tflite::TensorType_INT8) {
            SetError(operator_index, "INT16 input requires INT8 weights");
            return false;
        }
        if (bias->type() != tflite::TensorType_INT64) {
            SetError(operator_index, "INT16 input requires INT64 bias");
            return false;
        }
    } else {
        if (input->type() != weights->type()) {
            SetError(operator_index, "input and weights types do not match");
            return false;
        }
        if (bias->type() != tflite::TensorType_INT32 &&
            bias->type() != input->type()) {
            SetError(operator_index, "bias type is incompatible with input type");
            return false;
        }
    }
    const int32_t input_channels = input->shape()->Get(3);
    const int32_t output_channels = weights->shape()->Get(0);
    const int32_t weight_input_channels = weights->shape()->Get(3);
    const int32_t bias_channels = bias->shape()->Get(0);
    const int32_t output_depth = output->shape()->Get(3);
    if (input_channels != weight_input_channels) {
        SetError(operator_index, "input channels do not match weights input channels");
        return false;
    }
    if (output_channels != bias_channels) {
        SetError(operator_index, "weights output channels do not match bias size");
        return false;
    }
    if (output_channels != output_depth) {
        SetError(operator_index, "weights output channels do not match output channels");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateConv2DOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_Conv2DOptions) {
        SetError(operator_index, "invalid builtin options type");
        return false;
    }
    const auto* options = op->builtin_options_as_Conv2DOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing CONV_2D options");
        return false;
    }
    if (options->padding() != tflite::Padding_SAME && options->padding() != tflite::Padding_VALID) {
        SetError(operator_index, "invalid padding");
        return false;
    }
    if (options->stride_w() <= 0 || options->stride_h() <= 0) {
        SetError(operator_index, "stride must be positive");
        return false;
    }
    if (options->dilation_w_factor() <= 0 || options->dilation_h_factor() <= 0) {
        SetError(operator_index, "dilation factor must be positive");
        return false;
    }
    switch (options->fused_activation_function()) {
    case tflite::ActivationFunctionType_NONE:
    case tflite::ActivationFunctionType_RELU:
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
    case tflite::ActivationFunctionType_RELU6:
    case tflite::ActivationFunctionType_TANH:
    case tflite::ActivationFunctionType_SIGN_BIT:
        return true;
    default:
        SetError(operator_index, "invalid fused activation function");
        return false;
    }
}

bool ModelValidator::ValidateCos(size_t operator_index, const tflite::Operator* op) const
{
    if (!ValidateUnaryElementwise(operator_index, op))
        return false;
    if (op->builtin_options_type() != tflite::BuiltinOptions_CosOptions) {
        SetError(operator_index, "invalid COS options type");
        return false;
    }
    if (op->builtin_options_as_CosOptions() == nullptr) {
        SetError(operator_index, "missing COS options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateCumsum(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and axis tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t axis_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || axis_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(axis_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* axis = tensors->Get(static_cast<size_t>(axis_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || axis == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* axis_shape = axis->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || axis_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input_shape->size() == 0) {
        SetError(operator_index, "input tensor must have rank greater than zero");
        return false;
    }
    if (axis->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "axis tensor must be INT32");
        return false;
    }
    if (axis_shape->size() != 0 &&
        (axis_shape->size() != 1 || axis_shape->Get(0) != 1)) {
        SetError(operator_index, "axis tensor must be scalar");
        return false;
    }
    if (output->type() != input->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    switch (input->type()) {
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_INT32:
    case tflite::TensorType_INT64:
        break;
    default:
        SetError(operator_index, "unsupported CUMSUM tensor type");
        return false;
    }
    if (output_shape->size() != input_shape->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    for (size_t i = 0; i < input_shape->size(); ++i) {
        if (output_shape->Get(i) != input_shape->Get(i)) {
            SetError(operator_index, "input and output shapes do not match");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateCumsumOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_CumsumOptions) {
        SetError(operator_index, "invalid CUMSUM options type");
        return false;
    }
    const auto* options = op->builtin_options_as_CumsumOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing CUMSUM options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateDecode(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (outputs->size() == 0) {
        SetError(operator_index, "DECODE requires at least one output tensor");
        return false;
    }
    if (inputs->size() != outputs->size() * 2) {
        SetError(operator_index, "DECODE requires exactly two inputs per output tensor");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const auto* buffers = document_.Buffers();
    if (buffers == nullptr) {
        SetError(operator_index, "missing buffers");
        return false;
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
        const int32_t input_index = inputs->Get(i * 2);
        const int32_t ancillary_index = inputs->Get(i * 2 + 1);
        const int32_t output_index = outputs->Get(i);
        if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
            SetError(operator_index, "contains an invalid input tensor index");
            return false;
        }
        if (ancillary_index < 0 || static_cast<size_t>(ancillary_index) >= tensors->size()) {
            SetError(operator_index, "contains an invalid ancillary tensor index");
            return false;
        }
        if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
            SetError(operator_index, "contains an invalid output tensor index");
            return false;
        }
        const auto* input = tensors->Get(static_cast<size_t>(input_index));
        const auto* ancillary = tensors->Get(static_cast<size_t>(ancillary_index));
        const auto* output = tensors->Get(static_cast<size_t>(output_index));
        if (input == nullptr || ancillary == nullptr || output == nullptr) {
            SetError(operator_index, "contains a null tensor");
            return false;
        }
        if (input->shape() == nullptr || ancillary->shape() == nullptr || output->shape() == nullptr) {
            SetError(operator_index, "contains a tensor without shape");
            return false;
        }
        const size_t input_buffer_index = static_cast<size_t>(input->buffer());
        const size_t ancillary_buffer_index = static_cast<size_t>(ancillary->buffer());
        if (input_buffer_index >= buffers->size()) {
            SetError(operator_index, "DECODE input tensor has an invalid buffer");
            return false;
        }
        if (ancillary_buffer_index >= buffers->size()) {
            SetError(operator_index, "DECODE ancillary tensor has an invalid buffer");
            return false;
        }
        const auto* input_buffer = buffers->Get(input_buffer_index);
        const auto* ancillary_buffer = buffers->Get(ancillary_buffer_index);
        if (input_buffer == nullptr || input_buffer->data() == nullptr) {
            SetError(operator_index, "DECODE input tensor has no constant buffer data");
            return false;
        }
        if (ancillary_buffer == nullptr || ancillary_buffer->data() == nullptr) {
            SetError(operator_index, "DECODE ancillary tensor has no constant buffer data");
            return false;
        }
        if (ancillary_buffer->data()->size() < 16) {
            SetError(operator_index, "DECODE ancillary buffer is too small");
            return false;
        }
        const uint8_t type = ancillary_buffer->data()->Get(0);
        if (ancillary_buffer->data()->Get(4) != 1) {
            SetError(operator_index, "DECODE ancillary version must be 1");
            return false;
        }
        switch (type) {
        case 0: {
            const uint8_t bit_width = ancillary_buffer->data()->Get(5) & 0x07;
            if (bit_width == 0 || bit_width > 7) {
                SetError(operator_index, "DECODE LUT bit width is invalid");
                return false;
            }
            const uint8_t value_table_stride = ancillary_buffer->data()->Get(6);
            if (value_table_stride == 0) {
                SetError(operator_index, "DECODE LUT value table stride is invalid");
                return false;
            }
            break;
        }
        case 1: {
            const uint8_t table_size = ancillary_buffer->data()->Get(5);
            const bool use_32bit_table = (table_size & 0x01) != 0;
            if (!use_32bit_table && output->type() != tflite::TensorType_INT8) {
                SetError(operator_index, "DECODE Huffman 16-bit table requires INT8 output");
                return false;
            }
            break;
        }
        case 2: {
            const auto* quantization = output->quantization();
            if (quantization == nullptr)
                break;
            const auto* scales = quantization->scale();
            const auto* zero_points = quantization->zero_point();
            if (scales == nullptr || zero_points == nullptr)
                break;
            const size_t num_channels = scales->size();
            const int32_t quantized_dimension = quantization->quantized_dimension();
            const int32_t last_dimension = static_cast<int32_t>(output->shape()->size()) - 1;
            if (quantized_dimension != 0 &&
                !(quantized_dimension == last_dimension && num_channels > 1)) {
                SetError(operator_index, "DECODE Prune quantization axis is invalid");
                return false;
            }
            if (num_channels != zero_points->size()) {
                SetError(operator_index, "DECODE Prune quantization scales and zero points do not match");
                return false;
            }
            break;
        }
        case 127:
            SetError(operator_index, "DECODE custom type is not supported");
            return false;
        default:
            SetError(operator_index, "DECODE ancillary type is invalid");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateDelay(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "DELAY requires exactly one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "DELAY requires exactly one output tensor");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid input tensor index");
        return false;
    }
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || input->shape() == nullptr) {
        SetError(operator_index, "contains an invalid input tensor");
        return false;
    }
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (input->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "DELAY input tensor must be INT16");
        return false;
    }
    if (output->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "DELAY output tensor must be INT16");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateDepthToSpace(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input_shape->size() != 4) {
        SetError(operator_index, "input tensor must be 4-dimensional");
        return false;
    }
    if (output_shape->size() != 4) {
        SetError(operator_index, "output tensor must be 4-dimensional");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_DepthToSpaceOptions) {
        SetError(operator_index, "invalid DEPTH_TO_SPACE options type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateDepthToSpaceOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_DepthToSpaceOptions) {
        SetError(operator_index, "invalid DEPTH_TO_SPACE options type");
        return false;
    }
    const auto* options = op->builtin_options_as_DepthToSpaceOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing DEPTH_TO_SPACE options");
        return false;
    }
    if (options->block_size() <= 1) {
        SetError(operator_index, "DEPTH_TO_SPACE block size must be greater than one");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateDepthwiseConv2D(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() < 3) {
        SetError(operator_index, "requires input, weights and bias tensors");
        return false;
    }
    if (outputs->size() < 1) {
        SetError(operator_index, "has no output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t weights_index = inputs->Get(1);
    const int32_t bias_index = inputs->Get(2);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || weights_index < 0 || bias_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(weights_index) >= tensors->size() || static_cast<size_t>(bias_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* weights = tensors->Get(static_cast<size_t>(weights_index));
    const auto* bias = tensors->Get(static_cast<size_t>(bias_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || weights == nullptr || bias == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* weights_shape = weights->shape();
    const auto* bias_shape = bias->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || weights_shape == nullptr || bias_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "missing tensor shape");
        return false;
    }
    if (input_shape->size() != 4) {
        SetError(operator_index, "input tensor must be 4-dimensional");
        return false;
    }
    if (weights_shape->size() != 4) {
        SetError(operator_index, "weights tensor must be 4-dimensional");
        return false;
    }
    if (bias_shape->size() != 1) {
        SetError(operator_index, "bias tensor must be 1-dimensional");
        return false;
    }
    if (output_shape->size() != 4) {
        SetError(operator_index, "output tensor must be 4-dimensional");
        return false;
    }
    const auto* options = op->builtin_options_as_DepthwiseConv2DOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing DEPTHWISE_CONV_2D options");
        return false;
    }
    const size_t input_channels = static_cast<size_t>(input_shape->Get(3));
    const size_t depth_multiplier = static_cast<size_t>(options->depth_multiplier());
    const size_t output_channels = static_cast<size_t>(weights_shape->Get(3));
    if (static_cast<size_t>(weights_shape->Get(0)) != 1) {
        SetError(operator_index, "weights first dimension must be 1");
        return false;
    }
    if (weights_shape->Get(3) != static_cast<int32_t>(input_channels * depth_multiplier)) {
        SetError(operator_index, "weights output channels do not match input channels and depth multiplier");
        return false;
    }
    if (static_cast<size_t>(bias_shape->Get(0)) != output_channels) {
        SetError(operator_index, "bias size does not match output channels");
        return false;
    }
    if (static_cast<size_t>(output_shape->Get(3)) != output_channels) {
        SetError(operator_index, "output channels do not match weights output channels");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateDepthwiseConv2DOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_DepthwiseConv2DOptions) {
        SetError(operator_index, "invalid builtin options type");
        return false;
    }
    const auto* options = op->builtin_options_as_DepthwiseConv2DOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing DEPTHWISE_CONV_2D options");
        return false;
    }
    if (options->stride_w() <= 0 || options->stride_h() <= 0) {
        SetError(operator_index, "stride must be positive");
        return false;
    }
    if (options->dilation_w_factor() <= 0 || options->dilation_h_factor() <= 0) {
        SetError(operator_index, "dilation must be positive");
        return false;
    }
    switch (options->fused_activation_function()) {
    case tflite::ActivationFunctionType_NONE:
    case tflite::ActivationFunctionType_RELU:
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
    case tflite::ActivationFunctionType_RELU6:
    case tflite::ActivationFunctionType_TANH:
    case tflite::ActivationFunctionType_SIGN_BIT:
        return true;
    default:
        SetError(operator_index, "invalid fused activation function");
        return false;
    }
}

bool ModelValidator::ValidateDequantize(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input->shape() == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input->shape()->size() != output->shape()->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    for (size_t i = 0; i < input->shape()->size(); ++i) {
        if (input->shape()->Get(i) != output->shape()->Get(i)) {
            SetError(operator_index, "input and output shapes do not match");
            return false;
        }
    }
    if (op->builtin_options_type() !=
        tflite::BuiltinOptions_DequantizeOptions) {
        SetError(operator_index, "invalid DEQUANTIZE options type");
        return false;
    }
    if (op->builtin_options_as_DequantizeOptions() == nullptr) {
        SetError(operator_index, "missing DEQUANTIZE options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateDetectionPostprocess(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 3) {
        SetError(operator_index, "requires box encodings, class predictions and anchors tensors");
        return false;
    }
    if (outputs->size() != 4) {
        SetError(operator_index, "requires four output tensors");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t box_encodings_index = inputs->Get(0);
    const int32_t class_predictions_index = inputs->Get(1);
    const int32_t anchors_index = inputs->Get(2);
    if (box_encodings_index < 0 || static_cast<size_t>(box_encodings_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid box encodings tensor index");
        return false;
    }
    if (class_predictions_index < 0 || static_cast<size_t>(class_predictions_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid class predictions tensor index");
        return false;
    }
    if (anchors_index < 0 || static_cast<size_t>(anchors_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid anchors tensor index");
        return false;
    }
    const auto* box_encodings = tensors->Get(static_cast<size_t>(box_encodings_index));
    const auto* class_predictions = tensors->Get(static_cast<size_t>(class_predictions_index));
    const auto* anchors = tensors->Get(static_cast<size_t>(anchors_index));
    if (box_encodings == nullptr || box_encodings->shape() == nullptr) {
        SetError(operator_index, "contains an invalid box encodings tensor");
        return false;
    }
    if (class_predictions == nullptr || class_predictions->shape() == nullptr) {
        SetError(operator_index, "contains an invalid class predictions tensor");
        return false;
    }
    if (anchors == nullptr || anchors->shape() == nullptr) {
        SetError(operator_index, "contains an invalid anchors tensor");
        return false;
    }
    if (box_encodings->shape()->size() != 3) {
        SetError(operator_index, "DETECTION_POSTPROCESS box encodings must be 3D");
        return false;
    }
    if (class_predictions->shape()->size() != 3) {
        SetError(operator_index, "DETECTION_POSTPROCESS class predictions must be 3D");
        return false;
    }
    if (anchors->shape()->size() != 2) {
        SetError(operator_index, "DETECTION_POSTPROCESS anchors must be 2D");
        return false;
    }
    if (box_encodings->shape()->Get(0) != 1) {
        SetError(operator_index, "DETECTION_POSTPROCESS box encodings batch size must be one");
        return false;
    }
    if (class_predictions->shape()->Get(0) != 1) {
        SetError(operator_index, "DETECTION_POSTPROCESS class predictions batch size must be one");
        return false;
    }
    if (box_encodings->shape()->Get(1) <= 0) {
        SetError(operator_index, "DETECTION_POSTPROCESS box count must be positive");
        return false;
    }
    if (box_encodings->shape()->Get(2) < 4) {
        SetError(operator_index, "DETECTION_POSTPROCESS box encodings must contain at least four coordinates");
        return false;
    }
    if (class_predictions->shape()->Get(1) != box_encodings->shape()->Get(1)) {
        SetError(operator_index, "DETECTION_POSTPROCESS class prediction count does not match box count");
        return false;
    }
    if (anchors->shape()->Get(0) != box_encodings->shape()->Get(1)) {
        SetError(operator_index, "DETECTION_POSTPROCESS anchors do not match box count");
        return false;
    }
    if (box_encodings->type() != tflite::TensorType_FLOAT32) {
        SetError(operator_index, "DETECTION_POSTPROCESS box encodings must be FLOAT32");
        return false;
    }
    if (class_predictions->type() != tflite::TensorType_FLOAT32) {
        SetError(operator_index, "DETECTION_POSTPROCESS class predictions must be FLOAT32");
        return false;
    }
    if (anchors->type() != tflite::TensorType_FLOAT32) {
        SetError(operator_index, "DETECTION_POSTPROCESS anchors must be FLOAT32");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateDiv(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input1_index = inputs->Get(0);
    const int32_t input2_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input1_index < 0 || input2_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input1_index) >= tensors->size() || static_cast<size_t>(input2_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input1 = tensors->Get(static_cast<size_t>(input1_index));
    const auto* input2 = tensors->Get(static_cast<size_t>(input2_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input1 == nullptr || input2 == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input1->type() != input2->type() || input1->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (!ValidateBinaryBroadcast(operator_index, input1, input2, output)) {
        return false;
    }
    return true;
}

bool ModelValidator::ValidateDivOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_DivOptions) {
        SetError(operator_index, "invalid DIV options type");
        return false;
    }
    const auto* options = op->builtin_options_as_DivOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing DIV options");
        return false;
    }
    switch (options->fused_activation_function()) {
    case tflite::ActivationFunctionType_NONE:
    case tflite::ActivationFunctionType_RELU:
    case tflite::ActivationFunctionType_RELU6:
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
        break;
    default:
        SetError(operator_index, "invalid fused activation function");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateDynamicUpdateSlice(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 3) {
        SetError(operator_index, "requires input, update and start indices tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t update_index = inputs->Get(1);
    const int32_t start_indices_index = inputs->Get(2);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || update_index < 0 || start_indices_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(update_index) >= tensors->size() || static_cast<size_t>(start_indices_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* update = tensors->Get(static_cast<size_t>(update_index));
    const auto* start_indices = tensors->Get(static_cast<size_t>(start_indices_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || update == nullptr || start_indices == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* update_shape = update->shape();
    const auto* start_indices_shape = start_indices->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || update_shape == nullptr || start_indices_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input->type() != update->type() || input->type() != output->type()) {
        SetError(operator_index, "input, update and output tensor types do not match");
        return false;
    }
    if (start_indices->type() != tflite::TensorType_INT32 && start_indices->type() != tflite::TensorType_INT64) {
        SetError(operator_index, "start indices tensor must be INT32 or INT64");
        return false;
    }
    const size_t rank = input_shape->size();
    if (rank == 0) {
        SetError(operator_index, "input tensor must have rank greater than zero");
        return false;
    }
    if (update_shape->size() != rank) {
        SetError(operator_index, "update tensor rank does not match input rank");
        return false;
    }
    if (output_shape->size() != rank) {
        SetError(operator_index, "output tensor rank does not match input rank");
        return false;
    }
    if (start_indices_shape->size() != 1 || start_indices_shape->Get(0) != static_cast<int32_t>(rank)) {
        SetError(operator_index, "start indices tensor must have shape [input rank]");
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        const int32_t input_dimension = input_shape->Get(i);
        const int32_t update_dimension = update_shape->Get(i);
        const int32_t output_dimension = output_shape->Get(i);
        if (input_dimension <= 0 || update_dimension <= 0 || output_dimension <= 0) {
            SetError(operator_index, "contains invalid tensor dimension");
            return false;
        }
        if (update_dimension > input_dimension) {
            SetError(operator_index, "update tensor dimension exceeds input dimension");
            return false;
        }
        if (output_dimension != input_dimension) {
            SetError(operator_index, "output shape does not match input shape");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateDynamicUpdateSliceOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_DynamicUpdateSliceOptions) {
        SetError(operator_index, "invalid DYNAMIC_UPDATE_SLICE options type");
        return false;
    }
    const auto* options = op->builtin_options_as_DynamicUpdateSliceOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing DYNAMIC_UPDATE_SLICE options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateElu(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateUnaryElementwise(operator_index, op);
}

bool ModelValidator::ValidateEmbeddingLookup(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires ids and weights tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t ids_index = inputs->Get(0);
    const int32_t weights_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (ids_index < 0 || weights_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(ids_index) >= tensors->size() || static_cast<size_t>(weights_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* ids = tensors->Get(static_cast<size_t>(ids_index));
    const auto* weights = tensors->Get(static_cast<size_t>(weights_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (ids == nullptr || weights == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* ids_shape = ids->shape();
    const auto* weights_shape = weights->shape();
    const auto* output_shape = output->shape();
    if (ids_shape == nullptr || weights_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (ids->type() != tflite::TensorType_INT32 && ids->type() != tflite::TensorType_INT64) {
        SetError(operator_index, "ids tensor must be INT32 or INT64");
        return false;
    }
    if (weights_shape->size() < 2) {
        SetError(operator_index, "weights tensor must have rank at least two");
        return false;
    }
    if (output_shape->size() != ids_shape->size() + weights_shape->size() - 1) {
        SetError(operator_index, "output rank does not match EMBEDDING_LOOKUP");
        return false;
    }
    if (output->type() != weights->type()) {
        SetError(operator_index, "output and weights types do not match");
        return false;
    }
    for (size_t i = 0; i < ids_shape->size(); ++i) {
        if (output_shape->Get(i) != ids_shape->Get(i)) {
            SetError(operator_index, "output shape does not match ids shape");
            return false;
        }
    }
    for (size_t i = 1; i < weights_shape->size(); ++i) {
        if (output_shape->Get(ids_shape->size() + i - 1) != weights_shape->Get(i)) {
            SetError(operator_index, "output shape does not match weights shape");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateEnergy(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "ENERGY requires exactly one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "ENERGY requires exactly one output tensor");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid input tensor index");
        return false;
    }
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || input->shape() == nullptr) {
        SetError(operator_index, "contains an invalid input tensor");
        return false;
    }
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (input->shape()->size() != 1) {
        SetError(operator_index, "ENERGY input tensor must be 1D");
        return false;
    }
    if (output->shape()->size() != 1) {
        SetError(operator_index, "ENERGY output tensor must be 1D");
        return false;
    }
    if (input->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "ENERGY input tensor must be INT16");
        return false;
    }
    if (output->type() != tflite::TensorType_UINT32) {
        SetError(operator_index, "ENERGY output tensor must be UINT32");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateEqual(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateComparison(operator_index, op);
}

bool ModelValidator::ValidateEthosU(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() == 0) {
        SetError(operator_index, "ETHOS_U requires at least one input tensor");
        return false;
    }
    const auto* custom_options = op->custom_options();
    if (custom_options == nullptr || custom_options->size() == 0) {
        SetError(operator_index, "ETHOS_U requires custom options");
        return false;
    }
    const uint8_t* custom_data = custom_options->Data();
    const size_t custom_data_size = custom_options->size();
    auto root = flexbuffers::GetRoot(custom_data, custom_data_size);
    if (root.IsNull()) {
        SetError(operator_index, "ETHOS_U custom options are invalid");
        return false;
    }
    if (root.AsInt8() != 1) {
        SetError(operator_index, "ETHOS_U custom option type is invalid");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    for (size_t i = 0; i < inputs->size(); ++i) {
        const int32_t input_index = inputs->Get(i);
        if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
            SetError(operator_index, "contains an invalid input tensor index");
            return false;
        }
        const auto* input = tensors->Get(static_cast<size_t>(input_index));
        if (input == nullptr || input->shape() == nullptr) {
            SetError(operator_index, "contains an invalid input tensor");
            return false;
        }
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
        const int32_t output_index = outputs->Get(i);
        if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
            SetError(operator_index, "contains an invalid output tensor index");
            return false;
        }
        const auto* output = tensors->Get(static_cast<size_t>(output_index));
        if (output == nullptr || output->shape() == nullptr) {
            SetError(operator_index, "contains an invalid output tensor");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateExp(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateUnaryElementwise(operator_index, op);
}

bool ModelValidator::ValidateExpandDims(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and axis tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t axis_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || axis_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(axis_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* axis = tensors->Get(static_cast<size_t>(axis_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || axis == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input->shape() == nullptr || axis->shape() == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (axis->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "axis tensor must be INT32");
        return false;
    }
    if (axis->shape()->size() != 0 && (axis->shape()->size() != 1 || axis->shape()->Get(0) != 1)) {
        SetError(operator_index, "axis tensor must be scalar");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    const size_t input_rank = input->shape()->size();
    const size_t output_rank = output->shape()->size();
    if (output_rank != input_rank + 1) {
        SetError(operator_index, "output rank must be input rank plus one");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateFftAutoScale(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "FFT_AUTO_SCALE requires exactly one input tensor");
        return false;
    }
    if (outputs->size() != 2) {
        SetError(operator_index, "FFT_AUTO_SCALE requires exactly two output tensors");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    const int32_t scale_bit_index = outputs->Get(1);
    if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid input tensor index");
        return false;
    }
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    if (scale_bit_index < 0 || static_cast<size_t>(scale_bit_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid scale bit tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    const auto* scale_bit = tensors->Get(static_cast<size_t>(scale_bit_index));
    if (input == nullptr || input->shape() == nullptr) {
        SetError(operator_index, "contains an invalid input tensor");
        return false;
    }
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (scale_bit == nullptr || scale_bit->shape() == nullptr) {
        SetError(operator_index, "contains an invalid scale bit tensor");
        return false;
    }
    if (input->shape()->size() != 1) {
        SetError(operator_index, "FFT_AUTO_SCALE input tensor must be 1D");
        return false;
    }
    if (output->shape()->size() != 1) {
        SetError(operator_index, "FFT_AUTO_SCALE output tensor must be 1D");
        return false;
    }
    if (scale_bit->shape()->size() != 0) {
        SetError(operator_index, "FFT_AUTO_SCALE scale bit tensor must be scalar");
        return false;
    }
    if (input->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "FFT_AUTO_SCALE input tensor must be INT16");
        return false;
    }
    if (output->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "FFT_AUTO_SCALE output tensor must be INT16");
        return false;
    }
    if (scale_bit->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "FFT_AUTO_SCALE scale bit tensor must be INT32");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateFill(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires shape and value tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t shape_index = inputs->Get(0);
    const int32_t value_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (shape_index < 0 || value_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(shape_index) >= tensors->size() || static_cast<size_t>(value_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* shape = tensors->Get(static_cast<size_t>(shape_index));
    const auto* value = tensors->Get(static_cast<size_t>(value_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (shape == nullptr || value == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (shape->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "FILL shape tensor must be INT32");
        return false;
    }
    if (shape->shape() == nullptr || value->shape() == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (shape->shape()->size() != 1) {
        SetError(operator_index, "FILL shape tensor must be one-dimensional");
        return false;
    }
    if (value->type() != output->type()) {
        SetError(operator_index, "FILL value and output types do not match");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateFilterBank(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 6) {
        SetError(operator_index, "FILTER_BANK requires exactly six input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "FILTER_BANK requires exactly one output tensor");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const tflite::TensorType input_types[] = {
        tflite::TensorType_UINT32,
        tflite::TensorType_INT16,
        tflite::TensorType_INT16,
        tflite::TensorType_INT16,
        tflite::TensorType_INT16,
        tflite::TensorType_INT16
    };
    for (size_t i = 0; i < inputs->size(); ++i) {
        const int32_t input_index = inputs->Get(i);
        if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
            SetError(operator_index, "contains an invalid input tensor index");
            return false;
        }
        const auto* input = tensors->Get(static_cast<size_t>(input_index));
        if (input == nullptr || input->shape() == nullptr) {
            SetError(operator_index, "contains an invalid input tensor");
            return false;
        }
        if (input->shape()->size() != 1) {
            SetError(operator_index, "FILTER_BANK input tensor must be 1D");
            return false;
        }
        if (input->type() != input_types[i]) {
            SetError(operator_index, "FILTER_BANK input tensor has an invalid type");
            return false;
        }
    }
    const int32_t output_index = outputs->Get(0);
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (output->shape()->size() != 1) {
        SetError(operator_index, "FILTER_BANK output tensor must be 1D");
        return false;
    }
    if (output->type() != tflite::TensorType_UINT64) {
        SetError(operator_index, "FILTER_BANK output tensor must be UINT64");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateFilterBankLog(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "FILTER_BANK_LOG requires exactly one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "FILTER_BANK_LOG requires exactly one output tensor");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid input tensor index");
        return false;
    }
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || input->shape() == nullptr) {
        SetError(operator_index, "contains an invalid input tensor");
        return false;
    }
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (input->shape()->size() != 1) {
        SetError(operator_index, "FILTER_BANK_LOG input tensor must be 1D");
        return false;
    }
    if (output->shape()->size() != 1) {
        SetError(operator_index, "FILTER_BANK_LOG output tensor must be 1D");
        return false;
    }
    if (input->type() != tflite::TensorType_UINT32) {
        SetError(operator_index, "FILTER_BANK_LOG input tensor must be UINT32");
        return false;
    }
    if (output->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "FILTER_BANK_LOG output tensor must be INT16");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateFilterBankSpectralSubtraction(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "FILTER_BANK_SPECTRAL_SUBTRACTION requires exactly one input tensor");
        return false;
    }
    if (outputs->size() != 2) {
        SetError(operator_index, "FILTER_BANK_SPECTRAL_SUBTRACTION requires exactly two output tensors");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    const int32_t noise_estimate_index = outputs->Get(1);
    if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid input tensor index");
        return false;
    }
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    if (noise_estimate_index < 0 || static_cast<size_t>(noise_estimate_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid noise estimate tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    const auto* noise_estimate = tensors->Get(static_cast<size_t>(noise_estimate_index));
    if (input == nullptr || input->shape() == nullptr) {
        SetError(operator_index, "contains an invalid input tensor");
        return false;
    }
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (noise_estimate == nullptr || noise_estimate->shape() == nullptr) {
        SetError(operator_index, "contains an invalid noise estimate tensor");
        return false;
    }
    if (input->shape()->size() != 1) {
        SetError(operator_index, "FILTER_BANK_SPECTRAL_SUBTRACTION input tensor must be 1D");
        return false;
    }
    if (output->shape()->size() != 1) {
        SetError(operator_index, "FILTER_BANK_SPECTRAL_SUBTRACTION output tensor must be 1D");
        return false;
    }
    if (noise_estimate->shape()->size() != 1) {
        SetError(operator_index, "FILTER_BANK_SPECTRAL_SUBTRACTION noise estimate tensor must be 1D");
        return false;
    }
    if (input->type() != tflite::TensorType_UINT32) {
        SetError(operator_index, "FILTER_BANK_SPECTRAL_SUBTRACTION input tensor must be UINT32");
        return false;
    }
    if (output->type() != tflite::TensorType_UINT32) {
        SetError(operator_index, "FILTER_BANK_SPECTRAL_SUBTRACTION output tensor must be UINT32");
        return false;
    }
    if (noise_estimate->type() != tflite::TensorType_UINT32) {
        SetError(operator_index, "FILTER_BANK_SPECTRAL_SUBTRACTION noise estimate tensor must be UINT32");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateFilterBankSquareRoot(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "FILTER_BANK_SQUARE_ROOT requires exactly two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "FILTER_BANK_SQUARE_ROOT requires exactly one output tensor");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t input_index = inputs->Get(0);
    const int32_t scale_bits_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid input tensor index");
        return false;
    }
    if (scale_bits_index < 0 || static_cast<size_t>(scale_bits_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid scale bits tensor index");
        return false;
    }
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* scale_bits = tensors->Get(static_cast<size_t>(scale_bits_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || input->shape() == nullptr) {
        SetError(operator_index, "contains an invalid input tensor");
        return false;
    }
    if (scale_bits == nullptr || scale_bits->shape() == nullptr) {
        SetError(operator_index, "contains an invalid scale bits tensor");
        return false;
    }
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (input->shape()->size() != 1) {
        SetError(operator_index, "FILTER_BANK_SQUARE_ROOT input tensor must be 1D");
        return false;
    }
    if (scale_bits->shape()->size() != 0) {
        SetError(operator_index, "FILTER_BANK_SQUARE_ROOT scale bits tensor must be scalar");
        return false;
    }
    if (output->shape()->size() != 1) {
        SetError(operator_index, "FILTER_BANK_SQUARE_ROOT output tensor must be 1D");
        return false;
    }
    if (input->type() != tflite::TensorType_UINT64) {
        SetError(operator_index, "FILTER_BANK_SQUARE_ROOT input tensor must be UINT64");
        return false;
    }
    if (scale_bits->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "FILTER_BANK_SQUARE_ROOT scale bits tensor must be INT32");
        return false;
    }
    if (output->type() != tflite::TensorType_UINT32) {
        SetError(operator_index, "FILTER_BANK_SQUARE_ROOT output tensor must be UINT32");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateFloor(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateUnaryElementwise(operator_index, op);
}

bool ModelValidator::ValidateFloorDiv(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input1_index = inputs->Get(0);
    const int32_t input2_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input1_index < 0 || input2_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input1_index) >= tensors->size() || static_cast<size_t>(input2_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input1 = tensors->Get(static_cast<size_t>(input1_index));
    const auto* input2 = tensors->Get(static_cast<size_t>(input2_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input1 == nullptr || input2 == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input1->type() != input2->type() || input1->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    switch (input1->type()) {
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_FLOAT16:
    case tflite::TensorType_INT32:
    case tflite::TensorType_INT64:
        break;
    default:
        SetError(operator_index, "unsupported FLOOR_DIV tensor type");
        return false;
    }
    if (!ValidateBinaryBroadcast(operator_index, input1, input2, output)) {
        return false;
    }
    return true;
}

bool ModelValidator::ValidateFloorMod(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input1_index = inputs->Get(0);
    const int32_t input2_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input1_index < 0 || input2_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input1_index) >= tensors->size() || static_cast<size_t>(input2_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input1 = tensors->Get(static_cast<size_t>(input1_index));
    const auto* input2 = tensors->Get(static_cast<size_t>(input2_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input1 == nullptr || input2 == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input1->type() != input2->type() || input1->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    switch (input1->type()) {
    case tflite::TensorType_INT32:
    case tflite::TensorType_INT64:
        break;
    default:
        SetError(operator_index, "unsupported FLOOR_MOD tensor type");
        return false;
    }
    if (!ValidateBinaryBroadcast(operator_index, input1, input2, output)) {
        return false;
    }
    return true;
}

bool ModelValidator::ValidateFramer(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "FRAMER requires exactly one input tensor");
        return false;
    }
    if (outputs->size() != 2) {
        SetError(operator_index, "FRAMER requires exactly two output tensors");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    const int32_t output_valid_index = outputs->Get(1);
    if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid input tensor index");
        return false;
    }
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    if (output_valid_index < 0 || static_cast<size_t>(output_valid_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output valid tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    const auto* output_valid = tensors->Get(static_cast<size_t>(output_valid_index));
    if (input == nullptr || input->shape() == nullptr) {
        SetError(operator_index, "contains an invalid input tensor");
        return false;
    }
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (output_valid == nullptr || output_valid->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output valid tensor");
        return false;
    }
    if (input->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "FRAMER input tensor must be INT16");
        return false;
    }
    if (output->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "FRAMER output tensor must be INT16");
        return false;
    }
    if (output_valid->type() != tflite::TensorType_BOOL) {
        SetError(operator_index, "FRAMER valid output tensor must be BOOL");
        return false;
    }
    if (output->shape()->size() != input->shape()->size() + 1) {
        SetError(operator_index, "FRAMER output rank must equal input rank plus one");
        return false;
    }
    if (output_valid->shape()->size() != 0) {
        SetError(operator_index, "FRAMER valid output must be scalar");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateFullyConnected(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    /*
     * FULLY_CONNECTED inputs:
     *
     *   0 = input
     *   1 = weights
     *   2 = bias
     */
    if (inputs->size() < 3) {
        SetError(operator_index, "requires input, weights and bias tensors");
        return false;
    }
    if (outputs->size() < 1) {
        SetError(operator_index, "has no output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t weights_index = inputs->Get(1);
    const int32_t bias_index = inputs->Get(2);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || weights_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(weights_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size() || (bias_index >= 0 && static_cast<size_t>(bias_index) >= tensors->size())) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* weights = tensors->Get(static_cast<size_t>(weights_index));
    const auto* bias = bias_index >= 0 ? tensors->Get(static_cast<size_t>(bias_index)) : nullptr;
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || weights == nullptr || output == nullptr || (bias_index >= 0 && bias == nullptr)) {
        SetError(operator_index, "references a null tensor");
        return false;
    }
    /*
     * FULLY_CONNECTED tensors must have
     * compatible element types.
     */
    const auto input_type = input->type();
    const auto weights_type = weights->type();
    const auto output_type = output->type();
    if (input_type != weights_type || input_type != output_type) {
        SetError(operator_index, "input, weights and output types differ");
        return false;
    }
    if (bias_index >= 0) {
        const auto bias_type = bias->type();
        if (bias_type != tflite::TensorType_INT32 && bias_type != input_type) {
            SetError(operator_index, "bias type is incompatible with input type");
            return false;
        }
    }
    const auto* input_shape = input->shape();
    const auto* weights_shape = weights->shape();
    const auto* bias_shape = bias_index >= 0 ? bias->shape() : nullptr;
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || weights_shape == nullptr || output_shape == nullptr || (bias_index >= 0 && bias_shape == nullptr)) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    /*
     * weights = [output_size, input_size]
     */
    if (weights_shape->size() != 2) {
        SetError(operator_index, "weights tensor must be rank 2");
        return false;
    }
    /*
     * bias = [output_size]
     */
    if (bias_index >= 0 && bias_shape->size() != 1) {
        SetError(operator_index, "bias tensor must be rank 1");
        return false;
    }
    if (input_shape->size() == 0) {
        SetError(operator_index, "input tensor must have at least one dimension");
        return false;
    }
    if (output_shape->size() == 0) {
        SetError(operator_index, "output tensor must have at least one dimension");
        return false;
    }
    const int32_t output_size = weights_shape->Get(0);
    const int32_t input_size = weights_shape->Get(1);
    const int32_t actual_input_size = input_shape->Get(input_shape->size() - 1);
    const int32_t actual_output_size = output_shape->Get(output_shape->size() - 1);
    if (output_size < 0 || input_size < 0 || actual_input_size < 0 || actual_output_size < 0) {
        SetError(operator_index, "contains a negative tensor dimension");
        return false;
    }
    if (bias_index >= 0) {
        const int32_t bias_size = bias_shape->Get(0);
        if (bias_size < 0) {
            SetError(operator_index, "contains a negative tensor dimension");
            return false;
        }
        if (bias_size != output_size) {
            SetError(operator_index, "bias size does not match weights output size");
            return false;
        }
    }
    size_t input_elements = 1;
    if (input->shape() == nullptr || input->shape()->size() == 0) {
        SetError(operator_index, "input tensor has no shape");
        return false;
    }
    for (size_t i = 0; i < input->shape()->size(); ++i) {
        const int32_t dimension = input->shape()->Get(i);
        if (dimension < 0) {
            SetError(operator_index, "input tensor has invalid dimension");
            return false;
        }
        input_elements *= static_cast<size_t>(dimension);
    }
    if (weights->shape() == nullptr || weights->shape()->size() != 2) {
        SetError(operator_index, "weights tensor must be 2-dimensional");
        return false;
    }
    const size_t weights_input_size = static_cast<size_t>(weights->shape()->Get(1));
    if (input_elements != weights_input_size) {
        SetError(operator_index, "input size does not match weights input size");
        return false;
    }
    if (actual_output_size != output_size) {
        SetError(operator_index, "output size does not match weights output size");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateFullyConnectedOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_FullyConnectedOptions) {
        SetError(operator_index, "invalid builtin options type");
        return false;
    }
    const auto* options = op->builtin_options_as_FullyConnectedOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing FULLY_CONNECTED options");
        return false;
    }
    const auto activation = options->fused_activation_function();
    switch (activation) {
    case tflite::ActivationFunctionType_NONE:
    case tflite::ActivationFunctionType_RELU:
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
    case tflite::ActivationFunctionType_RELU6:
    case tflite::ActivationFunctionType_TANH:
    case tflite::ActivationFunctionType_SIGN_BIT:
        return true;
    default:
        SetError(operator_index, "invalid fused activation function");
        return false;
    }
}

bool ModelValidator::ValidateGather(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and indices tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t indices_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || indices_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(indices_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* indices = tensors->Get(static_cast<size_t>(indices_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || indices == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input->shape() == nullptr || indices->shape() == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (indices->type() != tflite::TensorType_INT32 && indices->type() != tflite::TensorType_INT64) {
        SetError(operator_index, "GATHER indices tensor must be INT32 or INT64");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_GatherOptions) {
        SetError(operator_index, "invalid GATHER options type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateGatherOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_GatherOptions) {
        SetError(operator_index, "invalid GATHER options type");
        return false;
    }
    const auto* options = op->builtin_options_as_GatherOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing GATHER options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateGatherNd(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and indices tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t indices_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || indices_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(indices_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* indices = tensors->Get(static_cast<size_t>(indices_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || indices == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* indices_shape = indices->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || indices_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (indices->type() != tflite::TensorType_INT32 && indices->type() != tflite::TensorType_INT64) {
        SetError(operator_index, "GATHER_ND indices tensor must be INT32 or INT64");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    const size_t indices_rank = indices_shape->size();
    if (indices_rank == 0) {
        SetError(operator_index, "GATHER_ND indices tensor must have rank greater than zero");
        return false;
    }
    const int32_t index_depth = indices_shape->Get(indices_rank - 1);
    if (index_depth <= 0) {
        SetError(operator_index, "GATHER_ND index depth is invalid");
        return false;
    }
    if (static_cast<size_t>(index_depth) > input_shape->size()) {
        SetError(operator_index, "GATHER_ND index depth exceeds input rank");
        return false;
    }
    const size_t expected_output_rank = indices_rank - 1 + input_shape->size() - static_cast<size_t>(index_depth);
    if (output_shape->size() != expected_output_rank) {
        SetError(operator_index, "output rank does not match GATHER_ND");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateGreater(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateComparison(operator_index, op);
}

bool ModelValidator::ValidateGreaterEqual(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateComparison(operator_index, op);
}

bool ModelValidator::ValidateHardSwish(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() ||static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (output_shape->size() != input_shape->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    for (size_t i = 0; i < input_shape->size(); ++i) {
        if (output_shape->Get(i) != input_shape->Get(i)) {
            SetError(operator_index, "input and output shapes do not match");
            return false;
        }
    }
    switch (input->type()) {
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_FLOAT16:
    case tflite::TensorType_INT8:
    case tflite::TensorType_INT16:
        break;
    default:
        SetError(operator_index, "unsupported HARD_SWISH tensor type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateIf(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* model = document_.Model();
    if (model == nullptr || model->subgraphs() == nullptr) {
        SetError(operator_index, "missing model or subgraphs");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() < 1) {
        SetError(operator_index, "requires condition input");
        return false;
    }
    if (outputs->size() == 0) {
        SetError(operator_index, "requires at least one output tensor");
        return false;
    }
    const int32_t condition_index = inputs->Get(0);
    if (condition_index < 0 || static_cast<size_t>(condition_index) >= subgraph->tensors()->size()) {
        SetError(operator_index, "condition tensor index is out of range");
        return false;
    }
    const auto* condition = subgraph->tensors()->Get(static_cast<size_t>(condition_index));
    if (condition == nullptr) {
        SetError(operator_index, "condition tensor is null");
        return false;
    }
    if (condition->type() != tflite::TensorType_BOOL) {
        SetError(operator_index, "condition tensor must be BOOL");
        return false;
    }
    const auto* condition_shape = condition->shape();
    if (condition_shape == nullptr) {
        SetError(operator_index, "condition tensor has no shape");
        return false;
    }
    if (condition_shape->size() != 0) {
        SetError(operator_index, "condition tensor must be scalar");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    for (size_t i = 1; i < inputs->size(); ++i) {
        const int32_t input_index = inputs->Get(i);
        if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
            SetError(operator_index, "contains an out-of-range branch input tensor index");
            return false;
        }
        if (tensors->Get(static_cast<size_t>(input_index)) == nullptr) {
            SetError(operator_index, "contains a null branch input tensor");
            return false;
        }
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
        const int32_t output_index = outputs->Get(i);
        if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
            SetError(operator_index, "contains an out-of-range output tensor index");
            return false;
        }
        if (tensors->Get(static_cast<size_t>(output_index)) == nullptr) {
            SetError(operator_index, "contains a null output tensor");
            return false;
        }
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_IfOptions) {
        SetError(operator_index, "invalid IF options type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateIfOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_IfOptions) {
        SetError(operator_index, "invalid IF options type");
        return false;
    }
    const auto* options = op->builtin_options_as_IfOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing IF options");
        return false;
    }
    const auto* model = document_.Model();
    if (model == nullptr || model->subgraphs() == nullptr) {
        SetError(operator_index, "missing model or subgraphs");
        return false;
    }
    const int32_t then_subgraph_index = options->then_subgraph_index();
    const int32_t else_subgraph_index = options->else_subgraph_index();
    if (then_subgraph_index < 0 || static_cast<size_t>(then_subgraph_index) >= model->subgraphs()->size()) {
        SetError(operator_index, "IF then subgraph index is out of range");
        return false;
    }
    if (else_subgraph_index < 0 || static_cast<size_t>(else_subgraph_index) >= model->subgraphs()->size()) {
        SetError(operator_index, "IF else subgraph index is out of range");
        return false;
    }
    if (model->subgraphs()->Get(static_cast<size_t>(then_subgraph_index)) == nullptr) {
        SetError(operator_index, "IF then subgraph is null");
        return false;
    }
    if (model->subgraphs()->Get(static_cast<size_t>(else_subgraph_index)) == nullptr) {
        SetError(operator_index, "IF else subgraph is null");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateIfSubgraphs(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* model = document_.Model();
    if (model == nullptr || model->subgraphs() == nullptr) {
        SetError(operator_index, "missing model or subgraphs");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_IfOptions) {
        SetError(operator_index, "invalid IF options type");
        return false;
    }
    const auto* options = op->builtin_options_as_IfOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing IF options");
        return false;
    }
    const int32_t then_index = options->then_subgraph_index();
    const int32_t else_index = options->else_subgraph_index();
    if (then_index < 0 || static_cast<size_t>(then_index) >= model->subgraphs()->size() || else_index < 0 || static_cast<size_t>(else_index) >= model->subgraphs()->size()) {
        SetError(operator_index, "IF subgraph index is out of range");
        return false;
    }
    const auto* then_subgraph = model->subgraphs()->Get(static_cast<size_t>(then_index));
    const auto* else_subgraph = model->subgraphs()->Get(static_cast<size_t>(else_index));
    if (then_subgraph == nullptr || else_subgraph == nullptr) {
        SetError(operator_index, "IF subgraph is null");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    const auto* then_inputs = then_subgraph->inputs();
    const auto* else_inputs = else_subgraph->inputs();
    const auto* then_outputs = then_subgraph->outputs();
    const auto* else_outputs = else_subgraph->outputs();
    if (inputs == nullptr || outputs == nullptr || then_inputs == nullptr || else_inputs == nullptr || then_outputs == nullptr || else_outputs == nullptr) {
        SetError(operator_index, "IF is missing tensor lists");
        return false;
    }
    if (then_inputs->size() != inputs->size() - 1 || else_inputs->size() != inputs->size() - 1) {
        SetError(operator_index, "IF branch input count does not match operator inputs");
        return false;
    }
    if (then_outputs->size() != outputs->size() || else_outputs->size() != outputs->size()) {
        SetError(operator_index, "IF branch output count does not match operator outputs");
        return false;
    }
    const auto* then_tensors = then_subgraph->tensors();
    const auto* else_tensors = else_subgraph->tensors();
    const auto* current_subgraph = CurrentSubgraph();
    if (then_tensors == nullptr || else_tensors == nullptr || current_subgraph == nullptr || current_subgraph->tensors() == nullptr) {
        SetError(operator_index, "IF is missing tensor lists");
        return false;
    }
    const auto* tensors = current_subgraph->tensors();
    for (size_t i = 0; i < then_inputs->size(); ++i) {
        const int32_t operator_input_index = inputs->Get(i + 1);
        const int32_t then_input_index = then_inputs->Get(i);
        const int32_t else_input_index = else_inputs->Get(i);
        if (operator_input_index < 0 || static_cast<size_t>(operator_input_index) >= tensors->size() || then_input_index < 0 || static_cast<size_t>(then_input_index) >= then_tensors->size() || else_input_index < 0 || static_cast<size_t>(else_input_index) >= else_tensors->size()) {
            SetError(operator_index, "IF branch input tensor index is out of range");
            return false;
        }
        const auto* operator_input = tensors->Get(static_cast<size_t>(operator_input_index));
        const auto* then_input = then_tensors->Get(static_cast<size_t>(then_input_index));
        const auto* else_input = else_tensors->Get(static_cast<size_t>(else_input_index));
        if (operator_input == nullptr || then_input == nullptr || else_input == nullptr) {
            SetError(operator_index, "IF branch input tensor is null");
            return false;
        }
        if (operator_input->type() != then_input->type() || operator_input->type() != else_input->type()) {
            SetError(operator_index, "IF branch input types do not match operator inputs");
            return false;
        }
        if (operator_input->shape() == nullptr || then_input->shape() == nullptr || else_input->shape() == nullptr) {
            SetError(operator_index, "IF branch input tensor has no shape");
            return false;
        }
        if (operator_input->shape()->size() != then_input->shape()->size() || operator_input->shape()->size() != else_input->shape()->size()) {
            SetError(operator_index, "IF branch input ranks do not match operator inputs");
            return false;
        }
        for (size_t j = 0; j < operator_input->shape()->size(); ++j) {
            if (operator_input->shape()->Get(j) != then_input->shape()->Get(j) || operator_input->shape()->Get(j) != else_input->shape()->Get(j)) {
                SetError(operator_index, "IF branch input shapes do not match operator inputs");
                return false;
            }
        }
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
        const int32_t operator_output_index = outputs->Get(i);
        const int32_t then_output_index = then_outputs->Get(i);
        const int32_t else_output_index = else_outputs->Get(i);
        if (operator_output_index < 0 || static_cast<size_t>(operator_output_index) >= tensors->size() || then_output_index < 0 || static_cast<size_t>(then_output_index) >= then_tensors->size() || else_output_index < 0 || static_cast<size_t>(else_output_index) >= else_tensors->size()) {
            SetError(operator_index, "IF branch output tensor index is out of range");
            return false;
        }
        const auto* operator_output = tensors->Get(static_cast<size_t>(operator_output_index));
        const auto* then_output = then_tensors->Get(static_cast<size_t>(then_output_index));
        const auto* else_output = else_tensors->Get(static_cast<size_t>(else_output_index));
        if (operator_output == nullptr || then_output == nullptr || else_output == nullptr) {
            SetError(operator_index, "IF branch output tensor is null");
            return false;
        }
        if (operator_output->type() != then_output->type() || operator_output->type() != else_output->type()) {
            SetError(operator_index, "IF branch output types do not match operator output");
            return false;
        }
        if (operator_output->shape() == nullptr || then_output->shape() == nullptr || else_output->shape() == nullptr) {
            SetError(operator_index, "IF branch output tensor has no shape");
            return false;
        }
        if (operator_output->shape()->size() != then_output->shape()->size() || operator_output->shape()->size() != else_output->shape()->size()) {
            SetError(operator_index, "IF branch output ranks do not match operator output");
            return false;
        }
        for (size_t j = 0; j < operator_output->shape()->size(); ++j) {
            if (operator_output->shape()->Get(j) != then_output->shape()->Get(j) || operator_output->shape()->Get(j) != else_output->shape()->Get(j)) {
                SetError(operator_index, "IF branch output shapes do not match operator output");
                return false;
            }
        }
    }
    return true;
}

bool ModelValidator::ValidateIrfft(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "IRFFT requires exactly one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "IRFFT requires exactly one output tensor");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid input tensor index");
        return false;
    }
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || input->shape() == nullptr) {
        SetError(operator_index, "contains an invalid input tensor");
        return false;
    }
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (input->shape()->size() != output->shape()->size()) {
        SetError(operator_index, "IRFFT input and output ranks do not match");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "IRFFT input and output tensor types do not match");
        return false;
    }
    if (input->type() != tflite::TensorType_INT16 &&
        input->type() != tflite::TensorType_INT32 &&
        input->type() != tflite::TensorType_FLOAT32) {
        SetError(operator_index, "IRFFT tensor type is unsupported");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateL2Normalization(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (output_shape->size() != input_shape->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    for (size_t i = 0; i < input_shape->size(); ++i) {
        if (input_shape->Get(i) != output_shape->Get(i)) {
            SetError(operator_index, "input and output shapes do not match");
            return false;
        }
    }
    switch (input->type()) {
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_FLOAT16:
    case tflite::TensorType_INT8:
    case tflite::TensorType_INT16:
        break;
    default:
        SetError(operator_index, "unsupported L2_NORMALIZATION tensor type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateL2Pool2D(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input_shape->size() != 4 || output_shape->size() != 4) {
        SetError(operator_index, "L2_POOL_2D tensors must be 4-dimensional");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    switch (input->type()) {
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_FLOAT16:
    case tflite::TensorType_INT8:
    case tflite::TensorType_INT16:
        break;
    default:
        SetError(operator_index, "unsupported L2_POOL_2D tensor type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateL2Pool2DOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_Pool2DOptions) {
        SetError(operator_index, "invalid L2_POOL_2D options type");
        return false;
    }
    const auto* options = op->builtin_options_as_Pool2DOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing L2_POOL_2D options");
        return false;
    }
    if (options->stride_w() <= 0 || options->stride_h() <= 0) {
        SetError(operator_index, "pool stride must be positive");
        return false;
    }
    if (options->filter_width() <= 0 || options->filter_height() <= 0) {
        SetError(operator_index, "pool filter size must be positive");
        return false;
    }
    switch (options->padding()) {
    case tflite::Padding_VALID:
    case tflite::Padding_SAME:
        break;
    default:
        SetError(operator_index, "invalid pool padding");
        return false;
    }
    if (options->fused_activation_function() != tflite::ActivationFunctionType_NONE) {
        SetError(operator_index, "L2_POOL_2D must not use fused activation");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateLeakyRelu(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "input or output has no shape");
        return false;
    }
    if (input_shape->size() != output_shape->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    for (size_t i = 0; i < input_shape->size(); ++i) {
        if (input_shape->Get(i) != output_shape->Get(i)) {
            SetError(operator_index, "input and output shapes do not match");
            return false;
        }
        if (input_shape->Get(i) < 0) {
            SetError(operator_index, "tensor contains a negative dimension");
            return false;
        }
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output types do not match");
        return false;
    }
    switch (input->type()) {
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_FLOAT16:
    case tflite::TensorType_INT8:
    case tflite::TensorType_INT16:
        break;
    default:
        SetError(operator_index, "unsupported LEAKY_RELU tensor type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateLeakyReluOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_LeakyReluOptions) {
        SetError(operator_index, "invalid LEAKY_RELU options type");
        return false;
    }
    const auto* options = op->builtin_options_as_LeakyReluOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing LEAKY_RELU options");
        return false;
    }
    const float alpha = options->alpha();
    if (!std::isfinite(alpha) || alpha < 0.0f) {
        SetError(operator_index, "invalid LEAKY_RELU alpha");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateLess(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateComparison(operator_index, op);
}

bool ModelValidator::ValidateLessEqual(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateComparison(operator_index, op);
}

bool ModelValidator::ValidateLog(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateUnaryElementwise(operator_index, op);
}

bool ModelValidator::ValidateLogicalAnd(size_t operator_index, const tflite::Operator* op) const
{
    if (!ValidateLogicalBinary(operator_index, op))
        return false;
    if (op->builtin_options_type() != tflite::BuiltinOptions_LogicalAndOptions) {
        SetError(operator_index, "invalid LOGICAL_AND options type");
        return false;
    }
    if (op->builtin_options_as_LogicalAndOptions() == nullptr) {
        SetError(operator_index, "missing LOGICAL_AND options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateLogicalNot(size_t operator_index, const tflite::Operator* op) const
{
    if (!ValidateUnaryElementwise(operator_index, op))
        return false;
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const auto* input = tensors->Get(static_cast<size_t>(inputs->Get(0)));
    const auto* output = tensors->Get(static_cast<size_t>(outputs->Get(0)));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input->type() != tflite::TensorType_BOOL || output->type() != tflite::TensorType_BOOL) {
        SetError(operator_index, "LOGICAL_NOT tensors must be BOOL");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_LogicalNotOptions) {
        SetError(operator_index, "invalid LOGICAL_NOT options type");
        return false;
    }
    if (op->builtin_options_as_LogicalNotOptions() == nullptr) {
        SetError(operator_index, "missing LOGICAL_NOT options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateLogicalOr(size_t operator_index, const tflite::Operator* op) const
{
    if (!ValidateLogicalBinary(operator_index, op))
        return false;
    if (op->builtin_options_type() != tflite::BuiltinOptions_LogicalOrOptions) {
        SetError(operator_index, "invalid LOGICAL_OR options type");
        return false;
    }
    if (op->builtin_options_as_LogicalOrOptions() == nullptr) {
        SetError(operator_index, "missing LOGICAL_OR options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateLogistic(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input->shape() == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "input or output tensor has no shape");
        return false;
    }
    if (input->shape()->size() != output->shape()->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    for (size_t i = 0; i < input->shape()->size(); ++i) {
        if (input->shape()->Get(i) != output->shape()->Get(i)) {
            SetError(operator_index, "input and output shapes do not match");
            return false;
        }
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    switch (input->type()) {
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_INT8:
    case tflite::TensorType_INT16:
    case tflite::TensorType_UINT8:
        break;
    default:
        SetError(operator_index, "unsupported LOGISTIC tensor type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateLogSoftmax(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (output_shape->size() != input_shape->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    for (size_t i = 0; i < input_shape->size(); ++i) {
        if (output_shape->Get(i) != input_shape->Get(i)) {
            SetError(operator_index, "input and output shapes do not match");
            return false;
        }
    }
    switch (input->type()) {
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_FLOAT16:
    case tflite::TensorType_INT8:
    case tflite::TensorType_INT16:
        break;
    default:
        SetError(operator_index, "unsupported LOG_SOFTMAX tensor type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateLogSoftmaxOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_LogSoftmaxOptions) {
        SetError(operator_index, "invalid LOG_SOFTMAX options type");
        return false;
    }
    const auto* options = op->builtin_options_as_LogSoftmaxOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing LOG_SOFTMAX options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateMaximum(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input1_index = inputs->Get(0);
    const int32_t input2_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input1_index < 0 || input2_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input1_index) >= tensors->size() || static_cast<size_t>(input2_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input1 = tensors->Get(static_cast<size_t>(input1_index));
    const auto* input2 = tensors->Get(static_cast<size_t>(input2_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input1 == nullptr || input2 == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input1->type() != input2->type() || input1->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (!ValidateBinaryBroadcast(operator_index, input1, input2, output)) {
        return false;
    }
    return true;
}

bool ModelValidator::ValidateMaximumOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() == tflite::BuiltinOptions_NONE)
        return true;
    if (op->builtin_options_type() != tflite::BuiltinOptions_MaximumMinimumOptions) {
        SetError(operator_index, "invalid MAXIMUM options type");
        return false;
    }
    const auto* options = op->builtin_options_as_MaximumMinimumOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing MAXIMUM options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateMaxPool2D(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input_shape->size() != 4) {
        SetError(operator_index, "input tensor must be 4-dimensional");
        return false;
    }
    if (output_shape->size() != 4) {
        SetError(operator_index, "output tensor must be 4-dimensional");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateMaxPool2DOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_Pool2DOptions) {
        SetError(operator_index, "invalid MAX_POOL_2D options type");
        return false;
    }
    const auto* options = op->builtin_options_as_Pool2DOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing MAX_POOL_2D options");
        return false;
    }
    if (options->stride_w() <= 0 || options->stride_h() <= 0) {
        SetError(operator_index, "pool stride must be positive");
        return false;
    }
    if (options->filter_width() <= 0 || options->filter_height() <= 0) {
        SetError(operator_index, "pool filter size must be positive");
        return false;
    }
    switch (options->padding()) {
    case tflite::Padding_VALID:
    case tflite::Padding_SAME:
        break;
    default:
        SetError(operator_index, "invalid pool padding");
        return false;
    }
    switch (options->fused_activation_function()) {
    case tflite::ActivationFunctionType_NONE:
    case tflite::ActivationFunctionType_RELU:
    case tflite::ActivationFunctionType_RELU6:
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
        break;
    default:
        SetError(operator_index, "invalid fused activation function");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateMean(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and axis tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t axis_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || axis_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(axis_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* axis = tensors->Get(static_cast<size_t>(axis_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || axis == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (axis->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "axis tensor must be INT32");
        return false;
    }
    if (axis->shape() == nullptr || axis->shape()->size() != 1) {
        SetError(operator_index, "axis tensor must be one-dimensional");
        return false;
    }
    if (axis->shape()->Get(0) <= 0) {
        SetError(operator_index, "axis tensor must contain at least one axis");
        return false;
    }
    if (axis->buffer() >= document_.BufferCount()) {
        SetError(operator_index, "axis tensor buffer index out of range");
        return false;
    }
    const auto* buffers = document_.Buffers();
    if (buffers == nullptr || axis->buffer() >= buffers->size()) {
        SetError(operator_index, "axis tensor buffer index out of range");
        return false;
    }
    const auto* axis_buffer = buffers->Get(axis->buffer());
    if (axis_buffer == nullptr ||
        axis_buffer->data() == nullptr) {
        SetError(operator_index, "axis tensor has no constant data");
        return false;
    }
    const size_t axis_count = static_cast<size_t>(axis->shape()->Get(0));
    if (axis_buffer->data()->size() <
        axis_count * sizeof(int32_t)) {
        SetError(operator_index, "axis tensor buffer is too small");
        return false;
    }
    const auto* axis_data = reinterpret_cast<const int32_t*>(axis_buffer->data()->data());
    const auto* input_shape = input->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "input or output has no shape");
        return false;
    }
    const size_t input_rank = input_shape->size();
    if (input_rank == 0) {
        SetError(operator_index, "input tensor has no shape");
        return false;
    }
    /*
     * Get keep_dims from ReducerOptions.
     */
    bool keep_dims = false;
    if (op->builtin_options_type() == tflite::BuiltinOptions_ReducerOptions) {
        const auto* options = op->builtin_options_as_ReducerOptions();
        if (options == nullptr) {
            SetError(operator_index, "missing MEAN options");
            return false;
        }
        keep_dims = options->keep_dims();
    } else if (op->builtin_options_type() != tflite::BuiltinOptions_NONE) {
        SetError(operator_index, "invalid MEAN options type");
        return false;
    }
    /*
     * Normalize axes and reject duplicates.
     */
    for (size_t i = 0; i < axis_count; ++i) {
        int32_t axis_value = axis_data[i];
        if (axis_value < 0)
            axis_value += static_cast<int32_t>(input_rank);
        if (axis_value < 0 || axis_value >= static_cast<int32_t>(input_rank)) {
            SetError(operator_index, "axis value is out of range");
            return false;
        }
        for (size_t j = 0; j < i; ++j) {
            int32_t previous = axis_data[j];
            if (previous < 0)
                previous += static_cast<int32_t>(input_rank);
            if (previous == axis_value) {
                SetError(operator_index, "axis tensor contains duplicate axes");
                return false;
            }
        }
    }
    /*
     * Calculate expected output rank.
     *
     * keep_dims == false:
     *     reduced dimensions disappear.
     *
     * keep_dims == true:
     *     reduced dimensions become 1.
     */
    size_t expected_output_rank = input_rank;
    if (!keep_dims)
        expected_output_rank -= axis_count;
    if (output_shape->size() != expected_output_rank) {
        SetError(operator_index, "output rank does not match reduced input rank");
        return false;
    }
    if (keep_dims) {
        /*
         * Output has the same rank as input.
         */
        for (size_t i = 0; i < input_rank; ++i) {
            bool reduced = false;
            for (size_t j = 0; j < axis_count; ++j) {
                int32_t axis_value = axis_data[j];
                if (axis_value < 0)
                    axis_value += static_cast<int32_t>(input_rank);
                if (static_cast<size_t>(axis_value) == i) {
                    reduced = true;
                    break;
                }
            }
            const int32_t expected_dimension = reduced ? 1 : input_shape->Get(i);
            if (output_shape->Get(i) != expected_dimension) {
                SetError(operator_index, "output shape does not match MEAN reduction");
                return false;
            }
        }
    } else {
        /*
         * Reduced dimensions disappear from the output.
         */
        size_t output_index_position = 0;
        for (size_t i = 0; i < input_rank; ++i) {
            bool reduced = false;
            for (size_t j = 0; j < axis_count; ++j) {
                int32_t axis_value = axis_data[j];
                if (axis_value < 0)
                    axis_value += static_cast<int32_t>(input_rank);
                if (static_cast<size_t>(axis_value) == i) {
                    reduced = true;
                    break;
                }
            }
            if (!reduced) {
                if (output_shape->Get(output_index_position) !=
                    input_shape->Get(i)) {
                    SetError(operator_index, "output shape does not match unreduced input dimensions");
                    return false;
                }
                ++output_index_position;
            }
        }
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output types do not match");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateMeanOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_ReducerOptions) {
        SetError(operator_index, "invalid MEAN options type");
        return false;
    }
    const auto* options = op->builtin_options_as_ReducerOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing MEAN options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateMinimum(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input1_index = inputs->Get(0);
    const int32_t input2_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input1_index < 0 || input2_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input1_index) >= tensors->size() || static_cast<size_t>(input2_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input1 = tensors->Get(static_cast<size_t>(input1_index));
    const auto* input2 = tensors->Get(static_cast<size_t>(input2_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input1 == nullptr || input2 == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input1->type() != input2->type() || input1->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (!ValidateBinaryBroadcast(operator_index, input1, input2, output)) {
        return false;
    }
    return true;
}

bool ModelValidator::ValidateMinimumOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() == tflite::BuiltinOptions_NONE)
        return true;
    if (op->builtin_options_type() != tflite::BuiltinOptions_MaximumMinimumOptions) {
        SetError(operator_index, "invalid MINIMUM options type");
        return false;
    }
    const auto* options = op->builtin_options_as_MaximumMinimumOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing MINIMUM options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateMirrorPad(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and paddings tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t paddings_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || paddings_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(paddings_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* paddings = tensors->Get(static_cast<size_t>(paddings_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || paddings == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* paddings_shape = paddings->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || paddings_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (paddings->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "paddings tensor must be INT32");
        return false;
    }
    if (paddings_shape->size() != 2 || paddings_shape->Get(0) != static_cast<int32_t>(input_shape->size()) || paddings_shape->Get(1) != 2) {
        SetError(operator_index, "paddings tensor shape is invalid");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (output_shape->size() != input_shape->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateMirrorPadOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_MirrorPadOptions) {
        SetError(operator_index, "invalid MIRROR_PAD options type");
        return false;
    }
    const auto* options = op->builtin_options_as_MirrorPadOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing MIRROR_PAD options");
        return false;
    }
    switch (options->mode()) {
    case tflite::MirrorPadMode_REFLECT:
    case tflite::MirrorPadMode_SYMMETRIC:
        break;
    default:
        SetError(operator_index, "invalid MIRROR_PAD mode");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateMul(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input1_index = inputs->Get(0);
    const int32_t input2_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input1_index < 0 || input2_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input1_index) >= tensors->size() || static_cast<size_t>(input2_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input1 = tensors->Get(static_cast<size_t>(input1_index));
    const auto* input2 = tensors->Get(static_cast<size_t>(input2_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input1 == nullptr || input2 == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input1->type() != input2->type() || input1->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (!ValidateBinaryBroadcast(operator_index, input1, input2, output)) {
        return false;
    }
    return true;
}

bool ModelValidator::ValidateMulOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_MulOptions) {
        SetError(operator_index, "invalid MUL options type");
        return false;
    }
    const auto* options = op->builtin_options_as_MulOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing MUL options");
        return false;
    }
    switch (options->fused_activation_function()) {
    case tflite::ActivationFunctionType_NONE:
    case tflite::ActivationFunctionType_RELU:
    case tflite::ActivationFunctionType_RELU6:
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
        break;
    default:
        SetError(operator_index, "invalid fused activation function");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateNeg(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateUnaryElementwise(operator_index, op);
}

bool ModelValidator::ValidateNotEqual(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateComparison(operator_index, op);
}

bool ModelValidator::ValidateOverlapAdd(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "OVERLAP_ADD requires exactly one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "OVERLAP_ADD requires exactly one output tensor");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid input tensor index");
        return false;
    }
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || input->shape() == nullptr) {
        SetError(operator_index, "contains an invalid input tensor");
        return false;
    }
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (input->shape()->size() < 2) {
        SetError(operator_index, "OVERLAP_ADD input tensor must have rank at least two");
        return false;
    }
    if (input->shape()->size() != output->shape()->size() + 1) {
        SetError(operator_index, "OVERLAP_ADD input rank must equal output rank plus one");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "OVERLAP_ADD input and output tensor types do not match");
        return false;
    }
    if (input->type() != tflite::TensorType_INT16 &&
        input->type() != tflite::TensorType_FLOAT32) {
        SetError(operator_index, "OVERLAP_ADD tensor type is unsupported");
        return false;
    }
    return true;
}

bool ModelValidator::ValidatePack(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() < 1) {
        SetError(operator_index, "requires at least one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t output_index = outputs->Get(0);
    if (output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (output == nullptr) {
        SetError(operator_index, "contains null output tensor");
        return false;
    }
    const auto* first_input = tensors->Get(static_cast<size_t>(inputs->Get(0)));
    if (inputs->Get(0) < 0 || static_cast<size_t>(inputs->Get(0)) >= tensors->size() || first_input == nullptr) {
        SetError(operator_index, "contains an invalid input tensor");
        return false;
    }
    if (first_input->shape() == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "missing tensor shape");
        return false;
    }
    for (size_t i = 0; i < inputs->size(); ++i) {
        const int32_t input_index = inputs->Get(i);
        if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
            SetError(operator_index, "contains an invalid input tensor index");
            return false;
        }
        const auto* input = tensors->Get(static_cast<size_t>(input_index));
        if (input == nullptr) {
            SetError(operator_index, "contains null input tensor");
            return false;
        }
        if (input->type() != first_input->type()) {
            SetError(operator_index, "input tensor types do not match");
            return false;
        }
        if (input->shape() == nullptr || input->shape()->size() != first_input->shape()->size()) {
            SetError(operator_index, "input tensor ranks do not match");
            return false;
        }
        for (size_t j = 0; j < first_input->shape()->size(); ++j) {
            if (input->shape()->Get(j) != first_input->shape()->Get(j)) {
                SetError(operator_index, "input tensor shapes do not match");
                return false;
            }
        }
    }
    if (output->type() != first_input->type()) {
        SetError(operator_index, "output tensor type does not match inputs");
        return false;
    }
    return true;
}

bool ModelValidator::ValidatePackOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_PackOptions) {
        SetError(operator_index, "invalid PACK options type");
        return false;
    }
    const auto* options = op->builtin_options_as_PackOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing PACK options");
        return false;
    }
    if (options->values_count() <= 0) {
        SetError(operator_index, "PACK values count must be positive");
        return false;
    }
    const auto* inputs = op->inputs();
    if (inputs == nullptr || inputs->size() == 0) {
        SetError(operator_index, "PACK has no inputs");
        return false;
    }
    if (static_cast<size_t>(options->values_count()) != inputs->size()) {
        SetError(operator_index, "PACK values count does not match input count");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t input_index = inputs->Get(0);
    if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "PACK input tensor index is invalid");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    if (input == nullptr || input->shape() == nullptr) {
        SetError(operator_index, "PACK input has no shape");
        return false;
    }
    const int32_t rank = static_cast<int32_t>(input->shape()->size());
    if (options->axis() < -rank || options->axis() > rank) {
        SetError(operator_index, "PACK axis is out of range");
        return false;
    }
    return true;
}

bool ModelValidator::ValidatePad(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and paddings tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t paddings_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || paddings_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(paddings_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* paddings = tensors->Get(static_cast<size_t>(paddings_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || paddings == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* paddings_shape = paddings->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || paddings_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    const size_t rank = input_shape->size();
    /*
     * paddings must be INT32 with shape [rank, 2].
     */
    if (paddings->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "paddings tensor must be INT32");
        return false;
    }
    if (paddings_shape->size() != 2 || paddings_shape->Get(0) != static_cast<int32_t>(rank) || paddings_shape->Get(1) != 2) {
        SetError(operator_index, "paddings tensor must have shape [rank, 2]");
        return false;
    }
    if (rank == 0) {
        SetError(operator_index, "input tensor must have rank greater than zero");
        return false;
    }
    /*
     * PAD preserves the input tensor type.
     */
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    /*
     * Paddings are constant tensor data.
     */
    if (paddings->buffer() >= document_.BufferCount()) {
        SetError(operator_index, "paddings tensor buffer index out of range");
        return false;
    }
    const auto* buffers = document_.Buffers();
    if (buffers == nullptr || paddings->buffer() >= buffers->size()) {
        SetError(operator_index, "paddings tensor buffer index out of range");
        return false;
    }
    const auto* padding_buffer = buffers->Get(paddings->buffer());
    if (padding_buffer == nullptr || padding_buffer->data() == nullptr) {
        SetError(operator_index, "paddings tensor has no constant data");
        return false;
    }
    const size_t padding_count = rank * 2;
    if (padding_buffer->data()->size() < padding_count * sizeof(int32_t)) {
        SetError(operator_index, "paddings tensor buffer is too small");
        return false;
    }
    const auto* padding_data = reinterpret_cast<const int32_t*>(padding_buffer->data()->data());
    /*
     * Compute the expected output shape.
     *
     * output_dim = input_dim + pad_before + pad_after
     */
    for (size_t i = 0; i < rank; ++i) {
        const int32_t input_dimension = input_shape->Get(i);
        const int32_t pad_before = padding_data[i * 2];
        const int32_t pad_after = padding_data[i * 2 + 1];
        if (input_dimension < 0) {
            SetError(operator_index, "input tensor contains a negative dimension");
            return false;
        }
        if (pad_before < 0 || pad_after < 0) {
            SetError(operator_index, "padding values must not be negative");
            return false;
        }
        const int64_t expected_dimension = static_cast<int64_t>(input_dimension) + static_cast<int64_t>(pad_before) + static_cast<int64_t>(pad_after);
        if (expected_dimension > INT32_MAX) {
            SetError(operator_index, "padded dimension overflows");
            return false;
        }
        if (output_shape->Get(i) != static_cast<int32_t>(expected_dimension)) {
            SetError(operator_index, "output shape does not match padding");
            return false;
        }
    }
    if (output_shape->size() != rank) {
        SetError(operator_index, "output rank does not match input rank");
        return false;
    }
    return true;
}

bool ModelValidator::ValidatePadOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() == tflite::BuiltinOptions_NONE)
        return true;
    if (op->builtin_options_type() != tflite::BuiltinOptions_PadOptions) {
        SetError(operator_index, "invalid PAD options type");
        return false;
    }
    const auto* options = op->builtin_options_as_PadOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing PAD options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidatePadV2(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 3) {
        SetError(operator_index, "requires input, paddings and constant value tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t paddings_index = inputs->Get(1);
    const int32_t constant_value_index = inputs->Get(2);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || paddings_index < 0 || constant_value_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(paddings_index) >= tensors->size() || static_cast<size_t>(constant_value_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* paddings = tensors->Get(static_cast<size_t>(paddings_index));
    const auto* constant_value = tensors->Get(static_cast<size_t>(constant_value_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || paddings == nullptr || constant_value == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* paddings_shape = paddings->shape();
    const auto* constant_value_shape = constant_value->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || paddings_shape == nullptr || constant_value_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (paddings->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "PAD_V2 paddings tensor must be INT32");
        return false;
    }
    if (paddings_shape->size() != 2 ||
        paddings_shape->Get(0) != static_cast<int32_t>(input_shape->size()) ||
        paddings_shape->Get(1) != 2) {
        SetError(operator_index, "PAD_V2 paddings shape is invalid");
        return false;
    }
    if (constant_value->type() != input->type()) {
        SetError(operator_index, "PAD_V2 constant value type does not match input");
        return false;
    }
    if (constant_value_shape->size() != 0 && (constant_value_shape->size() != 1 || constant_value_shape->Get(0) != 1)) {
        SetError(operator_index, "PAD_V2 constant value must be scalar");
        return false;
    }
    if (output->type() != input->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (output_shape->size() != input_shape->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    return true;
}

bool ModelValidator::ValidatePadV2Options(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() == tflite::BuiltinOptions_NONE)
        return true;
    if (op->builtin_options_type() != tflite::BuiltinOptions_PadV2Options) {
        SetError(operator_index, "invalid PAD_V2 options type");
        return false;
    }
    const auto* options = op->builtin_options_as_PadV2Options();
    if (options == nullptr) {
        SetError(operator_index, "missing PAD_V2 options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidatePcan(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 3) {
        SetError(operator_index, "PCAN requires exactly three input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "PCAN requires exactly one output tensor");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const tflite::TensorType input_types[] = {
        tflite::TensorType_UINT32,
        tflite::TensorType_UINT32,
        tflite::TensorType_INT16
    };
    for (size_t i = 0; i < inputs->size(); ++i) {
        const int32_t input_index = inputs->Get(i);
        if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
            SetError(operator_index, "contains an invalid input tensor index");
            return false;
        }
        const auto* input = tensors->Get(static_cast<size_t>(input_index));
        if (input == nullptr || input->shape() == nullptr) {
            SetError(operator_index, "contains an invalid input tensor");
            return false;
        }
        if (input->shape()->size() != 1) {
            SetError(operator_index, "PCAN input tensor must be 1D");
            return false;
        }
        if (input->type() != input_types[i]) {
            SetError(operator_index, "PCAN input tensor has an invalid type");
            return false;
        }
    }
    const int32_t output_index = outputs->Get(0);
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (output->shape()->size() != 1) {
        SetError(operator_index, "PCAN output tensor must be 1D");
        return false;
    }
    if (output->type() != tflite::TensorType_UINT32) {
        SetError(operator_index, "PCAN output tensor must be UINT32");
        return false;
    }
    return true;
}

bool ModelValidator::ValidatePrelu(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and alpha tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t alpha_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || alpha_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(alpha_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* alpha = tensors->Get(static_cast<size_t>(alpha_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || alpha == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input->shape() == nullptr || alpha->shape() == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input->type() != alpha->type() || input->type() != output->type()) {
        SetError(operator_index, "input, alpha and output types do not match");
        return false;
    }
    if (input->shape()->size() != output->shape()->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    for (size_t i = 0; i < input->shape()->size(); ++i) {
        if (input->shape()->Get(i) != output->shape()->Get(i)) {
            SetError(operator_index, "input and output shapes do not match");
            return false;
        }
    }
    const size_t alpha_rank = alpha->shape()->size();
    if (alpha_rank == 0)
        return true;
    if (alpha_rank != 1) {
        SetError(operator_index, "PRELU alpha tensor must be scalar or one-dimensional");
        return false;
    }
    if (input->shape()->size() == 0 || alpha->shape()->Get(0) != input->shape()->Get(input->shape()->size() - 1)) {
        SetError(operator_index, "PRELU alpha size does not match input channels");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateQuantize(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input->shape() == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "input or output tensor has no shape");
        return false;
    }
    if (input->shape()->size() != output->shape()->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    for (size_t i = 0; i < input->shape()->size(); ++i) {
        if (input->shape()->Get(i) != output->shape()->Get(i)) {
            SetError(operator_index, "input and output shapes do not match");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateReadVariable(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one resource handle tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t resource_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (resource_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(resource_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* resource = tensors->Get(static_cast<size_t>(resource_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (resource == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (resource->type() != tflite::TensorType_RESOURCE) {
        SetError(operator_index, "resource handle tensor must be RESOURCE");
        return false;
    }
    if (output->shape() == nullptr) {
        SetError(operator_index, "output tensor has no shape");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateReduceAll(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and axis tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t axis_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || axis_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(axis_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* axis = tensors->Get(static_cast<size_t>(axis_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || axis == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* axis_shape = axis->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || axis_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input->type() != tflite::TensorType_BOOL) {
        SetError(operator_index, "REDUCE_ALL input tensor must be BOOL");
        return false;
    }
    if (axis->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "axis tensor must be INT32");
        return false;
    }
    if (output->type() != tflite::TensorType_BOOL) {
        SetError(operator_index, "REDUCE_ALL output tensor must be BOOL");
        return false;
    }
    if (axis_shape->size() != 1 || axis_shape->Get(0) <= 0) {
        SetError(operator_index, "axis tensor must be one-dimensional");
        return false;
    }
    const size_t input_rank = input_shape->size();
    if (input_rank == 0) {
        SetError(operator_index, "input tensor must have rank greater than zero");
        return false;
    }
    if (axis_shape->Get(0) > static_cast<int32_t>(input_rank)) {
        SetError(operator_index, "axis count exceeds input rank");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_ReducerOptions) {
        SetError(operator_index, "invalid REDUCE_ALL options type");
        return false;
    }
    const auto* options = op->builtin_options_as_ReducerOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing REDUCE_ALL options");
        return false;
    }
    if (options->keep_dims()) {
        if (output_shape->size() != input_rank) {
            SetError(operator_index, "output rank does not match REDUCE_ALL keep_dims");
            return false;
        }
    } else {
        if (output_shape->size() != input_rank - static_cast<size_t>(axis_shape->Get(0))) {
            SetError(operator_index, "output rank does not match REDUCE_ALL reduction");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateReduceAllOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_ReducerOptions) {
        SetError(operator_index, "invalid REDUCE_ALL options type");
        return false;
    }
    const auto* options = op->builtin_options_as_ReducerOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing REDUCE_ALL options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateReduceMax(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and axis tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t axis_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || axis_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(axis_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* axis = tensors->Get(static_cast<size_t>(axis_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || axis == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* axis_shape = axis->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || axis_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (axis->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "axis tensor must be INT32");
        return false;
    }
    if (axis_shape->size() != 1 || axis_shape->Get(0) <= 0) {
        SetError(operator_index, "axis tensor must be one-dimensional");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    switch (input->type()) {
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_FLOAT16:
    case tflite::TensorType_INT8:
    case tflite::TensorType_UINT8:
    case tflite::TensorType_INT16:
    case tflite::TensorType_INT32:
    case tflite::TensorType_INT64:
        break;
    default:
        SetError(operator_index, "unsupported REDUCE_MAX tensor type");
        return false;
    }
    const size_t input_rank = input_shape->size();
    if (input_rank == 0) {
        SetError(operator_index, "input tensor must have rank greater than zero");
        return false;
    }
    if (axis_shape->Get(0) > static_cast<int32_t>(input_rank)) {
        SetError(operator_index, "axis count exceeds input rank");
        return false;
    }
    const auto* options = op->builtin_options_as_ReducerOptions();
    if (op->builtin_options_type() != tflite::BuiltinOptions_ReducerOptions || options == nullptr) {
        SetError(operator_index, "invalid REDUCE_MAX options");
        return false;
    }
    if (options->keep_dims()) {
        if (output_shape->size() != input_rank) {
            SetError(operator_index, "output rank does not match REDUCE_MAX keep_dims");
            return false;
        }
    } else {
        if (output_shape->size() != input_rank - static_cast<size_t>(axis_shape->Get(0))) {
            SetError(operator_index, "output rank does not match REDUCE_MAX reduction");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateReduceMaxOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_ReducerOptions) {
        SetError(operator_index, "invalid REDUCE_MAX options type");
        return false;
    }
    const auto* options = op->builtin_options_as_ReducerOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing REDUCE_MAX options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateReduceMin(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and axis tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t axis_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || axis_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(axis_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* axis = tensors->Get(static_cast<size_t>(axis_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || axis == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* axis_shape = axis->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || axis_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (axis->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "axis tensor must be INT32");
        return false;
    }
    if (axis_shape->size() != 1 || axis_shape->Get(0) <= 0) {
        SetError(operator_index, "axis tensor must be one-dimensional");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    switch (input->type()) {
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_FLOAT16:
    case tflite::TensorType_INT8:
    case tflite::TensorType_UINT8:
    case tflite::TensorType_INT16:
    case tflite::TensorType_INT32:
    case tflite::TensorType_INT64:
        break;
    default:
        SetError(operator_index, "unsupported REDUCE_MIN tensor type");
        return false;
    }
    const size_t input_rank = input_shape->size();
    if (input_rank == 0) {
        SetError(operator_index, "input tensor must have rank greater than zero");
        return false;
    }
    if (axis_shape->Get(0) > static_cast<int32_t>(input_rank)) {
        SetError(operator_index, "axis count exceeds input rank");
        return false;
    }
    const auto* options = op->builtin_options_as_ReducerOptions();
    if (op->builtin_options_type() != tflite::BuiltinOptions_ReducerOptions || options == nullptr) {
        SetError(operator_index, "invalid REDUCE_MIN options");
        return false;
    }
    if (options->keep_dims()) {
        if (output_shape->size() != input_rank) {
            SetError(operator_index, "output rank does not match REDUCE_MIN keep_dims");
            return false;
        }
    } else {
        if (output_shape->size() != input_rank - static_cast<size_t>(axis_shape->Get(0))) {
            SetError(operator_index, "output rank does not match REDUCE_MIN reduction");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateReduceMinOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_ReducerOptions) {
        SetError(operator_index, "invalid REDUCE_MIN options type");
        return false;
    }
    const auto* options = op->builtin_options_as_ReducerOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing REDUCE_MIN options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateRelu(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateUnaryElementwise(operator_index, op);
}

bool ModelValidator::ValidateRelu6(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateUnaryElementwise(operator_index, op);
}

bool ModelValidator::ValidateReshape(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() < 2) {
        SetError(operator_index, "requires input and shape tensors");
        return false;
    }
    if (outputs->size() < 1) {
        SetError(operator_index, "has no output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t shape_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || shape_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(shape_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* shape = tensors->Get(static_cast<size_t>(shape_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || shape == nullptr || output == nullptr) {
        SetError(operator_index, "missing input, shape or output tensor");
        return false;
    }
    if (shape->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "shape tensor must be INT32");
        return false;
    }
    if (shape->shape() == nullptr || shape->shape()->size() != 1) {
        SetError(operator_index, "shape tensor must be one-dimensional");
        return false;
    }
    if (shape->shape()->Get(0) <= 0) {
        SetError(operator_index, "invalid shape tensor size");
        return false;
    }
    if (input->shape() == nullptr || input->shape()->size() == 0) {
        SetError(operator_index, "input tensor has no shape");
        return false;
    }
    if (output->shape() == nullptr || output->shape()->size() == 0) {
        SetError(operator_index, "output tensor has no shape");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output types do not match");
        return false;
    }
    if (shape->buffer() >= document_.BufferCount()) {
        SetError(operator_index, "shape tensor buffer index out of range");
        return false;
    }
    const auto* buffers = document_.Buffers();
    if (buffers == nullptr || shape->buffer() >= buffers->size()) {
        SetError(operator_index, "shape tensor buffer index out of range");
        return false;
    }
    const auto* shape_buffer = buffers->Get(shape->buffer());
    if (shape_buffer != nullptr && shape_buffer->data() != nullptr) {
        if (shape_buffer->data()->size() < static_cast<size_t>(shape->shape()->Get(0)) * sizeof(int32_t)) {
            SetError(operator_index, "shape tensor buffer is too small");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateResizeBilinear(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and size tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t size_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || size_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(size_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* size = tensors->Get(static_cast<size_t>(size_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || size == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* size_shape = size->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || size_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input_shape->size() != 4) {
        SetError(operator_index, "input tensor must be 4-dimensional");
        return false;
    }
    if (output_shape->size() != 4) {
        SetError(operator_index, "output tensor must be 4-dimensional");
        return false;
    }
    if (size->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "resize size tensor must be INT32");
        return false;
    }
    if (size_shape->size() != 1 || size_shape->Get(0) != 2) {
        SetError(operator_index, "resize size tensor must contain two values");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateResizeBilinearOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_ResizeBilinearOptions) {
        SetError(operator_index, "invalid RESIZE_BILINEAR options type");
        return false;
    }
    const auto* options = op->builtin_options_as_ResizeBilinearOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing RESIZE_BILINEAR options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateResizeNearestNeighbor(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and size tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t size_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || size_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(size_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* size = tensors->Get(static_cast<size_t>(size_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || size == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* size_shape = size->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || size_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input_shape->size() != 4) {
        SetError(operator_index, "input tensor must be 4-dimensional");
        return false;
    }
    if (output_shape->size() != 4) {
        SetError(operator_index, "output tensor must be 4-dimensional");
        return false;
    }
    if (size->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "resize size tensor must be INT32");
        return false;
    }
    if (size_shape->size() != 1 || size_shape->Get(0) != 2) {
        SetError(operator_index, "resize size tensor must contain two values");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateResizeNearestNeighborOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_ResizeNearestNeighborOptions) {
        SetError(operator_index, "invalid RESIZE_NEAREST_NEIGHBOR options type");
        return false;
    }
    const auto* options = op->builtin_options_as_ResizeNearestNeighborOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing RESIZE_NEAREST_NEIGHBOR options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateReverseV2(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and axis tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t axis_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || axis_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(axis_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* axis = tensors->Get(static_cast<size_t>(axis_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || axis == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* axis_shape = axis->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || axis_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (axis->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "axis tensor must be INT32");
        return false;
    }
    if (axis_shape->size() != 1) {
        SetError(operator_index, "axis tensor must be one-dimensional");
        return false;
    }
    if (axis_shape->Get(0) <= 0) {
        SetError(operator_index, "axis tensor must contain at least one axis");
        return false;
    }
    if (input_shape->size() != output_shape->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    for (size_t i = 0; i < input_shape->size(); ++i) {
        if (input_shape->Get(i) != output_shape->Get(i)) {
            SetError(operator_index, "input and output shapes do not match");
            return false;
        }
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateRfft(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "RFFT requires exactly one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "RFFT requires exactly one output tensor");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid input tensor index");
        return false;
    }
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || input->shape() == nullptr) {
        SetError(operator_index, "contains an invalid input tensor");
        return false;
    }
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (input->shape()->size() != output->shape()->size()) {
        SetError(operator_index, "RFFT input and output ranks do not match");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "RFFT input and output tensor types do not match");
        return false;
    }
    if (input->type() != tflite::TensorType_INT16 &&
        input->type() != tflite::TensorType_INT32 &&
        input->type() != tflite::TensorType_FLOAT32) {
        SetError(operator_index, "RFFT tensor type is unsupported");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateRfft2D(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and FFT length tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t fft_length_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || fft_length_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(fft_length_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* fft_length = tensors->Get(static_cast<size_t>(fft_length_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || fft_length == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* fft_length_shape = fft_length->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || fft_length_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input->type() != tflite::TensorType_FLOAT32) {
        SetError(operator_index, "RFFT input tensor must be FLOAT32");
        return false;
    }
    if (fft_length->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "RFFT FFT length tensor must be INT32");
        return false;
    }
    if (output->type() != tflite::TensorType_COMPLEX64) {
        SetError(operator_index, "RFFT output tensor must be COMPLEX64");
        return false;
    }
    if (input_shape->size() == 0) {
        SetError(operator_index, "RFFT input tensor must have rank greater than zero");
        return false;
    }
    if (fft_length_shape->size() != 0 && (fft_length_shape->size() != 1 || fft_length_shape->Get(0) != 1)) {
        SetError(operator_index, "RFFT FFT length tensor must be scalar");
        return false;
    }
    if (output_shape->size() != input_shape->size()) {
        SetError(operator_index, "RFFT input and output ranks do not match");
        return false;
    }
    for (size_t i = 0; i + 1 < input_shape->size(); ++i) {
        if (output_shape->Get(i) != input_shape->Get(i)) {
            SetError(operator_index, "RFFT input and output batch dimensions do not match");
            return false;
        }
    }
    if (output_shape->Get(output_shape->size() - 1) <= 0) {
        SetError(operator_index, "RFFT output FFT dimension is invalid");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateRound(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateUnaryElementwise(operator_index, op);
}

bool ModelValidator::ValidateRsqrt(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateUnaryElementwise(operator_index, op);
}

bool ModelValidator::ValidateSelectV2(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 3) {
        SetError(operator_index, "requires condition, then and else tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t condition_index = inputs->Get(0);
    const int32_t then_index = inputs->Get(1);
    const int32_t else_index = inputs->Get(2);
    const int32_t output_index = outputs->Get(0);
    if (condition_index < 0 || then_index < 0 || else_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(condition_index) >= tensors->size() || static_cast<size_t>(then_index) >= tensors->size() || static_cast<size_t>(else_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* condition = tensors->Get(static_cast<size_t>(condition_index));
    const auto* then_value = tensors->Get(static_cast<size_t>(then_index));
    const auto* else_value = tensors->Get(static_cast<size_t>(else_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (condition == nullptr || then_value == nullptr || else_value == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (condition->shape() == nullptr || then_value->shape() == nullptr || else_value->shape() == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (condition->type() != tflite::TensorType_BOOL) {
        SetError(operator_index, "SELECT_V2 condition tensor must be BOOL");
        return false;
    }
    if (then_value->type() != else_value->type() || then_value->type() != output->type()) {
        SetError(operator_index, "SELECT_V2 value and output types do not match");
        return false;
    }
    if (!ValidateBinaryBroadcast(operator_index, then_value, else_value, output)) {
        return false;
    }
    const auto* condition_shape = condition->shape();
    const auto* then_shape = then_value->shape();
    const auto* output_shape = output->shape();
    const size_t condition_rank = condition_shape->size();
    const size_t then_rank = then_shape->size();
    const size_t output_rank = output_shape->size();
    const size_t broadcast_rank = condition_rank > then_rank ? condition_rank : then_rank;
    if (output_rank != broadcast_rank) {
        SetError(operator_index, "output rank does not match SELECT_V2 broadcast rank");
        return false;
    }
    for (size_t i = 0; i < broadcast_rank; ++i) {
        const int32_t condition_dimension = i < broadcast_rank - condition_rank ? 1 : condition_shape->Get(i - (broadcast_rank - condition_rank));
        const int32_t then_dimension = i < broadcast_rank - then_rank ? 1 : then_shape->Get(i - (broadcast_rank - then_rank));
        const int32_t output_dimension = output_shape->Get(i);
        if (condition_dimension < 0 || then_dimension < 0 || output_dimension < 0) {
            SetError(operator_index, "contains a negative tensor dimension");
            return false;
        }
        if (condition_dimension != then_dimension && condition_dimension != 1 && then_dimension != 1) {
            SetError(operator_index, "condition and value shapes are not broadcast compatible");
            return false;
        }
        const int32_t expected_dimension = condition_dimension > then_dimension ? condition_dimension : then_dimension;
        if (output_dimension != expected_dimension) {
            SetError(operator_index, "output shape does not match SELECT_V2 broadcast shape");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateShape(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input->shape() == nullptr) {
        SetError(operator_index, "input tensor has no shape");
        return false;
    }
    if (output->type() != tflite::TensorType_INT32 && output->type() != tflite::TensorType_INT64) {
        SetError(operator_index, "output tensor must be INT32 or INT64");
        return false;
    }
    if (output->shape() == nullptr || output->shape()->size() != 1) {
        SetError(operator_index, "output tensor must be one-dimensional");
        return false;
    }
    if (output->shape()->Get(0) != static_cast<int32_t>(input->shape()->size())) {
        SetError(operator_index, "output size does not match input rank");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateShapeOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_ShapeOptions) {
        SetError(operator_index, "invalid SHAPE options type");
        return false;
    }
    const auto* options = op->builtin_options_as_ShapeOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing SHAPE options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSin(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateUnaryElementwise(operator_index, op);
}

bool ModelValidator::ValidateSlice(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 3) {
        SetError(operator_index, "requires input, begin and size tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t begin_index = inputs->Get(1);
    const int32_t size_index = inputs->Get(2);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || begin_index < 0 || size_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(begin_index) >= tensors->size() || static_cast<size_t>(size_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* begin = tensors->Get(static_cast<size_t>(begin_index));
    const auto* size = tensors->Get(static_cast<size_t>(size_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || begin == nullptr || size == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* begin_shape = begin->shape();
    const auto* size_shape = size->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || begin_shape == nullptr || size_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (begin->type() != tflite::TensorType_INT32 || size->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "begin and size tensors must be INT32");
        return false;
    }
    if (begin_shape->size() != 1 || size_shape->size() != 1) {
        SetError(operator_index, "begin and size tensors must be one-dimensional");
        return false;
    }
    if (begin_shape->Get(0) != static_cast<int32_t>(input_shape->size()) || size_shape->Get(0) != static_cast<int32_t>(input_shape->size())) {
        SetError(operator_index, "begin and size lengths must match input rank");
        return false;
    }
    if (output_shape->size() != input_shape->size()) {
        SetError(operator_index, "output rank does not match input rank");
        return false;
    }
    if (output->type() != input->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSoftmax(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() < 1) {
        SetError(operator_index, "requires input tensor");
        return false;
    }
    if (outputs->size() < 1) {
        SetError(operator_index, "has no output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "missing input or output tensor");
        return false;
    }
    if (input->type() != tflite::TensorType_FLOAT32 &&
        input->type() != tflite::TensorType_INT8 &&
        input->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "unsupported SOFTMAX tensor type");
        return false;
    }
    if (output->type() != tflite::TensorType_FLOAT32 &&
        output->type() != tflite::TensorType_INT8 &&
        output->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "unsupported output tensor type");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_SoftmaxOptions) {
        SetError(operator_index, "invalid builtin options type");
        return false;
    }
    const auto* options = op->builtin_options_as_SoftmaxOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing SOFTMAX options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSpaceToBatchNd(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 3) {
        SetError(operator_index, "requires input, block shape and paddings tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t block_shape_index = inputs->Get(1);
    const int32_t paddings_index = inputs->Get(2);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || block_shape_index < 0 || paddings_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(block_shape_index) >= tensors->size() || static_cast<size_t>(paddings_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* block_shape = tensors->Get(static_cast<size_t>(block_shape_index));
    const auto* paddings = tensors->Get(static_cast<size_t>(paddings_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || block_shape == nullptr || paddings == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* block_shape_shape = block_shape->shape();
    const auto* paddings_shape = paddings->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || block_shape_shape == nullptr || paddings_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (block_shape->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "block shape tensor must be INT32");
        return false;
    }
    if (paddings->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "paddings tensor must be INT32");
        return false;
    }
    if (block_shape_shape->size() != 1) {
        SetError(operator_index, "block shape tensor must be one-dimensional");
        return false;
    }
    const int32_t block_rank = block_shape_shape->Get(0);
    if (block_rank <= 0) {
        SetError(operator_index, "block shape tensor must contain at least one dimension");
        return false;
    }
    if (paddings_shape->size() != 2 || paddings_shape->Get(0) != block_rank || paddings_shape->Get(1) != 2) {
        SetError(operator_index, "paddings tensor shape is invalid");
        return false;
    }
    if (input_shape->size() <= static_cast<size_t>(block_rank)) {
        SetError(operator_index, "block shape rank is incompatible with input rank");
        return false;
    }
    if (output->type() != input->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (output_shape->size() != input_shape->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSpaceToBatchNdOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() == tflite::BuiltinOptions_NONE)
        return true;
    if (op->builtin_options_type() != tflite::BuiltinOptions_SpaceToBatchNDOptions) {
        SetError(operator_index, "invalid SPACE_TO_BATCH_ND options type");
        return false;
    }
    const auto* options = op->builtin_options_as_SpaceToBatchNDOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing SPACE_TO_BATCH_ND options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSpaceToDepth(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input_shape->size() != 4) {
        SetError(operator_index, "input tensor must be 4-dimensional");
        return false;
    }
    if (output_shape->size() != 4) {
        SetError(operator_index, "output tensor must be 4-dimensional");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_SpaceToDepthOptions) {
        SetError(operator_index, "invalid SPACE_TO_DEPTH options type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSpaceToDepthOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_SpaceToDepthOptions) {
        SetError(operator_index, "invalid SPACE_TO_DEPTH options type");
        return false;
    }
    const auto* options = op->builtin_options_as_SpaceToDepthOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing SPACE_TO_DEPTH options");
        return false;
    }
    if (options->block_size() <= 1) {
        SetError(operator_index, "SPACE_TO_DEPTH block size must be greater than one");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSplit(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires axis and input tensors");
        return false;
    }
    if (outputs->size() == 0) {
        SetError(operator_index, "requires at least one output tensor");
        return false;
    }
    const int32_t axis_index = inputs->Get(0);
    const int32_t input_index = inputs->Get(1);
    if (axis_index < 0 || input_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(axis_index) >= tensors->size() || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* axis = tensors->Get(static_cast<size_t>(axis_index));
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    if (axis == nullptr || input == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (axis->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "axis tensor must be INT32");
        return false;
    }
    const auto* axis_shape = axis->shape();
    const auto* input_shape = input->shape();
    if (axis_shape == nullptr || input_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (axis_shape->size() != 0 && (axis_shape->size() != 1 || axis_shape->Get(0) != 1)) {
        SetError(operator_index, "axis tensor must be scalar");
        return false;
    }
    const size_t input_rank = input_shape->size();
    if (input_rank == 0) {
        SetError(operator_index, "input tensor must have rank greater than zero");
        return false;
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
        const int32_t output_index = outputs->Get(i);
        if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
            SetError(operator_index, "contains an invalid output tensor index");
            return false;
        }
        const auto* output = tensors->Get(static_cast<size_t>(output_index));
        if (output == nullptr) {
            SetError(operator_index, "contains null output tensor");
            return false;
        }
        const auto* output_shape = output->shape();
        if (output_shape == nullptr) {
            SetError(operator_index, "output tensor has no shape");
            return false;
        }
        if (output->type() != input->type()) {
            SetError(operator_index, "input and output tensor types do not match");
            return false;
        }
        if (output_shape->size() != input_rank) {
            SetError(operator_index, "output rank does not match input rank");
            return false;
        }
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_SplitOptions) {
        SetError(operator_index, "invalid SPLIT options type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSplitOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_SplitOptions) {
        SetError(operator_index, "invalid SPLIT options type");
        return false;
    }
    const auto* options = op->builtin_options_as_SplitOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing SPLIT options");
        return false;
    }
    if (options->num_splits() <= 0) {
        SetError(operator_index, "SPLIT num_splits must be positive");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSplitV(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 3) {
        SetError(operator_index, "requires input, size splits and axis tensors");
        return false;
    }
    if (outputs->size() == 0) {
        SetError(operator_index, "requires at least one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t size_splits_index = inputs->Get(1);
    const int32_t axis_index = inputs->Get(2);
    if (input_index < 0 || size_splits_index < 0 || axis_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(size_splits_index) >= tensors->size() || static_cast<size_t>(axis_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* size_splits = tensors->Get(static_cast<size_t>(size_splits_index));
    const auto* axis = tensors->Get(static_cast<size_t>(axis_index));
    if (input == nullptr || size_splits == nullptr || axis == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* size_splits_shape = size_splits->shape();
    const auto* axis_shape = axis->shape();
    if (input_shape == nullptr || size_splits_shape == nullptr || axis_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input_shape->size() == 0) {
        SetError(operator_index, "input tensor must have rank greater than zero");
        return false;
    }
    if (size_splits->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "size splits tensor must be INT32");
        return false;
    }
    if (axis->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "axis tensor must be INT32");
        return false;
    }
    if (size_splits_shape->size() != 1) {
        SetError(operator_index, "size splits tensor must be one-dimensional");
        return false;
    }
    if (size_splits_shape->Get(0) != static_cast<int32_t>(outputs->size())) {
        SetError(operator_index, "size splits count does not match output count");
        return false;
    }
    if (axis_shape->size() != 0 &&
        (axis_shape->size() != 1 || axis_shape->Get(0) != 1)) {
        SetError(operator_index, "axis tensor must be scalar");
        return false;
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
        const int32_t output_index = outputs->Get(i);
        if (output_index < 0 ||
            static_cast<size_t>(output_index) >= tensors->size()) {
            SetError(operator_index, "contains an invalid output tensor index");
            return false;
        }
        const auto* output = tensors->Get(static_cast<size_t>(output_index));
        if (output == nullptr || output->shape() == nullptr) {
            SetError(operator_index, "output tensor has no shape");
            return false;
        }
        if (output->type() != input->type()) {
            SetError(operator_index, "input and output tensor types do not match");
            return false;
        }
        if (output->shape()->size() != input_shape->size()) {
            SetError(operator_index, "output rank does not match input rank");
            return false;
        }
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_SplitVOptions) {
        SetError(operator_index, "invalid SPLIT_V options type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSplitVOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_SplitVOptions) {
        SetError(operator_index, "invalid SPLIT_V options type");
        return false;
    }
    const auto* options = op->builtin_options_as_SplitVOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing SPLIT_V options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSqrt(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateUnaryElementwise(operator_index, op);
}

bool ModelValidator::ValidateSquare(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateUnaryElementwise(operator_index, op);
}

bool ModelValidator::ValidateSquaredDifference(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input1_index = inputs->Get(0);
    const int32_t input2_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input1_index < 0 || input2_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input1_index) >= tensors->size() || static_cast<size_t>(input2_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input1 = tensors->Get(static_cast<size_t>(input1_index));
    const auto* input2 = tensors->Get(static_cast<size_t>(input2_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input1 == nullptr || input2 == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input1->type() != input2->type() || input1->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    switch (input1->type()) {
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_FLOAT16:
    case tflite::TensorType_INT32:
    case tflite::TensorType_INT64:
    case tflite::TensorType_INT8:
    case tflite::TensorType_UINT8:
    case tflite::TensorType_INT16:
        break;
    default:
        SetError(operator_index, "unsupported SQUARED_DIFFERENCE tensor type");
        return false;
    }
    if (!ValidateBinaryBroadcast(operator_index, input1, input2, output)) {
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSqueeze(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (output_shape->size() > input_shape->size()) {
        SetError(operator_index, "output rank is greater than input rank");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSqueezeOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_SqueezeOptions) {
        SetError(operator_index, "invalid SQUEEZE options type");
        return false;
    }
    const auto* options = op->builtin_options_as_SqueezeOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing SQUEEZE options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateStacker(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "STACKER requires exactly one input tensor");
        return false;
    }
    if (outputs->size() != 2) {
        SetError(operator_index, "STACKER requires exactly two output tensors");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    const int32_t output_valid_index = outputs->Get(1);
    if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid input tensor index");
        return false;
    }
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    if (output_valid_index < 0 || static_cast<size_t>(output_valid_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output valid tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    const auto* output_valid = tensors->Get(static_cast<size_t>(output_valid_index));
    if (input == nullptr || input->shape() == nullptr) {
        SetError(operator_index, "contains an invalid input tensor");
        return false;
    }
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (output_valid == nullptr || output_valid->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output valid tensor");
        return false;
    }
    if (input->shape()->size() != 1) {
        SetError(operator_index, "STACKER input tensor must be 1D");
        return false;
    }
    if (output->shape()->size() != 1) {
        SetError(operator_index, "STACKER output tensor must be 1D");
        return false;
    }
    if (output_valid->shape()->size() != 0) {
        SetError(operator_index, "STACKER valid output tensor must be scalar");
        return false;
    }
    if (input->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "STACKER input tensor must be INT16");
        return false;
    }
    if (output->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "STACKER output tensor must be INT16");
        return false;
    }
    if (output_valid->type() != tflite::TensorType_BOOL) {
        SetError(operator_index, "STACKER valid output tensor must be BOOL");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateStridedSlice(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 4) {
        SetError(operator_index, "requires input, begin, end and strides tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t begin_index = inputs->Get(1);
    const int32_t end_index = inputs->Get(2);
    const int32_t strides_index = inputs->Get(3);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || begin_index < 0 || end_index < 0 || strides_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(begin_index) >= tensors->size() || static_cast<size_t>(end_index) >= tensors->size() || static_cast<size_t>(strides_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* begin = tensors->Get(static_cast<size_t>(begin_index));
    const auto* end = tensors->Get(static_cast<size_t>(end_index));
    const auto* strides = tensors->Get(static_cast<size_t>(strides_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || begin == nullptr || end == nullptr || strides == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (begin->type() != tflite::TensorType_INT32 || end->type() != tflite::TensorType_INT32 || strides->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "begin, end and strides tensors must be INT32");
        return false;
    }
    if (begin->shape() == nullptr || end->shape() == nullptr || strides->shape() == nullptr) {
        SetError(operator_index, "missing slice parameter shape");
        return false;
    }
    if (begin->shape()->size() != 1 || end->shape()->size() != 1 || strides->shape()->size() != 1) {
        SetError(operator_index, "slice parameter tensors must be one-dimensional");
        return false;
    }
    if (begin->shape()->Get(0) != end->shape()->Get(0) || begin->shape()->Get(0) != strides->shape()->Get(0)) {
        SetError(operator_index, "slice parameter sizes do not match");
        return false;
    }
    if (input->shape() == nullptr || input->shape()->size() == 0) {
        SetError(operator_index, "input tensor has no shape");
        return false;
    }
    if (output->shape() == nullptr) {
        SetError(operator_index, "output tensor has no shape");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output types do not match");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateStridedSliceOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_StridedSliceOptions) {
        SetError(operator_index, "invalid STRIDED_SLICE options type");
        return false;
    }
    const auto* options = op->builtin_options_as_StridedSliceOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing STRIDED_SLICE options");
        return false;
    }
    if (options->ellipsis_mask() != 0 && (options->ellipsis_mask() & (options->ellipsis_mask() - 1)) != 0) {
        SetError(operator_index, "ellipsis mask must contain at most one bit");
        return false;
    }
    if (options->new_axis_mask() != 0 && (options->new_axis_mask() & options->ellipsis_mask()) != 0) {
        SetError(operator_index, "new axis and ellipsis masks overlap");
        return false;
    }
    if (options->shrink_axis_mask() != 0 && (options->shrink_axis_mask() & options->new_axis_mask()) != 0) {
        SetError(operator_index, "shrink axis and new axis masks overlap");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSub(size_t operator_index,const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input1_index = inputs->Get(0);
    const int32_t input2_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input1_index < 0 || input2_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input1_index) >= tensors->size() || static_cast<size_t>(input2_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input1 = tensors->Get(static_cast<size_t>(input1_index));
    const auto* input2 = tensors->Get(static_cast<size_t>(input2_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input1 == nullptr || input2 == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (input1->type() != input2->type() || input1->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    switch (input1->type()) {
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_FLOAT16:
    case tflite::TensorType_INT8:
    case tflite::TensorType_UINT8:
    case tflite::TensorType_INT16:
        break;
    default:
        SetError(operator_index, "unsupported SUB tensor type");
        return false;
    }
    if (!ValidateBinaryBroadcast(operator_index, input1, input2, output)) {
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSubOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() == tflite::BuiltinOptions_NONE)
        return true;
    if (op->builtin_options_type() != tflite::BuiltinOptions_SubOptions) {
        SetError(operator_index, "invalid SUB options type");
        return false;
    }
    const auto* options = op->builtin_options_as_SubOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing SUB options");
        return false;
    }
    switch (options->fused_activation_function()) {
    case tflite::ActivationFunctionType_NONE:
    case tflite::ActivationFunctionType_RELU:
    case tflite::ActivationFunctionType_RELU6:
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
        break;
    default:
        SetError(operator_index, "invalid fused activation function");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSum(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and axis tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t axis_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || axis_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(axis_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* axis = tensors->Get(static_cast<size_t>(axis_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || axis == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* axis_shape = axis->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || axis_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (axis->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "axis tensor must be INT32");
        return false;
    }
    if (axis_shape->size() != 1 || axis_shape->Get(0) <= 0) {
        SetError(operator_index, "axis tensor must be one-dimensional");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_ReducerOptions) {
        SetError(operator_index, "invalid SUM options type");
        return false;
    }
    const auto* options = op->builtin_options_as_ReducerOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing SUM options");
        return false;
    }
    const size_t input_rank = input_shape->size();
    if (input_rank == 0) {
        SetError(operator_index, "input tensor must have rank greater than zero");
        return false;
    }
    const size_t axis_count = static_cast<size_t>(axis_shape->Get(0));
    if (axis_count > input_rank) {
        SetError(operator_index, "SUM axis count exceeds input rank");
        return false;
    }
    if (options->keep_dims()) {
        if (output_shape->size() != input_rank) {
            SetError(operator_index, "output rank does not match SUM keep_dims");
            return false;
        }
    } else {
        if (output_shape->size() != input_rank - axis_count) {
            SetError(operator_index, "output rank does not match SUM reduction");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateSvdf(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 5) {
        SetError(operator_index, "requires input, feature weights, time weights, bias and state tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t feature_weights_index = inputs->Get(1);
    const int32_t time_weights_index = inputs->Get(2);
    const int32_t bias_index = inputs->Get(3);
    const int32_t state_index = inputs->Get(4);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || feature_weights_index < 0 || time_weights_index < 0 || bias_index < 0 || state_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(feature_weights_index) >= tensors->size() || static_cast<size_t>(time_weights_index) >= tensors->size() || static_cast<size_t>(bias_index) >= tensors->size() || static_cast<size_t>(state_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* feature_weights = tensors->Get(static_cast<size_t>(feature_weights_index));
    const auto* time_weights = tensors->Get(static_cast<size_t>(time_weights_index));
    const auto* bias = tensors->Get(static_cast<size_t>(bias_index));
    const auto* state = tensors->Get(static_cast<size_t>(state_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || feature_weights == nullptr || time_weights == nullptr || bias == nullptr || state == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (feature_weights->shape() == nullptr || feature_weights->shape()->size() != 2) {
        SetError(operator_index, "feature weights tensor must be rank 2");
        return false;
    }
    if (time_weights->shape() == nullptr || time_weights->shape()->size() != 2) {
        SetError(operator_index, "time weights tensor must be rank 2");
        return false;
    }
    if (bias->shape() == nullptr || bias->shape()->size() != 1) {
        SetError(operator_index, "bias tensor must be rank 1");
        return false;
    }
    if (state->shape() == nullptr || state->shape()->size() != 2) {
        SetError(operator_index, "state tensor must be rank 2");
        return false;
    }
    if (output->shape() == nullptr || output->shape()->size() != 2) {
        SetError(operator_index, "output tensor must be rank 2");
        return false;
    }
    if (input->shape() == nullptr || input->shape()->size() != 2) {
        SetError(operator_index, "input tensor must be rank 2");
        return false;
    }
    if (feature_weights->type() != input->type()) {
        SetError(operator_index, "SVDF feature weights type does not match input type");
        return false;
    }
    if (output->type() != input->type()) {
        SetError(operator_index, "SVDF output type does not match input type");
        return false;
    }
    if (time_weights->type() != tflite::TensorType_INT16 && time_weights->type() != input->type()) {
        SetError(operator_index, "SVDF time weights type is incompatible with input type");
        return false;
    }
    if (state->type() != time_weights->type()) {
        SetError(operator_index, "SVDF state type does not match time weights type");
        return false;
    }
    if (bias->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "SVDF bias must be INT32");
        return false;
    }
    if (bias->type() != tflite::TensorType_INT32 && bias->type() != input->type()) {
        SetError(operator_index, "SVDF bias type is incompatible with input type");
        return false;
    }
    const int32_t batch_size = input->shape()->Get(0);
    const int32_t input_size = input->shape()->Get(1);
    const int32_t units = feature_weights->shape()->Get(0);
    const int32_t feature_input_size = feature_weights->shape()->Get(1);
    const int32_t time_units = time_weights->shape()->Get(1);
    if (batch_size <= 0 || input_size <= 0 || units <= 0 || feature_input_size <= 0 || time_units <= 0) {
        SetError(operator_index, "contains invalid SVDF tensor dimensions");
        return false;
    }
    if (input_size != feature_input_size) {
        SetError(operator_index, "SVDF feature weights input size does not match input size");
        return false;
    }
    if (bias->shape()->Get(0) != units) {
        SetError(operator_index, "SVDF bias size does not match feature weight output size");
        return false;
    }
    if (output->shape()->Get(0) != batch_size) {
        SetError(operator_index, "SVDF output batch size does not match input batch size");
        return false;
    }
    if (output->shape()->Get(1) != units) {
        SetError(operator_index, "SVDF output size does not match feature weight output size");
        return false;
    }
    if (state->shape()->Get(0) != batch_size) {
        SetError(operator_index, "SVDF state batch size does not match input batch size");
        return false;
    }
    if (state->shape()->Get(1) != units * time_units) {
        SetError(operator_index, "SVDF state size does not match units and time weights");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateSvdfOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_SVDFOptions) {
        SetError(operator_index, "invalid SVDF options type");
        return false;
    }
    const auto* options = op->builtin_options_as_SVDFOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing SVDF options");
        return false;
    }
    if (options->rank() <= 0) {
        SetError(operator_index, "SVDF rank must be positive");
        return false;
    }
    switch (options->fused_activation_function()) {
    case tflite::ActivationFunctionType_NONE:
    case tflite::ActivationFunctionType_RELU:
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
    case tflite::ActivationFunctionType_RELU6:
    case tflite::ActivationFunctionType_TANH:
        break;
    default:
        SetError(operator_index, "invalid fused activation function");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateTanh(size_t operator_index, const tflite::Operator* op) const
{
    return ValidateUnaryElementwise(operator_index, op);
}

bool ModelValidator::ValidateTranspose(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "requires input and permutation tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t permutation_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || permutation_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(permutation_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* permutation = tensors->Get(static_cast<size_t>(permutation_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || permutation == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* permutation_shape = permutation->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || permutation_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    const size_t rank = input_shape->size();
    if (rank == 0) {
        SetError(operator_index, "input tensor must have rank greater than zero");
        return false;
    }
    if (permutation->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "permutation tensor must be INT32");
        return false;
    }
    if (permutation_shape->size() != 1 || permutation_shape->Get(0) != static_cast<int32_t>(rank)) {
        SetError(operator_index, "permutation tensor must have shape [rank]");
        return false;
    }
    if (output_shape->size() != rank) {
        SetError(operator_index, "output rank does not match input rank");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output types do not match");
        return false;
    }
    if (permutation->buffer() >= document_.BufferCount()) {
        SetError(operator_index, "permutation tensor buffer index out of range");
        return false;
    }
    const auto* buffers = document_.Buffers();
    if (buffers == nullptr || permutation->buffer() >= buffers->size()) {
        SetError(operator_index, "permutation tensor buffer index out of range");
        return false;
    }
    const auto* permutation_buffer = buffers->Get(permutation->buffer());
    if (permutation_buffer == nullptr || permutation_buffer->data() == nullptr) {
        SetError(operator_index, "permutation tensor has no constant data");
        return false;
    }
    const size_t permutation_count = rank;
    if (permutation_buffer->data()->size() < permutation_count * sizeof(int32_t)) {
        SetError(operator_index, "permutation tensor buffer is too small");
        return false;
    }
    const auto* permutation_data = reinterpret_cast<const int32_t*>(permutation_buffer->data()->data());
    for (size_t i = 0; i < rank; ++i) {
        const int32_t permutation_value = permutation_data[i];
        if (permutation_value < 0 || permutation_value >= static_cast<int32_t>(rank)) {
            SetError(operator_index, "permutation value is out of range");
            return false;
        }
        for (size_t j = 0; j < i; ++j) {
            if (permutation_data[j] == permutation_value) {
                SetError(operator_index, "permutation contains duplicate values");
                return false;
            }
        }
        if (output_shape->Get(i) != input_shape->Get(static_cast<size_t>(permutation_value))) {
            SetError(operator_index, "output shape does not match permutation");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateTransposeOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() == tflite::BuiltinOptions_NONE)
        return true;
    if (op->builtin_options_type() != tflite::BuiltinOptions_TransposeOptions) {
        SetError(operator_index, "invalid TRANSPOSE options type");
        return false;
    }
    const auto* options = op->builtin_options_as_TransposeOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing TRANSPOSE options");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateTransposeConv(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 4) {
        SetError(operator_index, "requires output shape, weights, input and bias tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t output_shape_index = inputs->Get(0);
    const int32_t weights_index = inputs->Get(1);
    const int32_t input_index = inputs->Get(2);
    const int32_t bias_index = inputs->Get(3);
    const int32_t output_index = outputs->Get(0);
    if (output_shape_index < 0 || weights_index < 0 || input_index < 0 || bias_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(output_shape_index) >= tensors->size() || static_cast<size_t>(weights_index) >= tensors->size() || static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(bias_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* output_shape_tensor = tensors->Get(static_cast<size_t>(output_shape_index));
    const auto* weights = tensors->Get(static_cast<size_t>(weights_index));
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* bias = tensors->Get(static_cast<size_t>(bias_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (output_shape_tensor == nullptr || weights == nullptr || input == nullptr || bias == nullptr || output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (output_shape_tensor->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "output shape tensor must be INT32");
        return false;
    }
    const auto* output_shape_tensor_shape = output_shape_tensor->shape();
    const auto* weights_shape = weights->shape();
    const auto* input_shape = input->shape();
    const auto* bias_shape = bias->shape();
    const auto* output_tensor_shape = output->shape();
    if (output_shape_tensor_shape == nullptr || weights_shape == nullptr || input_shape == nullptr || bias_shape == nullptr || output_tensor_shape == nullptr) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (output_shape_tensor_shape->size() != 1) {
        SetError(operator_index, "output shape tensor must be one-dimensional");
        return false;
    }
    if (output_shape_tensor_shape->Get(0) != 4) {
        SetError(operator_index, "output shape tensor must contain four dimensions");
        return false;
    }
    if (input_shape->size() != 4) {
        SetError(operator_index, "input tensor must be rank 4");
        return false;
    }
    if (weights_shape->size() != 4) {
        SetError(operator_index, "weights tensor must be rank 4");
        return false;
    }
    if (bias_shape->size() != 1) {
        SetError(operator_index, "bias tensor must be rank 1");
        return false;
    }
    if (output_tensor_shape->size() != 4) {
        SetError(operator_index, "output tensor must be rank 4");
        return false;
    }
        if (input->type() != output->type()) {
        SetError(operator_index, "input and output types do not match");
        return false;
    }
    if (input->type() == tflite::TensorType_INT16) {
        if (weights->type() != tflite::TensorType_INT8) {
            SetError(operator_index, "INT16 input requires INT8 weights");
            return false;
        }
        if (bias->type() != tflite::TensorType_INT64) {
            SetError(operator_index, "INT16 input requires INT64 bias");
            return false;
        }
    } else {
        if (weights->type() != input->type()) {
            SetError(operator_index, "input and weights types do not match");
            return false;
        }
        if (bias->type() != tflite::TensorType_INT32 &&
            bias->type() != input->type()) {
            SetError(operator_index, "bias type is incompatible with input type");
            return false;
        }
    }
    const int32_t output_channels = weights_shape->Get(0);
    const int32_t input_channels = weights_shape->Get(3);
    if (input_channels <= 0 || output_channels <= 0) {
        SetError(operator_index, "weights tensor contains an invalid dimension");
        return false;
    }
    if (input_shape->Get(3) != input_channels) {
        SetError(operator_index, "input channels do not match weights");
        return false;
    }
    if (bias_shape->Get(0) != output_channels) {
        SetError(operator_index, "bias size does not match weights");
        return false;
    }
    if (output_tensor_shape->Get(3) != output_channels) {
        SetError(operator_index, "output channels do not match weights");
        return false;
    }
    if (bias->type() != tflite::TensorType_INT32 && bias->type() != input->type() && bias->type() != tflite::TensorType_FLOAT32 && bias->type() != tflite::TensorType_INT64) {
        SetError(operator_index, "bias type is incompatible with input type");
        return false;
    }
    if (output_shape_tensor->buffer() >= document_.BufferCount()) {
        SetError(operator_index, "output shape tensor buffer index out of range");
        return false;
    }
    const auto* buffers = document_.Buffers();
    if (buffers == nullptr || output_shape_tensor->buffer() >= buffers->size()) {
        SetError(operator_index, "output shape tensor buffer index out of range");
        return false;
    }
    const auto* shape_buffer = buffers->Get(output_shape_tensor->buffer());
    /*
    * The output shape may be supplied at runtime.
    * In that case the tensor legitimately has no
    * constant buffer data, so its values cannot be
    * checked statically.
    */
    if (shape_buffer == nullptr) {
        SetError(operator_index, "output shape tensor buffer is null");
        return false;
    }
    if (shape_buffer->data() != nullptr && shape_buffer->data()->size() != 0) {
        if (shape_buffer->data()->size() < 4 * sizeof(int32_t)) {
            SetError(operator_index, "output shape tensor buffer is too small");
            return false;
        }
        const auto* requested_shape = reinterpret_cast<const int32_t*>(shape_buffer->data()->data());
        for (size_t i = 0; i < 4; ++i) {
            if (requested_shape[i] <= 0) {
                SetError(operator_index, "output shape contains an invalid dimension");
                return false;
            }
            if (output_tensor_shape->Get(i) != requested_shape[i]) {
                SetError(operator_index, "output tensor shape does not match output shape tensor");
                return false;
            }
        }
    }
    return true;
}

bool ModelValidator::ValidateTransposeConvOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() == tflite::BuiltinOptions_NONE)
        return true;
    if (op->builtin_options_type() != tflite::BuiltinOptions_TransposeConvOptions) {
        SetError(operator_index, "invalid TRANSPOSE_CONV options type");
        return false;
    }
    const auto* options = op->builtin_options_as_TransposeConvOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing TRANSPOSE_CONV options");
        return false;
    }
    if (options->stride_w() <= 0 || options->stride_h() <= 0) {
        SetError(operator_index, "TRANSPOSE_CONV strides must be positive");
        return false;
    }
    if (options->padding() != tflite::Padding_SAME && options->padding() != tflite::Padding_VALID) {
        SetError(operator_index, "invalid TRANSPOSE_CONV padding");
        return false;
    }
    switch (options->fused_activation_function()) {
    case tflite::ActivationFunctionType_NONE:
    case tflite::ActivationFunctionType_RELU:
    case tflite::ActivationFunctionType_RELU6:
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
        break;
    default:
        SetError(operator_index, "invalid fused activation function");
        return false;
    }
    switch (options->quantized_bias_type()) {
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_INT32:
    case tflite::TensorType_INT64:
        break;
    default:
        SetError(operator_index, "invalid quantized bias type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateUnidirectionalSequenceLSTM(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 24) {
        SetError(operator_index, "requires 24 input slots");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    auto get_tensor = [&](size_t slot, const tflite::Tensor** tensor) -> bool {
        const int32_t index = inputs->Get(slot);
        if (index < 0)
            return true;
        if (static_cast<size_t>(index) >= tensors->size()) {
            SetError(operator_index, "contains an out-of-range tensor index");
            return false;
        }
        *tensor = tensors->Get(static_cast<size_t>(index));
        if (*tensor == nullptr) {
            SetError(operator_index, "contains a null tensor");
            return false;
        }
        return true;
    };
    const tflite::Tensor* input = nullptr;
    const tflite::Tensor* input_to_input = nullptr;
    const tflite::Tensor* input_to_forget = nullptr;
    const tflite::Tensor* input_to_cell = nullptr;
    const tflite::Tensor* input_to_output = nullptr;
    const tflite::Tensor* recurrent_to_input = nullptr;
    const tflite::Tensor* recurrent_to_forget = nullptr;
    const tflite::Tensor* recurrent_to_cell = nullptr;
    const tflite::Tensor* recurrent_to_output = nullptr;
    const tflite::Tensor* cell_to_input = nullptr;
    const tflite::Tensor* cell_to_forget = nullptr;
    const tflite::Tensor* cell_to_output = nullptr;
    const tflite::Tensor* input_bias = nullptr;
    const tflite::Tensor* forget_bias = nullptr;
    const tflite::Tensor* cell_bias = nullptr;
    const tflite::Tensor* output_bias = nullptr;
    const tflite::Tensor* projection_weights = nullptr;
    const tflite::Tensor* projection_bias = nullptr;
    const tflite::Tensor* output_state = nullptr;
    const tflite::Tensor* cell_state = nullptr;
    const tflite::Tensor* input_layer_norm = nullptr;
    const tflite::Tensor* forget_layer_norm = nullptr;
    const tflite::Tensor* cell_layer_norm = nullptr;
    const tflite::Tensor* output_layer_norm = nullptr;
    if (inputs->Get(0) < 0) {
        SetError(operator_index, "input tensor is required");
        return false;
    }
    if (!get_tensor(0, &input) ||
        !get_tensor(1, &input_to_input) ||
        !get_tensor(2, &input_to_forget) ||
        !get_tensor(3, &input_to_cell) ||
        !get_tensor(4, &input_to_output) ||
        !get_tensor(5, &recurrent_to_input) ||
        !get_tensor(6, &recurrent_to_forget) ||
        !get_tensor(7, &recurrent_to_cell) ||
        !get_tensor(8, &recurrent_to_output) ||
        !get_tensor(9, &cell_to_input) ||
        !get_tensor(10, &cell_to_forget) ||
        !get_tensor(11, &cell_to_output) ||
        !get_tensor(12, &input_bias) ||
        !get_tensor(13, &forget_bias) ||
        !get_tensor(14, &cell_bias) ||
        !get_tensor(15, &output_bias) ||
        !get_tensor(16, &projection_weights) ||
        !get_tensor(17, &projection_bias) ||
        !get_tensor(18, &output_state) ||
        !get_tensor(19, &cell_state) ||
        !get_tensor(20, &input_layer_norm) ||
        !get_tensor(21, &forget_layer_norm) ||
        !get_tensor(22, &cell_layer_norm) ||
        !get_tensor(23, &output_layer_norm)) {
        return false;
    }
    const int32_t output_index = outputs->Get(0);
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (output == nullptr) {
        SetError(operator_index, "output tensor is null");
        return false;
    }
    if (input->shape() == nullptr || input->shape()->size() != 3) {
        SetError(operator_index, "input tensor must be rank 3");
        return false;
    }
    if (output->shape() == nullptr || output->shape()->size() != 3) {
        SetError(operator_index, "output tensor must be rank 3");
        return false;
    }
    const int32_t input_dim = input->shape()->Get(2);
    const int32_t batch_size = input->shape()->Get(0);
    const int32_t sequence_size = input->shape()->Get(1);
    if (batch_size <= 0 || sequence_size <= 0 || input_dim <= 0) {
        SetError(operator_index, "input tensor contains an invalid dimension");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (input_to_input == nullptr ||
        input_to_forget == nullptr ||
        input_to_cell == nullptr ||
        input_to_output == nullptr ||
        recurrent_to_input == nullptr ||
        recurrent_to_forget == nullptr ||
        recurrent_to_cell == nullptr ||
        recurrent_to_output == nullptr ||
        input_bias == nullptr ||
        forget_bias == nullptr ||
        cell_bias == nullptr ||
        output_bias == nullptr ||
        output_state == nullptr ||
        cell_state == nullptr) {
        SetError(operator_index, "required LSTM tensors are missing");
        return false;
    }
    const auto* input_to_input_shape = input_to_input->shape();
    const auto* input_to_forget_shape = input_to_forget->shape();
    const auto* input_to_cell_shape = input_to_cell->shape();
    const auto* input_to_output_shape = input_to_output->shape();
    const auto* recurrent_to_input_shape = recurrent_to_input->shape();
    const auto* recurrent_to_forget_shape = recurrent_to_forget->shape();
    const auto* recurrent_to_cell_shape = recurrent_to_cell->shape();
    const auto* recurrent_to_output_shape = recurrent_to_output->shape();
    if (input_to_input_shape == nullptr || input_to_forget_shape == nullptr ||
        input_to_cell_shape == nullptr || input_to_output_shape == nullptr ||
        recurrent_to_input_shape == nullptr || recurrent_to_forget_shape == nullptr ||
        recurrent_to_cell_shape == nullptr || recurrent_to_output_shape == nullptr) {
        SetError(operator_index, "required LSTM weight tensor has no shape");
        return false;
    }
    if (input_to_input_shape->size() != 2 ||
        input_to_forget_shape->size() != 2 ||
        input_to_cell_shape->size() != 2 ||
        input_to_output_shape->size() != 2 ||
        recurrent_to_input_shape->size() != 2 ||
        recurrent_to_forget_shape->size() != 2 ||
        recurrent_to_cell_shape->size() != 2 ||
        recurrent_to_output_shape->size() != 2) {
        SetError(operator_index, "LSTM weight tensors must be rank 2");
        return false;
    }
    const int32_t units = input_to_input_shape->Get(0);
    if (units <= 0 || input_to_input_shape->Get(1) != input_dim ||
        input_to_forget_shape->Get(0) != units || input_to_forget_shape->Get(1) != input_dim ||
        input_to_cell_shape->Get(0) != units || input_to_cell_shape->Get(1) != input_dim ||
        input_to_output_shape->Get(0) != units || input_to_output_shape->Get(1) != input_dim) {
        SetError(operator_index, "LSTM input weight shapes are inconsistent");
        return false;
    }
    if (recurrent_to_input_shape->Get(0) != units || recurrent_to_input_shape->Get(1) != units ||
        recurrent_to_forget_shape->Get(0) != units || recurrent_to_forget_shape->Get(1) != units ||
        recurrent_to_cell_shape->Get(0) != units || recurrent_to_cell_shape->Get(1) != units ||
        recurrent_to_output_shape->Get(0) != units || recurrent_to_output_shape->Get(1) != units) {
        SetError(operator_index, "LSTM recurrent weight shapes are inconsistent");
        return false;
    }
    const auto check_bias = [&](const tflite::Tensor* bias) -> bool {
        return bias != nullptr && bias->shape() != nullptr && bias->shape()->size() == 1 && bias->shape()->Get(0) == units;
    };
    if (!check_bias(input_bias) || !check_bias(forget_bias) || !check_bias(cell_bias) || !check_bias(output_bias)) {
        SetError(operator_index, "LSTM bias shapes are invalid");
        return false;
    }
    if (output_state->shape() == nullptr || cell_state->shape() == nullptr ||
        output_state->shape()->size() != 2 || cell_state->shape()->size() != 2) {
        SetError(operator_index, "LSTM state tensors must be rank 2");
        return false;
    }
    if (output_state->shape()->Get(0) != batch_size || output_state->shape()->Get(1) != units ||
        cell_state->shape()->Get(0) != batch_size || cell_state->shape()->Get(1) != units) {
        SetError(operator_index, "LSTM state tensor shapes are invalid");
        return false;
    }
    if (output->shape()->Get(0) != batch_size || output->shape()->Get(1) != sequence_size || output->shape()->Get(2) != units) {
        SetError(operator_index, "LSTM output shape is invalid");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateUnidirectionalSequenceLSTMOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_UnidirectionalSequenceLSTMOptions) {
        SetError(operator_index, "invalid UNIDIRECTIONAL_SEQUENCE_LSTM options type");
        return false;
    }
    const auto* options = op->builtin_options_as_UnidirectionalSequenceLSTMOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing UNIDIRECTIONAL_SEQUENCE_LSTM options");
        return false;
    }
    switch (options->fused_activation_function()) {
    case tflite::ActivationFunctionType_NONE:
    case tflite::ActivationFunctionType_RELU:
    case tflite::ActivationFunctionType_RELU6:
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
    case tflite::ActivationFunctionType_TANH:
        break;
    default:
        SetError(operator_index, "invalid LSTM fused activation function");
        return false;
    }
    if (!std::isfinite(options->cell_clip()) || options->cell_clip() < 0.0f) {
        SetError(operator_index, "invalid LSTM cell clip");
        return false;
    }
    if (!std::isfinite(options->proj_clip()) || options->proj_clip() < 0.0f) {
        SetError(operator_index, "invalid LSTM projection clip");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateUnpack(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    if (input_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    if (input == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    if (input_shape == nullptr) {
        SetError(operator_index, "input tensor has no shape");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_UnpackOptions) {
        SetError(operator_index, "invalid UNPACK options type");
        return false;
    }
    const auto* options = op->builtin_options_as_UnpackOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing UNPACK options");
        return false;
    }
    const int32_t num = options->num();
    const int32_t axis = options->axis();
    const int32_t rank = static_cast<int32_t>(input_shape->size());
    if (num <= 0) {
        SetError(operator_index, "UNPACK num must be positive");
        return false;
    }
    if (rank == 0) {
        SetError(operator_index, "UNPACK input must have rank greater than zero");
        return false;
    }
    if (axis < -rank || axis >= rank) {
        SetError(operator_index, "UNPACK axis is out of range");
        return false;
    }
    if (outputs->size() != static_cast<size_t>(num)) {
        SetError(operator_index, "UNPACK output count does not match num");
        return false;
    }
    int32_t normalized_axis = axis;
    if (normalized_axis < 0)
        normalized_axis += rank;
    const int32_t axis_dimension = input_shape->Get(static_cast<size_t>(normalized_axis));
    if (axis_dimension <= 0) {
        SetError(operator_index, "UNPACK axis dimension is invalid");
        return false;
    }
    if (axis_dimension != num) {
        SetError(operator_index, "UNPACK num does not match axis dimension");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateVarHandle(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 0) {
        SetError(operator_index, "VAR_HANDLE must not have input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t output_index = outputs->Get(0);
    if (output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (output == nullptr) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    if (output->type() != tflite::TensorType_RESOURCE) {
        SetError(operator_index, "VAR_HANDLE output tensor must be RESOURCE");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateWhile(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* model = document_.Model();
    if (model == nullptr || model->subgraphs() == nullptr) {
        SetError(operator_index, "missing model or subgraphs");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() == 0) {
        SetError(operator_index, "requires at least one input tensor");
        return false;
    }
    if (outputs->size() == 0) {
        SetError(operator_index, "requires at least one output tensor");
        return false;
    }
    if (inputs->size() != outputs->size()) {
        SetError(operator_index, "input and output counts must match");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    for (size_t i = 0; i < inputs->size(); ++i) {
        const int32_t input_index = inputs->Get(i);
        if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
            SetError(operator_index, "contains an out-of-range input tensor index");
            return false;
        }
        if (tensors->Get(static_cast<size_t>(input_index)) == nullptr) {
            SetError(operator_index, "contains a null input tensor");
            return false;
        }
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
        const int32_t output_index = outputs->Get(i);
        if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
            SetError(operator_index, "contains an out-of-range output tensor index");
            return false;
        }
        if (tensors->Get(static_cast<size_t>(output_index)) == nullptr) {
            SetError(operator_index, "contains a null output tensor");
            return false;
        }
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_WhileOptions) {
        SetError(operator_index, "invalid WHILE options type");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateWhileOptions(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_WhileOptions) {
        SetError(operator_index, "invalid WHILE options type");
        return false;
    }
    const auto* options = op->builtin_options_as_WhileOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing WHILE options");
        return false;
    }
    const auto* model = document_.Model();
    if (model == nullptr || model->subgraphs() == nullptr) {
        SetError(operator_index, "missing model or subgraphs");
        return false;
    }
    const int32_t cond_subgraph_index = options->cond_subgraph_index();
    const int32_t body_subgraph_index = options->body_subgraph_index();
    if (cond_subgraph_index < 0 || static_cast<size_t>(cond_subgraph_index) >= model->subgraphs()->size()) {
        SetError(operator_index, "WHILE condition subgraph index is out of range");
        return false;
    }
    if (body_subgraph_index < 0 || static_cast<size_t>(body_subgraph_index) >= model->subgraphs()->size()) {
        SetError(operator_index, "WHILE body subgraph index is out of range");
        return false;
    }
    if (model->subgraphs()->Get(static_cast<size_t>(cond_subgraph_index)) == nullptr) {
        SetError(operator_index, "WHILE condition subgraph is null");
        return false;
    }
    if (model->subgraphs()->Get(static_cast<size_t>(body_subgraph_index)) == nullptr) {
        SetError(operator_index, "WHILE body subgraph is null");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateWhileSubgraphs(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* model = document_.Model();
    if (model == nullptr || model->subgraphs() == nullptr) {
        SetError(operator_index, "missing model or subgraphs");
        return false;
    }
    if (op->builtin_options_type() != tflite::BuiltinOptions_WhileOptions) {
        SetError(operator_index, "invalid WHILE options type");
        return false;
    }
    const auto* options = op->builtin_options_as_WhileOptions();
    if (options == nullptr) {
        SetError(operator_index, "missing WHILE options");
        return false;
    }
    const int32_t cond_index = options->cond_subgraph_index();
    const int32_t body_index = options->body_subgraph_index();
    if (cond_index < 0 || static_cast<size_t>(cond_index) >= model->subgraphs()->size() || body_index < 0 || static_cast<size_t>(body_index) >= model->subgraphs()->size()) {
        SetError(operator_index, "WHILE subgraph index is out of range");
        return false;
    }
    const auto* cond_subgraph = model->subgraphs()->Get(static_cast<size_t>(cond_index));
    const auto* body_subgraph = model->subgraphs()->Get(static_cast<size_t>(body_index));
    const auto* current_subgraph = CurrentSubgraph();
    if (cond_subgraph == nullptr || body_subgraph == nullptr || current_subgraph == nullptr) {
        SetError(operator_index, "WHILE subgraph is null");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    const auto* cond_inputs = cond_subgraph->inputs();
    const auto* cond_outputs = cond_subgraph->outputs();
    const auto* body_inputs = body_subgraph->inputs();
    const auto* body_outputs = body_subgraph->outputs();
    if (inputs == nullptr || outputs == nullptr || cond_inputs == nullptr || cond_outputs == nullptr || body_inputs == nullptr || body_outputs == nullptr) {
        SetError(operator_index, "WHILE is missing tensor lists");
        return false;
    }
    if (inputs->size() == 0 || outputs->size() == 0) {
        SetError(operator_index, "WHILE requires loop variables");
        return false;
    }
    if (inputs->size() != outputs->size()) {
        SetError(operator_index, "WHILE input and output counts must match");
        return false;
    }
    if (cond_inputs->size() != inputs->size()) {
        SetError(operator_index, "WHILE condition input count does not match loop variables");
        return false;
    }
    if (body_inputs->size() != inputs->size() || body_outputs->size() != outputs->size()) {
        SetError(operator_index, "WHILE body tensor counts do not match loop variables");
        return false;
    }
    const auto* current_tensors = current_subgraph->tensors();
    const auto* cond_tensors = cond_subgraph->tensors();
    const auto* body_tensors = body_subgraph->tensors();
    if (current_tensors == nullptr || cond_tensors == nullptr || body_tensors == nullptr) {
        SetError(operator_index, "WHILE is missing tensor lists");
        return false;
    }
    if (cond_outputs->size() != 1) {
        SetError(operator_index, "WHILE condition subgraph must have one output");
        return false;
    }
    const int32_t cond_output_index = cond_outputs->Get(0);
    if (cond_output_index < 0 || static_cast<size_t>(cond_output_index) >= cond_tensors->size()) {
        SetError(operator_index, "WHILE condition output tensor index is out of range");
        return false;
    }
    const auto* cond_output = cond_tensors->Get(static_cast<size_t>(cond_output_index));
    if (cond_output == nullptr || cond_output->shape() == nullptr) {
        SetError(operator_index, "WHILE condition output tensor is invalid");
        return false;
    }
    if (cond_output->type() != tflite::TensorType_BOOL || cond_output->shape()->size() != 0) {
        SetError(operator_index, "WHILE condition output must be a BOOL scalar");
        return false;
    }
    for (size_t i = 0; i < inputs->size(); ++i) {
        const int32_t input_index = inputs->Get(i);
        const int32_t cond_input_index = cond_inputs->Get(i);
        const int32_t body_input_index = body_inputs->Get(i);
        const int32_t output_index = outputs->Get(i);
        const int32_t body_output_index = body_outputs->Get(i);
        if (input_index < 0 || static_cast<size_t>(input_index) >= current_tensors->size() || cond_input_index < 0 || static_cast<size_t>(cond_input_index) >= cond_tensors->size() || body_input_index < 0 || static_cast<size_t>(body_input_index) >= body_tensors->size() || output_index < 0 || static_cast<size_t>(output_index) >= current_tensors->size() || body_output_index < 0 || static_cast<size_t>(body_output_index) >= body_tensors->size()) {
            SetError(operator_index, "WHILE loop tensor index is out of range");
            return false;
        }
        const auto* input = current_tensors->Get(static_cast<size_t>(input_index));
        const auto* cond_input = cond_tensors->Get(static_cast<size_t>(cond_input_index));
        const auto* body_input = body_tensors->Get(static_cast<size_t>(body_input_index));
        const auto* output = current_tensors->Get(static_cast<size_t>(output_index));
        const auto* body_output = body_tensors->Get(static_cast<size_t>(body_output_index));
        if (input == nullptr || cond_input == nullptr || body_input == nullptr || output == nullptr || body_output == nullptr) {
            SetError(operator_index, "WHILE loop tensor is null");
            return false;
        }
        if (input->shape() == nullptr || cond_input->shape() == nullptr || body_input->shape() == nullptr || output->shape() == nullptr || body_output->shape() == nullptr) {
            SetError(operator_index, "WHILE loop tensor has no shape");
            return false;
        }
        if (input->type() != cond_input->type() || input->type() != body_input->type() || input->type() != output->type() || input->type() != body_output->type()) {
            SetError(operator_index, "WHILE loop tensor types do not match");
            return false;
        }
        if (input->shape()->size() != cond_input->shape()->size() || input->shape()->size() != body_input->shape()->size() || input->shape()->size() != output->shape()->size() || input->shape()->size() != body_output->shape()->size()) {
            SetError(operator_index, "WHILE loop tensor ranks do not match");
            return false;
        }
        for (size_t j = 0; j < input->shape()->size(); ++j) {
            if (input->shape()->Get(j) != cond_input->shape()->Get(j) || input->shape()->Get(j) != body_input->shape()->Get(j) || input->shape()->Get(j) != output->shape()->Get(j) || input->shape()->Get(j) != body_output->shape()->Get(j)) {
                SetError(operator_index, "WHILE loop tensor shapes do not match");
                return false;
            }
        }
    }
    return true;
}

bool ModelValidator::ValidateWindow(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 2) {
        SetError(operator_index, "WINDOW requires exactly two input tensors");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "WINDOW requires exactly one output tensor");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    const int32_t input_index = inputs->Get(0);
    const int32_t weights_index = inputs->Get(1);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || static_cast<size_t>(input_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid input tensor index");
        return false;
    }
    if (weights_index < 0 || static_cast<size_t>(weights_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid weights tensor index");
        return false;
    }
    if (output_index < 0 || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an invalid output tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* weights = tensors->Get(static_cast<size_t>(weights_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || input->shape() == nullptr) {
        SetError(operator_index, "contains an invalid input tensor");
        return false;
    }
    if (weights == nullptr || weights->shape() == nullptr) {
        SetError(operator_index, "contains an invalid weights tensor");
        return false;
    }
    if (output == nullptr || output->shape() == nullptr) {
        SetError(operator_index, "contains an invalid output tensor");
        return false;
    }
    if (input->shape()->size() < 1) {
        SetError(operator_index, "WINDOW input tensor must have rank at least one");
        return false;
    }
    if (weights->shape()->size() != 1) {
        SetError(operator_index, "WINDOW weights tensor must be 1D");
        return false;
    }
    if (output->shape()->size() != input->shape()->size()) {
        SetError(operator_index, "WINDOW input and output ranks do not match");
        return false;
    }
    if (input->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "WINDOW input tensor must be INT16");
        return false;
    }
    if (weights->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "WINDOW weights tensor must be INT16");
        return false;
    }
    if (output->type() != tflite::TensorType_INT16) {
        SetError(operator_index, "WINDOW output tensor must be INT16");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateZerosLike(size_t operator_index, const tflite::Operator* op) const
{
    if (!op) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = CurrentSubgraph();
    if (!subgraph || !subgraph->tensors()) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (!inputs || !outputs) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (inputs->size() != 1) {
        SetError(operator_index, "requires one input tensor");
        return false;
    }
    if (outputs->size() != 1) {
        SetError(operator_index, "requires one output tensor");
        return false;
    }
    const int32_t input_index = inputs->Get(0);
    const int32_t output_index = outputs->Get(0);
    if (input_index < 0 || output_index < 0) {
        SetError(operator_index, "contains a negative tensor index");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() || static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (!input || !output) {
        SetError(operator_index, "contains null tensor");
        return false;
    }
    const auto* input_shape = input->shape();
    const auto* output_shape = output->shape();
    if (!input_shape || !output_shape) {
        SetError(operator_index, "one or more tensors have no shape");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output tensor types do not match");
        return false;
    }
    if (input_shape->size() != output_shape->size()) {
        SetError(operator_index, "input and output ranks do not match");
        return false;
    }
    for (size_t i = 0; i < input_shape->size(); ++i) {
        if (input_shape->Get(i) != output_shape->Get(i)) {
            SetError(operator_index, "input and output shapes do not match");
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateCustom(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* opcode = document_.OperatorCode(op->opcode_index());
    if (opcode == nullptr) {
        SetError(operator_index, "missing custom operator code");
        return false;
    }
    if (opcode->builtin_code() != tflite::BuiltinOperator_CUSTOM) {
        SetError(operator_index, "operator is not CUSTOM");
        return false;
    }
    const auto* custom_name = opcode->custom_code();
    if (custom_name == nullptr || custom_name->str().empty()) {
        SetError(operator_index, "missing custom operator name");
        return false;
    }
    const auto* inputs = op->inputs();
    const auto* outputs = op->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        SetError(operator_index, "missing inputs or outputs");
        return false;
    }
    if (custom_name->str() == "CIRCULAR_BUFFER")
        return ValidateCircularBuffer(operator_index, op);
    if (custom_name->str() == "TFLM_DECODE")
        return ValidateDecode(operator_index, op);
    if (custom_name->str() == "SignalDelay")
        return ValidateDelay(operator_index, op);
    if (custom_name->str() == "TFLite_Detection_PostProcess")
        return ValidateDetectionPostprocess(operator_index, op);
    if (custom_name->str() == "ethos-u")
        return ValidateEthosU(operator_index, op);
    if (custom_name->str() == "SignalFramer")
        return ValidateFramer(operator_index, op);
    if (custom_name->str() == "SignalFftAutoScale")
        return ValidateFftAutoScale(operator_index, op);
    if (custom_name->str() == "SignalEnergy")
        return ValidateEnergy(operator_index, op);
    if (custom_name->str() == "SignalFilterBank")
        return ValidateFilterBank(operator_index, op);
    if (custom_name->str() == "SignalFilterBankSquareRoot")
        return ValidateFilterBankSquareRoot(operator_index, op);
    if (custom_name->str() == "SignalFilterBankSpectralSubtraction")
        return ValidateFilterBankSpectralSubtraction(operator_index, op);
    if (custom_name->str() == "SignalFilterBankLog")
        return ValidateFilterBankLog(operator_index, op);
    if (custom_name->str() == "SignalIrfft")
        return ValidateIrfft(operator_index, op);
    if (custom_name->str() == "SignalOverlapAdd")
        return ValidateOverlapAdd(operator_index, op);
    if (custom_name->str() == "SignalPCAN")
        return ValidatePcan(operator_index, op);
    if (custom_name->str() == "SignalRfft")
        return ValidateRfft(operator_index, op);
    if (custom_name->str() == "SignalStacker")
        return ValidateStacker(operator_index, op);
    if (custom_name->str() == "SignalWindow")
        return ValidateWindow(operator_index, op);
    SetError(operator_index, ("unsupported custom operator: " + custom_name->str()).c_str());
    return false;
}