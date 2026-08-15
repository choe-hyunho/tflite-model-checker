#include "model_validator.h"

#include <cmath>
#include <sstream>

#include "model_utils.h"

ModelValidator::ModelValidator(const ModelDocument& document)
    : document_(document)
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
        last_error_ = "model interface validation failed";
        return false;
    }
    if (!ValidateOperatorCodes()) {
        last_error_ = "operator code validation failed";
        return false;
    }
    if (!ValidateTensors()) {
        last_error_ = "tensor validation failed";
        return false;
    }
    if (!ValidateGraph()) {
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
    const auto* subgraph = document_.PrimarySubgraph();

    if (subgraph == nullptr)
        return false;
    const auto* tensors = subgraph->tensors();
    if (tensors == nullptr)
        return false;
    const auto* inputs = subgraph->inputs();
    const auto* outputs = subgraph->outputs();
    if (inputs == nullptr || outputs == nullptr) {
        return false;
    }
    if (!ValidateTensorIndexList(inputs, tensors->size())) {
        return false;
    }
    if (!ValidateTensorIndexList(outputs, tensors->size())) {
        return false;
    }
    return true;
}

bool ModelValidator::ValidateOperatorCodes() const
{
    const auto* model = document_.Model();
    if (model == nullptr)
        return false;
    const auto* operator_codes = model->operator_codes();
    if (operator_codes == nullptr)
        return false;
    const auto* subgraph = document_.PrimarySubgraph();
    if (subgraph == nullptr)
        return false;
    const auto* operators = subgraph->operators();
    if (operators == nullptr)
        return false;
    for (size_t i = 0; i < operators->size(); ++i) {
        const auto* op = operators->Get(i);
        if (op == nullptr)
            return false;
        const uint32_t opcode_index = op->opcode_index();
        if (opcode_index >= operator_codes->size()) {
            return false;
        }
        const auto* opcode = operator_codes->Get(opcode_index);
        if (opcode == nullptr)
            return false;
        //
        // The builtin opcode must be within the
        // generated TensorFlow Lite enum range.
        //
        const int builtin_code = BuiltinOperatorCode(opcode);
        if (builtin_code < static_cast<int>(tflite::BuiltinOperator_MIN) || builtin_code > static_cast<int>(tflite::BuiltinOperator_MAX)) {
            return false;
        }
    }
    return true;
}

bool ModelValidator::ValidateTensors() const
{
    const auto* subgraphs = document_.Model()->subgraphs();
    if (subgraphs == nullptr || subgraphs->size() == 0) {
        return false;
    }
    const auto* subgraph = subgraphs->Get(0);
    if (subgraph == nullptr)
        return false;
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
        if (!ValidateTensor(i, tensor))
            valid = false;
    }
    return valid;
}

bool ModelValidator::ValidateTensor(size_t tensor_index, const tflite::Tensor* tensor) const
{
    (void)tensor_index;
    if (tensor == nullptr)
        return false;
    if (!ValidateQuantization(tensor))
        return false;
    //
    // Tensor type
    //
    const int type = static_cast<int>(tensor->type());
    if (type < tflite::TensorType_MIN || type > tflite::TensorType_MAX) {
        return false;
    }
    //
    // Tensor shape
    //
    const auto* shape = tensor->shape();
    if (shape == nullptr)
        return false;
    for (size_t i = 0; i < shape->size(); ++i) {
        if (shape->Get(i) < 0)
            return false;
    }
    //
    // Buffer index
    //
    const auto* buffers = document_.Buffers();
    if (buffers == nullptr)
        return false;
    const size_t buffer_index = static_cast<size_t>(tensor->buffer());
    if (buffer_index >= buffers->size())
        return false;
    //
    // Dynamic/intermediate tensors are allowed
    // to have an empty model buffer.
    //
    const auto* buffer = buffers->Get(buffer_index);
    if (buffer == nullptr)
        return false;
    if (buffer->data() == nullptr)
        return true;
    //
    // For tensors containing constant data,
    // verify that the buffer is large enough.
    //
    size_t element_count = 1;
    for (size_t i = 0; i < shape->size(); ++i) {
        const size_t dimension = static_cast<size_t>(shape->Get(i));
        if (dimension != 0 && element_count > static_cast<size_t>(-1) / dimension) {
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
        if (element_size != 0 && element_count > static_cast<size_t>(-1) / element_size) {
            return false;
        }
        expected_bytes = element_count * element_size;
    }
    const size_t actual_bytes = buffer->data()->size();
    return actual_bytes >= expected_bytes;
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

bool ModelValidator::ValidateGraph() const
{
    const auto* subgraphs = document_.Model()->subgraphs();
    if (subgraphs == nullptr || subgraphs->size() == 0) {
        return false;
    }
    const auto* subgraph = subgraphs->Get(0);
    if (subgraph == nullptr)
        return false;
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
        if (!ValidateTensorGraph(i, tensor))
            valid = false;
    }
    return valid;
}

bool ModelValidator::ValidateTensorGraph(size_t tensor_index, const tflite::Tensor* tensor) const
{
    (void)tensor;
    const auto* subgraph = document_.PrimarySubgraph();
    if (subgraph == nullptr)
        return false;
    const auto* operators = subgraph->operators();
    if (operators == nullptr)
        return false;
    const auto* inputs = subgraph->inputs();
    const auto* outputs = subgraph->outputs();
    //
    // Determine whether this tensor is a
    // model input/output.
    //
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
    //
    // Find producers and consumers directly
    // from the operator graph.
    //
    size_t producer_count = 0;
    size_t consumer_count = 0;
    for (size_t op_index = 0; op_index < operators->size(); ++op_index) {
        const auto* op = operators->Get(op_index);
        if (op == nullptr)
            return false;
        const auto* op_inputs = op->inputs();
        if (op_inputs != nullptr) {
            for (size_t i = 0; i < op_inputs->size(); ++i) {
                if (op_inputs->Get(i) == static_cast<int32_t>(tensor_index)) {
                    ++consumer_count;
                }
            }
        }
        const auto* op_outputs = op->outputs();
        if (op_outputs != nullptr) {
            for (size_t i = 0; i < op_outputs->size(); ++i) {
                if (op_outputs->Get(i) == static_cast<int32_t>(tensor_index)) {
                    ++producer_count;
                }
            }
        }
    }
    //
    // A tensor must not be both a model input and
    // an operator output.
    //
    if (is_input && producer_count != 0) {
        SetError("model input tensor has an operator producer");
        return false;
    }
    //
    // A model output must be produced unless it is
    // also directly a model input.
    //
    if (is_output && !is_input && producer_count == 0) {
        SetError("model output tensor has no producer");
        return false;
    }
    //
    // A tensor can have at most one producer.
    //
    if (producer_count > 1)
        return false;
    //
    // A graph input should not be produced by an
    // operator.
    //
    if (is_input && producer_count != 0)
        return false;
    //
    // Every non-input tensor must either:
    //
    //   1. have an operator producer, or
    //   2. contain constant buffer data.
    //
    // Model inputs are allowed to have no producer.
    //

    if (!is_input && producer_count == 0) {
        const auto* buffers = document_.Buffers();
        if (buffers == nullptr)
            return false;
        const size_t buffer_index = static_cast<size_t>(tensor->buffer());
        if (buffer_index >= buffers->size())
            return false;
        const auto* buffer = buffers->Get(buffer_index);
        if (buffer == nullptr || buffer->data() == nullptr || buffer->data()->size() == 0) {
            return false;
        }
    }
    //
    // A model output must have a producer unless
    // it is also a model input.
    //
    if (is_output && !is_input && producer_count == 0) {
        return false;
    }
    if (!is_output && producer_count > 0 && consumer_count == 0) {
        SetError("intermediate tensor has no consumer");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateOperators() const
{
    const auto* subgraphs = document_.Model()->subgraphs();
    if (subgraphs == nullptr || subgraphs->size() == 0) {
        return false;
    }
    const auto* subgraph = subgraphs->Get(0);
    if (subgraph == nullptr)
        return false;
    const auto* operators = subgraph->operators();
    const auto* tensors = subgraph->tensors();
    if (operators == nullptr || tensors == nullptr) {
        return false;
    }
    bool valid = true;
    for (size_t op_index = 0; op_index < operators->size(); ++op_index) {
        const auto* op = operators->Get(op_index);
        if (op == nullptr) {
            valid = false;
            continue;
        }
        if (!ValidateOperatorTensorReferences(op_index, op)) {
            valid = false;
            continue;
        }
        const auto* opcode = document_.OperatorCode(op->opcode_index());
        if (opcode == nullptr) {
            valid = false;
            continue;
        }
        switch (BuiltinOperatorCode(opcode)) {
        case tflite::BuiltinOperator_CONV_2D:
            if (!ValidateConv2D(op_index, op)) {
                valid = false;
                continue;
            }
            if (!ValidateConv2DOptions(op_index, op)) {
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
        case tflite::BuiltinOperator_STRIDED_SLICE:
            if (!ValidateStridedSlice(op_index, op)) {
                valid = false;
                continue;
            }
            if (!ValidateStridedSliceOptions(op_index, op)) {
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
        case tflite::BuiltinOperator_FULLY_CONNECTED:
            if (!ValidateFullyConnected(op_index, op)) {
                valid = false;
                continue;
            }
            if (!ValidateFullyConnectedOptions( op_index, op)) {
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
        case tflite::BuiltinOperator_RESHAPE:
            if (!ValidateReshape(op_index, op)) {
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
        case tflite::BuiltinOperator_SOFTMAX:
            if (!ValidateSoftmax(op_index, op)) {
                valid = false;
            }
            break;
        default:
            SetError(op_index, "unsupported operator");
            valid = false;
            break;
        }
    }
    return valid;
}

bool ModelValidator::ValidateOperatorTensorReferences(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = document_.PrimarySubgraph();
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

bool ModelValidator::ValidateFullyConnected(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = document_.PrimarySubgraph();
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
    if (static_cast<size_t>(input_index) >= tensors->size() ||
        static_cast<size_t>(weights_index) >= tensors->size() ||
        static_cast<size_t>(output_index) >= tensors->size() ||
        (bias_index >= 0 && static_cast<size_t>(bias_index) >= tensors->size())) {
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
}   const auto* input_shape = input->shape();
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

bool ModelValidator::ValidateConv2D(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = document_.PrimarySubgraph();
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
    if (input->type() != weights->type()) {
        SetError(operator_index, "input and weights types do not match");
        return false;
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output types do not match");
        return false;
    }
    if (bias->type() != tflite::TensorType_INT32) {
        SetError(operator_index, "bias tensor must be INT32");
        return false;
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

bool ModelValidator::ValidateDepthwiseConv2D(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = document_.PrimarySubgraph();
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
    if (static_cast<size_t>(input_index) >= tensors->size() ||
        static_cast<size_t>(weights_index) >= tensors->size() ||
        static_cast<size_t>(bias_index) >= tensors->size() ||
        static_cast<size_t>(output_index) >= tensors->size()) {
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

bool ModelValidator::ValidateStridedSlice(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = document_.PrimarySubgraph();
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
    const auto* subgraph = document_.PrimarySubgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        SetError(operator_index, "missing subgraph or tensors");
        return false;
    }
    const auto* tensors = subgraph->tensors();
    if (static_cast<size_t>(input_index) >= tensors->size() ||
        static_cast<size_t>(axis_index) >= tensors->size() ||
        static_cast<size_t>(output_index) >= tensors->size()) {
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
    if (axis_buffer == nullptr || axis_buffer->data() == nullptr) {
        SetError(operator_index, "axis tensor has no constant data");
        return false;
    }
    const size_t axis_count = static_cast<size_t>(axis->shape()->Get(0));
    if (axis_buffer->data()->size() < axis_count * sizeof(int32_t)) {
        SetError(operator_index, "axis tensor buffer is too small");
        return false;
    }
    const auto* axis_data = reinterpret_cast<const int32_t*>(axis_buffer->data()->data());
    const size_t input_rank = input->shape() != nullptr ? input->shape()->size() : 0;
    if (input_rank == 0) {
        SetError(operator_index, "input tensor has no shape");
        return false;
    }
    for (size_t i = 0; i < axis_count; ++i) {
        const int32_t axis_value = axis_data[i];
        if (axis_value < -static_cast<int32_t>(input_rank) ||
            axis_value >= static_cast<int32_t>(input_rank)) {
            SetError(operator_index, "axis value is out of range");
            return false;
        }
    }
    const auto* input_shape = input->shape();
    const auto* output_shape = output->shape();
    if (input_shape == nullptr || output_shape == nullptr) {
        SetError(operator_index, "input or output has no shape");
        return false;
    }
    size_t expected_output_rank = 0;
    for (size_t i = 0; i < input_shape->size(); ++i) {
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
        if (!reduced)
            ++expected_output_rank;
    }
    if (output_shape->size() != expected_output_rank) {
        SetError(operator_index, "output rank does not match reduced input rank");
        return false;
    }
    size_t output_index_position = 0;
    for (size_t i = 0; i < input_shape->size(); ++i) {
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
            if (output_shape->Get(output_index_position) != input_shape->Get(i)) {
                SetError(operator_index, "output shape does not match unreduced input dimensions");
                return false;
            }
            ++output_index_position;
        }
    }
    if (input->type() != output->type()) {
        SetError(operator_index, "input and output types do not match");
        return false;
    }
    return true;
}

bool ModelValidator::ValidateShape(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = document_.PrimarySubgraph();
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

bool ModelValidator::ValidateReshape(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = document_.PrimarySubgraph();
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

bool ModelValidator::ValidatePack(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = document_.PrimarySubgraph();
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

bool ModelValidator::ValidateSoftmax(size_t operator_index, const tflite::Operator* op) const
{
    if (op == nullptr) {
        SetError(operator_index, "null operator");
        return false;
    }
    const auto* subgraph = document_.PrimarySubgraph();
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
    if (static_cast<size_t>(input_index) >= tensors->size() ||
        static_cast<size_t>(output_index) >= tensors->size()) {
        SetError(operator_index, "contains an out-of-range tensor index");
        return false;
    }
    const auto* input = tensors->Get(static_cast<size_t>(input_index));
    const auto* output = tensors->Get(static_cast<size_t>(output_index));
    if (input == nullptr || output == nullptr) {
        SetError(operator_index, "missing input or output tensor");
        return false;
    }
    if (input->type() != tflite::TensorType_INT8) {
        SetError(operator_index, "input tensor must be INT8");
        return false;
    }
    if (output->type() != tflite::TensorType_INT8) {
        SetError(operator_index, "output tensor must be INT8");
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
    const auto* subgraph = document_.PrimarySubgraph();
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