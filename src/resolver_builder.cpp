#include "resolver_builder.h"

#include "model_document.h"

std::set<tflite::BuiltinOperator> CollectModelOperators(const ModelDocument& document)
{
    std::set<tflite::BuiltinOperator> ops;
    const auto* subgraph = document.PrimarySubgraph();
    if (subgraph == nullptr || subgraph->operators() == nullptr) {
        return ops;
    }
    const auto* opcodes = document.OperatorCodes();
    if (opcodes == nullptr) {
        return ops;
    }
    const auto* operators = subgraph->operators();
    for (size_t i = 0; i < operators->size(); ++i) {
        const auto* op = operators->Get(i);
        if (op == nullptr) {
            continue;
        }
        const auto* opcode = opcodes->Get(op->opcode_index());
        if (opcode == nullptr) {
            continue;
        }
        const auto builtin = static_cast<tflite::BuiltinOperator>(
            opcode->deprecated_builtin_code());
        if (builtin == tflite::BuiltinOperator_CUSTOM) {
            continue;
        }
        ops.insert(builtin);
    }
    return ops;
}

TfLiteStatus RegisterModelOperators(tflite::TflmOpResolver& resolver, const std::set<tflite::BuiltinOperator>& operators, tflite::BuiltinOperator* unsupported_operator)
{
    for (const auto op : operators) {
        switch (op) {
        case tflite::BuiltinOperator_ABS:
            TF_LITE_ENSURE_STATUS(resolver.AddAbs());
            break;
        case tflite::BuiltinOperator_ADD:
            TF_LITE_ENSURE_STATUS(resolver.AddAdd());
            break;
        case tflite::BuiltinOperator_ADD_N:
            TF_LITE_ENSURE_STATUS(resolver.AddAddN());
            break;
        case tflite::BuiltinOperator_ARG_MAX:
            TF_LITE_ENSURE_STATUS(resolver.AddArgMax());
            break;
        case tflite::BuiltinOperator_ARG_MIN:
            TF_LITE_ENSURE_STATUS(resolver.AddArgMin());
            break;
        case tflite::BuiltinOperator_ASSIGN_VARIABLE:
            TF_LITE_ENSURE_STATUS(resolver.AddAssignVariable());
            break;
        case tflite::BuiltinOperator_AVERAGE_POOL_2D:
            TF_LITE_ENSURE_STATUS(resolver.AddAveragePool2D());
            break;
        case tflite::BuiltinOperator_BATCH_MATMUL:
            TF_LITE_ENSURE_STATUS(resolver.AddBatchMatMul());
            break;
        case tflite::BuiltinOperator_BATCH_TO_SPACE_ND:
            TF_LITE_ENSURE_STATUS(resolver.AddBatchToSpaceNd());
            break;
        case tflite::BuiltinOperator_BROADCAST_ARGS:
            TF_LITE_ENSURE_STATUS(resolver.AddBroadcastArgs());
            break;
        case tflite::BuiltinOperator_BROADCAST_TO:
            TF_LITE_ENSURE_STATUS(resolver.AddBroadcastTo());
            break;
        case tflite::BuiltinOperator_CALL_ONCE:
            TF_LITE_ENSURE_STATUS(resolver.AddCallOnce());
            break;
        case tflite::BuiltinOperator_CAST:
            TF_LITE_ENSURE_STATUS(resolver.AddCast());
            break;
        case tflite::BuiltinOperator_CEIL:
            TF_LITE_ENSURE_STATUS(resolver.AddCeil());
            break;
        case tflite::BuiltinOperator_CONCATENATION:
            TF_LITE_ENSURE_STATUS(resolver.AddConcatenation());
            break;
        case tflite::BuiltinOperator_CONV_2D:
            TF_LITE_ENSURE_STATUS(resolver.AddConv2D());
            break;
        case tflite::BuiltinOperator_COS:
            TF_LITE_ENSURE_STATUS(resolver.AddCos());
            break;
        case tflite::BuiltinOperator_CUMSUM:
            TF_LITE_ENSURE_STATUS(resolver.AddCumSum());
            break;
        case tflite::BuiltinOperator_DEPTH_TO_SPACE:
            TF_LITE_ENSURE_STATUS(resolver.AddDepthToSpace());
            break;
        case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
            TF_LITE_ENSURE_STATUS(resolver.AddDepthwiseConv2D());
            break;
        case tflite::BuiltinOperator_DEQUANTIZE:
            TF_LITE_ENSURE_STATUS(resolver.AddDequantize());
            break;
        case tflite::BuiltinOperator_DIV:
            TF_LITE_ENSURE_STATUS(resolver.AddDiv());
            break;
        case tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE:
            TF_LITE_ENSURE_STATUS(resolver.AddDynamicUpdateSlice());
            break;
        case tflite::BuiltinOperator_ELU:
            TF_LITE_ENSURE_STATUS(resolver.AddElu());
            break;
        case tflite::BuiltinOperator_EMBEDDING_LOOKUP:
            TF_LITE_ENSURE_STATUS(resolver.AddEmbeddingLookup());
            break;
        case tflite::BuiltinOperator_EQUAL:
            TF_LITE_ENSURE_STATUS(resolver.AddEqual());
            break;
        case tflite::BuiltinOperator_EXP:
            TF_LITE_ENSURE_STATUS(resolver.AddExp());
            break;
        case tflite::BuiltinOperator_EXPAND_DIMS:
            TF_LITE_ENSURE_STATUS(resolver.AddExpandDims());
            break;
        case tflite::BuiltinOperator_FILL:
            TF_LITE_ENSURE_STATUS(resolver.AddFill());
            break;
        case tflite::BuiltinOperator_FLOOR:
            TF_LITE_ENSURE_STATUS(resolver.AddFloor());
            break;
        case tflite::BuiltinOperator_FLOOR_DIV:
            TF_LITE_ENSURE_STATUS(resolver.AddFloorDiv());
            break;
        case tflite::BuiltinOperator_FLOOR_MOD:
            TF_LITE_ENSURE_STATUS(resolver.AddFloorMod());
            break;
        case tflite::BuiltinOperator_FULLY_CONNECTED:
            TF_LITE_ENSURE_STATUS(resolver.AddFullyConnected());
            break;
        case tflite::BuiltinOperator_GATHER:
            TF_LITE_ENSURE_STATUS(resolver.AddGather());
            break;
        case tflite::BuiltinOperator_GATHER_ND:
            TF_LITE_ENSURE_STATUS(resolver.AddGatherNd());
            break;
        case tflite::BuiltinOperator_GREATER:
            TF_LITE_ENSURE_STATUS(resolver.AddGreater());
            break;
        case tflite::BuiltinOperator_GREATER_EQUAL:
            TF_LITE_ENSURE_STATUS(resolver.AddGreaterEqual());
            break;
        case tflite::BuiltinOperator_HARD_SWISH:
            TF_LITE_ENSURE_STATUS(resolver.AddHardSwish());
            break;
        case tflite::BuiltinOperator_IF:
            TF_LITE_ENSURE_STATUS(resolver.AddIf());
            break;
        case tflite::BuiltinOperator_L2_NORMALIZATION:
            TF_LITE_ENSURE_STATUS(resolver.AddL2Normalization());
            break;
        case tflite::BuiltinOperator_L2_POOL_2D:
            TF_LITE_ENSURE_STATUS(resolver.AddL2Pool2D());
            break;
        case tflite::BuiltinOperator_LEAKY_RELU:
            TF_LITE_ENSURE_STATUS(resolver.AddLeakyRelu());
            break;
        case tflite::BuiltinOperator_LESS:
            TF_LITE_ENSURE_STATUS(resolver.AddLess());
            break;
        case tflite::BuiltinOperator_LESS_EQUAL:
            TF_LITE_ENSURE_STATUS(resolver.AddLessEqual());
            break;
        case tflite::BuiltinOperator_LOG:
            TF_LITE_ENSURE_STATUS(resolver.AddLog());
            break;
        case tflite::BuiltinOperator_LOG_SOFTMAX:
            TF_LITE_ENSURE_STATUS(resolver.AddLogSoftmax());
            break;
        case tflite::BuiltinOperator_LOGICAL_AND:
            TF_LITE_ENSURE_STATUS(resolver.AddLogicalAnd());
            break;
        case tflite::BuiltinOperator_LOGICAL_NOT:
            TF_LITE_ENSURE_STATUS(resolver.AddLogicalNot());
            break;
        case tflite::BuiltinOperator_LOGICAL_OR:
            TF_LITE_ENSURE_STATUS(resolver.AddLogicalOr());
            break;
        case tflite::BuiltinOperator_LOGISTIC:
            TF_LITE_ENSURE_STATUS(resolver.AddLogistic());
            break;
        case tflite::BuiltinOperator_MAX_POOL_2D:
            TF_LITE_ENSURE_STATUS(resolver.AddMaxPool2D());
            break;
        case tflite::BuiltinOperator_MAXIMUM:
            TF_LITE_ENSURE_STATUS(resolver.AddMaximum());
            break;
        case tflite::BuiltinOperator_MEAN:
            TF_LITE_ENSURE_STATUS(resolver.AddMean());
            break;
        case tflite::BuiltinOperator_MINIMUM:
            TF_LITE_ENSURE_STATUS(resolver.AddMinimum());
            break;
        case tflite::BuiltinOperator_MIRROR_PAD:
            TF_LITE_ENSURE_STATUS(resolver.AddMirrorPad());
            break;
        case tflite::BuiltinOperator_MUL:
            TF_LITE_ENSURE_STATUS(resolver.AddMul());
            break;
        case tflite::BuiltinOperator_NEG:
            TF_LITE_ENSURE_STATUS(resolver.AddNeg());
            break;
        case tflite::BuiltinOperator_NOT_EQUAL:
            TF_LITE_ENSURE_STATUS(resolver.AddNotEqual());
            break;
        case tflite::BuiltinOperator_PACK:
            TF_LITE_ENSURE_STATUS(resolver.AddPack());
            break;
        case tflite::BuiltinOperator_PAD:
            TF_LITE_ENSURE_STATUS(resolver.AddPad());
            break;
        case tflite::BuiltinOperator_PADV2:
            TF_LITE_ENSURE_STATUS(resolver.AddPadV2());
            break;
        case tflite::BuiltinOperator_PRELU:
            TF_LITE_ENSURE_STATUS(resolver.AddPrelu());
            break;
        case tflite::BuiltinOperator_QUANTIZE:
            TF_LITE_ENSURE_STATUS(resolver.AddQuantize());
            break;
        case tflite::BuiltinOperator_READ_VARIABLE:
            TF_LITE_ENSURE_STATUS(resolver.AddReadVariable());
            break;
        case tflite::BuiltinOperator_REDUCE_ALL:
            TF_LITE_ENSURE_STATUS(resolver.AddReduceAll());
            break;
        case tflite::BuiltinOperator_REDUCE_MAX:
            TF_LITE_ENSURE_STATUS(resolver.AddReduceMax());
            break;
        case tflite::BuiltinOperator_REDUCE_MIN:
            TF_LITE_ENSURE_STATUS(resolver.AddReduceMin());
            break;
        case tflite::BuiltinOperator_RELU:
            TF_LITE_ENSURE_STATUS(resolver.AddRelu());
            break;
        case tflite::BuiltinOperator_RELU6:
            TF_LITE_ENSURE_STATUS(resolver.AddRelu6());
            break;
        case tflite::BuiltinOperator_RESHAPE:
            TF_LITE_ENSURE_STATUS(resolver.AddReshape());
            break;
        case tflite::BuiltinOperator_RESIZE_BILINEAR:
            TF_LITE_ENSURE_STATUS(resolver.AddResizeBilinear());
            break;
        case tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR:
            TF_LITE_ENSURE_STATUS(resolver.AddResizeNearestNeighbor());
            break;
        case tflite::BuiltinOperator_REVERSE_V2:
            TF_LITE_ENSURE_STATUS(resolver.AddReverseV2());
            break;
        case tflite::BuiltinOperator_ROUND:
            TF_LITE_ENSURE_STATUS(resolver.AddRound());
            break;
        case tflite::BuiltinOperator_RSQRT:
            TF_LITE_ENSURE_STATUS(resolver.AddRsqrt());
            break;
        case tflite::BuiltinOperator_SELECT_V2:
            TF_LITE_ENSURE_STATUS(resolver.AddSelectV2());
            break;
        case tflite::BuiltinOperator_SHAPE:
            TF_LITE_ENSURE_STATUS(resolver.AddShape());
            break;
        case tflite::BuiltinOperator_SIN:
            TF_LITE_ENSURE_STATUS(resolver.AddSin());
            break;
        case tflite::BuiltinOperator_SLICE:
            TF_LITE_ENSURE_STATUS(resolver.AddSlice());
            break;
        case tflite::BuiltinOperator_SOFTMAX:
            TF_LITE_ENSURE_STATUS(resolver.AddSoftmax());
            break;
        case tflite::BuiltinOperator_SPACE_TO_BATCH_ND:
            TF_LITE_ENSURE_STATUS(resolver.AddSpaceToBatchNd());
            break;
        case tflite::BuiltinOperator_SPACE_TO_DEPTH:
            TF_LITE_ENSURE_STATUS(resolver.AddSpaceToDepth());
            break;
        case tflite::BuiltinOperator_SPLIT:
            TF_LITE_ENSURE_STATUS(resolver.AddSplit());
            break;
        case tflite::BuiltinOperator_SPLIT_V:
            TF_LITE_ENSURE_STATUS(resolver.AddSplitV());
            break;
        case tflite::BuiltinOperator_SQRT:
            TF_LITE_ENSURE_STATUS(resolver.AddSqrt());
            break;
        case tflite::BuiltinOperator_SQUARE:
            TF_LITE_ENSURE_STATUS(resolver.AddSquare());
            break;
        case tflite::BuiltinOperator_SQUARED_DIFFERENCE:
            TF_LITE_ENSURE_STATUS(resolver.AddSquaredDifference());
            break;
        case tflite::BuiltinOperator_SQUEEZE:
            TF_LITE_ENSURE_STATUS(resolver.AddSqueeze());
            break;
        case tflite::BuiltinOperator_STRIDED_SLICE:
            TF_LITE_ENSURE_STATUS(resolver.AddStridedSlice());
            break;
        case tflite::BuiltinOperator_SUB:
            TF_LITE_ENSURE_STATUS(resolver.AddSub());
            break;
        case tflite::BuiltinOperator_SUM:
            TF_LITE_ENSURE_STATUS(resolver.AddSum());
            break;
        case tflite::BuiltinOperator_SVDF:
            TF_LITE_ENSURE_STATUS(resolver.AddSvdf());
            break;
        case tflite::BuiltinOperator_TANH:
            TF_LITE_ENSURE_STATUS(resolver.AddTanh());
            break;
        case tflite::BuiltinOperator_TRANSPOSE:
            TF_LITE_ENSURE_STATUS(resolver.AddTranspose());
            break;
        case tflite::BuiltinOperator_TRANSPOSE_CONV:
            TF_LITE_ENSURE_STATUS(resolver.AddTransposeConv());
            break;
        case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
            TF_LITE_ENSURE_STATUS(resolver.AddUnidirectionalSequenceLSTM());
            break;
        case tflite::BuiltinOperator_UNPACK:
            TF_LITE_ENSURE_STATUS(resolver.AddUnpack());
            break;
        case tflite::BuiltinOperator_VAR_HANDLE:
            TF_LITE_ENSURE_STATUS(resolver.AddVarHandle());
            break;
        case tflite::BuiltinOperator_WHILE:
            TF_LITE_ENSURE_STATUS(resolver.AddWhile());
            break;
        case tflite::BuiltinOperator_ZEROS_LIKE:
            TF_LITE_ENSURE_STATUS(resolver.AddZerosLike());
            break;
        default:
            if (unsupported_operator != nullptr) {
                *unsupported_operator = op;
            }
            return kTfLiteError;
        }
    }
    return kTfLiteOk;
}