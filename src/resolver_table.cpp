#include "resolver_table.h"

const ResolverEntry kResolverTable[] = {

    { tflite::BuiltinOperator_ABS,                     "AddAbs" },
    { tflite::BuiltinOperator_ADD,                     "AddAdd" },
    { tflite::BuiltinOperator_ADD_N,                   "AddAddN" },
    { tflite::BuiltinOperator_ARG_MAX,                 "AddArgMax" },
    { tflite::BuiltinOperator_ARG_MIN,                 "AddArgMin" },
    { tflite::BuiltinOperator_AVERAGE_POOL_2D,         "AddAveragePool2D" },

    { tflite::BuiltinOperator_BATCH_MATMUL,            "AddBatchMatMul" },
    { tflite::BuiltinOperator_BATCH_TO_SPACE_ND,       "AddBatchToSpaceNd" },

    { tflite::BuiltinOperator_CAST,                    "AddCast" },
    { tflite::BuiltinOperator_CEIL,                    "AddCeil" },
    { tflite::BuiltinOperator_CONCATENATION,           "AddConcatenation" },
    { tflite::BuiltinOperator_CONV_2D,                 "AddConv2D" },
    { tflite::BuiltinOperator_COS,                     "AddCos" },

    { tflite::BuiltinOperator_DEPTH_TO_SPACE,          "AddDepthToSpace" },
    { tflite::BuiltinOperator_DEPTHWISE_CONV_2D,       "AddDepthwiseConv2D" },
    { tflite::BuiltinOperator_DEQUANTIZE,              "AddDequantize" },
    { tflite::BuiltinOperator_DIV,                     "AddDiv" },

    { tflite::BuiltinOperator_ELU,                     "AddElu" },
    { tflite::BuiltinOperator_EQUAL,                   "AddEqual" },
    { tflite::BuiltinOperator_EXP,                     "AddExp" },
    { tflite::BuiltinOperator_EXPAND_DIMS,             "AddExpandDims" },

    { tflite::BuiltinOperator_FILL,                    "AddFill" },
    { tflite::BuiltinOperator_FLOOR,                   "AddFloor" },
    { tflite::BuiltinOperator_FULLY_CONNECTED,         "AddFullyConnected" },

    { tflite::BuiltinOperator_GATHER,                  "AddGather" },
    { tflite::BuiltinOperator_GATHER_ND,               "AddGatherNd" },
    { tflite::BuiltinOperator_GREATER,                 "AddGreater" },
    { tflite::BuiltinOperator_GREATER_EQUAL,           "AddGreaterEqual" },

    { tflite::BuiltinOperator_HARD_SWISH,              "AddHardSwish" },

    { tflite::BuiltinOperator_L2_NORMALIZATION,        "AddL2Normalization" },
    { tflite::BuiltinOperator_LEAKY_RELU,              "AddLeakyRelu" },
    { tflite::BuiltinOperator_LESS,                    "AddLess" },
    { tflite::BuiltinOperator_LESS_EQUAL,              "AddLessEqual" },
    { tflite::BuiltinOperator_LOG,                     "AddLog" },
    { tflite::BuiltinOperator_LOG_SOFTMAX,             "AddLogSoftmax" },
    { tflite::BuiltinOperator_LOGISTIC,                "AddLogistic" },

    { tflite::BuiltinOperator_MAX_POOL_2D,             "AddMaxPool2D" },
    { tflite::BuiltinOperator_MAXIMUM,                 "AddMaximum" },
    { tflite::BuiltinOperator_MEAN,                    "AddMean" },
    { tflite::BuiltinOperator_MINIMUM,                 "AddMinimum" },
    { tflite::BuiltinOperator_MUL,                     "AddMul" },

    { tflite::BuiltinOperator_NEG,                     "AddNeg" },
    { tflite::BuiltinOperator_NOT_EQUAL,               "AddNotEqual" },

    { tflite::BuiltinOperator_PACK,                    "AddPack" },
    { tflite::BuiltinOperator_PAD,                     "AddPad" },
    { tflite::BuiltinOperator_PADV2,                   "AddPadV2" },
    { tflite::BuiltinOperator_PRELU,                   "AddPrelu" },

    { tflite::BuiltinOperator_QUANTIZE,                "AddQuantize" },

    { tflite::BuiltinOperator_REDUCE_MAX,              "AddReduceMax" },
    { tflite::BuiltinOperator_REDUCE_MIN,              "AddReduceMin" },
    { tflite::BuiltinOperator_RELU,                    "AddRelu" },
    { tflite::BuiltinOperator_RELU6,                   "AddRelu6" },
    { tflite::BuiltinOperator_RESHAPE,                 "AddReshape" },
    { tflite::BuiltinOperator_RESIZE_BILINEAR,         "AddResizeBilinear" },
    { tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR, "AddResizeNearestNeighbor" },
    { tflite::BuiltinOperator_REVERSE_V2,              "AddReverseV2" },
    { tflite::BuiltinOperator_RSQRT,                   "AddRsqrt" },

    { tflite::BuiltinOperator_SHAPE,                   "AddShape" },
    { tflite::BuiltinOperator_SIN,                     "AddSin" },
    { tflite::BuiltinOperator_SLICE,                   "AddSlice" },
    { tflite::BuiltinOperator_SOFTMAX,                 "AddSoftmax" },
    { tflite::BuiltinOperator_SPACE_TO_BATCH_ND,       "AddSpaceToBatchNd" },
    { tflite::BuiltinOperator_SPACE_TO_DEPTH,          "AddSpaceToDepth" },
    { tflite::BuiltinOperator_SPLIT,                   "AddSplit" },
    { tflite::BuiltinOperator_SPLIT_V,                 "AddSplitV" },
    { tflite::BuiltinOperator_SQRT,                    "AddSqrt" },
    { tflite::BuiltinOperator_SQUARE,                  "AddSquare" },
    { tflite::BuiltinOperator_SQUARED_DIFFERENCE,      "AddSquaredDifference" },
    { tflite::BuiltinOperator_SQUEEZE,                 "AddSqueeze" },
    { tflite::BuiltinOperator_STRIDED_SLICE,           "AddStridedSlice" },
    { tflite::BuiltinOperator_SUB,                     "AddSub" },
    { tflite::BuiltinOperator_SUM,                     "AddSum" },
    { tflite::BuiltinOperator_SVDF,                    "AddSvdf" },

    { tflite::BuiltinOperator_TANH,                    "AddTanh" },
    { tflite::BuiltinOperator_TRANSPOSE,               "AddTranspose" },
    { tflite::BuiltinOperator_TRANSPOSE_CONV,          "AddTransposeConv" },
    { tflite::BuiltinOperator_UNPACK,                  "AddUnpack" },
};

const size_t kResolverTableSize = sizeof(kResolverTable) / sizeof(kResolverTable[0]);

const char* ResolverMethodName(tflite::BuiltinOperator op)
{
    for (size_t i = 0; i < kResolverTableSize; ++i) {
        if (kResolverTable[i].op == op)
            return kResolverTable[i].method;
    }
    return nullptr;
}