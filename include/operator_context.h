#ifndef OPERATOR_CONTEXT_H
#define OPERATOR_CONTEXT_H

#include "tensorflow/lite/schema/schema_generated.h"

struct OperatorContext {
    const tflite::Model* model;
    const tflite::SubGraph* subgraph;
    const tflite::Operator* op;
    const tflite::OperatorCode* opcode;
    int index;
};

#endif