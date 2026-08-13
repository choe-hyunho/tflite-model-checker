#ifndef RESOLVER_TABLE_H
#define RESOLVER_TABLE_H

#include "tensorflow/lite/schema/schema_generated.h"

struct ResolverEntry {
    tflite::BuiltinOperator op;
    const char* method;
};

extern const ResolverEntry kResolverTable[];
extern const size_t kResolverTableSize;

const char* ResolverMethodName(tflite::BuiltinOperator op);

#endif