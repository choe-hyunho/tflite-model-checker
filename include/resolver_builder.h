#ifndef RESOLVER_BUILDER_H
#define RESOLVER_BUILDER_H

#include <set>

#include "op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

class ModelDocument;

std::set<tflite::BuiltinOperator> CollectModelOperators(const ModelDocument& document);
TfLiteStatus RegisterModelOperators(tflite::TflmOpResolver& resolver, const std::set<tflite::BuiltinOperator>& operators, tflite::BuiltinOperator* unsupported_operator = nullptr);
#endif