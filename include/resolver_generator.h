#ifndef RESOLVER_GENERATOR_H
#define RESOLVER_GENERATOR_H

#include "tensorflow/lite/schema/schema_generated.h"

class ConsolePrinter;
class ModelDocument;

class ResolverGenerator {
public:
    ResolverGenerator(ConsolePrinter& printer, const ModelDocument& document);
    void Print() const;

private:
    ConsolePrinter& printer_;
    const ModelDocument& document_;
};

#endif