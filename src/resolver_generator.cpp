#include "resolver_generator.h"

#include <cstdio>
#include <set>

#include "console_printer.h"
#include "model_document.h"
#include "resolver_builder.h"
#include "resolver_table.h"

ResolverGenerator::ResolverGenerator(ConsolePrinter& printer, const ModelDocument& document)
    : printer_(printer), document_(document) {}

void ResolverGenerator::Print() const
{
    printer_.Section("Resolver");
    if (!document_.IsValid()) {
        printer_.Line("(invalid model)");
        return;
    }
    const auto ops = CollectModelOperators(document_);
    if (ops.empty()) {
        printer_.Line("(no operators)");
        return;
    }
    char line[128];
    snprintf(line, sizeof(line), "using OpResolver = tflite::MicroMutableOpResolver<%zu>;", ops.size());
    printer_.Line(line);
    printer_.Blank();
    printer_.Line("OpResolver resolver;");
    printer_.Blank();
    for (const auto op : ops) {
        const char* method = ResolverMethodName(op);
        if (method == nullptr) {
            printer_.Line("// Unsupported operator");
            continue;
        }
        snprintf(line, sizeof(line), "resolver.%s();", method);
        printer_.Line(line);
    }
}