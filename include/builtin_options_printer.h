#ifndef BUILTIN_OPTIONS_PRINTER_H
#define BUILTIN_OPTIONS_PRINTER_H

#include "console_printer.h"

#include "tensorflow/lite/schema/schema_generated.h"

class BuiltinOptionsPrinter {
public:
    BuiltinOptionsPrinter(ConsolePrinter& printer);
    void Print(const tflite::Operator* op) const;

private:
    void PrintFullyConnected(const tflite::Operator* op) const;
    void PrintConv2D(const tflite::Operator* op) const;
    void PrintDepthwiseConv2D(const tflite::Operator* op) const;

private:
    ConsolePrinter& printer_;
};

#endif