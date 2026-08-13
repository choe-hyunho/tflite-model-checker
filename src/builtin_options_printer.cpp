#include "builtin_options_printer.h"

#include "console_printer.h"

#include "tensorflow/lite/schema/schema_generated.h"

BuiltinOptionsPrinter::BuiltinOptionsPrinter(ConsolePrinter& printer)
    : printer_(printer) {}

void BuiltinOptionsPrinter::Print(const tflite::Operator* op) const
{
    if (op == nullptr) {
        return;
    }
    switch (op->builtin_options_type()) {
    case tflite::BuiltinOptions_FullyConnectedOptions:
        PrintFullyConnected(op);
        break;
    case tflite::BuiltinOptions_Conv2DOptions:
        PrintConv2D(op);
        break;
    case tflite::BuiltinOptions_DepthwiseConv2DOptions:
        PrintDepthwiseConv2D(op);
        break;
    default:
        break;
    }
}

void BuiltinOptionsPrinter::PrintFullyConnected(const tflite::Operator* op) const
{
    const auto* options = op->builtin_options_as_FullyConnectedOptions();

    if (options == nullptr) {
        return;
    }

    printer_.Key("Activation", tflite::EnumNameActivationFunctionType(options->fused_activation_function()));
    printer_.Key("Keep Num Dims", options->keep_num_dims() ? "Yes" : "No");
    printer_.Key("Asymmetric Quant", options->asymmetric_quantize_inputs() ? "Yes" : "No");
}

void BuiltinOptionsPrinter::PrintConv2D(const tflite::Operator* op) const
{
    const auto* options = op->builtin_options_as_Conv2DOptions();

    if (options == nullptr) {
        return;
    }

    printer_.Key("Padding", tflite::EnumNamePadding(options->padding()));
    printer_.Key("Stride Width", options->stride_w());
    printer_.Key("Stride Height", options->stride_h());
    printer_.Key("Activation", tflite::EnumNameActivationFunctionType(options->fused_activation_function()));
    printer_.Key("Dilation Width", options->dilation_w_factor());
    printer_.Key("Dilation Height", options->dilation_h_factor());
}

void BuiltinOptionsPrinter::PrintDepthwiseConv2D(const tflite::Operator* op) const
{
    const auto* options = op->builtin_options_as_DepthwiseConv2DOptions();

    if (options == nullptr) {
        return;
    }

    printer_.Key("Padding", tflite::EnumNamePadding(options->padding()));
    printer_.Key("Stride Width", options->stride_w());
    printer_.Key("Stride Height", options->stride_h());
    printer_.Key("Depth Multiplier", options->depth_multiplier());
    printer_.Key("Activation", tflite::EnumNameActivationFunctionType(options->fused_activation_function()));
    printer_.Key("Dilation Width", options->dilation_w_factor());
    printer_.Key("Dilation Height", options->dilation_h_factor());
}