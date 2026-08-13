#include "application.h"

#include <cstdlib>
#include <cstring>

#include "op_resolver.h"
#include "console_printer.h"
#include "model_printer.h"
#include "metadata_printer.h"
#include "operator_printer.h"
#include "operator_detail_printer.h"
#include "tensor_graph.h"
#include "tensor_printer.h"
#include "validation_printer.h"
#include "resolver_generator.h"
#include "model_validator.h"
#include "model_summary_printer.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

Application::Application(const CommandLine& cmd) : command_line_(cmd) {}

Application::~Application() = default;

bool Application::LoadModel()
{
    if (!loader_.Load(command_line_.model_filename)) {
        return false;
    }
    document_ = ModelDocument(loader_.GetModel());
    return document_.IsValid();
}

bool Application::Verify()
{
    using namespace tflite;

    TflmOpResolver resolver;

    if (CreateOpResolver(resolver) != kTfLiteOk) {
        return false;
    }

    constexpr size_t kTensorArenaSize = 1024 * 1024;
    auto* tensor_arena = static_cast<uint8_t*>(malloc(kTensorArenaSize));

    if (!tensor_arena) {
        return false;
    }
    memset(tensor_arena, 0, kTensorArenaSize);

    bool result = true;
    {
        MicroInterpreter interpreter(loader_.GetModel(), resolver, tensor_arena, kTensorArenaSize);
        if (interpreter.AllocateTensors() != kTfLiteOk) {
            result = false;
        } else {
            arena_used_ = interpreter.arena_used_bytes();
            if (interpreter.Invoke() != kTfLiteOk) {
                result = false;
            }
        }
    }
    free(tensor_arena);
    return result;
}

void Application::PrintTensorArena() const
{
    ConsolePrinter printer;

    printer.Section("Tensor Arena");

    constexpr size_t kTensorArenaSize = 1024 * 1024;

    printer.KeyBytes("Configured", kTensorArenaSize);
    printer.KeyBytes("Used", arena_used_);
    printer.KeyBytes("Free", kTensorArenaSize - arena_used_);
    printer.KeyPercent("Utilization", static_cast<double>(arena_used_) * 100.0 / static_cast<double>(kTensorArenaSize));
}

void Application::PrintReport() const
{
    ConsolePrinter console;
    ModelPrinter model(console, document_);
    ValidationPrinter validation(console, document_);
    MetadataPrinter metadata(console, document_);
    OperatorPrinter operators(console, document_);
    OperatorDetailPrinter details(console, document_);
    ResolverGenerator resolver(console, document_);
    TensorGraph tensor_graph(document_);
    tensor_graph.Build();
    TensorPrinter tensors(console, document_, tensor_graph);
    model.Print();
    validation.Print();
    metadata.Print();
    operators.Print();
    details.Print();
    resolver.Print();
    tensors.Print();
    PrintTensorArena();
    ModelSummaryPrinter summary(console, document_, tensor_graph);
    summary.Print();
}

int Application::Run()
{
    if (!LoadModel()) {
        return 2;
    }
    PrintReport();

    ModelValidator validator(document_);

    if (!validator.Validate()) {
        ConsolePrinter printer;
        printer.Error("Model validation failed");
        return 1;
    }
    if (!Verify()) {
        return 3;
    }
    return 0;
}