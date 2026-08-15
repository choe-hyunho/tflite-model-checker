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

    verify_error_.clear();
    allocation_passed_ = false;
    invoke_passed_ = false;

    TflmOpResolver resolver;

    if (CreateOpResolver(resolver) != kTfLiteOk) {
        verify_error_ = "failed to create operator resolver";
        return false;
    }

    constexpr size_t kTensorArenaSize = 1024 * 1024;
    auto* tensor_arena = static_cast<uint8_t*>(malloc(kTensorArenaSize));

    if (!tensor_arena) {
        verify_error_ = "failed to allocate tensor arena";
        return false;
    }
    memset(tensor_arena, 0, kTensorArenaSize);

    bool result = true;
    {
        MicroInterpreter interpreter(loader_.GetModel(), resolver, tensor_arena, kTensorArenaSize);
        if (interpreter.AllocateTensors() != kTfLiteOk) {
            verify_error_ = "AllocateTensors() failed";
            result = false;
        } else {
            allocation_passed_ = true;
            arena_used_ = interpreter.arena_used_bytes();
            if (!InitializeInput(interpreter)) {
                verify_error_ = "failed to initialize model input";
                result = false;
            } else if (interpreter.Invoke() != kTfLiteOk) {
                verify_error_ = "Invoke() failed";
                result = false;
            } else {
                invoke_passed_ = true;
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

bool Application::InitializeInput(tflite::MicroInterpreter& interpreter)
{
    auto* input = interpreter.input(0);

    if (input == nullptr)
        return false;
    if (input->data.raw == nullptr)
        return false;
    if (input->type == kTfLiteFloat32) {
        const size_t element_count = input->bytes / sizeof(float);
        float* data = input->data.f;

        for (size_t i = 0; i < element_count; ++i)
            data[i] = 0.0f;
        return true;
    }
    if (input->type == kTfLiteInt8) {
        memset(input->data.int8, 0, input->bytes);
        return true;
    }
    if (input->type == kTfLiteUInt8) {
        memset(input->data.uint8, 0, input->bytes);
        return true;
    }
    if (input->type == kTfLiteInt16) {
        memset(input->data.i16, 0, input->bytes);
        return true;
    }
    return false;
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
    const bool verified = Verify();
    ConsolePrinter console;
    console.Section("Runtime Verification");
    console.Key("AllocateTensors", allocation_passed_ ? "PASS" : "FAIL");
    console.Key("Invoke", invoke_passed_ ? "PASS" : "FAIL");
    if (!verify_error_.empty())
        console.Key("Error", verify_error_.c_str());
    return verified ? 0 : 1;
}