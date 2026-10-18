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
#include "resolver_builder.h"
#include "resolver_generator.h"
#include "resolver_table.h"
#include "model_validator.h"
#include "model_summary_printer.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_utils.h"

Application::Application(const CommandLine& cmd) : command_line_(cmd) {}

Application::~Application() = default;

namespace {
constexpr size_t kTensorArenaSize = 4 * 1024 * 1024;
}

bool Application::LoadModel()
{
    if (!loader_.Load(command_line_.model_filename)) {
        return false;
    }
    document_ = ModelDocument(loader_.GetModel(), loader_.Data(), loader_.Size());
    return document_.IsValid();
}

bool Application::Verify()
{
    using namespace tflite;
    verify_error_.clear();
    allocation_passed_ = false;
    invoke_passed_ = false;
    std::string custom_names;
    if (GetCustomOperatorNames(custom_names)) {
        runtime_verification_skipped_ = true;
        verify_error_ = "custom operators require external runtime";
        custom_operator_names_ = custom_names;
        return true;
    }
    runtime_verification_skipped_ = false;
    custom_operator_names_.clear();
    TflmOpResolver resolver;
    const auto operators = CollectModelOperators(document_);
    BuiltinOperator unsupported_operator = BuiltinOperator_ADD;
    if (RegisterModelOperators(resolver, operators, &unsupported_operator) != kTfLiteOk) {
        verify_error_ = "unsupported built-in operator: ";
        verify_error_ += tflite::EnumNameBuiltinOperator(unsupported_operator);
        return false;
    }
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
    if (input->type == kTfLiteInt32) {
        const size_t element_count = input->bytes / sizeof(int32_t);
        for (size_t i = 0; i < element_count; ++i)
            input->data.i32[i] = 0;
        return true;
    }
    return false;
}

void Application::PrintRuntimeTensorArena() const
{
    if (runtime_verification_skipped_)
        return;
    ConsolePrinter printer;
    printer.Section("Runtime Tensor Arena");
    printer.KeyBytes("Configured", kTensorArenaSize);
    printer.KeyBytes("Used", arena_used_);
    const size_t free_bytes = arena_used_ <= kTensorArenaSize ? kTensorArenaSize - arena_used_ : 0;
    printer.KeyBytes("Free", free_bytes);
    printer.KeyPercent("Utilization", static_cast<double>(arena_used_) * 100.0 / static_cast<double>(kTensorArenaSize));
}

void Application::PrintReport() const
{
    ConsolePrinter console;
    ModelPrinter model(console, document_);
    ValidationPrinter validation(console, document_);
    MetadataPrinter metadata(console, document_);
    OperatorPrinter operators(console, document_, command_line_.verbose);
    OperatorDetailPrinter details(console, document_);
    ResolverGenerator resolver(console, document_);
    TensorGraph tensor_graph(document_);
    tensor_graph.Build();
    TensorPrinter tensors(console, document_, tensor_graph);
    model.Print();
    validation.Print();
    metadata.Print();
    operators.Print();
    if (command_line_.verbose) {
        details.Print();
        tensors.Print();
    }
    resolver.Print();
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
        printer.Error(validator.LastError().empty() ? "Model validation failed" : validator.LastError().c_str());
        return 1;
    }
    const bool verified = Verify();
    if (allocation_passed_)
        PrintRuntimeTensorArena();
    ConsolePrinter console;
    console.Section("Runtime Verification");
    if (runtime_verification_skipped_) {
    console.Key("Status", "SKIPPED");
    console.Key("Reason", verify_error_.c_str());
    if (!custom_operator_names_.empty())
        console.Key("Custom Operators", custom_operator_names_.c_str());
    } else {
        console.Key("AllocateTensors", allocation_passed_ ? "PASS" : "FAIL");
        console.Key("Invoke", !allocation_passed_ ? "NOT RUN" : (invoke_passed_ ? "PASS" : "FAIL"));
        if (!verify_error_.empty())
            console.Key("Error", verify_error_.c_str());
    }
    return verified ? 0 : 1;
}

bool Application::HasCustomOperator(std::string& custom_name) const
{
    custom_name.clear();
    const auto* subgraph = document_.PrimarySubgraph();
    if (subgraph == nullptr || subgraph->operators() == nullptr)
        return false;
    const auto* operators = subgraph->operators();
    for (size_t i = 0; i < operators->size(); ++i) {
        const auto* op = operators->Get(i);
        if (op == nullptr)
            continue;
        const auto* opcode = document_.OperatorCode(op->opcode_index());
        if (opcode == nullptr)
            continue;
        if (tflite::GetBuiltinCode(opcode) != tflite::BuiltinOperator_CUSTOM)
            continue;
        const auto* name = opcode->custom_code();
        if (name != nullptr)
            custom_name = name->str();
        return true;
    }
    return false;
}

bool Application::GetCustomOperatorNames(std::string& custom_names) const
{
    custom_names.clear();
    const auto* subgraph = document_.PrimarySubgraph();
    if (subgraph == nullptr || subgraph->operators() == nullptr)
        return false;
    const auto* operators = subgraph->operators();
    for (size_t i = 0; i < operators->size(); ++i) {
        const auto* op = operators->Get(i);
        if (op == nullptr)
            continue;
        const auto* opcode = document_.OperatorCode(op->opcode_index());
        if (opcode == nullptr || tflite::GetBuiltinCode(opcode) != tflite::BuiltinOperator_CUSTOM)
            continue;
        const auto* name = opcode->custom_code();
        if (name == nullptr || name->str().empty())
            continue;
        if (!custom_names.empty())
            custom_names += ", ";
        custom_names += name->str();
    }
    return !custom_names.empty();
}