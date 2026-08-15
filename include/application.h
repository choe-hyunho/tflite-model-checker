#ifndef APPLICATION_H
#define APPLICATION_H

#include "command_line.h"
#include "model_loader.h"
#include "model_document.h"

namespace tflite {
class MicroInterpreter;
}

class ModelSummaryPrinter;

class Application {
public:
    explicit Application(const CommandLine& cmd);
    ~Application();
    int Run();

private:
    bool LoadModel();
    bool Verify();
    void PrintReport() const;
    void PrintTensorArena() const;
    bool InitializeInput(tflite::MicroInterpreter& interpreter);

private:
    CommandLine command_line_;
    ModelLoader loader_;
    ModelDocument document_;
    std::unique_ptr<ModelSummaryPrinter> summary_printer_;
    size_t arena_used_ = 0;
    std::string verify_error_;
    bool allocation_passed_ = false;
    bool invoke_passed_ = false;
};

#endif