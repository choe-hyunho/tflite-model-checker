#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "command_line.h"

void PrintUsage(const char* program)
{
    printf("Usage:\n");
    printf("    %s [options] model.tflite\n", program);
    printf("\n");

    printf("Options:\n");
    printf("    -h, --help          Show this help\n");
    printf("    -v, --verbose       Verbose output\n");
}

bool ParseCommandLine(int argc, char* argv[], CommandLine* cmd)
{
    *cmd = {};

    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];

        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            PrintUsage(argv[0]);
            return false;
        }
        if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            cmd->verbose = true;
            continue;
        }
        if (arg[0] == '-') {
            fprintf(stderr, "Unknown option: %s\n\n", arg);
            PrintUsage(argv[0]);
            return false;
        }
        if (cmd->model_filename != nullptr) {
            fprintf(stderr, "Only one model file may be specified.\n\n");
            PrintUsage(argv[0]);
            return false;
        }
        cmd->model_filename = arg;
    }

    if (cmd->model_filename == nullptr) {
        fprintf(stderr, "No model file specified.\n\n");
        PrintUsage(argv[0]);
        return false;
    }

    return true;
}