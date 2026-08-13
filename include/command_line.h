#ifndef COMMAND_LINE_H
#define COMMAND_LINE_H

#include <stddef.h>

struct CommandLine {
    bool verbose;
    const char* model_filename;
};

/* Parse command-line arguments. */
bool ParseCommandLine(int argc, char* argv[], CommandLine* cmd);

/* Print command-line usage. */
void PrintUsage(const char* program);

#endif