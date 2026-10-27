#include "console_printer.h"

#include <cstdarg>
#include <cstdio>

ConsolePrinter::ConsolePrinter(FILE* fp)
    : fp_(fp)
{
}

/******************************************************************************
 * Private
 *****************************************************************************/

void ConsolePrinter::Print(const char* fmt, ...) const
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp_, fmt, ap);
    va_end(ap);
}

void ConsolePrinter::PrintMessage(const char* prefix, const char* fmt, va_list ap) const
{
    fprintf(fp_, "%s: ", prefix);
    vfprintf(fp_, fmt, ap);
    fprintf(fp_, "\n");
}

/******************************************************************************
 * Document
 *****************************************************************************/

void ConsolePrinter::Banner(const char* title, const char* version) const
{
    Separator('=');
    Print("%s v%s\n", title, version);
    Separator('=');
}

void ConsolePrinter::Section(const char* title) const
{
    Blank();
    Print("%s\n", title);
    Separator('-');
}

void ConsolePrinter::SubSection(const char* title) const
{
    Blank();
    Print("%s\n", title);
    for (const char* p = title; *p; ++p)
        fputc('-', fp_);
    fputc('\n', fp_);
}

void ConsolePrinter::Blank() const
{
    Print("\n");
}

void ConsolePrinter::Line(const char* text) const
{
    Print("%s\n", text);
}

void ConsolePrinter::Separator(char ch) const
{
    for (int i = 0; i < kWidth; i++)
        fputc(ch, fp_);
    fputc('\n', fp_);
}

/******************************************************************************
 * Key / Value
 *****************************************************************************/

void ConsolePrinter::Key(const char* name, const char* value) const
{
    Print("%-*s : %s\n", kKeyWidth, name, value ? value : "");
}

void ConsolePrinter::Key(const char* name, const std::string& value) const
{
    Key(name, value.c_str());
}

void ConsolePrinter::Key(const char* name, size_t value) const
{
    Print("%-*s : %zu\n", kKeyWidth, name, value);
}

void ConsolePrinter::KeyBytes(const char* name, size_t bytes) const
{
    if (bytes < 1024) {
        Print("%-*s : %zu bytes\n", kKeyWidth, name, bytes);
        return;
    }
    double value = static_cast<double>(bytes);
    const char* unit;
    if (bytes >= 1024ULL * 1024ULL * 1024ULL) {
        value /= 1024.0 * 1024.0 * 1024.0;
        unit = "GB";
    }
    else if (bytes >= 1024ULL * 1024ULL) {
        value /= 1024.0 * 1024.0;
        unit = "MB";
    }
    else {
        value /= 1024.0;
        unit = "KB";
    }
    Print("%-*s : %.1f %s (%zu bytes)\n", kKeyWidth, name, value, unit, bytes);
}

void ConsolePrinter::KeyPercent(const char* name, double value) const
{
    Print("%-*s : %.1f %%\n", kKeyWidth, name, value);
}

/******************************************************************************
 * Status
 *****************************************************************************/

void ConsolePrinter::Status(const char* name, bool pass) const
{
    Status(name, pass ? "PASS" : "FAIL");
}

void ConsolePrinter::Status(const char* name, const char* status) const
{
    Key(name, status);
}

/******************************************************************************
 * Messages
 *****************************************************************************/

void ConsolePrinter::Info(const char* fmt, ...) const
{
    va_list ap;
    va_start(ap, fmt);
    PrintMessage("Info", fmt, ap);
    va_end(ap);
}

void ConsolePrinter::Warning(const char* fmt, ...) const
{
    va_list ap;
    va_start(ap, fmt);
    PrintMessage("Warning", fmt, ap);
    va_end(ap);
}

void ConsolePrinter::Error(const char* fmt, ...) const
{
    va_list ap;
    va_start(ap, fmt);
    PrintMessage("Error", fmt, ap);
    va_end(ap);
}

/******************************************************************************
 * Table
 *****************************************************************************/

void ConsolePrinter::TableHeader(const char* c1, const char* c2, const char* c3, const char* c4) const
{
    TableRow(c1, c2, c3, c4);
    Separator('-');
}

void ConsolePrinter::TableRow(const char* c1, const char* c2, const char* c3, const char* c4) const
{
    Print("%-*s", kCol1, c1 ? c1 : "");
    if (c2)
        Print("%-*s", kCol2, c2);
    if (c3)
        Print("%-*s", kCol3, c3);
    if (c4)
        Print("%s", c4);
    Print("\n");
}