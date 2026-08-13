#ifndef CONSOLE_PRINTER_H
#define CONSOLE_PRINTER_H

#include <cstddef>
#include <cstdarg>
#include <cstdio>
#include <string>

class ConsolePrinter {
public:
    explicit ConsolePrinter(FILE* fp = stdout);

    /**************************************************************************
     * Document
     *************************************************************************/

    void Banner(const char* title,
                const char* version) const;

    void Section(const char* title) const;

    void SubSection(const char* title) const;

    void Blank() const;

    void Line(const char* text) const;

    void Separator(char ch = '=') const;

    /**************************************************************************
     * Key / Value
     *************************************************************************/

    void Key(const char* name,
             const char* value) const;

    void Key(const char* name,
             const std::string& value) const;

    void Key(const char* name,
             size_t value) const;

    void KeyBytes(const char* name,
                  size_t value) const;

    void KeyPercent(const char* name,
                    double value) const;

    // Prevent accidental std::string/std::string usage.
    void Key(const std::string&,
             const std::string&) const = delete;

    /**************************************************************************
     * Status
     *************************************************************************/

    void Status(const char* name,
                bool pass) const;

    void Status(const char* name,
                const char* status) const;

    /**************************************************************************
     * Messages
     *************************************************************************/

    void Info(const char* fmt, ...) const;

    void Warning(const char* fmt, ...) const;

    void Error(const char* fmt, ...) const;

    /**************************************************************************
     * Table
     *************************************************************************/

    void TableHeader(const char* c1,
                     const char* c2,
                     const char* c3 = nullptr,
                     const char* c4 = nullptr) const;

    void TableRow(const char* c1,
                  const char* c2,
                  const char* c3 = nullptr,
                  const char* c4 = nullptr) const;

private:
    FILE* fp_;

    void Print(const char* fmt, ...) const;

    void PrintMessage(const char* prefix,
                      const char* fmt,
                      va_list ap) const;

    static constexpr int kWidth    = 80;
    static constexpr int kKeyWidth = 22;

    static constexpr int kCol1 = 8;
    static constexpr int kCol2 = 16;
    static constexpr int kCol3 = 24;
};

#endif