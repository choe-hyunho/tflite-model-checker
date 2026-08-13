# Model Checker Design Guide

Version: 1.0
Status: Frozen
Last Updated: 2026-08-04

---

# 1. Purpose

This document describes the implementation conventions used throughout
Model Checker.

Its purpose is to ensure that all source files follow consistent design,
naming, ownership, and coding patterns.

This document complements Architecture.md.

---

# 2. General Principles

The project follows several simple rules.

1. One class has one responsibility.

2. One source file implements one class.

3. Business logic shall not appear inside printers.

4. Console output shall only be performed by ConsolePrinter.

5. Utility functions shall be stateless.

6. Public interfaces shall remain small.

---

# 3. Directory Layout

```
include/

src/

docs/

tests/

samples/
```

Headers and sources use matching names.

Example

```
tensor_printer.h

tensor_printer.cpp
```

---

# 4. Naming Conventions

## Classes

PascalCase

```
ModelLoader

TensorPrinter

MemoryAnalyzer
```

---

## Methods

PascalCase

```
Load()

Print()

Analyze()

Validate()
```

---

## Variables

snake_case

```
tensor_index

arena_used

model_buffer
```

---

## Constants

```
kArenaSize

kWidth

kMaxColumns
```

---

## Files

snake_case

```
model_loader.cpp

tensor_printer.cpp

memory_analyzer.cpp
```

---

# 5. Class Categories

The project contains several kinds of classes.

---

## Loader

Responsible for reading files.

Example

```
ModelLoader
```

---

## Document

Represents immutable model data.

Example

```
ModelDocument
```

---

## Analyzer

Calculates information.

Examples

```
Statistics

Validator

MemoryAnalyzer

CoverageAnalyzer
```

Analyzers

- never print
- never own model memory

---

## Printer

Produces human-readable output.

Examples

```
ModelPrinter

TensorPrinter

OperatorPrinter
```

Printers

- never analyze
- never allocate resources
- only format output

---

## Utility

Stateless helper functions.

Examples

```
TensorShapeToString()

TensorTypeName()

BuiltinOperatorName()
```

---

# 6. Ownership

Ownership rules are strict.

```
Application

owns

↓

ModelLoader

owns

↓

model buffer
```

All remaining classes receive

```
const ModelDocument&
```

or

```
const Model*
```

No copies shall be made.

---

# 7. Dependency Rules

Allowed

```
Printer

↓

Analyzer

↓

ModelDocument
```

Not allowed

```
Analyzer

↓

Printer
```

---

Allowed

```
ModelPrinter

↓

ConsolePrinter
```

Not allowed

```
ConsolePrinter

↓

ModelPrinter
```

---

# 8. Console Output

All formatted console output shall use

```
ConsolePrinter
```

Never use

```
printf()

fprintf(stdout,...)

std::cout
```

inside application classes.

---

# 9. Error Handling

Version 1.0 uses return values.

Examples

```
bool

nullptr

empty vector
```

Exceptions are not used.

---

# 10. Formatting

Reports should

- fit within 80 columns
- align key/value pairs
- use consistent separators
- remain readable in plain terminals

---

# 11. Adding a Printer

Steps

1.

Create

```
xxx_printer.h

xxx_printer.cpp
```

2.

Accept

```
ConsolePrinter&
```

through the constructor.

3.

Accept

```
const ModelDocument&
```

or the appropriate analysis result.

4.

Implement

```
Print()
```

5.

No business logic.

---

# 12. Adding an Analyzer

Steps

1.

Create

```
xxx_analyzer.h

xxx_analyzer.cpp
```

2.

Accept

```
const ModelDocument&
```

3.

Return analysis results.

4.

Never print.

---

# 13. Utility Functions

Utility functions

- shall be pure
- shall not modify arguments
- shall not perform console output

Preferred

```
TensorShapeToString()

↓

std::string
```

Avoid

```
PrintTensorShape()
```

---

# 14. Include Rules

Header files include only what they require.

Forward declarations are preferred whenever possible.

Source files include corresponding headers first.

Example

```
#include "tensor_printer.h"

#include <string>

#include "console_printer.h"
```

---

# 15. Source File Structure

Recommended order

```
includes

namespace

constants

private helpers

constructor

public methods

private methods
```

---

# 16. Testing

Every analyzer should be testable independently.

Printers should be testable using known analysis results.

Golden output files are recommended.

---

# 17. Compiler Requirements

Minimum

```
C++17
```

Required warning level

```
-Wall

-Wextra
```

Warnings should be treated as defects.

---

# 18. Coding Style

Prefer

```
const

constexpr

std::array

std::vector
```

Avoid

```
macros

global variables

mutable shared state
```

---

# 19. Versioning

Public interfaces should remain stable throughout Version 1.x.

Breaking API changes belong to Version 2.0.

---

# 20. Summary

The project is organized around four concepts.

```
Load

↓

Analyze

↓

Print

↓

Verify
```

Keeping these responsibilities separate is the primary design goal of
Model Checker.