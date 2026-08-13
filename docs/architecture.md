# Model Checker Architecture

Version: 1.0
Status: Frozen
Last Updated: 2026-08-04

---

# 1. Overview

Model Checker is organized as a layered architecture.

Each layer has a single responsibility.

Higher layers never access TensorFlow Lite FlatBuffer objects directly.
Lower layers never produce user-visible output.

```
                +---------------------------+
                |      Application          |
                +-------------+-------------+
                              |
                              v
                +---------------------------+
                |      Model Loader         |
                +-------------+-------------+
                              |
                              v
                +---------------------------+
                |      Model Document       |
                +-------------+-------------+
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
      Analysis Layer                  Presentation Layer
             |                                 |
             v                                 v
      Statistics                    Console Printer
      Validation                    Model Printer
      Coverage                      Tensor Printer
      Memory                        Operator Printer
```

---

# 2. Design Principles

The architecture follows five principles.

## Single Responsibility

Each class performs one task only.

Examples

- ModelLoader loads models.
- TensorPrinter prints tensors.
- Validator validates models.
- MemoryAnalyzer analyzes arena usage.

---

## Separation of Analysis and Presentation

Analysis classes never print.

Printer classes never analyze.

Good

```
Statistics
    ↓
StatisticsPrinter
```

Bad

```
Statistics

↓

printf()
```

---

## Immutable Model

After loading succeeds,

ModelDocument becomes read-only.

No class modifies FlatBuffer contents.

---

## Dependency Direction

Dependencies always point downward.

```
Application

↓

ModelDocument

↓

Analysis

↓

Printer

↓

ConsolePrinter
```

Lower layers never depend on higher layers.

---

## Testability

Analysis classes shall be testable without console output.

Printer classes shall be testable using mock data.

---

# 3. Layer Overview

## Application Layer

Responsibilities

- parse command line
- load model
- execute verification
- invoke printers
- return exit code

Files

```
main.cpp
command_line.*
```

---

## Model Layer

Responsibilities

- load FlatBuffer
- validate schema
- own model memory

Files

```
model_loader.*
model_document.*
```

ModelDocument becomes the central object shared throughout the application.

---

## Analysis Layer

Responsibilities

- inspect model
- calculate statistics
- validate graph
- analyze memory
- determine operator coverage

Analysis classes never produce console output.

Examples

```
Statistics
Validator
MemoryAnalyzer
CoverageAnalyzer
CompatibilityChecker
```

---

## Presentation Layer

Responsibilities

Convert analysis results into readable reports.

Presentation classes contain no analysis logic.

Examples

```
ModelPrinter
TensorPrinter
OperatorPrinter
MetadataPrinter
StatisticsPrinter
ValidationPrinter
CoveragePrinter
MemoryPrinter
```

---

## Output Layer

Responsibilities

Provide output formatting.

Current implementation

```
ConsolePrinter
```

Future implementations

```
JsonPrinter
MarkdownPrinter
HtmlPrinter
```

Presentation classes should not depend on printf() directly.

---

# 4. Model Ownership

```
ModelLoader
      │
      ▼
allocated buffer
      │
      ▼
ModelDocument
      │
      ▼
FlatBuffer objects
```

Only ModelLoader owns model memory.

All other classes receive const pointers or references.

No class duplicates model data.

---

# 5. Printer Hierarchy

```
ConsolePrinter
        ▲
        │
        │
+-------+---------------------------+
|                                   |
ModelPrinter                TensorPrinter
OperatorPrinter             MetadataPrinter
StatisticsPrinter           ValidationPrinter
CoveragePrinter             MemoryPrinter
```

ConsolePrinter is the only class that performs formatted console output.

---

# 6. Analysis Hierarchy

```
ModelDocument
      │
      ▼
Statistics

Validator

MemoryAnalyzer

CoverageAnalyzer

CompatibilityChecker
```

Analysis classes are independent.

One analyzer may use another, but printers shall not.

---

# 7. Utility Layer

Utility functions are stateless.

Examples

```
TensorShapeToString()

TensorTypeName()

BufferSizeToString()

QuantizationToString()
```

Utilities never perform printing.

Utilities never allocate long-lived resources.

---

# 8. Error Handling

Functions return

- bool
- nullptr
- empty containers

depending on the operation.

Version 1.0 intentionally avoids exception-based error handling.

---

# 9. Memory Management

Ownership rules

- ModelLoader owns model buffer.
- Application owns tensor arena.
- Printers own no resources.
- Analysis classes own no resources.

Version 1.0 uses RAII whenever practical.

---

# 10. Thread Safety

Version 1.0 is single-threaded.

No synchronization primitives are required.

---

# 11. Future Extension

The architecture allows additional output formats without changing analysis code.

```
Statistics
      │
      ├────────► ConsolePrinter
      ├────────► JsonPrinter
      ├────────► HtmlPrinter
      └────────► MarkdownPrinter
```

Similarly,

new analyzers can be added without modifying existing printers.

---

# 12. Summary

The architecture separates

- loading
- analysis
- presentation
- formatting

into independent layers.

This separation minimizes coupling, improves maintainability, and allows new
features to be added with minimal impact on existing code.