# Model Checker v1.0 Specification

Version: 1.0
Status: Frozen
Last Updated: 2026-08-04

---

# 1. Introduction

Model Checker is a command-line utility for inspecting, validating, and verifying
TensorFlow Lite models, with particular emphasis on TensorFlow Lite Micro (TFLM).

The tool is intended for embedded software engineers who need to understand a
model before deploying it onto memory-constrained devices.

Model Checker is a read-only utility.
It never modifies a model.

---

# 2. Objectives

Version 1.0 provides:

- Model inspection
- Operator inspection
- Tensor inspection
- Metadata inspection
- Model validation
- Tensor arena verification
- Operator coverage analysis
- Human-readable reports

Version 1.0 does not optimize, modify, or convert models.

---

# 3. Supported Platforms

## Host

- Linux x86-64
- GCC
- Clang

## Target

- ARM Linux
- GNU Cross Compiler

---

# 4. Supported Model Format

TensorFlow Lite (.tflite)

Version 1.0 focuses on TensorFlow Lite Micro compatible models.

---

# 5. Command Line

```
model_checker [options] model.tflite
```

Example

```
model_checker mobilenet.tflite
```

---

# 6. Functional Requirements

## 6.1 Model Information

Display

- file name
- file size
- schema version
- model description
- number of subgraphs

---

## 6.2 Tensor Information

Display

- index
- name
- type
- shape
- buffer
- quantization
- producer
- consumers

---

## 6.3 Operator Information

Display

- index
- builtin/custom
- opcode
- version
- inputs
- outputs
- builtin options
- custom options

---

## 6.4 Metadata

Display

- metadata entries
- associated buffers

---

## 6.5 Validation

Verify

- schema version
- tensor indices
- operator indices
- buffers
- graph consistency

Validation shall not execute inference.

---

## 6.6 Verification

Verify execution using TensorFlow Lite Micro.

Operations performed

1. AllocateTensors()

2. Zero initialize input tensors

3. Invoke()

Display PASS or FAIL.

---

## 6.7 Tensor Arena

Display

- configured arena
- used arena
- free arena
- utilization percentage

---

## 6.8 Operator Coverage

Display

- required builtin operators
- required custom operators
- generated resolver code

---

# 7. Output

The default output format is plain text.

Reports shall be human-readable.

Output shall be suitable for

- console
- terminal
- text files

---

# 8. Exit Codes

| Code | Meaning |
|------:|---------|
| 0 | Success |
| 1 | Invalid command line |
| 2 | Model load failure |
| 3 | Verification failure |

---

# 9. Non-functional Requirements

The project shall

- compile as C++17
- build on host and target
- avoid compiler warnings
- avoid duplicated code
- avoid memory leaks
- separate analysis from presentation
- be unit-test friendly

---

# 10. Out of Scope

The following features are intentionally excluded from Version 1.0.

- HTML reports
- JSON reports
- Graphviz export
- SVG export
- Arena simulation
- Memory optimization
- Model comparison
- Performance estimation
- Resolver file generation
- Interactive mode
- GUI

These features belong to Version 2.0.

---

# 11. Project Layout

```
include/
src/
docs/
tests/
samples/
```

---

# 12. Documentation

Version 1.0 consists of

- README.md
- specification.md
- architecture.md
- design.md
- building.md
- cross_compile.md
- roadmap.md

---

# 13. Release Criteria

Version 1.0 is considered complete when

- Host build succeeds
- ARM build succeeds
- All compiler warnings are resolved
- Sample models execute successfully
- Documentation is complete
- Public interfaces are stable
- Version is tagged v1.0.0

---

# 14. Version History

## Version 1.0

Initial public release.