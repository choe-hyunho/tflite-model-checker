# Model Checker Roadmap

Version: Next
Status: Living Document
Last Updated: 2026-08-04

---

# Vision

Model Checker aims to become a complete inspection, validation, analysis,
and deployment tool for TensorFlow Lite Micro models.

Version 1.x focuses on inspection.

Version 2.x focuses on analysis and optimization.

---

# Release Plan

Version 1.0

✓ Model inspection

✓ Validation

✓ Verification

✓ Coverage

✓ Memory report

---

Version 2.0

□ Analysis

□ Optimization

□ Visualization

□ Export

---

Version 3.0

□ Deployment assistant

□ Performance estimation

□ Automatic optimization

---

# Version 2.0

## Analysis

Arena lifetime analysis

Peak memory analysis

Tensor lifetime analysis

Constant memory analysis

Activation memory analysis

Graph statistics

Operator statistics

Model complexity

---

## Visualization

Graphviz export

SVG export

Arena visualization

Memory heat map

Tensor dependency graph

Operator dependency graph

---

## Export

JSON

HTML

Markdown

CSV

---

## Comparison

Model comparison

Tensor comparison

Operator comparison

Arena comparison

Version comparison

---

## Resolver

Generate MicroMutableOpResolver

Generate source file

Generate header file

Operator dependency report

---

## Deployment

Compatibility report

MCU recommendation

Flash estimation

RAM estimation

Arena estimation

---

## Diagnostics

Risk report

Quantization report

Unsupported operator report

Operator version mismatch

Memory fragmentation report

---

## Interactive

Search tensor

Search operator

Trace tensor

Reverse trace

Dump tensor

Dump constant

---

# Version 3.0

Automatic arena optimization

Model scoring

Deployment checklist

Performance estimation

Model explanation

Automatic report generation

Plugin architecture

---

# Nice to Have

Dark terminal theme

Colorized console

Unicode box drawing

Progress bar

Configuration file

Project profiles

Batch mode

Directory scan

---

# Long-term Ideas

Support TensorFlow Lite desktop models

Support ONNX inspection

Support CMSIS-NN compatibility

Support Ethos-U compatibility

Support vendor-specific delegates

---

# Out of Scope

The following are intentionally excluded.

GUI

Qt

Electron

Python dependency

Embedded web server

Database

Cloud service

Model training

Model conversion

---

# Development Philosophy

Prefer

- simplicity
- portability
- readability
- maintainability

Avoid

- unnecessary dependencies
- hidden magic
- platform-specific code
- feature creep

---

# Success Criteria

Model Checker should become the first tool developers use after obtaining
a .tflite model.

Typical workflow

Train

↓

Convert

↓

Model Checker

↓

Deploy

↓

Benchmark

↓

Ship