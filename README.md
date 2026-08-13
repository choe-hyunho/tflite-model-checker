# Model Checker

A TensorFlow Lite Micro model inspection utility.

Model Checker is a command-line tool for inspecting and verifying
TensorFlow Lite (.tflite) models using the TensorFlow Lite Micro runtime.

The goal of the project is to provide a simple, portable, and easy-to-build
utility that helps developers understand model contents without writing
application-specific code.

---

## Features

Current

- Load TensorFlow Lite models
- Verify FlatBuffer integrity
- Verify TensorFlow Lite Micro compatibility
- Display model information
- Display input/output tensors
- Display metadata
- Display required operators

Planned

- Operator detail viewer
- Operator option decoder
- Tensor inspector
- Buffer inspector
- Quantization information
- Offline memory allocation decoder
- JSON export
- Markdown report

---

## Scope

Model Checker is an **inspection tool**.

Its purpose is to display the information already stored inside a
TensorFlow Lite model accurately and clearly.

Model Checker intentionally does **not**

- optimize models
- modify models
- perform graph analysis
- estimate performance

Those functions belong to separate tools.

---

## Supported Platforms

Current

- Linux x86_64
- Linux ARM
- Cross-compilation

Planned

- Windows
- macOS

---

## Build

```bash
make
```

Cross compilation

```bash
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf-
```

---

## Usage

```bash
model_checker <model.tflite>
```

Example

```text
$ model_checker person_detect.tflite

Model
------------------------------------------------------------
Version          3
Description      Person detection

Inputs           1
Outputs          1

...
```

---

## Project Structure

```
model_checker/
│
├── Makefile
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
│
├── docs/
│   ├── architecture.md
│   └── design.md
│
├── include/
├── src/
├── models/
└── tests/
```

---

## Current Milestone

Version **0.2**

Current work focuses on

- Stable model loading
- Model verification
- Operator inspection
- Tensor inspection

---

## License

TBD