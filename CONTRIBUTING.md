# Contributing

Thank you for your interest in Model Checker.

Contributions are welcome.

This document describes the development guidelines used throughout the project.

---

# Project Goals

Model Checker is a lightweight command-line utility for inspecting and
verifying TensorFlow Lite models.

The project emphasizes

- simplicity
- readability
- portability
- maintainability

over unnecessary abstraction or optimization.

---

# Project Scope

Model Checker is an inspection utility.

Its purpose is to display the information stored in a TensorFlow Lite
model accurately and clearly.

Model Checker intentionally does not

- optimize models
- modify models
- perform graph analysis
- estimate model performance

Those functions belong to separate tools.

---

# Development Principles

Every source file should have a single responsibility.

Functions should remain short and easy to understand.

Avoid unnecessary abstractions.

Prefer readable code over clever code.

Avoid hidden side effects.

When adding new features, extend existing modules rather than rewriting them.

---

# Coding Style

Language

- C++17

Indentation

- Four spaces
- No tabs

Braces

```cpp
if (...) {
    ...
}
```

Naming

Classes

```cpp
ModelLoader
```

Functions

```cpp
LoadModel()
```

Variables

```cpp
model_path
```

Constants

```cpp
kDefaultArenaSize
```

Macros

```cpp
MODEL_CHECKER_VERSION
```

---

# Directory Layout

```
model_checker/

include/
src/
docs/
models/
tests/
```

Public headers belong in

```
include/
```

Implementation files belong in

```
src/
```

Documentation belongs in

```
docs/
```

Example models belong in

```
models/
```

Tests belong in

```
tests/
```

---

# Build

Build using

```bash
make
```

Cross compilation

```bash
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf-
```

The project intentionally uses a simple Makefile.

Avoid introducing unnecessary build system complexity.

---

# Error Handling

Return errors to the caller whenever possible.

Do not terminate the application from library modules.

Error messages should explain

- what failed
- why it failed
- how the user can fix it (if applicable)

---

# Output

Console output should

- be human-readable
- use the full terminal width (80 columns)
- align related information vertically
- group information into logical sections

Avoid unnecessary verbosity.

---

# Dependencies

Keep dependencies to a minimum.

Prefer the C++ standard library.

Avoid introducing third-party libraries unless there is a clear benefit.

---

# Commits

Each commit should represent one logical change.

Examples

- Add operator printer
- Add Conv2D option decoder
- Improve tensor formatting

Avoid mixing unrelated changes in a single commit.

---

# Pull Requests

Pull requests should

- have a clear purpose
- build successfully
- include documentation updates when necessary
- avoid unrelated formatting changes

---

# Documentation

Whenever behavior changes,

update the corresponding documentation.

The following documents should remain synchronized.

- README.md
- CHANGELOG.md
- docs/architecture.md
- docs/design.md

---

# Philosophy

Write code that is easy to understand six months later.

Readable code is better than clever code.

Simple code is better than complicated code.

Consistency is more important than personal preference.