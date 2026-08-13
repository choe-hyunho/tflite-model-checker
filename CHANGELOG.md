# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Project documentation
- Build system
- Model loader
- FlatBuffer verification
- TensorFlow Lite Micro verification
- Model information printer
- Tensor information printer
- Metadata printer
- Required operator printer

### Changed

- Renamed project from **check_model** to **model_checker**.
- Reorganized source tree into `src/` and `include/`.
- Improved project documentation.

---

## [0.2.0] - In Development

### Planned

- Operator detail printer
- Operator option decoder
- Buffer information
- Quantization information
- Offline memory allocation information
- Improved console formatting

---

## [0.1.0]

### Added

- Initial project structure
- Makefile build system
- TensorFlow Lite Micro integration
- Basic model inspection
- AllocateTensors() verification
- Invoke() verification

---

## Versioning

This project follows Semantic Versioning.

- Major versions introduce significant new functionality or architectural changes.
- Minor versions add new features while maintaining compatibility.
- Patch versions contain bug fixes and documentation improvements.