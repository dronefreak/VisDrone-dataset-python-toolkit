# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Video sequence support for temporal tasks
- Integration with Weights & Biases for experiment tracking
- TensorRT optimization for faster inference
- Docker images for easy deployment
- Additional model architectures (DETR, YOLOv8, etc.)
- Mobile deployment guide (CoreML, TFLite)

## [2.0.0] - 2025-01-15

### Major Rewrite

Complete rewrite of the original 2019 toolkit with modern PyTorch and best practices.

### Added

- **Core Components**

  - Native PyTorch Dataset class (`VisDroneDataset`) with automatic filtering
  - Support for 4 detection models: Faster R-CNN (ResNet50, MobileNet), FCOS, RetinaNet
  - Model factory with `get_model()` function
  - Comprehensive visualization utilities
  - Format converters for COCO and YOLO (not just VOC)

- **CLI Tools**

  - `train.py` - Complete training pipeline with AMP support
  - `inference.py` - Batch inference on images/videos
  - `webcam_demo.py` - Real-time webcam detection demo
  - `evaluate.py` - Model evaluation with metrics
  - `convert_annotations.py` - Format conversion utility

- **Training Features**

  - Automatic mixed precision (AMP) for 2x faster training
  - Learning rate scheduling (StepLR)
  - Checkpointing and resuming
  - Training curve visualization
  - Validation during training

- **Documentation**

  - Comprehensive README with examples
  - Installation guide (INSTALL.md)
  - Quick reference guide (QUICKSTART.md)
  - Contributing guidelines (CONTRIBUTING.md)
  - Code of Conduct (CODE_OF_CONDUCT.md)
  - Security policy (SECURITY.md)
  - Test documentation

- **Development**

  - 66 unit tests with pytest
  - Test coverage >80%
  - Pre-commit hooks for code quality
  - Makefile for common tasks
  - Type hints throughout codebase
  - Modern packaging with pyproject.toml

- **Configuration**
  - YAML configuration files for all models
  - Virtualenv-first installation
  - Requirements files for different use cases
  - .gitignore for Python projects

### Changed

- **Framework**: Migrated from TensorFlow 1.x to PyTorch 2.0+
- **Models**: Replaced custom implementations with torchvision models
- **Format**: Moved from PASCAL VOC only to COCO and YOLO support
- **Installation**: Simplified with virtualenv instead of conda
- **Structure**: Proper Python package with CLI entry points

### Improved

- **Performance**: 2-3x faster training with AMP
- **Usability**: Simple CLI tools instead of complex scripts
- **Maintainability**: Comprehensive tests and documentation
- **Code Quality**: Type hints, linting, formatting with Black
- **Flexibility**: Easy to extend with new models and features

### Removed

- TensorFlow dependencies
- PASCAL VOC as primary format (still available if needed)
- Legacy training scripts
- Conda environment (replaced with virtualenv)

## [1.0.0] - 2019-XX-XX

### Initial Release (Legacy)

- Basic VisDrone annotation conversion to PASCAL VOC format
- TensorFlow 1.x implementation
- Faster R-CNN with Inception V3
- Simple conversion script (`convertVis_to_xml.py`)
- Basic visualization
- Training script for static images

---

## Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backwards compatible manner
- **PATCH** version for backwards compatible bug fixes

## Types of Changes

- **Added** - New features
- **Changed** - Changes in existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Security vulnerability fixes

## Links

- [Unreleased]: https://github.com/dronefreak/VisDrone-dataset-python-toolkit/compare/v2.0.0...HEAD
- [2.0.0]: https://github.com/dronefreak/VisDrone-dataset-python-toolkit/releases/tag/v2.0.0
- [1.0.0]: https://github.com/dronefreak/VisDrone-dataset-python-toolkit/releases/tag/v1.0.0
