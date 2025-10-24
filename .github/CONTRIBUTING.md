# Contributing to VisDrone Toolkit

First off, thanks for taking the time to contribute! 🎉

This document provides guidelines for contributing to the VisDrone Toolkit. Following these guidelines helps communicate that you respect the time of the developers managing this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)

## Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to <your.email@example.com>.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, PyTorch version, GPU)
- **Error messages** and stack traces
- **Screenshots** if applicable

**Bug Report Template:**

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:

1. Run command '...'
2. With configuration '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**

- OS: [e.g. Ubuntu 22.04]
- Python version: [e.g. 3.10]
- PyTorch version: [e.g. 2.0.1]
- CUDA version: [e.g. 11.8]
- GPU: [e.g. RTX 3090]

**Additional context**
Any other relevant information.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear title** that describes the enhancement
- **Provide detailed description** of the proposed enhancement
- **Explain why this would be useful** to most users
- **List similar features** in other projects if applicable

### Pull Requests

We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`
2. Add tests if you've added code
3. Update documentation if needed
4. Ensure tests pass
5. Make sure your code follows the style guidelines
6. Issue the pull request

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/VisDrone-dataset-python-toolkit.git
cd VisDrone-dataset-python-toolkit
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Install Development Dependencies

```bash
# Using make
make install-dev

# Or manually
pip install -e ".[dev]"
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

This will automatically run linters before each commit.

### 5. Create Feature Branch

```bash
git checkout -b feature/amazing-feature
```

## Pull Request Process

### Before Submitting

1. **Update tests**: Add or update tests for your changes
2. **Run tests**: Ensure all tests pass

   ```bash
   make test
   # or
   pytest tests/ -v
   ```

3. **Check code style**: Format and lint your code

   ```bash
   make format
   make lint
   ```

4. **Update documentation**: If you changed APIs or added features
5. **Update CHANGELOG.md**: Add entry under "Unreleased" section

### Submitting

1. **Push to your fork**

   ```bash
   git push origin feature/amazing-feature
   ```

2. **Open Pull Request** on GitHub with:

   - Clear title describing the change
   - Description of what changed and why
   - Link to related issues
   - Screenshots/demos if applicable

3. **Address review feedback**
   - Respond to comments
   - Make requested changes
   - Push updates to the same branch

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Commit messages follow guidelines
- [ ] No merge conflicts with main

## Style Guidelines

### Python Code Style

We use **Black** for formatting and **isort** for import sorting.

```bash
# Auto-format code
black visdrone_toolkit scripts tests
isort visdrone_toolkit scripts tests

# Or use make
make format
```

### Code Style Rules

1. **Line length**: Maximum 100 characters
2. **Imports**: Organized with isort

   ```python
   # Standard library
   import os
   from pathlib import Path

   # Third-party
   import torch
   import numpy as np

   # Local
   from visdrone_toolkit import VisDroneDataset
   ```

3. **Type hints**: Use type hints for function signatures

   ```python
   def process_image(image: np.ndarray, size: int = 640) -> torch.Tensor:
       """Process image to tensor."""
       pass
   ```

4. **Docstrings**: Use Google style

   ```python
   def my_function(param1: str, param2: int) -> bool:
       """
       Short description.

       Longer description if needed.

       Args:
           param1: Description of param1
           param2: Description of param2

       Returns:
           Description of return value

       Raises:
           ValueError: When something is wrong
       """
       pass
   ```

5. **Naming conventions**:
   - Classes: `PascalCase`
   - Functions/variables: `snake_case`
   - Constants: `UPPER_CASE`
   - Private methods: `_leading_underscore`

### Linting

We use **flake8** and **mypy** for linting:

```bash
# Check code
flake8 visdrone_toolkit scripts tests
mypy visdrone_toolkit scripts

# Or use make
make lint
```

## Testing Guidelines

### Writing Tests

1. **Location**: Put tests in `tests/` directory
2. **Naming**: Test files should start with `test_`
3. **Structure**: Organize tests in classes

   ```python
   class TestMyFeature:
       """Tests for my feature."""

       def test_basic_functionality(self):
           """Test basic case."""
           assert my_function() == expected_result

       def test_edge_case(self):
           """Test edge case."""
           with pytest.raises(ValueError):
               my_function(invalid_input)
   ```

4. **Fixtures**: Use pytest fixtures from `conftest.py`

   ```python
   def test_with_dataset(self, mock_visdrone_dataset):
       """Test using fixture."""
       dataset = VisDroneDataset(
           image_dir=str(mock_visdrone_dataset['image_dir']),
           annotation_dir=str(mock_visdrone_dataset['annotation_dir']),
       )
       assert len(dataset) > 0
   ```

### Running Tests

```bash
# All tests
pytest tests/

# Specific file
pytest tests/test_dataset.py

# With coverage
pytest tests/ --cov=visdrone_toolkit --cov-report=html

# Using make
make test
```

### Test Coverage

- Aim for **>80% coverage**
- All new features must include tests
- Bug fixes should include regression tests

## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/).

### Format

```html
<type
  >(<scope
    >):
    <subject>
      <body>
        <footer></footer></body></subject></scope
></type>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```text
feat(dataset): add support for video sequences

Add VideoSequenceDataset class to handle video frames with
temporal information. This enables training on video tasks.

Closes #123
```

```text
fix(converter): handle empty annotation files

Previously crashed when annotation file was empty.
Now returns empty annotations gracefully.

Fixes #456
```

```text
docs(readme): update installation instructions

Add section on CUDA version compatibility and
troubleshooting common installation issues.
```

### Subject Line Rules

- Use imperative mood ("add" not "added" or "adds")
- Don't capitalize first letter
- No period at the end
- Maximum 50 characters

## Documentation Guidelines

### Docstrings

Use Google style docstrings:

```python
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 10,
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """
    Train object detection model.

    This function handles the complete training loop including
    forward pass, loss computation, and backpropagation.

    Args:
        model: PyTorch model to train
        dataloader: Training data loader
        epochs: Number of training epochs (default: 10)
        device: Device to use for training (default: "cuda")

    Returns:
        Dictionary containing training metrics with keys:
            - 'train_loss': List of loss values per epoch
            - 'val_loss': List of validation loss values

    Raises:
        ValueError: If epochs < 1
        RuntimeError: If CUDA requested but not available

    Example:
        >>> model = get_model("fasterrcnn_resnet50")
        >>> metrics = train_model(model, train_loader, epochs=50)
        >>> print(f"Final loss: {metrics['train_loss'][-1]}")
    """
    pass
```

### README Updates

When adding features:

- Update main README.md
- Add examples if applicable
- Update relevant documentation files

## Project Structure

```bash
VisDrone-dataset-python-toolkit/
├── visdrone_toolkit/       # Core package
│   ├── __init__.py
│   ├── dataset.py          # Dataset classes
│   ├── utils.py            # Utilities
│   ├── visualization.py    # Plotting
│   └── converters/         # Format converters
├── scripts/                # CLI tools
│   ├── train.py
│   ├── inference.py
│   ├── webcam_demo.py
│   ├── evaluate.py
│   └── convert_annotations.py
├── tests/                  # Unit tests
├── configs/                # Training configs
├── docs/                   # Documentation
├── examples/               # Examples
└── .github/                # CI/CD workflows
```

## Getting Help

- **Documentation**: Check README.md and other docs
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Contact**: Email maintainers for sensitive matters

## Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

Don't hesitate to ask! We're here to help:

- Open an issue with the "question" label
- Start a discussion on GitHub Discussions
- Contact maintainers directly

---

**Thank you for contributing to VisDrone Toolkit! 🚀**
