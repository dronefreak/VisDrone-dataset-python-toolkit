.PHONY: help install install-dev test lint format clean setup-venv pre-commit

# Default target
help:
	@echo "VisDrone Toolkit - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup-venv     - Create virtual environment"
	@echo "  make install        - Install package in virtualenv"
	@echo "  make install-dev    - Install package with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run tests with pytest"
	@echo "  make lint           - Run linters (flake8, mypy)"
	@echo "  make format         - Format code (black, isort)"
	@echo "  make format-check   - Check code formatting without changes"
	@echo ""
	@echo "Pre-commit Hooks:"
	@echo "  make pre-commit-install  - Install pre-commit hooks"
	@echo "  make pre-commit-run      - Run pre-commit on all files"
	@echo "  make pre-commit-update   - Update pre-commit hook versions"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean          - Remove build artifacts and cache"
	@echo "  make clean-all      - Remove everything including venv"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup-venv     - Create virtual environment"
	@echo "  source venv/bin/activate"
	@echo "  make install-dev    - Install with dev dependencies"
	@echo "  make pre-commit-install  - Setup pre-commit hooks"

# Create virtual environment
setup-venv:
	@echo "Creating virtual environment..."
	python3 -m venv venv
	@echo "✓ Virtual environment created in ./venv"
	@echo ""
	@echo "Activate it with:"
	@echo "  source venv/bin/activate  # Linux/Mac"
	@echo "  venv\\Scripts\\activate     # Windows"

# Install package
install:
	pip install --upgrade pip
	pip install -e .
	@echo "✓ Package installed successfully"

# Install with development dependencies
install-dev:
	pip install --upgrade pip
	pip install -e ".[dev]"
	@echo "✓ Package installed with dev dependencies"
	@echo ""
	@echo "Setup pre-commit hooks with:"
	@echo "  pre-commit install"

# Run tests
test:
	pytest tests/ -v --cov=visdrone_toolkit --cov-report=term-missing

# Run linters
lint:
	@echo "Running flake8..."
	flake8 visdrone_toolkit scripts tests
	@echo "Running mypy..."
	mypy visdrone_toolkit scripts

# Format code
format:
	@echo "Formatting with black..."
	black visdrone_toolkit scripts tests
	@echo "Sorting imports with isort..."
	isort visdrone_toolkit scripts tests
	@echo "✓ Code formatted"

# Check formatting without making changes
format-check:
	black --check visdrone_toolkit scripts tests
	isort --check-only visdrone_toolkit scripts tests

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .eggs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	@echo "✓ Cleaned"

# Clean everything including virtualenv
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf venv/
	@echo "✓ Everything cleaned"

# Build distribution packages
build:
	python -m pip install --upgrade build
	python -m build
	@echo "✓ Distribution packages built in dist/"

# Upload to PyPI (test)
upload-test: build
	python -m pip install --upgrade twine
	python -m twine upload --repository testpypi dist/*

# Upload to PyPI (production)
upload: build
	python -m pip install --upgrade twine
	python -m twine upload dist/*

# Pre-commit hooks
pre-commit-install:
	@echo "Installing pre-commit hooks..."
	pip install pre-commit
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "✓ Pre-commit hooks installed"
	@echo ""
	@echo "Hooks will now run automatically on git commit"
	@echo "Run 'make pre-commit-run' to check all files now"

pre-commit-run:
	@echo "Running pre-commit on all files..."
	pre-commit run --all-files

pre-commit-update:
	@echo "Updating pre-commit hooks to latest versions..."
	pre-commit autoupdate
	@echo "✓ Pre-commit hooks updated"

pre-commit-uninstall:
	@echo "Uninstalling pre-commit hooks..."
	pre-commit uninstall
	pre-commit uninstall --hook-type commit-msg
	@echo "✓ Pre-commit hooks uninstalled"
