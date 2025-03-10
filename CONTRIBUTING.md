# Contributing to Mellon

Thank you for your interest in contributing to Mellon!

## Development Environment

### Setting up with micromamba or conda

We recommend using micromamba or conda for your development environment:

```bash
# Create development environment
micromamba create -n mellon_dev python=3.9 -y
micromamba activate mellon_dev

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=mellon
```

### Code Formatting

We use Black and isort for code formatting:

```bash
# Format code with Black
black mellon tests

# Sort imports with isort
isort mellon tests
```

### Type Checking

```bash
# Run static type checking with mypy
mypy mellon
```

### Linting

```bash
# Run flake8 for linting
flake8 mellon tests
```

## Building the Package

```bash
# Build wheel
python -m build --wheel

# Build source distribution
python -m build --sdist
```

## Documentation

To build the documentation:

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html
```

## Pull Request Process

1. Ensure your code passes all tests
2. Update documentation where necessary
3. Add an entry to CHANGELOG.md
4. Submit a pull request

Thank you for helping improve Mellon!