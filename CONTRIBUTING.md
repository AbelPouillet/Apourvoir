# Contributing to Apourvoir

Thank you for your interest in contributing to **Apourvoir**! This document provides guidelines for contributing to the project.

## 📋 Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## 🚀 How to Contribute

### 1. Reporting Bugs

- **Check existing issues** : Before creating a new issue, please check if the problem has already been reported.
- **Provide detailed information** : Include steps to reproduce, expected behavior, and actual behavior.
- **Include examples** : If possible, provide a minimal code example that demonstrates the issue.

### 2. Suggesting Enhancements

- **Open an issue** : Use GitHub issues to suggest new features or enhancements.
- **Describe the use case** : Explain why the enhancement would be valuable.
- **Provide examples** : Include examples of how the enhancement would be used.

### 3. Contributing Code

#### a. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/Apourvoir.git
   cd Apourvoir
   ```

#### b. Set Up Development Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

#### c. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use meaningful branch names:
- `feature/...` for new features
- `bugfix/...` for bug fixes
- `docs/...` for documentation improvements
- `refactor/...` for code refactoring

#### d. Write Code

- **Follow the existing style** : Use the same coding style as the rest of the project.
- **Add tests** : Include tests for new functionality.
- **Update documentation** : Update relevant documentation.
- **Keep commits atomic** : Each commit should represent a single logical change.

#### e. Test Your Changes

```bash
# Run tests
pytest tests/

# Run linting
black src/ tests/
flake8 src/ tests/

# Run type checking
mypy src/
```

#### f. Commit Your Changes

Use meaningful commit messages following the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat: add new alignment metric
fix: correct quantization error calculation
docs: update README with installation instructions
refactor: simplify matrix operations
```

#### g. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

#### h. Create a Pull Request

1. Go to the [Apourvoir repository](https://github.com/Albo-co/Apourvoir) on GitHub
2. Click "New Pull Request"
3. Select your fork and feature branch
4. Provide a clear description of your changes
5. Reference any related issues
6. Submit the pull request

## 📝 Code Style

### Python Code

- **PEP 8** : Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- **Type hints** : Use type hints for function parameters and return values
- **Docstrings** : Include docstrings for all public functions and classes
- **Line length** : Maximum line length is 88 characters
- **Imports** : Group imports (standard library, third-party, local) with blank lines between

### Example Code Style

```python
"""Module docstring."""

from typing import List, Optional
import numpy as np


class MyClass:
    """Class docstring."""

    def __init__(self, value: int):
        """Initialize the class.
        
        Args:
            value: Description of the parameter
        """
        self.value = value

    def my_method(self, x: float, y: Optional[float] = None) -> float:
        """Method docstring.
        
        Args:
            x: Description of x
            y: Description of y (optional)
            
        Returns:
            Description of return value
        """
        if y is None:
            y = 0.0
        return x + y
```

## 🧪 Testing

### Writing Tests

- Use `pytest` for testing
- Place tests in the `tests/` directory
- Mirror the package structure in the tests directory
- Use descriptive test function names

### Example Test

```python
"""Tests for the alignment module."""

import numpy as np
import pytest
from apourvoir.core.alignment import SavoirVector, SavoirSpace


def test_savoir_vector_norm():
    """Test that vector norm is calculated correctly."""
    space = SavoirSpace(3)
    vector = space.create_vector([3, 4, 0])
    
    assert np.isclose(vector.norm(), 5.0)


def test_savoir_vector_inner_product():
    """Test that inner product is calculated correctly."""
    space = SavoirSpace(2)
    v1 = space.create_vector([1, 0])
    v2 = space.create_vector([0, 1])
    
    assert v1.inner_product(v2) == 0.0
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_alignment.py

# Run specific test function
pytest tests/test_alignment.py::test_savoir_vector_norm
```

## 📚 Documentation

### Writing Documentation

- Use Markdown for documentation files
- Follow the existing documentation structure
- Include code examples where helpful
- Keep documentation up to date with code changes

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs
make html
```

## ✅ Pull Request Checklist

Before submitting a pull request, ensure that:

- [ ] Code follows the project's style guidelines
- [ ] All existing tests pass
- [ ] New functionality is tested
- [ ] Documentation is updated (if applicable)
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with the main branch

## 🎁 Recognition

All contributions are valued and recognized. Contributors will be:

- Listed in the [CONTRIBUTORS.md](CONTRIBUTORS.md) file
- Acknowledged in release notes
- Invited to become maintainers for significant contributions

## 📄 License

By contributing to Apourvoir, you agree that your contributions will be licensed under the [MIT License](LICENSE).

## 🙏 Thank You!

Thank you for contributing to Apourvoir! Your contributions help make this project better for everyone.
