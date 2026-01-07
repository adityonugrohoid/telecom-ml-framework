# Contributing to {PROJECT_NAME}

Thank you for your interest in contributing! This is a portfolio project, but suggestions and improvements are welcome.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. **Check existing issues** to avoid duplicates
2. **Open a new issue** with a clear title and description
3. **Include**:
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Environment details (Python version, OS)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:

1. **Describe the use case** - why would this be valuable?
2. **Propose the solution** - what should be added/changed?
3. **Consider alternatives** - what other approaches did you consider?

### Pull Requests

PRs are welcome for:
- Bug fixes
- Documentation improvements
- Test coverage improvements
- Code quality enhancements

**Before submitting a PR:**

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Run tests**: `uv run pytest tests/ -v`
5. **Run linter**: `uv run ruff check src/ tests/`
6. **Commit with clear messages**
7. **Push and create PR**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/{REPO_NAME}.git
cd {REPO_NAME}

# Install development dependencies
uv sync --all-extras

# Install pre-commit hooks (optional)
uv run pre-commit install
```

## Code Style

This project uses:
- **Ruff** for linting and formatting
- **Type hints** where applicable
- **Docstrings** for all public functions/classes

Format your code:
```bash
uv run ruff format src/ tests/
```

Check for issues:
```bash
uv run ruff check src/ tests/
```

## Testing

Write tests for new features:

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_data_quality.py -v

# Run with coverage
uv run pytest tests/ --cov={project_name}
```

## Documentation

Update documentation when:
- Adding new features
- Changing APIs
- Fixing bugs that affect behavior

Documentation files to consider:
- `README.md` - Main documentation
- `QUICKSTART.md` - Getting started guide
- `docs/USE_CASE_SPEC.md` - Use case specifications
- Docstrings in code

## Project Philosophy

When contributing, keep in mind:

1. **Domain expertise over code complexity** - Solutions should reflect telecom domain knowledge
2. **Clarity over cleverness** - Code should be easy to understand
3. **Practical over perfect** - This is a portfolio project, not production software
4. **Minimal but complete** - Avoid scope creep

## Questions?

Feel free to:
- Open a discussion on GitHub
- Email: {YOUR_EMAIL}
- Check existing issues/PRs for context

---

Thank you for contributing! 🎉
