# 🚀 TensorAeroSpace Developer Environment Guide

This guide helps you set up a local development environment for TensorAeroSpace.

## 📋 Prerequisites

- Python 3.10+ (3.11 or 3.12 recommended)
- Poetry for dependency management
- Git for version control
- Make for running development tasks

### Install Poetry

```bash
# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Or via pip
pip install poetry
```

## 🛠️ Quick setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/TensorAeroSpace.git
cd TensorAeroSpace
```

2. One‑command dev bootstrap:
```bash
make dev-setup
```

This will:
- Install all dependencies
- Show available commands

## 📝 Manual setup

If you prefer manual configuration:

1. Install dependencies:
```bash
poetry install --with dev,test
```

2. Activate the virtualenv:
```bash
poetry shell
```

## 🧪 Verify the setup

Run quick tests:
```bash
make test-quick
```

Or the full check suite:
```bash
make check-all
```

## 📚 Core development commands

### Testing
```bash
make test           # All tests with coverage
make test-quick     # Fast tests
make test-agents    # Agents tests
make test-envs      # Environment tests
```

### Code quality
```bash
make format         # Code formatting
make lint           # Linters
make security       # Security checks
```

### Documentation
```bash
make docs           # Build documentation
make docs-serve     # Serve docs locally
```

### Build & clean
```bash
make build          # Build the package
make clean          # Cleanup artifacts
```

### Versioning
```bash
make version        # Show current version
make bump-patch     # Increment patch version
make bump-minor     # Increment minor version
make bump-major     # Increment major version
```

## 🔧 IDE configuration

### VS Code

Recommended extensions (create `.vscode/extensions.json`):
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.isort",
    "ms-python.flake8",
    "ms-python.mypy-type-checker",
    "charliermarsh.ruff",
    "ms-python.pylint"
  ]
}
```

Settings (create `.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

### PyCharm

1. Open the project in PyCharm
2. Configure Python interpreter: Settings → Project → Python Interpreter
3. Select Poetry Environment
4. Configure formatting: Settings → Tools → External Tools

## 🐳 Docker-based development

For development in Docker:

```bash
# Build image
make docker_build

# Run in debug mode
make docker_debug
```

## 🔄 Development workflow

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make changes and run checks:
```bash
make format
make lint
make test
```

3. Commit your changes:
```bash
git add .
git commit -m "feat: add your feature description"
```

4. Push the branch:
```bash
git push origin feature/your-feature-name
```

5. Open a Pull Request via the GitHub UI.

## 🚨 Troubleshooting

### Poetry issues
```bash
# Clear Poetry cache
poetry cache clear pypi --all

# Reinstall dependencies
rm poetry.lock
poetry install
```



### Test issues
```bash
# Clear pytest cache
rm -rf .pytest_cache

# Verbose run
poetry run pytest -v --tb=long
```

## 📞 Getting help

- 📖 Project documentation: docs/
- 🐛 Report a bug: https://github.com/your-username/TensorAeroSpace/issues/new?template=bug_report.md
- ✨ Request a feature: https://github.com/your-username/TensorAeroSpace/issues/new?template=feature_request.md
- ❓ Ask a question: https://github.com/your-username/TensorAeroSpace/issues/new?template=question.md

## 🎯 Next steps

After the setup:

1. Explore the examples (example/) to learn the API
2. Read the docs (docs/) for deeper understanding
3. Run the tests (tests/) to verify your environment
4. Start building your features!

---

**Happy hacking! 🚀**