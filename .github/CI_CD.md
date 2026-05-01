# CI/CD Documentation

Этот документ описывает текущие GitHub Actions gates для TensorAeroSpace.

## Workflows

### Quick Check (`quick-check.yml`)
**Триггер:** push во все ветки.

**Jobs:**
- **⚡ Quick Test (Python 3.10, Ubuntu)**: `poetry check --lock`, установка runtime/test зависимостей, `compileall`, быстрые env-тесты и import smoke.
- **📝 Code Quality Check**: `black --check`, `isort --check-only`, fatal `flake8`, baseline gate для `flake8` и `ruff`.
- **🔒 Dependency Security Check**: `pip-audit` по `poetry.lock` для групп `main,dev,test` через baseline gate.

### Main CI (`action.yml`)
**Триггеры:** pull request в `main`/`develop`, ручной запуск.

**Required gates:**
- **🏷️ Version Tag Gate**: `pyproject.toml` должен совпадать с последним stable git tag (`vX.Y.Z`).
- **✅ All Python versions passed**: матрица Python 3.10, 3.11, 3.12, 3.13.
- **🧱 Quality Gates**: `black`, `isort`, fatal `flake8`, baseline gates для `flake8`, `ruff`, `mypy`.
- **📚 Documentation Coverage**: `docstr-coverage` с порогом 70%.
- **🔒 Security Scan**: baseline gate для `bandit` с конфигом из `pyproject.toml` и baseline gate для `pip-audit`.
- **🏗️ Build Package**: `poetry build`, `twine check`, package gate по `.github/package-gate.json`.
- **📦 Wheel installs on all Python versions**: собранный wheel устанавливается через `pip install dist/*.whl` и импортируется на Python 3.10, 3.11, 3.12, 3.13.

### Docs Build (`docs-build.yml`)
**Триггеры:** pull request в `main`/`develop`, push в `main`/`develop`, ручной запуск.

Собирает документацию через `mkdocs build --strict --clean`. Любые warnings MkDocs/mkdocstrings считаются ошибкой.

### Coverage Upload (`coverage-main.yml`, `coverage-develop.yml`)
**Триггеры:** push в соответствующую ветку, ручной запуск.

Запускает тесты с coverage и отправляет baseline-отчёт в Coveralls для `main` или `develop`.
После merge в `main` workflow `coverage-main.yml` заново прогоняет тесты, загружает `coverage.xml` в Coveralls и обновляет baseline для `branch=main`, чтобы coverage badge в README показывал актуальное покрытие основной ветки.

### Notebook Smoke (`notebooks-smoke.yml`)
**Триггеры:** push в `develop`, ручной запуск.

После merge в `develop` выполняет curated-набор Jupyter notebook examples через `jupyter nbconvert --execute`. Ошибка любой ячейки валит job.

### Publishing (`publish.yml`)
**Триггеры:** published GitHub Release, ручной запуск для `testpypi` или `pypi`.

Публикация работает по схеме build-once/publish-from-artifact:
- `test-before-publish` запускает тесты, quality/security gates, `poetry build`, `twine check` и package gate.
- перед релизной публикацией `pyproject.toml` сверяется с release tag; для ручной публикации в PyPI версия сверяется с последним stable git tag.
- собранный wheel устанавливается и импортируется на Python 3.10, 3.11, 3.12, 3.13.
- `publish-testpypi` публикует уже проверенный artifact в TestPyPI.
- `publish-pypi` публикует уже проверенный artifact в PyPI.
- `continue-on-error` для публикации не используется.

## Baseline Gates

Текущий технический долг зафиксирован в `.github/ci-baselines.json`. Gate проходит только если число findings не больше baseline:
- `flake8.max_total`
- `ruff.max_total`
- `mypy.max_total`
- `bandit.max_total`, `bandit.max_medium`, `bandit.max_high`

Новые PR не должны увеличивать baseline. Уменьшать значения можно отдельными PR после исправления долга.

Текущие dependency vulnerabilities зафиксированы в `.github/pip-audit-baseline.json` как точные `package + version + advisory id`. Gate по умолчанию строит временный pinned requirements из `poetry.lock` для групп `main,dev,test` и запускает `pip-audit --no-deps --disable-pip`, поэтому результат не зависит от случайного состояния локальной `.venv`. Если запись исчезла после обновления зависимости, gate показывает её как resolved и продолжает проходить.

Package thresholds находятся в `.github/package-gate.json`:
- количество wheel/sdist artifacts;
- максимальный размер одного artifact;
- общий размер `dist`;
- uncompressed wheel size;
- максимальный размер одного файла внутри пакета.

## Version Gate

Версионирование идёт через git tags. `scripts/version_gate.py` читает `[tool.poetry].version` из `pyproject.toml` и сравнивает его с последним stable semver tag вида `vX.Y.Z`.

Для текущего состояния репозитория latest stable tag: `v0.3.13`, поэтому `pyproject.toml` должен содержать `version = "0.3.13"`.

Перед публикацией по GitHub Release workflow проверяет точное совпадение release tag и `pyproject.toml`, а не меняет версию на лету.

## Local Commands

Все команды выполняются через Poetry:

```bash
poetry install --with dev,test
poetry check --lock
make version-check
make format-check
make lint
make security
make dependency-audit
make package-gate
```

Полезные прямые команды:

```bash
poetry run python scripts/ci_quality_gate.py flake8 ruff mypy bandit
poetry run python scripts/version_gate.py --check-latest-tag
poetry run python scripts/version_gate.py --sync-latest-tag
poetry run python -m pip install --quiet "pip-audit>=2.10,<3"
poetry run python scripts/dependency_audit_gate.py
poetry build
poetry run twine check dist/*
poetry run python scripts/package_gate.py
```

## Branch Protection

`.github/settings.yml` должен требовать для `main` и `develop` следующие PR checks:
- `🏷️ Version Tag Gate`
- `✅ All Python versions passed`
- `🧱 Quality Gates`
- `📚 Documentation Coverage`
- `🔒 Security Scan`
- `🏗️ Build Package`
- `📦 Wheel installs on all Python versions`
- `🏗️ mkdocs build --strict`

Не добавляйте push-only workflows в required PR checks, иначе PR может зависнуть в ожидании статуса, который не создаётся.

## Required Secrets

- `PYPI_TEST`: token для TestPyPI.
- `PYPI_PUBLISH`: token для PyPI.
- `READTHEDOCS_WEBHOOK_URL` и `READTHEDOCS_WEBHOOK_SECRET`: опционально, для Read the Docs webhook.
