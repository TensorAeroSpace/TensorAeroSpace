# GitHub Actions Workflows

Каталог `.github/workflows/` содержит CI, release и post-merge проверки TensorAeroSpace.

## Workflows

| File | Trigger | Purpose |
| --- | --- | --- |
| `action.yml` | PR to `main`/`develop`, manual | Main PR gate: tests, quality, docs coverage, security, package build |
| `quick-check.yml` | Push to any branch | Fast feedback on pushes |
| `docs-build.yml` | PR/push to `main`/`develop`, manual | Strict MkDocs build |
| `coverage-main.yml` | Push to `main`, manual | Coveralls baseline for `main` |
| `coverage-develop.yml` | Push to `develop`, manual | Coveralls baseline for `develop` |
| `notebooks-smoke.yml` | Push to `develop`, manual | Execute key notebooks |
| `publish.yml` | Published release, manual | TestPyPI/PyPI publication from verified artifacts |

## Main PR Gates

Required checks configured in `.github/settings.yml`:
- `🏷️ Version Tag Gate`
- `✅ All Python versions passed`
- `🧱 Quality Gates`
- `📚 Documentation Coverage`
- `🔒 Security Scan`
- `🏗️ Build Package`
- `📦 Wheel installs on all Python versions`
- `🏗️ mkdocs build --strict`

`🏷️ Version Tag Gate` requires `[tool.poetry].version` in `pyproject.toml` to match the latest stable `vX.Y.Z` git tag. `🧱 Quality Gates` uses `.github/ci-baselines.json` to prevent new `flake8`, `ruff`, and `mypy` regressions while existing debt is paid down incrementally. `🔒 Security Scan` applies the same baseline approach to Bandit and `pip-audit`. `📦 Wheel installs on all Python versions` installs the built wheel and imports `tensoraerospace` on Python 3.10, 3.11, 3.12, and 3.13.

## Publication Gates

`publish.yml` builds once in `test-before-publish`, verifies the artifact, uploads it as `dist`, and publishes that exact artifact to TestPyPI or PyPI. The publish jobs do not rebuild the package and do not use `continue-on-error`.

Before upload, the workflow runs:
- release/manual PyPI version checks against git tags;
- tests;
- black/isort;
- `scripts/ci_quality_gate.py flake8 ruff mypy bandit`;
- `scripts/dependency_audit_gate.py`;
- `poetry build`;
- `twine check dist/*`;
- `scripts/package_gate.py`;
- wheel install/import checks on Python 3.10, 3.11, 3.12, and 3.13.

## Secrets

- `PYPI_TEST`: TestPyPI token.
- `PYPI_PUBLISH`: PyPI token.
- `READTHEDOCS_WEBHOOK_URL`: optional RTD webhook URL.
- `READTHEDOCS_WEBHOOK_SECRET`: optional RTD webhook secret.

## Local Reproduction

```bash
poetry install --with dev,test
poetry check --lock
poetry run python scripts/version_gate.py --check-latest-tag
poetry run python scripts/ci_quality_gate.py flake8 ruff mypy bandit
poetry run python -m pip install --quiet pip-audit
poetry run python scripts/dependency_audit_gate.py
poetry build
poetry run twine check dist/*
poetry run python scripts/package_gate.py
```
