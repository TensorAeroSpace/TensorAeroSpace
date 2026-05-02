# GitHub Actions Workflows

Каталог `.github/workflows/` содержит CI, release и post-merge проверки TensorAeroSpace.

## Workflows

| File | Trigger | Purpose |
| --- | --- | --- |
| `action.yml` | PR to `main`/`develop`, manual | Main PR gate: tests, quality, docs coverage, security, package build |
| `quick-check.yml` | Push to any branch | Fast feedback on pushes |
| `docs-build.yml` | PR/push to `main`/`develop`, manual | Strict MkDocs build |
| `coverage-main.yml` | Push to `main`, manual | Post-merge tests + Coveralls baseline for the `main` badge |
| `coverage-develop.yml` | Push to `develop`, manual | Coveralls baseline for `develop` |
| `notebooks-smoke.yml` | Push to `develop`, manual | Post-merge execution of key notebooks on `develop` |
| `docker-image.yml` | PR to `main`/`develop`, push to `main`/`develop`, manual | Dockerfile build/smoke; publish merged branch images to GHCR |
| `publish.yml` | Published release, manual | TestPyPI/PyPI publication from verified artifacts |

## Post-Merge Gates

After a merge to `develop`, `notebooks-smoke.yml` executes the curated Jupyter notebook matrix with `jupyter nbconvert --execute`; any cell error fails the workflow.

After a merge to `main`, `coverage-main.yml` runs tests with coverage and uploads `coverage.xml` to Coveralls. This keeps the README/Coveralls `branch=main` coverage badge aligned with the current main branch.

After a merge to `develop`, `docker-image.yml` builds the Dockerfile, smoke-tests the resulting image, and publishes `ghcr.io/tensoraerospace/tensoraerospace:develop` plus an immutable `sha-*` tag.

After a merge to `main`, `docker-image.yml` publishes `ghcr.io/tensoraerospace/tensoraerospace:main`, `ghcr.io/tensoraerospace/tensoraerospace:latest`, and an immutable `sha-*` tag.

## Main PR Gates

Required checks configured in `.github/settings.yml`:
- `🏷️ Version Tag Gate`
- `✅ All Python versions passed`
- `🧱 Quality Gates`
- `📚 Documentation Coverage`
- `🔒 Security Scan`
- `🏗️ Build Package`
- `📦 Wheel installs on all Python versions`
- `🐳 Docker image builds`
- `🏗️ mkdocs build --strict`

`🏷️ Version Tag Gate` requires `[tool.poetry].version` in `pyproject.toml` to match the latest stable `vX.Y.Z` git tag. `🧱 Quality Gates` uses `.github/ci-baselines.json` to prevent new `flake8`, `ruff`, and `mypy` regressions while existing debt is paid down incrementally. `🔒 Security Scan` applies the same baseline approach to Bandit and runs `pip-audit` against pinned packages from `poetry.lock` for the `main,dev,test` groups. `📦 Wheel installs on all Python versions` installs the built wheel and imports `tensoraerospace` on Python 3.10, 3.11, 3.12, and 3.13.

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

## Docker Image Gates

`docker-image.yml` builds the repository `Dockerfile` from source and verifies that the final runtime image imports `tensoraerospace` from `site-packages`, contains `/workspace/examples`, contains `/workspace/projects`, and does not contain `/workspace/tensoraerospace`. It also runs a lightweight example smoke script from the image.

GHCR publication uses the built image from the same job and the default `GITHUB_TOKEN` with `packages: write`; no custom registry secret is required.

Published tags:
- `develop` and `sha-<short-sha>` after push/merge to `develop`;
- `main`, `latest`, and `sha-<short-sha>` after push/merge to `main`.

## Secrets

- `PYPI_TEST`: TestPyPI token.
- `PYPI_PUBLISH`: PyPI token.
- `READTHEDOCS_WEBHOOK_URL`: optional RTD webhook URL.
- `READTHEDOCS_WEBHOOK_SECRET`: optional RTD webhook secret.
- GHCR publishing uses `GITHUB_TOKEN`; no extra secret is required.

## Local Reproduction

```bash
poetry install --with dev,test
poetry check --lock
poetry run python scripts/version_gate.py --check-latest-tag
poetry run python scripts/ci_quality_gate.py flake8 ruff mypy bandit
poetry run python -m pip install --quiet "pip-audit>=2.10,<3"
poetry run python scripts/dependency_audit_gate.py
poetry build
poetry run twine check dist/*
poetry run python scripts/package_gate.py
docker build --platform=linux/amd64 -t tensoraerospace:local .
docker run --rm tensoraerospace:local python -c "import tensoraerospace"
```
