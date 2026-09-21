# Repository maintenance scripts

Run these tools from the repository root with the development dependencies
installed. The root of this directory contains CI/CD checks and asset utilities.

## CI/CD

| Script | Purpose | Local command |
|---|---|---|
| `ci_quality_gate.py` | Compare Flake8, Ruff, mypy and Bandit findings with the checked-in baseline. | `poetry run python scripts/ci_quality_gate.py flake8 ruff mypy bandit` |
| `dependency_audit_gate.py` | Audit locked dependencies against the vulnerability baseline. | `make dependency-audit` |
| `version_gate.py` | Validate package versions against stable release tags. | `poetry run python scripts/version_gate.py --check-release-ready` |
| `package_gate.py` | Inspect built distributions before publication. | `make package-gate` |

GitHub Actions, the Makefile and pre-commit use these paths. See
[the CI/CD guide](../.github/CI_CD.md) for setup and release procedures.
`__init__.py` keeps the maintenance tools importable by their tests.

## Asset utilities

| Script | Purpose | Local command |
|---|---|---|
| `extract_f16_aero.py` | Rebuild F-16 aerodynamic `.npz` tables when the MATLAB sources change. | `poetry run python -m scripts.extract_f16_aero all` |
| `generate_damage_docs_plots.py` | Regenerate numerical damage-model figures for English and Russian documentation. | `poetry run python scripts/generate_damage_docs_plots.py` |

These commands write repository assets. Review the generated changes before
committing them.

## Tests and examples

Regression checks live in [`tests/`](../tests/). The iADP interaction fixture and
transport balance checks are maintained with their tests. One-off validation
and training runners have been removed from this directory.

Complete user-facing SDK examples belong in [`example/`](../example/) and the
[English](../docs/en/index.md) / [Russian](../docs/ru/index.md) documentation.
The installed AIDI benchmark CLI remains
`python -m tensoraerospace.scripts.benchmark_aidi`.
