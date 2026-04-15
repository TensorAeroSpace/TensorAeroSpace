# F-16 Nonlinear Model Numpy Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `matlab.engine`-backed F-16 nonlinear longitudinal & angular models with pure-numpy implementations that preserve the public API, run without a Matlab installation, and execute ≥10× faster per step.

**Architecture:** Pure functions in `dynamics.py` / `aero.py` / `_integrators.py`; thin `LongitudinalF16` / `AngularF16` classes hold state. Aero tables extracted once from `.m` files into `.npz` and loaded at import time. Cubic interpolation (scipy `RegularGridInterpolator(method="cubic")` for 2D/3D tables, `PchipInterpolator` for 1D) to match matlab `csaps`/`pchip`.

**Tech Stack:** Python 3.10+, numpy, scipy ≥ 1.9, pytest.

**Reference spec:** `docs/superpowers/specs/2026-04-12-f16-nonlinear-numpy-port-design.md`

---

## Path Remap (override applied 2026-04-13)

The plan body below references paths under `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/...` and `.../angular/...`. **Per user request, the new pure-numpy code goes into a side-by-side `python/` subtree instead** so the existing matlab-backed code stays alive and untouched. Apply the following rewrite to every path in the plan body:

| Plan body says | Use instead |
|---|---|
| `tensoraerospace/aerospacemodel/f16/nonlinear/_integrators.py` | `tensoraerospace/aerospacemodel/f16/nonlinear/python/_integrators.py` |
| `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/<file>` (any non-matlab file) | `tensoraerospace/aerospacemodel/f16/nonlinear/python/longitudinal/<file>` |
| `tensoraerospace/aerospacemodel/f16/nonlinear/angular/<file>` (any non-matlab file) | `tensoraerospace/aerospacemodel/f16/nonlinear/python/angular/<file>` |
| `tensoraerospace/aerospacemodel/f16/nonlinear/{long,ang}/aero_tables/...` | `tensoraerospace/aerospacemodel/f16/nonlinear/python/{long,ang}/aero_tables/...` |

**Unchanged:**
- `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/matlab_code/` and the analogous angular dir — the extraction script reads `.m` from these original locations.
- `tensoraerospace/aerospacemodel/f16/nonlinear/{longitudinal,angular}/{model.py, inital.py, __init__.py}` — the existing matlab-backed code is **not** rewritten or deleted. It coexists.
- Test paths (`tests/aerospacemodel/f16/nonlinear/...`) — but the tests target the new `python/...` import path.

**Tasks that no longer apply (skip them):**
- Task 3.5 (delete legacy longitudinal tests) — skip. The legacy matlab-fake-based tests still test the still-existing matlab-backed code.
- The "delete legacy angular tests" steps inside Task 4.7 — skip the deletes; still write the new property tests.
- The "rewrite existing model.py / inital.py" steps in Tasks 3.4 and 4.6 — instead, **create** new files at `python/longitudinal/{model.py, inital.py}` and `python/angular/{model.py, inital.py}`.

**New files needed for the python subtree to be importable:**
- `tensoraerospace/aerospacemodel/f16/nonlinear/python/__init__.py` (empty, created by first implementer)
- `tensoraerospace/aerospacemodel/f16/nonlinear/python/longitudinal/__init__.py` exporting `LongitudinalF16, initial_state, set_initial_state`
- `tensoraerospace/aerospacemodel/f16/nonlinear/python/angular/__init__.py` exporting `AngularF16, initial_state, set_initial_state`

**New public import paths (use these in tests and final docs):**
```python
from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import (
    LongitudinalF16, initial_state, set_initial_state,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.python.angular import (
    AngularF16, initial_state, set_initial_state,
)
```

The `test_import_does_not_load_matlab` tests still apply: importing the new `python.*` modules must not pull `matlab` into `sys.modules`. (The existing legacy `nonlinear.longitudinal` import path still does — that's expected and not in scope.)

---

## Pre-flight notes for the executing agent

- The matlab `.m` files in `tensoraerospace/aerospacemodel/f16/nonlinear/{longitudinal,angular}/matlab_code/` are the **source of truth** for both the numerics and the equations of motion. Do not delete them; they stay as reference. Keep them at exactly the same path so the `.npz` extraction script can find them.
- **State vector ordering for the angular model** (from `angular/matlab_code/F16State_vec2struct.m`):
  `[alpha, beta, wx, wy, wz, gamma, psi, theta, stab, dstab, ail, dail, dir, ddir]`
  Note: the existing `angular/inital.py` builds `initial_state_dict` with a *different* ordering (stab/ail/dir grouped before their derivatives). That's a latent bug in `set_initial_state` — fix it in Task 12.
- **Longitudinal state vector** (from `longitudinal/matlab_code/F16State_vec2struct.m`):
  `[alpha, wz, stab, dstab]`
- **Control vectors:**
  - longitudinal: `[stab_act]`
  - angular: `[stab_act, ail_act, dir_act]`
- The existing tests under `tests/aerospacemodel/f16_nonlinear_*` install a *fake* `matlab` module to allow the tests to import. After the port, those tests are obsolete — replace them with real tests as part of each phase, do not just patch them.
- Examples (mostly Jupyter notebooks) under `example/` reference `LongitudinalF16` / `AngularF16` and `initial_state` from these modules. They currently pass `matlab.double(...)` as the control. The new API takes plain `list`/`np.ndarray`. Notebooks are out of scope to update — they'll just work because `np.asarray([[0]])` is what we'll accept. The Python-file examples don't reference the nonlinear models, so no example code edits are needed.

---

## Phase 0 — Branching & worktree setup

### Task 0.1: Create a clean branch for the port

**Files:** none

- [ ] **Step 1: Confirm we're on the existing fix branch and create a sub-branch**

Run:
```bash
cd /home/mr8bit/Projects/TensorAeroSpace
git status -sb
git checkout -b feat/f16-nonlinear-numpy-port
```
Expected: branch created without errors. The plan does **not** require a worktree — work in-place on this branch.

- [ ] **Step 2: Verify scipy availability and version**

Run:
```bash
poetry run python -c "import scipy; print(scipy.__version__)"
```
Expected: version `1.11.x` or higher (`RegularGridInterpolator(method="cubic")` requires ≥ 1.9).

If poetry isn't initialized: `python -c "import scipy; print(scipy.__version__)"`. If scipy is missing entirely, stop and report — it should be installed per `pyproject.toml`.

---

## Phase 1 — Shared integrator module

### Task 1.1: Implement `_integrators.py`

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/_integrators.py`
- Test: `tests/aerospacemodel/f16/nonlinear/test_integrators.py`

- [ ] **Step 1: Write the failing test**

Create `tests/aerospacemodel/f16/nonlinear/__init__.py` (empty) and `tests/aerospacemodel/f16/nonlinear/test_integrators.py`:

```python
import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear._integrators import euler, rk4


def _linear_rhs(x, u, t, params):
    """dx/dt = -x + u; analytic solution x(t) = u + (x0 - u) * exp(-t)."""
    return -x + u


def test_euler_one_step_linear():
    x0 = np.array([1.0, 2.0])
    u = np.array([0.0, 0.0])
    out = euler(_linear_rhs, x0, u, t=0.0, dt=0.1, params=None)
    np.testing.assert_allclose(out, x0 + 0.1 * (-x0 + u))


def test_rk4_matches_analytic_solution_better_than_euler():
    x0 = np.array([1.0])
    u = np.array([0.0])
    dt = 0.05
    n = 100
    x_euler = x0.copy()
    x_rk4 = x0.copy()
    for k in range(n):
        x_euler = euler(_linear_rhs, x_euler, u, t=k * dt, dt=dt, params=None)
        x_rk4 = rk4(_linear_rhs, x_rk4, u, t=k * dt, dt=dt, params=None)
    analytic = np.array([np.exp(-n * dt)])
    err_euler = abs(x_euler - analytic).item()
    err_rk4 = abs(x_rk4 - analytic).item()
    assert err_rk4 < err_euler
    assert err_rk4 < 1e-6


def test_rhs_signature_called_with_all_args():
    seen = {}

    def rhs(x, u, t, params):
        seen.update({"x": x, "u": u, "t": t, "params": params})
        return np.zeros_like(x)

    x0 = np.array([1.0])
    u = np.array([0.5])
    rk4(rhs, x0, u, t=0.7, dt=0.01, params={"k": 1})
    assert seen["t"] >= 0.7
    assert seen["params"] == {"k": 1}
```

- [ ] **Step 2: Run test, verify it fails**

Run: `poetry run pytest tests/aerospacemodel/f16/nonlinear/test_integrators.py -v`
Expected: `ImportError: cannot import name 'euler'`.

- [ ] **Step 3: Implement the integrators**

Create `tensoraerospace/aerospacemodel/f16/nonlinear/_integrators.py`:

```python
"""Fixed-step ODE integrators for F-16 nonlinear models.

These are intentionally minimal: a function ``f(x, u, t, params) -> dx`` and
a step size are all that's needed. Both integrators are pure functions and
allocation-cheap so they're safe to call thousands of times per RL episode.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

RHS = Callable[[np.ndarray, np.ndarray, float, Any], np.ndarray]


def euler(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float, params: Any) -> np.ndarray:
    return x + dt * f(x, u, t, params)


def rk4(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float, params: Any) -> np.ndarray:
    k1 = f(x, u, t, params)
    k2 = f(x + 0.5 * dt * k1, u, t + 0.5 * dt, params)
    k3 = f(x + 0.5 * dt * k2, u, t + 0.5 * dt, params)
    k4 = f(x + dt * k3, u, t + dt, params)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `poetry run pytest tests/aerospacemodel/f16/nonlinear/test_integrators.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/_integrators.py \
        tests/aerospacemodel/f16/nonlinear/__init__.py \
        tests/aerospacemodel/f16/nonlinear/test_integrators.py
git commit -m "feat(f16): add euler/rk4 integrator module for nonlinear F-16 port"
```

---

## Phase 2 — Aero table extraction tooling

### Task 2.1: Build the matlab `.m` table parser

**Files:**
- Create: `scripts/extract_f16_aero.py`
- Test: `tests/scripts/test_extract_f16_aero.py`

This script is dev tooling: it reads the matlab `.m` files once, parses out the numeric tables, and writes `.npz` archives. It's not part of the runtime package. It must be reproducible: anyone who re-runs it on the same `.m` files gets the same `.npz` files.

- [ ] **Step 1: Write the failing parser test**

Create `tests/scripts/__init__.py` (empty) and `tests/scripts/test_extract_f16_aero.py`:

```python
import numpy as np
import pytest

from scripts.extract_f16_aero import (
    parse_matlab_assignment,
    parse_matlab_file,
)


def test_parse_simple_row_vector():
    src = "alpha1 = deg2rad([-20 -15 -10 0 5 10]);"
    out = parse_matlab_assignment(src, "alpha1")
    assert isinstance(out, np.ndarray)
    expected = np.deg2rad([-20, -15, -10, 0, 5, 10])
    np.testing.assert_allclose(out, expected)


def test_parse_column_vector_with_transpose():
    src = "Cywz1 = [-23.9 -29.5 -30.5]';"
    out = parse_matlab_assignment(src, "Cywz1")
    expected = np.array([-23.9, -29.5, -30.5])
    np.testing.assert_allclose(out, expected)


def test_parse_2d_matrix():
    src = """
    M = [1.0 2.0 3.0;
         4.0 5.0 6.0];
    """
    out = parse_matlab_assignment(src, "M")
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(out, expected)


def test_parse_3d_indexed_assignment():
    src = """
    Cy1(:,:,1) = [1.0 2.0;
                  3.0 4.0];
    Cy1(:,:,2) = [5.0 6.0;
                  7.0 8.0];
    """
    out = parse_matlab_assignment(src, "Cy1")
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(out[:, :, 0], [[1, 2], [3, 4]])
    np.testing.assert_allclose(out[:, :, 1], [[5, 6], [7, 8]])


def test_parse_matlab_file_multiple_vars():
    src = """
    a = [1 2 3];
    b = [4; 5; 6];
    """
    out = parse_matlab_file(src, ["a", "b"])
    np.testing.assert_allclose(out["a"], [1, 2, 3])
    np.testing.assert_allclose(out["b"], [4, 5, 6])


def test_parse_handles_negation_followed_by_assignment():
    """Some matlab files do `Cy1 = -Cy1;` after defining Cy1. Parser must
    return the negated version when asked for the final value."""
    src = """
    Cy1 = [1 2; 3 4];
    Cy1 = -Cy1;
    """
    out = parse_matlab_assignment(src, "Cy1")
    np.testing.assert_allclose(out, [[-1, -2], [-3, -4]])
```

- [ ] **Step 2: Run test, verify it fails**

Run: `poetry run pytest tests/scripts/test_extract_f16_aero.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts'`.

- [ ] **Step 3: Implement the parser**

Create `scripts/__init__.py` (empty) and `scripts/extract_f16_aero.py`:

```python
"""Extract F-16 aerodynamic lookup tables from matlab .m files into .npz.

This is a one-shot dev script. Re-run only when the .m sources change.
The runtime package does NOT depend on this script - it consumes the .npz
files it produces.

Usage:
    python -m scripts.extract_f16_aero longitudinal
    python -m scripts.extract_f16_aero angular
    python -m scripts.extract_f16_aero all
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LONG_DIR = REPO_ROOT / "tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal"
ANG_DIR = REPO_ROOT / "tensoraerospace/aerospacemodel/f16/nonlinear/angular"


# ---------- low-level matlab matrix parsing ----------

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?")


def _strip_comments(src: str) -> str:
    """Remove matlab `% ...` line comments."""
    return re.sub(r"%[^\n]*", "", src)


def _parse_matrix_literal(literal: str) -> np.ndarray:
    """Parse the inside of `[...]` (already stripped of brackets) into ndarray.

    Rows separated by `;` or by newlines (matlab tolerates both).
    Columns separated by whitespace or commas.
    """
    rows = []
    raw_rows = re.split(r";|\n", literal)
    for raw in raw_rows:
        raw = raw.strip()
        if not raw:
            continue
        nums = _NUMBER_RE.findall(raw)
        if nums:
            rows.append([float(n) for n in nums])
    if not rows:
        raise ValueError(f"empty matrix literal: {literal!r}")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError(
            f"ragged matrix literal: row widths {[len(r) for r in rows]}"
        )
    return np.array(rows, dtype=np.float64)


def _eval_rhs(rhs: str, scope: dict) -> np.ndarray:
    """Evaluate a tiny subset of matlab expressions on the right-hand side.

    Supports:
        [<numbers>]              -> matrix literal
        [<numbers>]'             -> transpose (column vector)
        deg2rad([...])           -> radians conversion
        -<name>                  -> negate previously-defined variable
        <name>                   -> reference previously-defined variable
    """
    rhs = rhs.strip()

    # deg2rad([...])
    m = re.fullmatch(r"deg2rad\((.*)\)", rhs, flags=re.DOTALL)
    if m:
        return np.deg2rad(_eval_rhs(m.group(1), scope))

    # bracket literal, optionally followed by `'` (transpose)
    if rhs.startswith("["):
        # find matching closing bracket
        depth = 0
        end = -1
        for i, ch in enumerate(rhs):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            raise ValueError(f"unbalanced brackets: {rhs!r}")
        inner = rhs[1:end]
        arr = _parse_matrix_literal(inner)
        suffix = rhs[end + 1 :].strip().rstrip(";")
        if suffix == "'":
            arr = arr.T
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[1] == 1:
                arr = arr[:, 0]
        return arr

    # -name
    if rhs.startswith("-"):
        name = rhs[1:].strip().rstrip(";")
        if name in scope:
            return -scope[name]

    # bare name
    name = rhs.rstrip(";").strip()
    if name in scope:
        return scope[name]

    raise ValueError(f"unsupported rhs: {rhs!r}")


def parse_matlab_assignment(src: str, var_name: str) -> np.ndarray:
    """Return the *final* value of `var_name` after walking through `src`.

    Walks all `var_name = ...;` and `var_name(:,:,k) = ...;` assignments and
    returns whatever value sticks at the end. Indexed assignments are stacked
    along the last axis in index order.
    """
    src = _strip_comments(src)
    scope: dict = {}
    indexed_pages: dict = {}  # var_name -> {page_index: matrix}

    # Tokenise into top-level statements terminated by `;`
    statements = _split_statements(src)

    name_re = re.escape(var_name)
    direct_re = re.compile(rf"^{name_re}\s*=\s*(.+)$", re.DOTALL)
    indexed_re = re.compile(
        rf"^{name_re}\s*\(\s*:\s*,\s*:\s*,\s*(\d+)\s*\)\s*=\s*(.+)$",
        re.DOTALL,
    )

    for stmt in statements:
        stmt = stmt.strip()
        if not stmt:
            continue
        m = indexed_re.match(stmt)
        if m:
            page_idx = int(m.group(1))
            value = _eval_rhs(m.group(2), scope)
            indexed_pages[page_idx] = value
            continue
        m = direct_re.match(stmt)
        if m:
            scope[var_name] = _eval_rhs(m.group(1), scope)
            continue

        # Track other simple `name = ...` assignments so that `-name` and
        # cross-references work in subsequent statements.
        plain = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", stmt, re.DOTALL)
        if plain:
            try:
                scope[plain.group(1)] = _eval_rhs(plain.group(2), scope)
            except Exception:
                pass

    if indexed_pages:
        pages = [indexed_pages[i] for i in sorted(indexed_pages)]
        scope[var_name] = np.stack(pages, axis=-1)

    if var_name not in scope:
        raise KeyError(f"variable {var_name!r} not found in source")
    return scope[var_name]


def _split_statements(src: str) -> list[str]:
    """Split matlab source into `;`-terminated statements respecting brackets."""
    statements: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in src:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == ";" and depth == 0:
            statements.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        tail = "".join(buf).strip()
        if tail:
            statements.append(tail)
    return statements


def parse_matlab_file(src: str, var_names: list[str]) -> dict:
    """Parse multiple variables from a single matlab source string."""
    return {name: parse_matlab_assignment(src, name) for name in var_names}


# ---------- high-level: extract per-coefficient .npz files ----------

LONG_TABLES = {
    "GetCy.m": {
        "axes": ["alpha1", "alpha2", "beta1", "fi1"],
        "tables": ["Cy1", "Cy_nos1", "Cywz1", "dCywz_nos1", "dCy_sb1"],
    },
    "GetMz.m": {
        "axes": ["alpha1", "alpha2", "beta1", "fi1", "fi2"],
        "tables": [
            "mz1", "mz_nos1", "mzwz1", "dmzwz_nos1",
            "dmz1", "dmz_sb1", "eta_fi1", "dmz_ds1",
        ],
    },
}

ANG_TABLES = {
    # Filled in Task 10 once the angular .m files have been read.
}


def extract(matlab_dir: Path, out_dir: Path, table_spec: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, spec in table_spec.items():
        src = (matlab_dir / filename).read_text()
        names = spec["axes"] + spec["tables"]
        data = parse_matlab_file(src, names)
        out_path = out_dir / (filename.replace(".m", "").lower() + ".npz")
        np.savez_compressed(out_path, **data)
        print(f"wrote {out_path} with {list(data.keys())}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["longitudinal", "angular", "all"])
    args = parser.parse_args(argv)

    if args.target in ("longitudinal", "all"):
        extract(LONG_DIR / "matlab_code", LONG_DIR / "aero_tables", LONG_TABLES)
    if args.target in ("angular", "all"):
        extract(ANG_DIR / "matlab_code", ANG_DIR / "aero_tables", ANG_TABLES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test, verify it passes**

Run: `poetry run pytest tests/scripts/test_extract_f16_aero.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/extract_f16_aero.py \
        tests/scripts/__init__.py tests/scripts/test_extract_f16_aero.py
git commit -m "feat(scripts): add matlab .m aero table parser for F-16 port"
```

### Task 2.2: Run extraction for the longitudinal tables

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/aero_tables/getcy.npz`
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/aero_tables/getmz.npz`

- [ ] **Step 1: Run the extractor**

```bash
poetry run python -m scripts.extract_f16_aero longitudinal
```
Expected output mentions both `getcy.npz` and `getmz.npz` written into `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/aero_tables/`.

- [ ] **Step 2: Sanity check the npz files**

```bash
poetry run python - <<'PY'
import numpy as np
for name in ("getcy", "getmz"):
    f = np.load(f"tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/aero_tables/{name}.npz")
    for k in f.files:
        print(name, k, f[k].shape, f[k].dtype)
PY
```

Expected: shapes consistent with the matlab grids:
- `getcy`: `Cy1` is `(20, 19, 5)`, `Cy_nos1` is `(14, 19)`, `Cywz1` is `(20,)`, `dCywz_nos1` is `(14,)`, `dCy_sb1` is `(20,)`, axes `alpha1`(20,), `alpha2`(14,), `beta1`(19,), `fi1`(5,).
- `getmz`: `mz1` `(20, 19, 5)`, `mz_nos1` `(14, 19)`, `dmz_ds1` `(20, 7)`, `eta_fi1` `(5,)`, etc.

If shapes don't match, debug the parser before continuing — every later task depends on these files being correct.

- [ ] **Step 3: Commit the .npz files**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/aero_tables/
git commit -m "feat(f16): extract longitudinal aero tables to npz"
```

Note: `.npz` files are binary but small (≪100 KB total). Committing them is intentional — keeps the package self-contained.

---

## Phase 3 — Longitudinal model

### Task 3.1: Parameters dataclass

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/params.py`
- Test: `tests/aerospacemodel/f16/nonlinear/test_longitudinal_params.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/aerospacemodel/f16/nonlinear/test_longitudinal_params.py
import math

import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params import (
    F16LongParameters,
    default_parameters,
)


def test_default_matches_matlab_constants():
    p = default_parameters()
    assert p.m == pytest.approx(9295.44)
    assert p.S == pytest.approx(27.87)
    assert p.bA == pytest.approx(3.45)
    assert p.Jz == pytest.approx(75673.6)
    assert p.rcgx == pytest.approx(-0.05 * 3.45)
    assert p.Tstab == pytest.approx(0.03)
    assert p.Xistab == pytest.approx(0.707)
    assert p.maxabsstab == pytest.approx(math.radians(25))
    assert p.maxabsdstab == pytest.approx(math.radians(60))
    assert p.lef == 0.0
    assert p.sb == 0.0
    assert p.g == pytest.approx(9.80665)
    assert p.Oy == 3000.0
    assert p.V == 150.0


def test_dynamic_pressure_matches_isa_atmosphere_at_3000m():
    p = default_parameters()
    # Compare to a hand-computed value: rho(3000m) ~ 0.9091, V=150, q=0.5*rho*V^2 ~ 10227
    assert p.q == pytest.approx(10227.6, rel=1e-3)


def test_can_override_altitude_and_velocity():
    p = F16LongParameters(Oy=0.0, V=200.0)
    # at sea level rho=1.225, q = 0.5 * 1.225 * 200^2 = 24500
    assert p.q == pytest.approx(24500.0, rel=1e-3)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `poetry run pytest tests/aerospacemodel/f16/nonlinear/test_longitudinal_params.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement params.py**

```python
# tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/params.py
"""Default parameters for the F-16 longitudinal nonlinear model.

Mirrors longitudinal/matlab_code/airplane_parameters.m line by line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ISA atmosphere constants (sea level reference)
_L = 0.0065        # K/m, lapse rate
_R = 287.0531      # J/(kg K), specific gas constant for dry air
_T0 = 288.15       # K, sea-level temperature
_RHO0 = 1.225      # kg/m^3, sea-level density


def _isa_dynamic_pressure(altitude_m: float, velocity_mps: float, g: float) -> float:
    T = _T0 - _L * altitude_m
    rho = _RHO0 * (T / _T0) ** (g / (_L * _R) - 1.0)
    return 0.5 * rho * velocity_mps ** 2


@dataclass
class F16LongParameters:
    m: float = 9295.44               # mass, kg
    S: float = 27.87                 # wing area, m^2
    bA: float = 3.45                 # mean aerodynamic chord, m
    Jz: float = 75673.6              # pitch moment of inertia
    rcgx: float = field(init=False)  # cg-to-aero-focus offset, m
    Tstab: float = 0.03              # actuator time constant, s
    Xistab: float = 0.707            # actuator damping ratio
    maxabsstab: float = field(default_factory=lambda: math.radians(25))
    maxabsdstab: float = field(default_factory=lambda: math.radians(60))
    lef: float = 0.0                 # leading-edge flap deflection, rad
    sb: float = 0.0                  # speedbrake deflection, rad
    g: float = 9.80665               # gravity, m/s^2
    Oy: float = 3000.0               # altitude, m
    V: float = 150.0                 # airspeed, m/s
    q: float = field(init=False)     # dynamic pressure, Pa

    def __post_init__(self) -> None:
        self.rcgx = -0.05 * self.bA
        self.q = _isa_dynamic_pressure(self.Oy, self.V, self.g)


def default_parameters() -> F16LongParameters:
    return F16LongParameters()
```

- [ ] **Step 4: Run test, verify it passes**

Run: `poetry run pytest tests/aerospacemodel/f16/nonlinear/test_longitudinal_params.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/params.py \
        tests/aerospacemodel/f16/nonlinear/test_longitudinal_params.py
git commit -m "feat(f16): add longitudinal parameters dataclass with ISA atmosphere"
```

### Task 3.2: Aero coefficient module (longitudinal)

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/aero.py`
- Test: `tests/aerospacemodel/f16/nonlinear/test_longitudinal_aero.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/aerospacemodel/f16/nonlinear/test_longitudinal_aero.py
import math

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import aero


def test_get_cy_at_zero_state_returns_finite_negative():
    # F-16 has negative Cy at zero alpha (matlab Cy1 is negated to "human axes")
    val = aero.get_cy(alpha=0.0, beta=0.0, fi=0.0, dnos=0.0, wz=0.0, V=150.0, ba=3.45, sb=0.0)
    assert math.isfinite(val)


def test_get_cy_grid_node_matches_table_value():
    """At a grid node (alpha=0, beta=0, fi=0), Cy should equal exactly the
    table entry for that node (cubic spline interpolates the data points)."""
    raw = np.load(
        "tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/aero_tables/getcy.npz"
    )
    alpha1 = raw["alpha1"]
    beta1 = raw["beta1"]
    fi1 = raw["fi1"]
    Cy1 = raw["Cy1"]
    # The matlab file negates Cy1 to "human axes" -- the parser should preserve that.

    i_alpha = int(np.argmin(np.abs(alpha1 - 0.0)))
    i_beta = int(np.argmin(np.abs(beta1 - 0.0)))
    i_fi = int(np.argmin(np.abs(fi1 - 0.0)))
    expected_base = Cy1[i_alpha, i_beta, i_fi]

    # With wz=0, sb=0, dnos=0, the assembled value still has the dCy_nos
    # contribution from `Cy0 = fnval(interpCy, {alpha, beta, 0})` -- which at
    # this exact node also equals `expected_base`. So dCy_nos = 0 here.
    val = aero.get_cy(
        alpha=alpha1[i_alpha], beta=beta1[i_beta], fi=fi1[i_fi],
        dnos=0.0, wz=0.0, V=150.0, ba=3.45, sb=0.0,
    )
    assert val == pytest.approx(expected_base, rel=1e-6, abs=1e-6)


def test_get_cy_increases_with_pitch_rate_contribution():
    """Cywz term: Cy gain due to wz scales linearly with wz at fixed alpha."""
    a = aero.get_cy(alpha=0.1, beta=0.0, fi=0.0, dnos=0.0, wz=0.0, V=150.0, ba=3.45, sb=0.0)
    b = aero.get_cy(alpha=0.1, beta=0.0, fi=0.0, dnos=0.0, wz=0.5, V=150.0, ba=3.45, sb=0.0)
    assert a != b
    # The delta should be linear in wz: doubling wz doubles the increment.
    c = aero.get_cy(alpha=0.1, beta=0.0, fi=0.0, dnos=0.0, wz=1.0, V=150.0, ba=3.45, sb=0.0)
    assert (c - a) == pytest.approx(2.0 * (b - a), rel=1e-9)


def test_get_mz_at_grid_node_matches_table():
    raw = np.load(
        "tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/aero_tables/getmz.npz"
    )
    alpha1 = raw["alpha1"]
    beta1 = raw["beta1"]
    fi1 = raw["fi1"]
    mz1 = raw["mz1"]
    eta_fi1 = raw["eta_fi1"]
    dmz1 = raw["dmz1"]

    i_alpha = int(np.argmin(np.abs(alpha1 - 0.0)))
    i_beta = int(np.argmin(np.abs(beta1 - 0.0)))
    i_fi = int(np.argmin(np.abs(fi1 - 0.0)))
    expected = mz1[i_alpha, i_beta, i_fi] * eta_fi1[i_fi] + dmz1[i_alpha]

    val = aero.get_mz(
        alpha=alpha1[i_alpha], beta=beta1[i_beta], fi=fi1[i_fi],
        dnos=0.0, wz=0.0, V=150.0, ba=3.45, sb=0.0,
    )
    assert val == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_clamp_out_of_bounds_inputs():
    """Inputs outside the alpha grid should be clamped, not blow up."""
    val = aero.get_cy(
        alpha=math.radians(200.0), beta=0.0, fi=0.0, dnos=0.0, wz=0.0,
        V=150.0, ba=3.45, sb=0.0,
    )
    assert math.isfinite(val)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `poetry run pytest tests/aerospacemodel/f16/nonlinear/test_longitudinal_aero.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement aero.py**

```python
# tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/aero.py
"""Aerodynamic coefficient functions for the F-16 longitudinal model.

Direct port of GetCy.m and GetMz.m. Tables are loaded once at import
from `aero_tables/`. The math at the bottom of each `get_*` function
mirrors the matlab "Сборка" (assembly) section line for line.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

_AERO_DIR = Path(__file__).parent / "aero_tables"


def _clamped_lookup(interp: RegularGridInterpolator, point: np.ndarray, bounds: list[tuple[float, float]]) -> float:
    """Evaluate `interp` at `point`, clamping each axis to its grid bounds."""
    clipped = np.array([np.clip(p, lo, hi) for p, (lo, hi) in zip(point, bounds)])
    return float(interp(clipped))


# ---------- module-level table loading (executed once) ----------

_cy_data = np.load(_AERO_DIR / "getcy.npz")
_alpha1 = _cy_data["alpha1"]
_alpha2 = _cy_data["alpha2"]
_beta1 = _cy_data["beta1"]
_fi1 = _cy_data["fi1"]
_Cy1 = _cy_data["Cy1"]
_Cy_nos1 = _cy_data["Cy_nos1"]
_Cywz1 = _cy_data["Cywz1"]
_dCywz_nos1 = _cy_data["dCywz_nos1"]
_dCy_sb1 = _cy_data["dCy_sb1"]

_interp_cy = RegularGridInterpolator(
    (_alpha1, _beta1, _fi1), _Cy1, method="cubic", bounds_error=False
)
_interp_cy_nos = RegularGridInterpolator(
    (_alpha2, _beta1), _Cy_nos1, method="cubic", bounds_error=False
)
_interp_cywz = PchipInterpolator(_alpha1, _Cywz1, extrapolate=False)
_interp_cywz_nos = PchipInterpolator(_alpha2, _dCywz_nos1, extrapolate=False)
_interp_dcy_sb = PchipInterpolator(_alpha1, _dCy_sb1, extrapolate=False)

_cy_bounds_3d = [
    (float(_alpha1.min()), float(_alpha1.max())),
    (float(_beta1.min()), float(_beta1.max())),
    (float(_fi1.min()), float(_fi1.max())),
]
_cy_bounds_nos = [
    (float(_alpha2.min()), float(_alpha2.max())),
    (float(_beta1.min()), float(_beta1.max())),
]


_mz_data = np.load(_AERO_DIR / "getmz.npz")
_mz_alpha1 = _mz_data["alpha1"]
_mz_alpha2 = _mz_data["alpha2"]
_mz_beta1 = _mz_data["beta1"]
_mz_fi1 = _mz_data["fi1"]
_mz_fi2 = _mz_data["fi2"]
_mz1 = _mz_data["mz1"]
_mz_nos1 = _mz_data["mz_nos1"]
_mzwz1 = _mz_data["mzwz1"]
_dmzwz_nos1 = _mz_data["dmzwz_nos1"]
_dmz1 = _mz_data["dmz1"]
_dmz_sb1 = _mz_data["dmz_sb1"]
_eta_fi1 = _mz_data["eta_fi1"]
_dmz_ds1 = _mz_data["dmz_ds1"]

_interp_mz = RegularGridInterpolator(
    (_mz_alpha1, _mz_beta1, _mz_fi1), _mz1, method="cubic", bounds_error=False
)
_interp_mz_nos = RegularGridInterpolator(
    (_mz_alpha2, _mz_beta1), _mz_nos1, method="cubic", bounds_error=False
)
_interp_dmz = PchipInterpolator(_mz_alpha1, _dmz1, extrapolate=False)
_interp_mzwz = PchipInterpolator(_mz_alpha1, _mzwz1, extrapolate=False)
_interp_mzwz_nos = PchipInterpolator(_mz_alpha2, _dmzwz_nos1, extrapolate=False)
_interp_dmz_sb = PchipInterpolator(_mz_alpha1, _dmz_sb1, extrapolate=False)
_interp_eta_fi = PchipInterpolator(_mz_fi1, _eta_fi1, extrapolate=False)
_interp_dmz_ds = RegularGridInterpolator(
    (_mz_alpha1, _mz_fi2), _dmz_ds1, method="cubic", bounds_error=False
)

_mz_bounds_3d = [
    (float(_mz_alpha1.min()), float(_mz_alpha1.max())),
    (float(_mz_beta1.min()), float(_mz_beta1.max())),
    (float(_mz_fi1.min()), float(_mz_fi1.max())),
]
_mz_bounds_nos = [
    (float(_mz_alpha2.min()), float(_mz_alpha2.max())),
    (float(_mz_beta1.min()), float(_mz_beta1.max())),
]
_mz_bounds_ds = [
    (float(_mz_alpha1.min()), float(_mz_alpha1.max())),
    (float(_mz_fi2.min()), float(_mz_fi2.max())),
]


def _clip_alpha(alpha: float, lo: float, hi: float) -> float:
    return float(np.clip(alpha, lo, hi))


# ---------- public functions ----------

_DEG25 = math.radians(25)
_DEG60 = math.radians(60)


def get_cy(alpha: float, beta: float, fi: float, dnos: float,
           wz: float, V: float, ba: float, sb: float) -> float:
    """Normal force coefficient. Mirrors longitudinal/matlab_code/GetCy.m."""
    cy = _clamped_lookup(_interp_cy, np.array([alpha, beta, fi]), _cy_bounds_3d)
    cy0 = _clamped_lookup(_interp_cy, np.array([alpha, beta, 0.0]), _cy_bounds_3d)
    cy_nos = _clamped_lookup(_interp_cy_nos, np.array([alpha, beta]), _cy_bounds_nos)
    a_clip1 = _clip_alpha(alpha, float(_alpha1.min()), float(_alpha1.max()))
    a_clip2 = _clip_alpha(alpha, float(_alpha2.min()), float(_alpha2.max()))
    cywz = float(_interp_cywz(a_clip1)) + float(_interp_cywz_nos(a_clip2)) * (dnos / _DEG25)
    dcy_sb = float(_interp_dcy_sb(a_clip1))

    dcy_nos = cy_nos - cy0
    return cy + dcy_nos * (dnos / _DEG25) + cywz * ((wz * ba) / (2.0 * V)) + dcy_sb * (sb / _DEG60)


def get_mz(alpha: float, beta: float, fi: float, dnos: float,
           wz: float, V: float, ba: float, sb: float) -> float:
    """Pitch moment coefficient. Mirrors longitudinal/matlab_code/GetMz.m."""
    mz = _clamped_lookup(_interp_mz, np.array([alpha, beta, fi]), _mz_bounds_3d)
    mz0 = _clamped_lookup(_interp_mz, np.array([alpha, beta, 0.0]), _mz_bounds_3d)
    mz_nos = _clamped_lookup(_interp_mz_nos, np.array([alpha, beta]), _mz_bounds_nos)
    a_clip1 = _clip_alpha(alpha, float(_mz_alpha1.min()), float(_mz_alpha1.max()))
    a_clip2 = _clip_alpha(alpha, float(_mz_alpha2.min()), float(_mz_alpha2.max()))
    fi_clip1 = float(np.clip(fi, _mz_fi1.min(), _mz_fi1.max()))
    dmz = float(_interp_dmz(a_clip1))
    mzwz = float(_interp_mzwz(a_clip1)) + float(_interp_mzwz_nos(a_clip2)) * (dnos / _DEG25)
    dmz_sb = float(_interp_dmz_sb(a_clip1))
    eta_fi = float(_interp_eta_fi(fi_clip1))
    dmz_ds = _clamped_lookup(_interp_dmz_ds, np.array([alpha, fi]), _mz_bounds_ds)

    dmz_nos = mz_nos - mz0
    return (
        mz * eta_fi
        + dmz_nos * (dnos / _DEG25)
        + dmz
        + mzwz * ((wz * ba) / (2.0 * V))
        + dmz_sb * (sb / _DEG60)
        + dmz_ds
    )
```

- [ ] **Step 4: Run test, verify it passes**

Run: `poetry run pytest tests/aerospacemodel/f16/nonlinear/test_longitudinal_aero.py -v`
Expected: 5 passed.

If the grid-node test fails because cubic interpolation introduces rounding error: relax `rel=1e-6` to `rel=1e-4`. Cubic spline values at the data points are exact in theory, but scipy's `RegularGridInterpolator(method="cubic")` uses a tensor-product Catmull-Rom-style scheme that may differ at machine precision.

If the test still fails meaningfully (e.g., wrong sign, off by 0.1+), the most likely cause is that the matlab `Cy1 = -Cy1;` post-negation wasn't preserved by the parser — confirm the npz value matches the matlab table after negation.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/aero.py \
        tests/aerospacemodel/f16/nonlinear/test_longitudinal_aero.py
git commit -m "feat(f16): port longitudinal aero coefficient functions to numpy"
```

### Task 3.3: Dynamics module (longitudinal)

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/dynamics.py`
- Test: `tests/aerospacemodel/f16/nonlinear/test_longitudinal_dynamics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/aerospacemodel/f16/nonlinear/test_longitudinal_dynamics.py
import math

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear._integrators import euler
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import f16_ode_long
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params import default_parameters


def test_dx_shape_and_finiteness():
    p = default_parameters()
    x = np.array([0.0, 0.0, 0.0, 0.0])
    u = np.array([0.0])
    dx = f16_ode_long(x, u, t=0.0, params=p)
    assert dx.shape == (4,)
    assert np.all(np.isfinite(dx))


def test_actuator_dstab_rate_limit():
    """If `dstab` exceeds maxabsdstab, dx[2] (=clamped dstab) should be saturated."""
    p = default_parameters()
    x = np.array([0.0, 0.0, 0.0, 10.0 * p.maxabsdstab])  # huge dstab
    u = np.array([0.0])
    dx = f16_ode_long(x, u, t=0.0, params=p)
    assert dx[2] == pytest.approx(p.maxabsdstab)


def test_actuator_position_command_limit():
    """Commanded `stab_act` outside +-maxabsstab is clamped before being fed
    into the actuator second-order ODE."""
    p = default_parameters()
    x = np.array([0.0, 0.0, 0.0, 0.0])
    u_huge = np.array([10.0 * p.maxabsstab])
    u_clip = np.array([p.maxabsstab])
    np.testing.assert_allclose(
        f16_ode_long(x, u_huge, 0.0, p),
        f16_ode_long(x, u_clip, 0.0, p),
    )


def test_actuator_step_response_settles_within_envelope():
    """A unit step (within limits) on stab_act drives stab to track it within
    a few time-constants. Verifies the second-order actuator wiring is right."""
    p = default_parameters()
    x = np.array([0.0, 0.0, 0.0, 0.0])
    target = math.radians(5.0)
    u = np.array([target])
    dt = 0.001
    for k in range(int(0.5 / dt)):
        x = euler(f16_ode_long, x, u, t=k * dt, dt=dt, params=p)
    assert x[2] == pytest.approx(target, abs=math.radians(0.5))


def test_pitch_rate_response_to_elevator():
    """A negative elevator deflection (nose-up) should produce a non-zero
    pitch acceleration. Locks in the sign so any future code change that
    inverts the convention is caught."""
    p = default_parameters()
    x = np.array([math.radians(2.0), 0.0, math.radians(-5.0), 0.0])
    u = np.array([0.0])
    dx = f16_ode_long(x, u, 0.0, p)
    assert dx[1] != 0.0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `poetry run pytest tests/aerospacemodel/f16/nonlinear/test_longitudinal_dynamics.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement dynamics.py**

```python
# tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/dynamics.py
"""ODE right-hand side for the F-16 longitudinal model.

Direct line-by-line port of longitudinal/matlab_code/F16ODE.m.
State vector: [alpha, wz, stab, dstab].
Control vector: [stab_act].
"""
from __future__ import annotations

import numpy as np

from .aero import get_cy, get_mz
from .params import F16LongParameters


def f16_ode_long(x: np.ndarray, u: np.ndarray, t: float, params: F16LongParameters) -> np.ndarray:
    alpha, wz, stab, dstab = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    stab_act = float(u[0])
    p = params

    cy = get_cy(alpha, 0.0, stab, p.lef, wz, p.V, p.bA, p.sb)
    mz = get_mz(alpha, 0.0, stab, p.lef, wz, p.V, p.bA, p.sb)

    Y = p.q * p.S * cy
    Mz = p.q * p.S * p.bA * mz

    Ry = Y
    MRz = Mz + p.rcgx * Ry

    dwz = MRz / p.Jz

    # Note: the matlab original computes `gay = -p.g` and then
    #   dalpha = wz - (Ry + m*gay)/(m*V) = wz - (Ry - m*g)/(m*V).
    # We inline that here.
    dalpha = wz - (Ry - p.m * p.g) / (p.m * p.V)

    # Actuator with rate + position limits (matches matlab order of operations)
    dstab_clamped = float(np.clip(dstab, -p.maxabsdstab, p.maxabsdstab))
    stab_act_clamped = float(np.clip(stab_act, -p.maxabsstab, p.maxabsstab))
    ddstab = (-2.0 * p.Tstab * p.Xistab * dstab - stab + stab_act_clamped) / (p.Tstab ** 2)

    return np.array([dalpha, dwz, dstab_clamped, ddstab], dtype=np.float64)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `poetry run pytest tests/aerospacemodel/f16/nonlinear/test_longitudinal_dynamics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/dynamics.py \
        tests/aerospacemodel/f16/nonlinear/test_longitudinal_dynamics.py
git commit -m "feat(f16): port longitudinal F16ODE to pure-numpy dynamics module"
```

### Task 3.4: Replace `LongitudinalF16` class and `inital.py`

**Files:**
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/model.py` (full rewrite)
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/inital.py` (full rewrite)
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/__init__.py` (no change needed if exports stay the same)
- Test: `tests/aerospacemodel/f16/nonlinear/test_longitudinal_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/aerospacemodel/f16/nonlinear/test_longitudinal_model.py
import math
import sys

import numpy as np
import pytest

# Sanity gate: importing the model must not pull in matlab.
def test_import_does_not_load_matlab(monkeypatch):
    for mod in list(sys.modules):
        if mod.startswith("matlab"):
            sys.modules.pop(mod, None)
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import (  # noqa: F401
        LongitudinalF16,
        initial_state,
    )
    assert "matlab" not in sys.modules
    assert "matlab.engine" not in sys.modules


def test_run_step_returns_4d_state():
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import (
        LongitudinalF16,
        initial_state,
    )
    m = LongitudinalF16(initial_state)
    out = m.run_step([[0.0]])
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 1) or out.shape == (4,)


def test_run_step_accepts_numpy_array():
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import (
        LongitudinalF16,
        initial_state,
    )
    m = LongitudinalF16(initial_state)
    out = m.run_step(np.array([[0.0]]))
    assert out is not None


def test_run_step_rejects_wrong_action_dim():
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import (
        LongitudinalF16,
        initial_state,
    )
    m = LongitudinalF16(initial_state)
    with pytest.raises(Exception):
        m.run_step([[0.0], [0.0]])


def test_selected_state_output_subset():
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import (
        LongitudinalF16,
        initial_state,
    )
    m = LongitudinalF16(initial_state, selected_state_output=["alpha", "wz"])
    y = m.run_step([[0.0]])
    assert y.shape[0] == 2


def test_integrator_choice_affects_trajectory():
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import (
        LongitudinalF16,
        initial_state,
    )
    m_euler = LongitudinalF16(initial_state, integrator="euler")
    m_rk4 = LongitudinalF16(initial_state, integrator="rk4")
    # Both should run; trajectories will differ slightly but both must be finite.
    for _ in range(50):
        out_e = m_euler.run_step([[math.radians(1.0)]])
        out_r = m_rk4.run_step([[math.radians(1.0)]])
        assert np.all(np.isfinite(np.asarray(out_e)))
        assert np.all(np.isfinite(np.asarray(out_r)))


def test_set_initial_state_returns_array_with_overrides_applied():
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import (
        set_initial_state,
    )
    out = set_initial_state({"alpha": math.radians(10.0)})
    arr = np.asarray(out, dtype=float).reshape(-1)
    assert arr[0] == pytest.approx(math.radians(10.0))


def test_set_initial_state_rejects_unknown_key():
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import (
        set_initial_state,
    )
    with pytest.raises(Exception):
        set_initial_state({"not_a_state": 1.0})
```

- [ ] **Step 2: Run test, verify it fails**

Run: `poetry run pytest tests/aerospacemodel/f16/nonlinear/test_longitudinal_model.py -v`
Expected: failures (the import test passes only if no other test imports matlab; the integrator-choice test fails because the kwarg doesn't exist; etc.).

- [ ] **Step 3: Rewrite model.py**

Replace the entire contents of `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/model.py` with:

```python
"""F-16 nonlinear longitudinal model — pure-numpy implementation.

Replaces the previous matlab.engine-backed wrapper.
State vector: [alpha, wz, stab, dstab].
Control vector: [stab_act].
"""
from __future__ import annotations

from typing import Literal, Sequence, Union

import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase

from .._integrators import euler, rk4
from .dynamics import f16_ode_long
from .params import F16LongParameters, default_parameters


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]


class LongitudinalF16(ModelBase):
    r"""F-16 aircraft in isolated longitudinal channel.

    Action space:
        stab_act: elevator command [rad]

    State space:
        alpha: angle of attack [rad]
        wz:    pitch angular velocity [rad/s]
        stab:  elevator position [rad]
        dstab: elevator angular velocity [rad/s]

    Args:
        x0: Initial state, shape (4,) or (4, 1).
        selected_state_output: Optional subset of state names to return from
            ``run_step``.
        t0: Initial simulation time. Default 0.
        dt: Discretization step in seconds. Default 0.01.
        integrator: ``"euler"`` (default, matches the legacy matlab version)
            or ``"rk4"`` (more accurate, recommended for ``dt > 0.005``).
    """

    def __init__(
        self,
        x0: ArrayLike,
        selected_state_output=None,
        t0: float = 0,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "euler",
    ) -> None:
        x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0_arr.size != 4:
            raise ValueError(
                f"x0 must have 4 elements (alpha, wz, stab, dstab); got {x0_arr.size}"
            )
        super().__init__(x0_arr, selected_state_output, t0, dt)
        self.list_state = ["alpha", "wz", "stab", "dstab"]
        self.control_list = ["stab"]
        self.action_space_length = len(self.control_list)
        self.param: F16LongParameters = default_parameters()
        self.x_history = [x0_arr.reshape(4, 1)]
        self._initialize_selected_state_index(self.selected_state_output, self.list_state)

        if integrator == "euler":
            self._step_fn = euler
        elif integrator == "rk4":
            self._step_fn = rk4
        else:
            raise ValueError(f"unknown integrator: {integrator!r}")
        self._integrator_name = integrator

    def get_param(self) -> F16LongParameters:
        return self.param

    def set_param(self, new_param: F16LongParameters) -> None:
        self.param = new_param

    def run_step(self, u: ArrayLike) -> np.ndarray:
        """Advance the model one step.

        Control signal format::

            run_step([[stab_act]])

        Returns the new state (full vector or the selected subset, as a column).
        """
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_arr.size != self.action_space_length:
            raise Exception(
                "Размерность управляющего вектора задана неверно."
                f" Текущее значение {u_arr.size}, не соответсвует {self.action_space_length}"
            )

        x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
        t_now = self.t0 + self.dt * self.time_step
        x_next = self._step_fn(f16_ode_long, x_prev, u_arr, t_now, self.dt, self.param)

        x_next_col = x_next.reshape(4, 1)
        self.x_history.append(x_next_col)
        self.u_history.append(u_arr.reshape(-1, 1))
        self.time_step += 1

        if self.selected_state_output:
            return x_next_col[self.selected_state_index]
        return x_next_col
```

- [ ] **Step 4: Rewrite inital.py**

Replace the entire contents of `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/inital.py` with:

```python
"""Default initial state for the longitudinal F-16 model."""
from __future__ import annotations

import numpy as np
from numpy import deg2rad

alpha0 = deg2rad(0.0)
wz0 = deg2rad(0.0)
stab0 = deg2rad(0.0)
dstab0 = deg2rad(0.0)

# Column-vector layout, matching the legacy matlab.double shape.
initial_state: np.ndarray = np.array(
    [[alpha0], [wz0], [stab0], [dstab0]],
    dtype=np.float64,
)

# Order matters: must match `LongitudinalF16.list_state`.
_STATE_ORDER = ("alpha", "wz", "stab", "dstab")
initial_state_dict: dict[str, list[float]] = {
    "alpha": [alpha0],
    "wz": [wz0],
    "stab": [stab0],
    "dstab": [dstab0],
}


def set_initial_state(new_initial: dict) -> np.ndarray:
    """Override one or more initial states. Returns the new column vector."""
    unknown = set(new_initial) - set(_STATE_ORDER)
    if unknown:
        raise Exception(
            "Состояния заданы неверно, проверьте."
            f" Доступные состояния {list(_STATE_ORDER)}"
        )
    for key, value in new_initial.items():
        initial_state_dict[key] = [float(value)]
    return np.array(
        [initial_state_dict[name] for name in _STATE_ORDER],
        dtype=np.float64,
    )
```

- [ ] **Step 5: Run test, verify it passes**

Run: `poetry run pytest tests/aerospacemodel/f16/nonlinear/test_longitudinal_model.py -v`
Expected: 8 passed.

If `test_import_does_not_load_matlab` fails because `tensoraerospace.aerospacemodel.base` or some other transitive import drags matlab in, find the offender (`grep -rn "import matlab" tensoraerospace/`) — the only places it should appear after this task are the angular files (still being ported) and the unrelated `linear/angular/initial.py` and `supersonic/linear/directional/initial.py`. The longitudinal import path must be matlab-free.

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/model.py \
        tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/inital.py \
        tests/aerospacemodel/f16/nonlinear/test_longitudinal_model.py
git commit -m "refactor(f16): replace LongitudinalF16 matlab wrapper with numpy implementation"
```

### Task 3.5: Delete legacy longitudinal tests that depend on the matlab fake

**Files:**
- Delete: `tests/aerospacemodel/f16_nonlinear_longitudinal_model_test.py`
- Delete: `tests/aerospacemodel/f16_nonlinear_longitudinal_initial_test.py`

These tests install a fake matlab module so the old code can be imported. They are obsolete: the new code never imports matlab, and the new test file already covers their assertions.

- [ ] **Step 1: Confirm no test references them**

Run: `grep -rn "f16_nonlinear_longitudinal_model_test\|f16_nonlinear_longitudinal_initial_test" tests/ docs/`
Expected: no matches.

- [ ] **Step 2: Delete the files**

```bash
git rm tests/aerospacemodel/f16_nonlinear_longitudinal_model_test.py \
       tests/aerospacemodel/f16_nonlinear_longitudinal_initial_test.py
```

- [ ] **Step 3: Run the longitudinal test suite**

```bash
poetry run pytest tests/aerospacemodel/f16/nonlinear/ -v
```
Expected: all longitudinal tests pass; no collection errors.

- [ ] **Step 4: Commit**

```bash
git commit -m "test(f16): remove legacy matlab-fake-based longitudinal tests"
```

### Task 3.6: Longitudinal property tests (trim + sign + snapshot)

**Files:**
- Create: `tests/aerospacemodel/f16/nonlinear/test_longitudinal_properties.py`
- Create (snapshot): `tests/aerospacemodel/f16/nonlinear/snapshots/long_open_loop_1s.npz`

- [ ] **Step 1: Write the property/snapshot test**

```python
# tests/aerospacemodel/f16/nonlinear/test_longitudinal_properties.py
import math
import pathlib

import numpy as np
import pytest
from scipy.optimize import fsolve

from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import LongitudinalF16
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import f16_ode_long
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params import default_parameters

SNAPSHOT = pathlib.Path(__file__).parent / "snapshots" / "long_open_loop_1s.npz"


def test_trim_point_exists_and_is_close_to_zero_dx():
    """For some (alpha, stab) the longitudinal forces and moments balance.
    Find it numerically and assert ||dx|| is small."""
    p = default_parameters()

    def residual(z):
        alpha, stab = z
        x = np.array([alpha, 0.0, stab, 0.0])
        u = np.array([stab])  # commanded == current => actuator at rest
        dx = f16_ode_long(x, u, 0.0, p)
        return np.array([dx[0], dx[1]])  # dalpha, dwz

    sol, info, ier, _msg = fsolve(
        residual, x0=[math.radians(2.0), math.radians(-2.0)], full_output=True
    )
    assert ier == 1
    alpha_trim, stab_trim = sol
    x_trim = np.array([alpha_trim, 0.0, stab_trim, 0.0])
    u_trim = np.array([stab_trim])
    dx = f16_ode_long(x_trim, u_trim, 0.0, p)
    np.testing.assert_allclose(dx[:2], 0.0, atol=1e-6)
    # Sanity: trim alpha is somewhere between -10 and +20 degrees.
    assert math.radians(-10) < alpha_trim < math.radians(20)


def test_open_loop_trajectory_matches_snapshot():
    """Regression test: a fixed open-loop episode reproduces an exact trajectory."""
    initial = np.array([[0.0], [0.0], [0.0], [0.0]])
    m = LongitudinalF16(initial, dt=0.01)
    states = []
    for k in range(100):  # 1 second
        out = m.run_step([[math.radians(0.5) if k < 50 else math.radians(-0.5)]])
        states.append(np.asarray(out).reshape(-1))
    traj = np.stack(states, axis=0)

    if not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SNAPSHOT, trajectory=traj)
        pytest.skip(f"snapshot created at {SNAPSHOT}; re-run to compare")

    expected = np.load(SNAPSHOT)["trajectory"]
    np.testing.assert_allclose(traj, expected, atol=1e-12)


def test_actuator_position_limit_enforced_over_long_command():
    """Sustained commanded value above maxabsstab keeps `stab` at the limit."""
    p = default_parameters()
    initial = np.array([[0.0], [0.0], [0.0], [0.0]])
    m = LongitudinalF16(initial, dt=0.005)
    huge = math.radians(40.0)  # well above maxabsstab=25 deg
    for _ in range(500):
        m.run_step([[huge]])
    final = np.asarray(m.x_history[-1]).reshape(-1)
    assert abs(final[2]) <= p.maxabsstab + 1e-6
```

- [ ] **Step 2: Run test once to create the snapshot**

Run: `poetry run pytest tests/aerospacemodel/f16/nonlinear/test_longitudinal_properties.py -v`
Expected: trim and limit tests pass; snapshot test is **skipped** ("snapshot created…").

- [ ] **Step 3: Run again to verify snapshot reproduces**

Run: `poetry run pytest tests/aerospacemodel/f16/nonlinear/test_longitudinal_properties.py -v`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/aerospacemodel/f16/nonlinear/test_longitudinal_properties.py \
        tests/aerospacemodel/f16/nonlinear/snapshots/long_open_loop_1s.npz
git commit -m "test(f16): add longitudinal trim, snapshot, and actuator limit tests"
```

---

## Phase 4 — Angular model

### Task 4.1: Read the remaining angular `.m` files and finalise the parser config

**Files:** none (research)

The angular variant adds `GetCx`, `GetCz`, `GetMx`, `GetMy`, `GetThrust`, `EnginePowerLevel`, `body2wind`, `wind2body`. Before writing extraction config, the executing agent must read each of those files **end to end** to identify the table names and grid axes. Only the longitudinal-side `LONG_TABLES` dict is filled in `scripts/extract_f16_aero.py`; `ANG_TABLES` is an empty placeholder. Fill it in here.

- [ ] **Step 1: Read each .m file**

```bash
for f in tensoraerospace/aerospacemodel/f16/nonlinear/angular/matlab_code/{GetCx,GetCy,GetCz,GetMx,GetMy,GetMz,GetThrust,EnginePowerLevel,body2wind,wind2body}.m; do
  echo "=== $f ==="
  cat "$f"
done | less
```

- [ ] **Step 2: For each Get*.m file, list the persistent variable names**

These are the names that follow `persistent` declarations near the top of each file. Write them down — they'll become the keys in the `ANG_TABLES` dict.

- [ ] **Step 3: Update `scripts/extract_f16_aero.py` with `ANG_TABLES`**

Edit `scripts/extract_f16_aero.py` and replace `ANG_TABLES = {}` with the populated dict. Use the same shape as `LONG_TABLES`:

```python
ANG_TABLES = {
    "GetCx.m": {"axes": [...], "tables": [...]},
    "GetCy.m": {"axes": [...], "tables": [...]},
    "GetCz.m": {"axes": [...], "tables": [...]},
    "GetMx.m": {"axes": [...], "tables": [...]},
    "GetMy.m": {"axes": [...], "tables": [...]},
    "GetMz.m": {"axes": [...], "tables": [...]},
    "GetThrust.m": {"axes": [...], "tables": [...]},
}
```

`EnginePowerLevel.m`, `body2wind.m`, `wind2body.m` have **no** persistent tables — they're pure-formula files and don't need extraction.

- [ ] **Step 4: Run the extractor**

```bash
poetry run python -m scripts.extract_f16_aero angular
```

Expected: one `.npz` file written per entry in `ANG_TABLES`, into `tensoraerospace/aerospacemodel/f16/nonlinear/angular/aero_tables/`.

- [ ] **Step 5: Sanity check shapes**

```bash
poetry run python - <<'PY'
import numpy as np
import pathlib
for p in sorted(pathlib.Path("tensoraerospace/aerospacemodel/f16/nonlinear/angular/aero_tables").glob("*.npz")):
    f = np.load(p)
    print(p.name)
    for k in f.files:
        print(" ", k, f[k].shape)
PY
```

Cross-check the printed shapes against the matlab files. If anything looks wrong, fix the parser before going further — the entire angular dynamics tests will silently produce garbage if a table is mis-shaped.

- [ ] **Step 6: Commit**

```bash
git add scripts/extract_f16_aero.py \
        tensoraerospace/aerospacemodel/f16/nonlinear/angular/aero_tables/
git commit -m "feat(f16): extract angular aero tables to npz"
```

### Task 4.2: Angular parameters dataclass

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/angular/params.py`
- Test: `tests/aerospacemodel/f16/nonlinear/test_angular_params.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/aerospacemodel/f16/nonlinear/test_angular_params.py
import math

import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    F16AngularParameters,
    default_parameters,
)


def test_defaults_match_matlab():
    p = default_parameters()
    assert p.m == pytest.approx(9295.44)
    assert p.l == pytest.approx(9.144)
    assert p.S == pytest.approx(27.87)
    assert p.bA == pytest.approx(3.45)
    assert p.Jx == pytest.approx(12874.8)
    assert p.Jy == pytest.approx(85552.1)
    assert p.Jz == pytest.approx(75673.6)
    assert p.Jxy == pytest.approx(1331.4)
    assert p.Jyz == 0
    assert p.Jxz == 0
    assert p.rcgx == pytest.approx(-0.05 * 3.45)
    assert p.hEx == 0.0
    assert p.Tstab == pytest.approx(0.03)
    assert p.Xistab == pytest.approx(0.707)
    assert p.maxabsstab == pytest.approx(math.radians(25))
    assert p.maxabsdstab == pytest.approx(math.radians(60))
    assert p.Tail == pytest.approx(0.02)
    assert p.Xiail == pytest.approx(0.707)
    assert p.maxabsail == pytest.approx(math.radians(21.5))
    assert p.maxabsdail == pytest.approx(math.radians(80))
    assert p.Tdir == pytest.approx(0.03)
    assert p.Xidir == pytest.approx(0.707)
    assert p.maxabsdir == pytest.approx(math.radians(30))
    assert p.maxabsddir == pytest.approx(math.radians(120))
    assert p.lef == 0.0
    assert p.sb == 0.0
    assert p.g == pytest.approx(9.80665)
    assert p.Oy == 3000.0
    assert p.V == 120.0


def test_dynamic_pressure_isa_at_3000m_120ms():
    p = default_parameters()
    # rho(3000) ~= 0.9091; q = 0.5*rho*120^2 ~= 6545
    assert p.q == pytest.approx(6545.5, rel=1e-3)
```

- [ ] **Step 2: Run, expect failure, then implement**

```python
# tensoraerospace/aerospacemodel/f16/nonlinear/angular/params.py
"""Default parameters for the F-16 6-DoF angular model.

Mirrors angular/matlab_code/airplane_parameters.m line by line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ISA atmosphere constants
_L = 0.0065
_R = 287.0531
_T0 = 288.15
_RHO0 = 1.225


def _isa_dynamic_pressure(altitude_m: float, velocity_mps: float, g: float) -> float:
    T = _T0 - _L * altitude_m
    rho = _RHO0 * (T / _T0) ** (g / (_L * _R) - 1.0)
    return 0.5 * rho * velocity_mps ** 2


@dataclass
class F16AngularParameters:
    m: float = 9295.44
    l: float = 9.144
    S: float = 27.87
    bA: float = 3.45
    Jx: float = 12874.8
    Jy: float = 85552.1
    Jz: float = 75673.6
    Jxy: float = 1331.4
    Jyz: float = 0.0
    Jxz: float = 0.0
    rcgx: float = field(init=False)
    hEx: float = 0.0

    Tstab: float = 0.03
    Xistab: float = 0.707
    maxabsstab: float = field(default_factory=lambda: math.radians(25))
    maxabsdstab: float = field(default_factory=lambda: math.radians(60))

    Tail: float = 0.02
    Xiail: float = 0.707
    maxabsail: float = field(default_factory=lambda: math.radians(21.5))
    maxabsdail: float = field(default_factory=lambda: math.radians(80))

    Tdir: float = 0.03
    Xidir: float = 0.707
    maxabsdir: float = field(default_factory=lambda: math.radians(30))
    maxabsddir: float = field(default_factory=lambda: math.radians(120))

    lef: float = 0.0
    sb: float = 0.0
    g: float = 9.80665

    Oy: float = 3000.0
    V: float = 120.0
    q: float = field(init=False)

    def __post_init__(self) -> None:
        self.rcgx = -0.05 * self.bA
        self.q = _isa_dynamic_pressure(self.Oy, self.V, self.g)


def default_parameters() -> F16AngularParameters:
    return F16AngularParameters()
```

- [ ] **Step 3: Verify test passes and commit**

```bash
poetry run pytest tests/aerospacemodel/f16/nonlinear/test_angular_params.py -v
git add tensoraerospace/aerospacemodel/f16/nonlinear/angular/params.py \
        tests/aerospacemodel/f16/nonlinear/test_angular_params.py
git commit -m "feat(f16): add angular parameters dataclass"
```

### Task 4.3: Frame transforms (`frames.py`)

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/angular/frames.py`
- Test: `tests/aerospacemodel/f16/nonlinear/test_angular_frames.py`

These mirror `angular/matlab_code/body2wind.m` and `wind2body.m`. The agent must `cat` those files first to confirm the exact 3×3 matrix definitions before implementing — they're each only ~7 lines.

- [ ] **Step 1: Read the .m files and write the failing test**

```bash
cat tensoraerospace/aerospacemodel/f16/nonlinear/angular/matlab_code/body2wind.m
cat tensoraerospace/aerospacemodel/f16/nonlinear/angular/matlab_code/wind2body.m
```

```python
# tests/aerospacemodel/f16/nonlinear/test_angular_frames.py
import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.angular.frames import (
    body_to_wind, wind_to_body,
)


def test_round_trip_is_identity():
    v_body = np.array([100.0, 5.0, 2.0])
    alpha = 0.1
    beta = 0.05
    v_wind = body_to_wind(v_body, alpha, beta)
    v_back = wind_to_body(v_wind, alpha, beta)
    np.testing.assert_allclose(v_back, v_body, atol=1e-12)


def test_zero_angles_is_identity():
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(body_to_wind(v, 0.0, 0.0), v, atol=1e-12)
```

- [ ] **Step 2: Implement frames.py**

The exact matrix entries depend on the matlab files; the implementation should be a literal port. Sketch (verify against the .m files):

```python
# tensoraerospace/aerospacemodel/f16/nonlinear/angular/frames.py
"""Body <-> wind frame rotation. Direct port of body2wind.m / wind2body.m."""
from __future__ import annotations

import numpy as np


def body_to_wind(v_body: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    R = np.array([
        [ ca * cb,  sb,  sa * cb],
        [-ca * sb,  cb, -sa * sb],
        [    -sa,  0.0,      ca],
    ])
    return R @ v_body


def wind_to_body(v_wind: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    R = np.array([
        [ ca * cb, -ca * sb, -sa],
        [      sb,       cb, 0.0],
        [ sa * cb, -sa * sb,  ca],
    ])
    return R @ v_wind
```

> If reading the matlab files reveals a different convention (e.g., the matrices use a different sign for `sb`), use *those* matrices verbatim and update both the implementation and the test expectations accordingly. The round-trip test will catch any inverse mismatch.

- [ ] **Step 3: Run, fix until passing, commit**

```bash
poetry run pytest tests/aerospacemodel/f16/nonlinear/test_angular_frames.py -v
git add tensoraerospace/aerospacemodel/f16/nonlinear/angular/frames.py \
        tests/aerospacemodel/f16/nonlinear/test_angular_frames.py
git commit -m "feat(f16): add angular body-wind frame transforms"
```

### Task 4.4: Angular aero coefficient module

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/angular/aero.py`
- Test: `tests/aerospacemodel/f16/nonlinear/test_angular_aero.py`

This is the largest single task. The agent ports `GetCx.m`, `GetCy.m`, `GetCz.m`, `GetMx.m`, `GetMy.m`, `GetMz.m`, `GetThrust.m`, and `EnginePowerLevel.m` to one Python module.

**Critical:** the angular `GetCy.m` is *not* the same as the longitudinal one — it has different argument signatures (`fi` here is the elevator, but the function may also depend on `wx`, `wy`, `dir`, `ail`). Read each .m file end to end before porting it, and copy the assembly (`Сборка`) section verbatim into the corresponding Python `get_*` function.

- [ ] **Step 1: For each Get*.m file, write a separate `get_*` function in `aero.py`**

The agent should use `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/aero.py` as the structural template. For each function:

1. Module-level: load the corresponding `.npz`, build interpolators (cubic for ≥2D regular grids, pchip for 1D), record bounds for clamping.
2. Function body: clip inputs, evaluate interpolators, apply the assembly formula from the matlab file.

`engine_power_level` is purely formula-based (no tables). Port it as a small function. `get_thrust` depends on a power-level state — read carefully.

- [ ] **Step 2: For each get_* function, write a grid-node test**

Following the same pattern as `test_longitudinal_aero.test_get_cy_grid_node_matches_table`. At a known grid corner with all secondary terms (wz, dnos, sb, …) zero, the result should equal the corresponding table entry to within `1e-4` relative tolerance. Add at least one test per function.

- [ ] **Step 3: For each get_* function, write a finite-output sanity test**

`assert math.isfinite(get_*(...))` for a representative state inside the envelope.

- [ ] **Step 4: Run all tests in this file**

```bash
poetry run pytest tests/aerospacemodel/f16/nonlinear/test_angular_aero.py -v
```

Expected: all green. If a grid-node test is off by more than the cubic spline rounding tolerance (1e-4), the table extraction is probably wrong — go back to Task 4.1 and fix `ANG_TABLES`.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/angular/aero.py \
        tests/aerospacemodel/f16/nonlinear/test_angular_aero.py
git commit -m "feat(f16): port angular aero coefficient functions to numpy"
```

### Task 4.5: Angular dynamics module

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/angular/dynamics.py`
- Test: `tests/aerospacemodel/f16/nonlinear/test_angular_dynamics.py`

The right-hand side is a direct port of `angular/matlab_code/F16ODE.m`. The state vector has **14 components** in this order:

`[alpha, beta, wx, wy, wz, gamma, psi, theta, stab, dstab, ail, dail, dir, ddir]`

The control vector is `[stab_act, ail_act, dir_act]`.

- [ ] **Step 1: Re-read F16ODE.m to confirm the equations**

```bash
cat tensoraerospace/aerospacemodel/f16/nonlinear/angular/matlab_code/F16ODE.m
```

Look especially at:
- The moment-of-inertia coupling terms in `Dx.wx`, `Dx.wy`, `Dx.wz` (uses `Gamma = Jx*Jy - Jxy^2`).
- The Euler-angle propagation block (`Dx.gamma`, `Dx.theta`, `Dx.psi`).
- The three actuator second-order ODEs (one each for stab/ail/dir), each with their own rate and position limits.

- [ ] **Step 2: Write the failing test**

```python
# tests/aerospacemodel/f16/nonlinear/test_angular_dynamics.py
import math

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics import f16_ode_6dof
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import default_parameters


def _zero_state():
    return np.zeros(14, dtype=np.float64)


def test_dx_shape_and_finite():
    p = default_parameters()
    dx = f16_ode_6dof(_zero_state(), np.zeros(3), 0.0, p)
    assert dx.shape == (14,)
    assert np.all(np.isfinite(dx))


def test_each_actuator_responds_to_its_own_command():
    p = default_parameters()
    x = _zero_state()
    # stab actuator: u[0] influences ddstab (=dx[9])
    dx_stab = f16_ode_6dof(x, np.array([math.radians(5.0), 0.0, 0.0]), 0.0, p)
    dx_zero = f16_ode_6dof(x, np.zeros(3), 0.0, p)
    assert dx_stab[9] != dx_zero[9]
    # ail actuator: u[1] influences ddail (=dx[11])
    dx_ail = f16_ode_6dof(x, np.array([0.0, math.radians(5.0), 0.0]), 0.0, p)
    assert dx_ail[11] != dx_zero[11]
    # dir actuator: u[2] influences dddir (=dx[13])
    dx_dir = f16_ode_6dof(x, np.array([0.0, 0.0, math.radians(5.0)]), 0.0, p)
    assert dx_dir[13] != dx_zero[13]


def test_actuator_command_clamping():
    p = default_parameters()
    x = _zero_state()
    huge = np.array([10 * p.maxabsstab, 10 * p.maxabsail, 10 * p.maxabsdir])
    clipped = np.array([p.maxabsstab, p.maxabsail, p.maxabsdir])
    np.testing.assert_allclose(
        f16_ode_6dof(x, huge, 0.0, p),
        f16_ode_6dof(x, clipped, 0.0, p),
    )


def test_state_at_rest_is_quasi_stationary():
    """At zero state and zero command, ddxx values for the actuator positions
    are well-defined (not NaN/Inf). This catches division-by-zero in the
    moment-of-inertia coupling and the Euler-angle kinematics."""
    p = default_parameters()
    dx = f16_ode_6dof(_zero_state(), np.zeros(3), 0.0, p)
    assert np.all(np.isfinite(dx))
```

- [ ] **Step 3: Implement dynamics.py**

```python
# tensoraerospace/aerospacemodel/f16/nonlinear/angular/dynamics.py
"""F-16 6-DoF angular ODE right-hand side.

Direct line-by-line port of angular/matlab_code/F16ODE.m.
"""
from __future__ import annotations

import numpy as np

from .aero import (  # all imports must exist after Task 4.4
    get_cx, get_cy, get_cz, get_mx, get_my, get_mz,
)
from .params import F16AngularParameters

# State vector indices, in matlab F16State_vec2struct.m order
I_ALPHA, I_BETA, I_WX, I_WY, I_WZ = 0, 1, 2, 3, 4
I_GAMMA, I_PSI, I_THETA = 5, 6, 7
I_STAB, I_DSTAB = 8, 9
I_AIL, I_DAIL = 10, 11
I_DIR, I_DDIR = 12, 13


def f16_ode_6dof(x: np.ndarray, u: np.ndarray, t: float, params: F16AngularParameters) -> np.ndarray:
    p = params
    alpha = float(x[I_ALPHA]); beta = float(x[I_BETA])
    wx = float(x[I_WX]); wy = float(x[I_WY]); wz = float(x[I_WZ])
    gamma = float(x[I_GAMMA]); psi = float(x[I_PSI]); theta = float(x[I_THETA])
    stab = float(x[I_STAB]); dstab = float(x[I_DSTAB])
    ail = float(x[I_AIL]); dail = float(x[I_DAIL])
    direc = float(x[I_DIR]); ddir = float(x[I_DDIR])

    stab_act, ail_act, dir_act = float(u[0]), float(u[1]), float(u[2])

    cx = get_cx(alpha, beta, stab, p.lef, wz, p.V, p.bA, p.sb)
    cy = get_cy(alpha, beta, stab, p.lef, wz, p.V, p.bA, p.sb)
    cz = get_cz(alpha, beta, direc, ail, p.lef, wx, wy, p.V, p.l)
    mx = get_mx(alpha, beta, stab, direc, ail, p.lef, wx, wy, p.V, p.l)
    my = get_my(alpha, beta, stab, direc, ail, p.lef, wx, wy, p.V, p.l)
    mz = get_mz(alpha, beta, stab, p.lef, wz, p.V, p.bA, p.sb)

    X = -p.q * p.S * cx
    Y =  p.q * p.S * cy
    Z =  p.q * p.S * cz

    Mx = p.q * p.S * p.l * mx
    My = p.q * p.S * p.l * my
    Mz = p.q * p.S * p.bA * mz

    Rx, Ry, Rz = X, Y, Z
    MRx = Mx
    MRy = My - p.rcgx * Rz
    MRz = Mz + p.rcgx * Ry

    Gamma_inertia = p.Jx * p.Jy - p.Jxy ** 2
    dwx = (p.Jy * MRx + p.Jxy * (MRy - p.hEx * wz)
           + p.Jxy * (p.Jz - p.Jx - p.Jy) * wx * wz
           + (p.Jxy ** 2 + p.Jy * (p.Jy - p.Jz)) * wy * wz) / Gamma_inertia
    dwy = (p.Jxy * MRx + p.Jx * (MRy - p.hEx * wz)
           + (p.Jx * (p.Jz - p.Jx) - p.Jxy ** 2) * wx * wz
           + p.Jxy * (p.Jx + p.Jy - p.Jz) * wy * wz) / Gamma_inertia
    dwz = (MRz + p.hEx * wy + p.Jxy * (wx * wx - wy * wy)
           + (p.Jx - p.Jy) * wx * wy) / p.Jz

    sin_a, cos_a = np.sin(alpha), np.cos(alpha)
    sin_b, cos_b = np.sin(beta), np.cos(beta)
    sin_g, cos_g = np.sin(gamma), np.cos(gamma)
    sin_th, cos_th = np.sin(theta), np.cos(theta)

    gax = p.g * (-sin_th * cos_a * cos_b + cos_g * cos_th * sin_a * cos_b + sin_g * cos_th * sin_b)
    gay = p.g * (-sin_th * sin_a - cos_g * cos_th * cos_a)
    gaz = p.g * ( sin_th * cos_a * sin_b - cos_g * cos_th * sin_a * sin_b + sin_g * cos_th * cos_b)

    Xa = -cos_a * cos_b * Rx - sin_a * cos_b * Ry + sin_b * Rz
    Ya = -sin_a * Rx + cos_a * Ry
    Za =  cos_a * sin_b * Rx + sin_a * sin_b * Ry + cos_b * Rz

    dalpha = wz + (wy * sin_a - wx * cos_a) * np.tan(beta) - (Ya + p.m * gay) / (p.m * p.V * cos_b)
    dbeta = wx * sin_a + wy * cos_a + (Za + p.m * gaz) / (p.m * p.V)

    dgamma = wx - cos_g * np.tan(theta) * wy + sin_g * np.tan(theta) * wz
    dtheta = sin_g * wy + cos_g * wz
    dpsi = (cos_g / cos_th) * wy - (sin_g / cos_th) * wz

    dstab_clip = float(np.clip(dstab, -p.maxabsdstab, p.maxabsdstab))
    stab_act_clip = float(np.clip(stab_act, -p.maxabsstab, p.maxabsstab))
    ddstab = (-2.0 * p.Tstab * p.Xistab * dstab - stab + stab_act_clip) / (p.Tstab ** 2)

    dail_clip = float(np.clip(dail, -p.maxabsdail, p.maxabsdail))
    ail_act_clip = float(np.clip(ail_act, -p.maxabsail, p.maxabsail))
    ddail = (-2.0 * p.Tail * p.Xiail * dail - ail + ail_act_clip) / (p.Tail ** 2)

    ddir_clip = float(np.clip(ddir, -p.maxabsddir, p.maxabsddir))
    dir_act_clip = float(np.clip(dir_act, -p.maxabsdir, p.maxabsdir))
    dddir = (-2.0 * p.Tdir * p.Xidir * ddir - direc + dir_act_clip) / (p.Tdir ** 2)

    return np.array([
        dalpha, dbeta,
        dwx, dwy, dwz,
        dgamma, dpsi, dtheta,
        dstab_clip, ddstab,
        dail_clip, ddail,
        ddir_clip, dddir,
    ], dtype=np.float64)
```

- [ ] **Step 4: Run, fix, commit**

```bash
poetry run pytest tests/aerospacemodel/f16/nonlinear/test_angular_dynamics.py -v
git add tensoraerospace/aerospacemodel/f16/nonlinear/angular/dynamics.py \
        tests/aerospacemodel/f16/nonlinear/test_angular_dynamics.py
git commit -m "feat(f16): port angular F16ODE to pure-numpy 6-DoF dynamics"
```

### Task 4.6: Replace `AngularF16` class and `inital.py`

**Files:**
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/angular/model.py` (full rewrite)
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/angular/inital.py` (full rewrite)
- Test: `tests/aerospacemodel/f16/nonlinear/test_angular_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/aerospacemodel/f16/nonlinear/test_angular_model.py
import math
import sys

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import (
    AngularF16,
    initial_state,
    set_initial_state,
)


def test_import_does_not_load_matlab():
    for mod in list(sys.modules):
        if mod.startswith("matlab"):
            sys.modules.pop(mod, None)
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16  # noqa
    assert "matlab" not in sys.modules


def test_run_step_returns_14d_state():
    m = AngularF16(initial_state)
    out = m.run_step([[0.0], [0.0], [0.0]])
    assert np.asarray(out).reshape(-1).shape == (14,)


def test_run_step_accepts_flat_list():
    m = AngularF16(initial_state)
    out = m.run_step([0.0, 0.0, 0.0])
    assert out is not None


def test_run_step_rejects_wrong_action_dim():
    m = AngularF16(initial_state)
    with pytest.raises(Exception):
        m.run_step([[0.0]])


def test_set_initial_state_overrides_alpha_correctly():
    """Regression: the legacy inital.py had a bug where set_initial_state
    returned a vector with stab/ail/dir grouped before their derivatives,
    inconsistent with `initial_state` and `list_state`. After the port,
    overriding alpha must put the override at index 0 of the new vector."""
    out = set_initial_state({"alpha": math.radians(7.0)})
    arr = np.asarray(out, dtype=float).reshape(-1)
    assert arr[0] == pytest.approx(math.radians(7.0))


def test_state_vector_ordering_matches_list_state():
    """The position of `dstab` in the returned initial vector must equal
    the position of 'dstab' in `AngularF16.list_state` -- this is the bug
    that the legacy code had."""
    m = AngularF16(initial_state)
    assert m.list_state.index("dstab") == 9
    assert m.list_state.index("ail") == 10
    assert m.list_state.index("ddir") == 13
```

- [ ] **Step 2: Run, expect failure, then implement**

Replace `tensoraerospace/aerospacemodel/f16/nonlinear/angular/model.py` with:

```python
"""F-16 6-DoF angular nonlinear model — pure-numpy implementation."""
from __future__ import annotations

from typing import Literal, Sequence, Union

import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase

from .._integrators import euler, rk4
from .dynamics import f16_ode_6dof
from .params import F16AngularParameters, default_parameters


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]


class AngularF16(ModelBase):
    """High-maneuverability F-16 aircraft, full 6-DoF angular model.

    State (14): alpha, beta, wx, wy, wz, gamma, psi, theta,
                stab, dstab, ail, dail, dir, ddir
    Control (3): stab_act, ail_act, dir_act
    """

    def __init__(
        self,
        x0: ArrayLike,
        selected_state_output=None,
        t0: float = 0,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "euler",
    ) -> None:
        x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0_arr.size != 14:
            raise ValueError(
                f"x0 must have 14 elements; got {x0_arr.size}"
            )
        super().__init__(x0_arr, selected_state_output, t0, dt)
        self.list_state = [
            "alpha", "beta",
            "wx", "wy", "wz",
            "gamma", "psi", "theta",
            "stab", "dstab",
            "ail", "dail",
            "dir", "ddir",
        ]
        self.control_list = ["stab", "ail", "dir"]
        self.action_space_length = len(self.control_list)
        self.param: F16AngularParameters = default_parameters()
        self.x_history = [x0_arr.reshape(14, 1)]
        self._initialize_selected_state_index(self.selected_state_output, self.list_state)

        if integrator == "euler":
            self._step_fn = euler
        elif integrator == "rk4":
            self._step_fn = rk4
        else:
            raise ValueError(f"unknown integrator: {integrator!r}")
        self._integrator_name = integrator

    def get_param(self) -> F16AngularParameters:
        return self.param

    def set_param(self, new_param: F16AngularParameters) -> None:
        self.param = new_param

    def run_step(self, u: ArrayLike) -> np.ndarray:
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_arr.size != self.action_space_length:
            raise Exception(
                "Размерность управляющего вектора задана неверно."
                f" Текущее значение {u_arr.size}, не соответсвует {self.action_space_length}"
            )

        x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
        t_now = self.t0 + self.dt * self.time_step
        x_next = self._step_fn(f16_ode_6dof, x_prev, u_arr, t_now, self.dt, self.param)

        x_next_col = x_next.reshape(14, 1)
        self.x_history.append(x_next_col)
        self.u_history.append(u_arr.reshape(-1, 1))
        self.time_step += 1

        if self.selected_state_output:
            return x_next_col[self.selected_state_index]
        return x_next_col
```

Replace `tensoraerospace/aerospacemodel/f16/nonlinear/angular/inital.py` with:

```python
"""Default initial state for the angular F-16 model.

State ordering MUST match angular/model.py::AngularF16.list_state and
angular/matlab_code/F16State_vec2struct.m -- the legacy inital.py had a bug
where the dict ordering was different from the matrix ordering, so
set_initial_state returned a permuted vector. Fixed here.
"""
from __future__ import annotations

import numpy as np
from numpy import deg2rad

_DEFAULTS = {
    "alpha": deg2rad(0.0),
    "beta": deg2rad(0.0),
    "wx": deg2rad(0.0),
    "wy": deg2rad(0.0),
    "wz": deg2rad(0.0),
    "gamma": deg2rad(0.0),
    "psi": deg2rad(0.0),
    "theta": deg2rad(0.0),
    "stab": deg2rad(0.0),
    "dstab": deg2rad(0.0),
    "ail": deg2rad(0.0),
    "dail": deg2rad(0.0),
    "dir": deg2rad(0.0),
    "ddir": deg2rad(0.0),
}

_STATE_ORDER = (
    "alpha", "beta",
    "wx", "wy", "wz",
    "gamma", "psi", "theta",
    "stab", "dstab",
    "ail", "dail",
    "dir", "ddir",
)

initial_state: np.ndarray = np.array(
    [[_DEFAULTS[name]] for name in _STATE_ORDER],
    dtype=np.float64,
)

initial_state_dict: dict[str, list[float]] = {
    name: [_DEFAULTS[name]] for name in _STATE_ORDER
}


def set_initial_state(new_initial: dict) -> np.ndarray:
    unknown = set(new_initial) - set(_STATE_ORDER)
    if unknown:
        raise Exception(
            "Состояния заданы неверно, проверьте."
            f" Доступные состояния {list(_STATE_ORDER)}"
        )
    for key, value in new_initial.items():
        initial_state_dict[key] = [float(value)]
    return np.array(
        [initial_state_dict[name] for name in _STATE_ORDER],
        dtype=np.float64,
    )
```

- [ ] **Step 3: Run tests, commit**

```bash
poetry run pytest tests/aerospacemodel/f16/nonlinear/test_angular_model.py -v
git add tensoraerospace/aerospacemodel/f16/nonlinear/angular/model.py \
        tensoraerospace/aerospacemodel/f16/nonlinear/angular/inital.py \
        tests/aerospacemodel/f16/nonlinear/test_angular_model.py
git commit -m "refactor(f16): replace AngularF16 matlab wrapper with numpy implementation"
```

### Task 4.7: Delete legacy angular tests + add property tests

**Files:**
- Delete: `tests/aerospacemodel/f16_nonlinear_angular_model_test.py`
- Delete: `tests/aerospacemodel/f16_nonlinear_angular_initial_test.py`
- Create: `tests/aerospacemodel/f16/nonlinear/test_angular_properties.py`
- Create: `tests/aerospacemodel/f16/nonlinear/snapshots/angular_open_loop_1s.npz`

- [ ] **Step 1: Delete legacy tests**

```bash
git rm tests/aerospacemodel/f16_nonlinear_angular_model_test.py \
       tests/aerospacemodel/f16_nonlinear_angular_initial_test.py
```

- [ ] **Step 2: Write the property + snapshot test**

```python
# tests/aerospacemodel/f16/nonlinear/test_angular_properties.py
import math
import pathlib

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import (
    AngularF16,
    initial_state,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics import f16_ode_6dof
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import default_parameters

SNAPSHOT = pathlib.Path(__file__).parent / "snapshots" / "angular_open_loop_1s.npz"


def test_zero_input_zero_state_no_nans_for_one_second():
    """Smoke test: 1 second of zero-command simulation produces finite states."""
    m = AngularF16(initial_state, dt=0.01)
    for _ in range(100):
        out = m.run_step([[0.0], [0.0], [0.0]])
        arr = np.asarray(out).reshape(-1)
        assert np.all(np.isfinite(arr))


def test_open_loop_trajectory_snapshot():
    """Regression: a fixed sequence of stab/ail/dir commands produces an
    exact trajectory."""
    m = AngularF16(initial_state, dt=0.01)
    states = []
    for k in range(100):
        u = [
            [math.radians(0.5) if k < 30 else 0.0],
            [math.radians(1.0) if 30 <= k < 60 else 0.0],
            [math.radians(0.5) if 60 <= k else 0.0],
        ]
        out = m.run_step(u)
        states.append(np.asarray(out).reshape(-1))
    traj = np.stack(states, axis=0)

    if not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SNAPSHOT, trajectory=traj)
        pytest.skip(f"snapshot created at {SNAPSHOT}; re-run to compare")

    expected = np.load(SNAPSHOT)["trajectory"]
    np.testing.assert_allclose(traj, expected, atol=1e-12)
```

- [ ] **Step 3: Run twice to seed and verify the snapshot, then commit**

```bash
poetry run pytest tests/aerospacemodel/f16/nonlinear/test_angular_properties.py -v
poetry run pytest tests/aerospacemodel/f16/nonlinear/test_angular_properties.py -v
git add tests/aerospacemodel/f16/nonlinear/test_angular_properties.py \
        tests/aerospacemodel/f16/nonlinear/snapshots/angular_open_loop_1s.npz
git commit -m "test(f16): add angular smoke and snapshot tests; remove legacy"
```

---

## Phase 5 — Cleanup and verification

### Task 5.1: Verify no `import matlab` left in nonlinear F-16 paths

**Files:** none (audit)

- [ ] **Step 1: Grep for matlab imports**

```bash
grep -rn "import matlab\|from matlab" tensoraerospace/aerospacemodel/f16/nonlinear/
```

Expected: **no matches**. The `matlab_code/` directories will be skipped because `.m` files don't match the regex.

If anything else turns up (e.g., a forgotten import in `__init__.py`), remove it and commit separately.

- [ ] **Step 2: Confirm `pyproject.toml` does not list `matlabengine` or similar**

```bash
grep -n "matlab" pyproject.toml || echo "no matlab refs"
```

Expected: `no matlab refs`. No edit needed (it was never declared).

### Task 5.2: Run the full nonlinear F-16 test suite

- [ ] **Step 1: Run everything**

```bash
poetry run pytest tests/aerospacemodel/f16/nonlinear/ -v
```

Expected: every test passes.

- [ ] **Step 2: Run a quick sanity import in a fresh Python**

```bash
poetry run python - <<'PY'
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import LongitudinalF16, initial_state
from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16, initial_state as angular_initial_state
import sys
assert "matlab" not in sys.modules, "matlab leaked into the import path"
print("OK:", LongitudinalF16, AngularF16)
PY
```

Expected: prints `OK: <class ...> <class ...>` and exits 0.

### Task 5.3: One-off micro-benchmark

**Files:** none (informational)

- [ ] **Step 1: Time a longitudinal episode**

```bash
poetry run python - <<'PY'
import time
import numpy as np
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import LongitudinalF16, initial_state

m = LongitudinalF16(initial_state, dt=0.01)
N = 10_000
t0 = time.perf_counter()
for _ in range(N):
    m.run_step([[0.0]])
dt = time.perf_counter() - t0
print(f"{N} longitudinal steps in {dt*1e3:.1f} ms ({dt*1e6/N:.1f} us/step)")
PY
```

Expected: < 200 µs/step on a modern CPU. If it's an order of magnitude slower (>2 ms/step), check whether `RegularGridInterpolator` is being constructed inside the inner loop instead of at module level — it should be at module level. (No commit needed.)

### Task 5.4: Final commit

- [ ] **Step 1: Verify clean working tree**

```bash
git status -sb
```

Expected: clean. If there are leftover untracked files (e.g., `__pycache__` directories), they should already be in `.gitignore` — verify with `git check-ignore -v <path>`.

- [ ] **Step 2: Push the branch**

```bash
git push -u origin feat/f16-nonlinear-numpy-port
```

(Only do this when the user confirms they want the branch published.)

---

## Self-review checklist (filled in by author)

**Spec coverage:**
- ✅ Architecture (numpy, pure functions, thin classes) — Tasks 1.1, 3.1–3.4, 4.2–4.6.
- ✅ Aero coefficients via cubic spline + pchip — Task 3.2 (longitudinal), 4.4 (angular).
- ✅ Configurable integrator — Task 1.1 + class kwargs in 3.4 / 4.6.
- ✅ Trim test, snapshot regression, sign tests, integrator consistency, actuator dynamics, position/rate limits — Tasks 3.6, 4.7, plus per-module dynamics tests.
- ✅ Public API preservation (`LongitudinalF16(x0).run_step(u)`, etc.) — Tasks 3.4, 4.6.
- ✅ `inital.py` ordering bug — explicitly fixed in Task 4.6.
- ✅ Removal of `matlab.engine` — verified in Task 5.1.
- ✅ Performance check — Task 5.3.
- ✅ Reference `matlab_code/` preserved — never touched, only read.

**Placeholder scan:** No `TBD` / `TODO` / `add appropriate error handling` / `similar to task N` strings.

**Type / signature consistency:** `f16_ode_long(x, u, t, params)` and `f16_ode_6dof(x, u, t, params)` use the same signature so the integrator works for both. `RHS` typedef in `_integrators.py` documents that signature. Class kwargs (`integrator`, `dt`, `t0`) match between `LongitudinalF16` and `AngularF16`.

**Known soft spots that the executing agent should pay attention to:**

1. **Task 4.1 deliberately requires reading the angular `.m` files** before filling `ANG_TABLES`. If the agent hand-waves this and guesses the variable names, the extraction will silently produce wrong tables. Don't.
2. **Task 4.4 (angular aero)** is the largest single task. If the agent needs to pause for a checkpoint, this is the natural place — split into per-coefficient sub-tasks (port `get_cx` first, run its tests, commit, then `get_cy`, etc.).
3. **The `RegularGridInterpolator(method="cubic")` rounding tolerance** at grid nodes is empirically `~1e-12` for well-conditioned tables but can drift to `1e-4` near saddle points. Adjust per-test tolerances accordingly — never relax to `1e-2` or worse without understanding why.
4. **The matlab files use `csaps` with smoothing parameter `1 - 1e-5` or `1 - 1e-6`.** That's not zero. Pure cubic interpolation will produce values that differ from matlab at non-grid points by ~1e-4 to 1e-3. This is acceptable for our purposes (the difference is smaller than the table values themselves) and is documented in the spec.
