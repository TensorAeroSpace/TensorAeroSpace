"""Extract F-16 aerodynamic lookup tables from matlab .m files into .npz.

Dev tooling. Re-run only when the .m sources change. The runtime package
does NOT depend on this script.

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

# Inputs (matlab .m sources) live in the legacy paths.
LONG_MATLAB_DIR = REPO_ROOT / "tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/matlab_code"
ANG_MATLAB_DIR = REPO_ROOT / "tensoraerospace/aerospacemodel/f16/nonlinear/angular/matlab_code"

# Outputs (.npz tables) live in the new python/ subtree.
LONG_OUT_DIR = REPO_ROOT / "tensoraerospace/aerospacemodel/f16/nonlinear/python/longitudinal/aero_tables"
ANG_OUT_DIR = REPO_ROOT / "tensoraerospace/aerospacemodel/f16/nonlinear/python/angular/aero_tables"


# ---------- low-level matlab matrix parsing ----------

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?")


def _strip_comments(src: str) -> str:
    return re.sub(r"%[^\n]*", "", src)


def _parse_matrix_literal(literal: str) -> np.ndarray:
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
        raise ValueError(f"ragged matrix literal: row widths {[len(r) for r in rows]}")
    return np.array(rows, dtype=np.float64)


def _eval_rhs(rhs: str, scope: dict) -> np.ndarray:
    rhs = rhs.strip()

    m = re.fullmatch(r"deg2rad\((.*)\)", rhs, flags=re.DOTALL)
    if m:
        return np.deg2rad(_eval_rhs(m.group(1), scope))

    if rhs.startswith("["):
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
        suffix = rhs[end + 1:].strip().rstrip(";")
        if suffix == "'":
            arr = arr.T
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[1] == 1:
                arr = arr[:, 0]
        else:
            # Squeeze single-row and single-column matrices to 1-D arrays,
            # matching numpy convention for Matlab row/column vectors.
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            elif arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
        return arr

    if rhs.startswith("-"):
        name = rhs[1:].strip().rstrip(";")
        if name in scope:
            return -scope[name]

    name = rhs.rstrip(";").strip()
    if name in scope:
        return scope[name]

    raise ValueError(f"unsupported rhs: {rhs!r}")


def _split_statements(src: str) -> list[str]:
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


def parse_matlab_assignment(src: str, var_name: str) -> np.ndarray:
    """Return the *final* value of `var_name` after walking through `src`."""
    src = _strip_comments(src)
    scope: dict = {}
    indexed_pages: dict = {}

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
            # If we have accumulated indexed pages (e.g. Cy1(:,:,k) = ...)
            # but not yet assembled them, do so now before evaluating the rhs
            # (the rhs might reference var_name itself, e.g. `Cy1 = -Cy1;`).
            if indexed_pages and var_name not in scope:
                pages = [indexed_pages[i] for i in sorted(indexed_pages)]
                scope[var_name] = np.stack(pages, axis=-1)
            scope[var_name] = _eval_rhs(m.group(1), scope)
            continue

        plain = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", stmt, re.DOTALL)
        if plain:
            try:
                scope[plain.group(1)] = _eval_rhs(plain.group(2), scope)
            except Exception:
                pass

    if indexed_pages and var_name not in scope:
        # Only assemble indexed pages if no direct assignment superseded them
        # (e.g. `Cy1 = -Cy1;` after the indexed assignments). In that case the
        # direct_re branch above already assembled and negated, so we skip here.
        pages = [indexed_pages[i] for i in sorted(indexed_pages)]
        scope[var_name] = np.stack(pages, axis=-1)

    if var_name not in scope:
        raise KeyError(f"variable {var_name!r} not found in source")
    return scope[var_name]


def parse_matlab_file(src: str, var_names: list[str]) -> dict:
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

ANG_TABLES: dict = {}  # populated in Phase 4 Task 4.1


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
        extract(LONG_MATLAB_DIR, LONG_OUT_DIR, LONG_TABLES)
    if args.target in ("angular", "all"):
        extract(ANG_MATLAB_DIR, ANG_OUT_DIR, ANG_TABLES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
