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
LONG_OUT_DIR = REPO_ROOT / "tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/aero_tables"
ANG_OUT_DIR = REPO_ROOT / "tensoraerospace/aerospacemodel/f16/nonlinear/angular/aero_tables"


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


def _parse_range(s: str) -> np.ndarray | None:
    """Parse a Matlab range literal like ``start:step:end`` or ``start:end``.

    Returns a 1-D numpy array or *None* if ``s`` does not look like a range.
    Only handles purely numeric (no variable) range expressions.
    """
    s = s.strip().rstrip(";")
    parts = s.split(":")
    if len(parts) == 2:
        try:
            start, stop = float(parts[0]), float(parts[1])
            step = 1.0
        except ValueError:
            return None
    elif len(parts) == 3:
        try:
            start, step, stop = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return None
    else:
        return None
    # Replicate Matlab colon behaviour: include stop if within half a step.
    n = int(round((stop - start) / step)) + 1
    return np.linspace(start, start + step * (n - 1), n)


def _eval_rhs(rhs: str, scope: dict) -> np.ndarray:
    rhs = rhs.strip()

    m = re.fullmatch(r"deg2rad\((.*)\)", rhs, flags=re.DOTALL)
    if m:
        return np.deg2rad(_eval_rhs(m.group(1), scope))

    # Parenthesised range multiplied by a scalar: (start:step:end)*scalar
    m = re.fullmatch(r"\(([^)]+)\)\s*\*\s*(" + _NUMBER_RE.pattern + r")", rhs)
    if m:
        inner = _parse_range(m.group(1))
        if inner is not None:
            return inner * float(m.group(2))

    # Bare range: start:step:end or start:end
    if ":" in rhs and not rhs.startswith("["):
        arr = _parse_range(rhs)
        if arr is not None:
            return arr

    # Named variable multiplied by a scalar: varname * scalar
    m = re.fullmatch(r"([A-Za-z_]\w*)\s*\*\s*(" + _NUMBER_RE.pattern + r")", rhs)
    if m:
        name, scalar = m.group(1), float(m.group(2))
        if name in scope:
            return scope[name] * scalar

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
            # Side-variable assignment. Best-effort: matlab init blocks contain
            # things this parser doesn't handle (function calls beyond deg2rad,
            # arithmetic, etc). If we can't evaluate it, drop it from scope so
            # later references trigger an explicit error rather than using
            # stale data.
            try:
                scope[plain.group(1)] = _eval_rhs(plain.group(2), scope)
            except ValueError:
                scope.pop(plain.group(1), None)

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

ANG_TABLES: dict = {
    # GetCx(alpha, beta, fi, dnos, Wz, V, ba, sb)
    # axes: alpha1(20), alpha2(14), beta1(19), fi1(5)
    # Cx1(alpha1,beta1,fi1), Cx_nos1(alpha2,beta1), Cxwz1(alpha1), dCxwz_nos1(alpha2), dCx_sb1(alpha1)
    # post-negation: Cx1=-Cx1, Cx_nos1=-Cx_nos1, Cxwz1=-Cxwz1, dCxwz_nos1=-dCxwz_nos1, dCx_sb1=-dCx_sb1
    "GetCx.m": {
        "axes": ["alpha1", "alpha2", "beta1", "fi1"],
        "tables": ["Cx1", "Cx_nos1", "Cxwz1", "dCxwz_nos1", "dCx_sb1"],
    },
    # GetCy(alpha, beta, fi, dnos, Wz, V, ba, sb)
    # axes: alpha1(20), alpha2(14), beta1(19), fi1(5)
    # Cy1(alpha1,beta1,fi1), Cy_nos1(alpha2,beta1), Cywz1(alpha1), dCywz_nos1(alpha2), dCy_sb1(alpha1)
    # post-negation: Cy1=-Cy1, Cy_nos1=-Cy_nos1, Cywz1=-Cywz1, dCywz_nos1=-dCywz_nos1, dCy_sb1=-dCy_sb1
    "GetCy.m": {
        "axes": ["alpha1", "alpha2", "beta1", "fi1"],
        "tables": ["Cy1", "Cy_nos1", "Cywz1", "dCywz_nos1", "dCy_sb1"],
    },
    # GetCz(alpha, beta, drn, del, dnos, Wx, Wy, V, l)
    # axes: alpha1(20), alpha2(14), beta1(19)  [NO fi axis]
    # Cz1(alpha1,beta1), Cz_nos1(alpha2,beta1), Czdel20(alpha1,beta1), Czdel20_nos(alpha2,beta1)
    # Czdrn30(alpha1,beta1), Czwy1(alpha1), dCzwy_nos1(alpha2), Czwx1(alpha1), dCzwx_nos1(alpha2)
    # post-negation: Czwy1=-Czwy1, dCzwy_nos1=-dCzwy_nos1  [Czwx1 NOT negated]
    "GetCz.m": {
        "axes": ["alpha1", "alpha2", "beta1"],
        "tables": [
            "Cz1", "Cz_nos1", "Czdel20", "Czdel20_nos",
            "Czdrn30", "Czwy1", "dCzwy_nos1", "Czwx1", "dCzwx_nos1",
        ],
    },
    # GetMx(alpha, beta, fi, drn, del, dnos, Wx, Wy, V, l)
    # axes: alpha1(20), alpha2(14), beta1(19), fi2(3) [-25,0,25]
    # mx1(alpha1,beta1,fi2), mx_nos1(alpha2,beta1), mxdel20(alpha1,beta1), mxdel20_nos(alpha2,beta1)
    # mxdrn30(alpha1,beta1), dmxbt1(alpha1), mxwy1(alpha1), dmxwy_nos1(alpha2), mxwx1(alpha1), dmxwx_nos1(alpha2)
    # post-negation: mxwy1=-mxwy1, dmxwy_nos1=-dmxwy_nos1  [mxwx1 NOT negated; dmxwx_nos1 NOT negated]
    "GetMx.m": {
        "axes": ["alpha1", "alpha2", "beta1", "fi2"],
        "tables": [
            "mx1", "mx_nos1", "mxdel20", "mxdel20_nos",
            "mxdrn30", "dmxbt1", "mxwy1", "dmxwy_nos1", "mxwx1", "dmxwx_nos1",
        ],
    },
    # GetMy(alpha, beta, fi, drn, del, dnos, Wx, Wy, V, l)
    # axes: alpha1(20), alpha2(14), beta1(19), fi2(3) [-25,0,25]
    # my1(alpha1,beta1,fi2), my_nos1(alpha2,beta1), mydel20(alpha1,beta1), mydel20_nos(alpha2,beta1)
    # mydrn30(alpha1,beta1), dmybt1(alpha1), mywy1(alpha1), dmywy_nos1(alpha2), mywx1(alpha1), dmywx_nos1(alpha2)
    # post-negation: my1=-my1, my_nos1=-my_nos1, mydel20=-mydel20, mydel20_nos=-mydel20_nos,
    #                mydrn30=-mydrn30, dmybt1=-dmybt1, mywx1=-mywx1, dmywx_nos1=-dmywx_nos1
    "GetMy.m": {
        "axes": ["alpha1", "alpha2", "beta1", "fi2"],
        "tables": [
            "my1", "my_nos1", "mydel20", "mydel20_nos",
            "mydrn30", "dmybt1", "mywy1", "dmywy_nos1", "mywx1", "dmywx_nos1",
        ],
    },
    # GetMz(alpha, beta, fi, dnos, Wz, V, ba, sb)
    # axes: alpha1(20), alpha2(14), beta1(19), fi1(5) [-25,-10,0,10,25], fi2(7) [-25,-10,0,10,15,20,25]
    # mz1(alpha1,beta1,fi1), mz_nos1(alpha2,beta1), mzwz1(alpha1), dmzwz_nos1(alpha2),
    # dmz1(alpha1), dmz_sb1(alpha1), eta_fi1(fi1), dmz_ds1(alpha1,fi2)
    "GetMz.m": {
        "axes": ["alpha1", "alpha2", "beta1", "fi1", "fi2"],
        "tables": [
            "mz1", "mz_nos1", "mzwz1", "dmzwz_nos1",
            "dmz1", "dmz_sb1", "eta_fi1", "dmz_ds1",
        ],
    },
    # GetThrust(H, M, Pa)
    # axes: H1(6) [0..50000 ft in meters], M1(6) [0..1], Pa1(3) [0,50,100]
    # Pt1(H1,M1,Pa1) - after Pt1=Pt1*4.4482216 (lbf->N conversion)
    "GetThrust.m": {
        "axes": ["H1", "M1", "Pa1"],
        "tables": ["Pt1"],
    },
}


def extract(matlab_dir: Path, out_dir: Path, table_spec: dict) -> None:
    if not table_spec:
        print(f"no tables defined for {matlab_dir.name}; skipping")
        return
    if not matlab_dir.exists():
        raise FileNotFoundError(f"matlab source directory not found: {matlab_dir}")
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
