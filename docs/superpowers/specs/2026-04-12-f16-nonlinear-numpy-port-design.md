# F-16 Nonlinear Model: Matlab → NumPy Port

**Date:** 2026-04-12
**Scope:** `tensoraerospace/aerospacemodel/f16/nonlinear/{longitudinal,angular}`

## Problem

The two F-16 nonlinear models (`LongitudinalF16`, `AngularF16`) are thin Python wrappers around `.m` files executed via `matlab.engine`. Consequences:

- **Hard dependency on a paid Matlab installation** to run any RL example that uses F-16 dynamics.
- **Per-step latency dominated by IPC** to the Matlab engine, making RL training impractical.
- **Distribution friction:** the package cannot be installed and used in CI / cloud envs without Matlab.

## Goal

Replace the Matlab runtime with a pure-NumPy implementation that:

1. Removes `matlab.engine` from runtime dependencies.
2. Preserves the public API (`LongitudinalF16(x0).run_step(u)`, `AngularF16(...).run_step(u)`) so existing examples and trained agents keep working with minimal call-site changes.
3. Is fast enough for RL training on CPU (target: ≥10× faster than the matlab.engine version per step).
4. Retains the existing forward-Euler integration as the default (numerical compatibility), with an opt-in RK4 integrator.

## Non-goals

- GPU / batched / differentiable simulation. The architecture leaves a clean path to a torch port, but it is out of scope.
- Adding new aero data, new control surfaces, or new dynamics terms.
- Refactoring `ModelBase` or other shared infrastructure.

## Architecture

Pure functions for physics, thin classes for state management.

```
tensoraerospace/aerospacemodel/f16/nonlinear/
├── _integrators.py           NEW: euler(f, x, u, t, dt, p), rk4(...)
├── longitudinal/
│   ├── model.py              REWRITTEN: LongitudinalF16, same public API
│   ├── dynamics.py           NEW: f16_ode_long(x, u, t, params) -> dx
│   ├── aero.py               NEW: get_cy(...), get_mz(...) + lookup tables
│   ├── params.py             NEW: F16LongParameters dataclass + default_parameters()
│   └── matlab_code/          KEPT as reference, not imported at runtime
├── angular/
│   ├── model.py              REWRITTEN: AngularF16
│   ├── dynamics.py           NEW: f16_ode_6dof(x, u, t, params) -> dx
│   ├── aero.py               NEW: get_cx/cy/cz/mx/my/mz, get_thrust, engine_power_level
│   ├── frames.py             NEW: body_to_wind(), wind_to_body()
│   ├── params.py             NEW: F16AngularParameters dataclass + default_parameters()
│   └── matlab_code/          KEPT as reference
```

### Why this layout

- **Pure functions in `dynamics.py`/`aero.py`** are easier to unit-test (no class state to set up), easier to vectorize later, and trivially swappable to torch if needed.
- **`params.py` as dataclass** replaces the matlab struct. ISA-atmosphere computation moves into `default_parameters()`, mirroring `airplane_parameters.m`.
- **`_integrators.py` shared** between longitudinal and angular keeps the integrator implementation in one place.

### Data flow per step

```
LongitudinalF16.run_step(u)
    │
    └─> integrator(f16_ode_long, x_prev, u, t, dt, params)
            │
            └─> f16_ode_long(x, u, t, params)
                    │
                    ├─> get_cy(alpha, beta, stab, lef, wz, V, bA, sb)   ← aero.py
                    ├─> get_mz(alpha, beta, stab, lef, wz, V, bA, sb)   ← aero.py
                    └─> assembles dx = [dalpha, dwz, dstab, ddstab]
```

## Component design

### `aero.py` — aerodynamic coefficients

The matlab `GetC*.m` / `GetM*.m` files are 200–260 lines each and consist of:

1. Hard-coded numeric lookup tables (likely from NASA TP-1538 / AFIT F-16 datasets) over grids of `(alpha, beta, stab)` etc.
2. **Cubic smoothing spline** (`csaps` with smoothing parameter ≈ `1 - 1e-6`, i.e. essentially exact interpolation) for 2-D and 3-D tables, and **piecewise cubic Hermite** (`pchip`) for 1-D tables.
3. Linear combinations of base + derivative terms (e.g.,
   `Cy = Cy_base(alpha, beta, stab) + dCy_nos*(dnos/25°) + Cywz*(wz*ba)/(2V) + dCy_sb*(sb/60°)`).

**Porting strategy:**

- Parse each `.m` file with a small one-shot Python script (`scripts/extract_f16_aero.py`, not in the package) that extracts the numeric tables into `.npz` files (one per `Get*.m`). The parser is hand-rolled regex over the well-formed matlab matrix syntax used in these files; no external dependency.
- Tables are committed to the repo at `tensoraerospace/aerospacemodel/f16/nonlinear/{longitudinal,angular}/aero_tables/<name>.npz`.
- At import time, `aero.py` loads the `.npz` files once (module-level constants) and constructs interpolator objects:
  - **3-D and 2-D tables** (e.g., `Cy(alpha, beta, fi)`, `Cy_nos(alpha, beta)`): `scipy.interpolate.RegularGridInterpolator(method="cubic")` (scipy ≥ 1.9). This is the closest scipy equivalent to matlab `csaps` with near-zero smoothing — both produce a tensor-product cubic spline that interpolates the data. Edges clamped manually before lookup.
  - **1-D tables** (e.g., `Cywz(alpha)`, `dCy_sb(alpha)`): `scipy.interpolate.PchipInterpolator`. This matches matlab `pchip` byte-for-byte (same Fritsch-Carlson algorithm).
- Each `get_*` function is a thin wrapper that calls the interpolators and assembles the final coefficient using the same algebra as the matlab version.

**Edge handling:** Matlab interpolation evaluates the spline outside the grid as well, but in practice the F-16 sim stays inside the grid. To avoid extrapolation blow-ups, clamp inputs to grid bounds before lookup: `np.clip(point, grid_min, grid_max)`. Document this as a behavioral difference (matlab extrapolates the cubic; we clamp). For the published F-16 envelope this should produce identical results.

### `params.py` — parameters dataclass

```python
@dataclass
class F16LongParameters:
    m: float = 9295.44
    S: float = 27.87
    bA: float = 3.45
    Jz: float = 75673.6
    rcgx: float = -0.05 * 3.45
    Tstab: float = 0.03
    Xistab: float = 0.707
    maxabsstab: float = math.radians(25)
    maxabsdstab: float = math.radians(60)
    lef: float = 0.0
    sb: float = 0.0
    g: float = 9.80665
    Oy: float = 3000.0
    V: float = 150.0
    q: float = field(init=False)  # computed from ISA atmosphere

    def __post_init__(self):
        self.q = _isa_dynamic_pressure(self.Oy, self.V, self.g)
```

`F16AngularParameters` is the analogous superset (adds `Jx, Jy, Jxz, Sw, lA, ...`).

### `dynamics.py` — ODE right-hand side

`f16_ode_long(x, u, t, params)` is a direct line-by-line port of `F16ODE.m`:

```python
def f16_ode_long(x, u, t, params):
    alpha, wz, stab, dstab = x
    stab_act = u[0]
    p = params

    cy = get_cy(alpha, 0.0, stab, p.lef, wz, p.V, p.bA, p.sb)
    mz = get_mz(alpha, 0.0, stab, p.lef, wz, p.V, p.bA, p.sb)

    Y = p.q * p.S * cy
    Mz = p.q * p.S * p.bA * mz

    Ry = Y
    MRz = Mz + p.rcgx * Ry

    dwz = MRz / p.Jz
    dalpha = wz - (Ry - p.m * p.g) / (p.m * p.V)

    # Actuator with rate + position limits
    dstab_clamped = np.clip(dstab, -p.maxabsdstab, p.maxabsdstab)
    stab_act_clamped = np.clip(stab_act, -p.maxabsstab, p.maxabsstab)
    ddstab = (-2 * p.Tstab * p.Xistab * dstab - stab + stab_act_clamped) / (p.Tstab ** 2)

    return np.array([dalpha, dwz, dstab_clamped, ddstab])
```

`f16_ode_6dof` for the angular variant follows the matlab `F16ODE.m` in `angular/matlab_code/` (full 6-DoF: positions, body rates, Euler angles, three control surfaces, throttle/power state).

### `_integrators.py`

```python
def euler(f, x, u, t, dt, params):
    return x + dt * f(x, u, t, params)

def rk4(f, x, u, t, dt, params):
    k1 = f(x, u, t, params)
    k2 = f(x + 0.5 * dt * k1, u, t + 0.5 * dt, params)
    k3 = f(x + 0.5 * dt * k2, u, t + 0.5 * dt, params)
    k4 = f(x + dt * k3, u, t + dt, params)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
```

### `model.py` — public class

Same constructor signature as today. Internal changes:

- No `matlab.engine.start_matlab()`, no `addpath`, no `eng.airplane_parameters()`.
- `self.param = default_parameters()` returns the dataclass.
- `run_step(u)` accepts `list | np.ndarray` (no `matlab.double`). The dimensionality check stays.
- `integrator: Literal["euler","rk4"] = "euler"` added to `__init__`. Stores a reference to `_integrators.euler` or `_integrators.rk4`.
- State history stays as `list[np.ndarray]`.

## Public API impact

| Today | After |
|---|---|
| `LongitudinalF16(x0)` | `LongitudinalF16(x0)` — unchanged |
| `LongitudinalF16(x0, integrator=...)` | NEW optional kwarg |
| `model.run_step(matlab.double([[0]]))` | `model.run_step([[0]])` or `np.array(...)` |
| `model.eng`, `model.matlab_files_path` | REMOVED |
| `model.get_param()` returns matlab struct | returns `F16LongParameters` dataclass |

Examples in `example/reinforcement_learning/` and `example/dynamic-programming/` that currently pass `matlab.double` need a one-line change. We will scan and update them.

## Verification (no Matlab available)

Property-based tests, since we cannot diff against the matlab oracle:

1. **Trim test.** For a known trim point (level flight: `alpha=alpha_trim, wz=0, stab=stab_trim, dstab=0`), `f16_ode_long(...)` returns `||dx|| < 1e-3`. Trim point computed once via a 2-D `scipy.optimize.fsolve` over `(alpha, stab)` such that `dwz = 0` and `dalpha = 0`, hardcoded into the test.
2. **Sign tests.** Positive elevator deflection produces negative pitch acceleration (and vice versa). Increasing `alpha` from trim produces positive `dwz` (statically stable) — or negative, depending on the F-16's open-loop stability; whichever sign the matlab tables produce, we lock in.
3. **Actuator dynamics.** Step input on `stab_act`: with damping `Xi=0.707`, `stab` reaches 95% of target in `~3*Tstab` seconds; no overshoot beyond 5%. Limits `|stab| ≤ maxabsstab` and `|dstab| ≤ maxabsdstab` are enforced.
4. **Integrator consistency.** Run RK4 and Euler with `dt=1e-4` from the same x0 over 0.1 s with zero input; trajectories agree to `1e-4` (sanity check that both integrators see the same RHS).
5. **Aero table sanity.** For each `get_*` function: at the grid corners, the value equals the corresponding table entry exactly (verifies that interpolators are wired to the right axes).
6. **Determinism / regression.** After the first green CI run, snapshot a 1-second open-loop trajectory to `tests/.../snapshots/long_trajectory.npz`. Future runs must reproduce it bit-for-bit (catches accidental changes).
7. **Aero parity to matlab tables (extraction-time).** The extraction script that reads `.m` files into `.npz` also writes a `tests/aero_extraction_checksums.json` file with row-count and SHA256 of each table. The unit test asserts the loaded `.npz` matches.

The angular variant gets the same suite, adapted to its state vector.

## Performance target

The matlab.engine version pays an IPC round-trip per step (~ms range). The numpy port should hit ~20–50 µs per step on a modern laptop CPU, i.e., **~50× faster** than the current implementation. We will not write a microbenchmark suite, but the executing-plans phase will include a one-off timing check.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Aero tables transcribed incorrectly (3000+ numeric lines) | Automated extraction via `.m` parser script + checksum tests; never type tables by hand |
| `RegularGridInterpolator` edge behavior differs from matlab `interp2/interpn` | Explicit `np.clip` of inputs to grid bounds before lookup |
| Forward Euler with `dt=0.01` is unstable for actuator (`Tstab=0.03`) | Already used by matlab version; replicating same behavior is goal. Document RK4 as recommended for `dt > 0.005` |
| Examples break because they pass `matlab.double` | Grep all examples, update to plain lists/arrays, run them once |
| `matlab.engine` removed from deps but still imported elsewhere in package | Grep entire codebase for `import matlab` / `from matlab` and confirm all usages are inside the two model files we are rewriting |

## Out-of-scope (explicitly)

- Replacing the linear F-16 model.
- Replacing F-16 models in `tensoraerospace/envs/` (those should keep working transparently if the `LongitudinalF16` / `AngularF16` API is preserved).
- New unit / coordinate-frame conventions.
- Documentation rewrite beyond updating the docstrings already present in `model.py`.

## Definition of done

1. Importing `tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal` does not import `matlab`.
2. `pip install -e .` works in a clean venv with no Matlab installed.
3. `pytest tests/aerospacemodel/f16/nonlinear/` passes (new test files).
4. The two `example/` scripts that exercise F-16 (longitudinal / angular) run end-to-end on a numpy-only install.
5. `matlab.engine` removed from `pyproject.toml`.
6. `matlab_code/` directories preserved as reference but no Python file imports them.
