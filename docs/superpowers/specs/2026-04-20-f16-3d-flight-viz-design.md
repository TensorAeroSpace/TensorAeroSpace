# 3D Flight Visualization for Nonlinear F-16 — Design

**Date:** 2026-04-20
**Status:** Approved (brainstorming)
**Scope:** Minimalistic 3D flight visualization for the nonlinear F-16 models (longitudinal 2-DoF and angular 6-DoF), exposed via the standard Gymnasium `env.render()` API. Plotly backend, rotatable camera, fading trail, time-synced charts below.
**Related issue:** #211

---

## Problem

The nonlinear F-16 models live as plain Python classes (`AngularF16`, `LongitudinalF16` family) and have no visualization. To inspect a flight today the user has to manually `matplotlib.subplots` the time-series of `alpha`, `q`, `theta`, control deflections, etc. There is no view of how the aircraft actually moves through space — no trajectory plot, no attitude visualization, no link between the aircraft motion and the controller's commanded inputs.

## Goal

After running an episode (manual or via an agent), call `env.render()` and get a 3D Plotly figure showing:
- Aircraft trajectory in space with a fading trail.
- A simple aircraft glyph (arrow / triangle) at the current pose, oriented by the Euler angles.
- A row of time-series charts below the 3D view showing the canonical states (alpha, beta, body rates) and control deflections.
- Rotatable camera (Plotly's standard 3D interaction).

## Non-goals

- Failure injection / failure markers (issue #211's full scope; out of this minimalistic version).
- Photorealistic aircraft mesh.
- VR / standalone web server / multi-aircraft scenes.
- Real-time visualization during agent training.
- MP4 export (PNG via `rgb_array` mode is the only export path).

---

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Visualization backend | Plotly (already a project dependency, native rotatable 3D in Jupyter, no extra deps). |
| Integration point | Standard Gymnasium `env.render()` — both 6-DoF and longitudinal models get a Gym wrapper. |
| Render modes | Single `render_mode` enum: `"human"`, `"rgb_array"`, `"live"`, `None`. |
| Update strategy | End-of-episode for `"human"` and `"rgb_array"`; per-step for `"live"`. |
| Position reconstruction | Inside the env (per step), based on configurable `airspeed` kwarg. The viz layer never reconstructs — it consumes a `(T, 3)` array. |
| File layout | Visualization is env-agnostic in `tensoraerospace/visualization/`; envs in `tensoraerospace/envs/` call into it. Two envs, one viz library. |

---

## Architecture

### File structure

```
tensoraerospace/envs/f16_angular.py              NEW: NonlinearAngularF16 gym env
tensoraerospace/envs/f16_longitudinal_nonlin.py  NEW or MODIFY: NonlinearLongitudinalF16 gym env
                                                  (creates if not present; otherwise extends)
tensoraerospace/visualization/__init__.py        NEW: package init, public re-exports
tensoraerospace/visualization/kinematics.py      NEW: position reconstruction helpers
tensoraerospace/visualization/flight_3d.py       NEW: build_flight_3d_figure() — env-agnostic
tensoraerospace/visualization/live.py            NEW: LivePlotlyRenderer (FigureWidget update)
tensoraerospace/__init__.py                      MODIFY: register "NonlinearAngularF16-v0"
                                                  (longitudinal env_id likely already registered;
                                                  verify and only add the missing one)

tests/envs/f16_angular_test.py                   NEW
tests/envs/f16_longitudinal_render_test.py       NEW
tests/visualization/kinematics_test.py           NEW
tests/visualization/flight_3d_test.py            NEW

example/visualization/example_f16_3d_angular.ipynb       NEW
example/visualization/example_f16_3d_longitudinal.ipynb  NEW

docs/en/guide/3d-visualization.md                NEW (with Russian counterpart)
```

### Layered responsibilities

```
NonlinearAngularF16 (env)        NonlinearLongitudinalF16 (env)
        |                                 |
        +------------+--------------------+
                     |
                     v
            kinematics.reconstruct_position_*()  (numpy-only, pure function)
                     |
                     v
            flight_3d.build_flight_3d_figure(positions, attitudes, time, chart_data)
                     |
       +-------------+-------------+
       v                           v
  fig.show()                LivePlotlyRenderer
  fig.to_image()            (FigureWidget extend_traces)
```

The viz module never imports envs; envs import viz. One-way dependency.

---

## Public API

### Env wrappers

```python
# tensoraerospace/envs/f16_angular.py
class NonlinearAngularF16(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "live"]}

    def __init__(
        self,
        x0: np.ndarray | None = None,
        dt: float = 0.01,
        integrator: str = "rk4",
        airspeed: float = 200.0,        # m/s, used for position reconstruction
        render_mode: str | None = None,
        chart_states: tuple[str, ...] = (
            "alpha", "beta", "wx", "wy", "wz",
            "stab", "ail", "dir",
        ),
        trail_length: int | None = None,  # None = full trail
    ): ...

    # gym API
    def reset(self, *, seed=None, options=None): ...
    def step(self, action): ...
    def render(self): ...      # dispatches on self.render_mode
    def close(self): ...

# Same surface for NonlinearLongitudinalF16; chart_states defaults differ:
#   ("alpha", "q", "theta", "stab")
```

The env also exposes:
- `env.unwrapped.position_history` — `(T, 3)` numpy array, rebuilt each step.
- `env.unwrapped.attitude_history` — `(T, 3)` array of `(roll, pitch, yaw)` in radians; for longitudinal roll = yaw = 0.
- `env.unwrapped.time_history` — `(T,)` array of seconds since `reset()`.
- `env.unwrapped.chart_history` — `dict[str, np.ndarray of shape (T,)]` with one entry per `chart_states` item.

### Render mode behaviour

- `render_mode=None` — `render()` returns `None`. No buffering overhead beyond the position/attitude/chart histories that the env tracks anyway.
- `render_mode="human"` — `render()` builds the figure once and calls `fig.show()`. End-of-episode use case (call after the rollout loop completes).
- `render_mode="rgb_array"` — `render()` builds the figure and returns `np.ndarray` of shape `(H, W, 3)` via `fig.to_image(format="png")` decoded with `PIL.Image`. Requires `kaleido` (added as a dependency).
- `render_mode="live"` — first `render()` call constructs a `LivePlotlyRenderer` (a `go.FigureWidget` displayed inline in Jupyter); subsequent calls extend the trail and chart traces with new data points. Notebook-only.

### Visualization library — env-agnostic

```python
# tensoraerospace/visualization/flight_3d.py

def build_flight_3d_figure(
    positions: np.ndarray,         # (T, 3) — x, y, z (or x, 0, z for longitudinal)
    attitudes: np.ndarray,         # (T, 3) — roll, pitch, yaw in radians
    time: np.ndarray,              # (T,) seconds
    chart_data: dict[str, np.ndarray],  # {state_name: (T,) array}
    *,
    trail_length: int | None = None,
    height: int = 800,
) -> go.Figure:
    """Plotly figure with a 3D scatter trail + N stacked subplots below.

    The 3D subplot uses go.Scatter3d for the trail (line+markers, opacity
    fades along time), plus a small Mesh3d arrow at the current pose
    oriented by the latest attitude row. The chart strip uses 2D Scatter
    in shared x-axis subplots created via plotly.subplots.make_subplots.
    """
```

```python
# tensoraerospace/visualization/kinematics.py

def reconstruct_position_6dof(
    state_history: np.ndarray,   # (T, n>=8) cols: alpha, beta, _, _, _, gamma, psi, theta, ...
    *,
    airspeed: float,
    dt: float,
    initial_position: np.ndarray | None = None,  # (3,), defaults to origin
) -> np.ndarray:                  # (T, 3)
    """Integrate position from velocity reconstructed from
    (V, alpha, beta, gamma, psi, theta) using the body→inertial DCM."""

def reconstruct_position_longitudinal(
    state_history: np.ndarray,   # (T, 4) cols: alpha, wz, stab, dstab
    *,
    airspeed: float,
    dt: float,
    initial_pitch: float = 0.0,            # theta(t=0) in radians
    initial_position: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:        # (positions (T, 3) with y=0, attitudes (T, 3) with roll=yaw=0)
```

```python
# tensoraerospace/visualization/live.py

class LivePlotlyRenderer:
    def __init__(self, height: int = 800): ...
    def init_from(self, positions, attitudes, time, chart_data) -> "FigureWidget": ...
    def extend(self, position_row, attitude_row, t, chart_row): ...  # incremental update
```

---

## Position reconstruction (math)

For both models, the per-step recipe:

1. Body-frame velocity from airspeed, AoA, sideslip:
   ```
   u_b = V * cos(alpha) * cos(beta)
   v_b = V * sin(beta)
   w_b = V * sin(alpha) * cos(beta)
   ```
2. Inertial-frame velocity via the standard body→inertial DCM built from `(roll, pitch, yaw) = (gamma, theta, psi)`:
   ```
   v_inertial = R(gamma, theta, psi) @ (u_b, v_b, w_b)
   ```
3. Integrate position:
   ```
   pos[t+1] = pos[t] + v_inertial * dt
   ```

Longitudinal model: `gamma = psi = beta = 0`, so the integration collapses to the vertical plane (`y = 0`). The longitudinal state vector is `[alpha, wz, stab, dstab]` (no `theta` — pitch angle is integrated from `wz`):
```
theta[0] = initial_pitch
theta[t+1] = theta[t] + wz[t] * dt
```
Then the same body→inertial integration is applied with `(roll=0, pitch=theta, yaw=0)`. `reconstruct_position_longitudinal` does the integration inline and returns BOTH positions and attitudes (the 6-DoF helper only returns positions because attitudes are already in its state vector).

This is purely kinematic; we do not model wind, gravity, or thrust effects on speed. The user provides `airspeed` as a constant for the episode. This matches the "minimalistic" scope — a real trajectory simulator is out of scope.

---

## Aircraft glyph

A minimal 3D glyph constructed from `Mesh3d` triangles:
- One body triangle (3 m long, pointing along body x-axis).
- Two wing triangles (1 m span each).
- One tail triangle (0.5 m vertical).

Total ~10 vertices, 4 faces. Built once in `build_flight_3d_figure`, rotated by the latest attitude row, translated by the latest position row. No animation in `"human"`/`"rgb_array"` (single static glyph at the final pose).

In `"live"` mode the glyph is updated in-place every `render()` call via `FigureWidget.batch_update`.

---

## Trail rendering

`go.Scatter3d` line in 3D space with:
- Colour gradient: `marker.color = time` with a sequential colormap (Viridis), so direction of flight is obvious.
- Opacity fade: not natively supported by Scatter3d; achieved by clipping to `trail_length` (last N seconds) when set.
- `mode="lines+markers"` with sparse markers (every N-th point) to avoid clutter.

For 60 s of flight at dt=0.01 → 6000 points. Plotly handles ~10k 3D points smoothly. No decimation needed at this scale.

---

## Charts strip

Built with `plotly.subplots.make_subplots(rows=1+N, cols=1, specs=[[{"type": "scene"}], *N rows])`. The 3D scene takes the top row (large), each chart state takes a thin row below. Shared x-axis (time) across the chart rows.

For longitudinal model the default is 4 charts (alpha, q, theta, stab); for angular 8 (alpha, beta, wx, wy, wz, stab, ail, dir). Configurable via `chart_states` env kwarg.

---

## Dependencies

Add `kaleido` to `pyproject.toml` (only used by `rgb_array` mode). All other deps already present:
- `plotly` ^6.2.0 — already in deps
- `numpy`, `gymnasium` — already in deps

`kaleido` is import-on-use inside `_render_rgb_array()` (with an `ImportError` message pointing the user to install it). Without `rgb_array` mode, `kaleido` is never imported.

---

## Testing strategy

1. **Kinematics unit tests** (`tests/visualization/kinematics_test.py`):
   - Straight-and-level flight (alpha=beta=gamma=psi=theta=0, airspeed=V) → position grows linearly along inertial x.
   - Pure pitch (theta=π/4, others=0) → trajectory in x-z plane only.
   - Pure roll (gamma=π/4, others=0, lvl flight) → still straight along x (roll alone doesn't move CG).
   - Coordinated turn (psi growing linearly, theta=0) → circular trajectory in xy plane.

2. **Figure builder unit tests** (`tests/visualization/flight_3d_test.py`):
   - Returns a `go.Figure`.
   - Has the right number of subplots (1 scene + N chart rows).
   - Trail trace has expected length.
   - Asserts on figure structure, not pixel output.

3. **Env unit tests** (`tests/envs/f16_angular_test.py`, `tests/envs/f16_longitudinal_render_test.py`):
   - `gym.make("NonlinearAngularF16-v0")` succeeds.
   - `env.reset()` returns observation of correct shape.
   - `env.step(action)` advances state; `env.unwrapped.position_history` grows by one row.
   - `env.render()` with `render_mode=None` returns `None` (no figure built).
   - `env.render()` with `render_mode="human"` returns a `go.Figure` (don't call `.show()` in tests; just check return type).
   - `env.render()` with `render_mode="rgb_array"` returns a `np.ndarray` of shape `(H, W, 3)` (skip if `kaleido` not installed).
   - `env.render()` with `render_mode="live"` returns a `FigureWidget` (skip if `ipywidgets` not installed).

4. **No notebook-execution tests.** The example notebooks are demonstration material; their execution is checked by the existing nbqa/notebook CI lane if present, otherwise manually.

---

## Out of scope (explicitly)

- Failure injection layer (issue #211 long-term scope; not here).
- Failure markers on the chart strip / trail.
- MP4 export (`rgb_array` is the only export, returns one PNG of the final frame).
- Multi-episode comparison views.
- Adapting the visualization for other aircraft (B747, F4C, ELV, etc.). The `flight_3d` module is env-agnostic; adapting requires only writing the right env wrapper, but is not part of this PR.
- Wind / gust models in position reconstruction.
- Saving / replaying recorded episodes — env state buffer is in-memory only.

---

## Acceptance criteria

1. `gym.make("NonlinearAngularF16-v0", render_mode="human")` works; `env.reset()` and `env.step(action)` follow standard Gymnasium semantics.
2. After running a 30-second episode and calling `env.render()`, the user gets a Plotly figure with:
   - A 3D scene containing the aircraft trail.
   - Time-series charts below for the configured `chart_states`.
   - A rotatable camera (Plotly default).
3. Same for `NonlinearLongitudinalF16-v0` (existing env_id), with the trail collapsing to the vertical plane.
4. `render_mode="rgb_array"` returns a numpy array of shape `(H, W, 3)` (with `kaleido` installed).
5. `render_mode="live"` produces a `FigureWidget` that updates per-step in a Jupyter notebook.
6. Two example notebooks render inline and serve as the user-facing usage demo.
7. Documentation page at `docs/en/guide/3d-visualization.md` (with Russian counterpart) covers usage with at least one screenshot or inline plot.
8. New tests pass; `pytest tests/` continues to pass overall.
