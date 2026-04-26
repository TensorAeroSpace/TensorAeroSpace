# 3D Flight Visualization

Both nonlinear F-16 environments expose a 3D flight visualization via
the standard Gymnasium `env.render()` API. The view shows the inertial
trajectory as a fading trail in 3D, the aircraft as a glyph at the
current pose, and a strip of time-synced charts below for the canonical
states and control deflections.

## Quick start

```python
import gymnasium as gym
import numpy as np
import tensoraerospace  # registers env_ids

env = gym.make(
    "NonlinearAngularF16-v0",
    initial_state=np.zeros(14),
    number_time_steps=200,
    dt=0.02,
    airspeed=220.0,
    render_mode="human",
)
env.reset()
for _ in range(200):
    env.step(np.array([2.0, 5.0, 0.0]))  # stab, ail, dir (deg)

fig = env.render()
fig.show()
```

The same pattern works for `NonlinearLongitudinalF16-v0` — the action
shape is `(1,)` (just elevator) and the trail collapses to the vertical
plane.

## Render modes

| `render_mode` | What happens |
|---|---|
| `None` (default) | `render()` returns `None`. Position, attitude, and chart histories are still tracked, so you can render later by setting `env.unwrapped.render_mode` and calling `render()`. |
| `"human"` | `render()` returns a `plotly.graph_objects.Figure`. Call `.show()` to open in browser / inline in Jupyter. |
| `"rgb_array"` | `render()` returns a `numpy` array of shape `(H, W, 3)` (PNG decoded with PIL). Requires `kaleido` (already in deps) and `Pillow`. |
| `"live"` | `render()` returns a `plotly.graph_objects.FigureWidget` that updates per call. Display once via `from IPython.display import display; display(env.render())`, then call `env.render()` after each `env.step()` to extend the trail and chart traces in place. Notebook-only (requires `anywidget`, included in deps). |

## Configuration kwargs

| Kwarg | Default | Description |
|---|---|---|
| `airspeed` | `200.0` (m/s) | True airspeed used for kinematic position reconstruction. Constant per episode. |
| `render_mode` | `None` | One of the modes above. |
| `chart_states` | algo-specific | Names of model state channels to plot below the 3D view. Angular default: `("alpha", "beta", "wx", "wy", "wz", "stab", "ail", "dir")`. Longitudinal default: `("alpha", "wz", "stab")`. |
| `trail_length` | `None` | If set, keep only the last N points of the trail. Useful for long episodes where the full path becomes visually cluttered. |

## How position is reconstructed

The F-16 model state vectors do not contain inertial position or true
airspeed. The visualization reconstructs position by integrating
velocity from the assumed constant `airspeed` and the aerodynamic
angles + Euler angles from the state:

1. Body-frame velocity: `(V·cos α·cos β, V·sin β, V·sin α·cos β)`.
2. Body→inertial DCM from the Euler angles (yaw-pitch-roll).
3. Forward-Euler integration: `pos[t+1] = pos[t] + R · v_body · dt`.

For the longitudinal model the pitch angle θ is integrated from `wz`
(no θ in the state). Roll, yaw, and sideslip stay at zero.

This is intentionally a simple kinematic recipe — no wind, gravity, or
thrust modelling. For a high-fidelity trajectory, integrate the full
6-DoF translation outside the env.

## Tips

- **Rotate the camera:** Plotly's 3D scenes are rotatable by default
  (drag with the mouse).
- **Compare runs:** call `env.render()` to get a `go.Figure`, then add
  more traces to it before showing.
- **Export to PNG:** switch to `render_mode="rgb_array"` and save with
  `imageio` or `PIL.Image.fromarray(...).save("flight.png")`.
- **Only see the recent trail:** set `trail_length=200` to keep just
  the last 200 points (avoids visual clutter on long episodes).
- **Live updates during a notebook rollout:** see `render_mode="live"`
  in the Render modes table above.

## Example notebooks

- `example/visualization/example_f16_3d_angular.ipynb` — open-loop
  banked turn on the 6-DoF angular env.
- `example/visualization/example_f16_3d_longitudinal.ipynb` — pitch-up
  + neutral on the 2-DoF longitudinal env (trail in the vertical plane).
