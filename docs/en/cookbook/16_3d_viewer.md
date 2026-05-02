# Recipe 16 — Interactive 3D viewer for F-16 flights

The `tensoraerospace.visualization.three_d` package turns a completed
episode into a self-contained interactive WebGL viewer: parametric F-16
geometry, mouse-controlled camera, animated trajectory trail, and live
damage visualization driven by the same `DamageProfile` API used by
the simulation.

**Plumbing.** [Aircraft damage modeling](../model/aircraft-damage-modeling.md) ·
**Source.** `tensoraerospace/visualization/three_d/` ·
**Runnable.** `example/visualization/example_3d_viewer_f16.py`.

## Quick start

```python
import numpy as np
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageEvent, DamageProfile,
)
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

profile = DamageProfile(events=[
    DamageEvent(20.0, "section_loss",
                payload={"section": "left_tip", "loss_fraction": 1.0}),
])

env = NonlinearAngularF16(
    initial_state=np.zeros(14), number_time_steps=6010,
    dt=0.01, airspeed=200.0, split_stab=True,
    damage_profile=profile,
    render_mode="3d_web",   # NEW
)
env.reset()
for _ in range(6000):
    env.step(np.zeros(4))

env.render()
```

In a Python script, `env.render()` opens the user's default browser on
the generated `flight.html`. In a Jupyter notebook it returns an
`IPython.display.HTML` so the WebGL canvas embeds inline in the cell.

## What the viewer shows

* **Aircraft mesh.** Parametric 13-section F-16 built from the same
  `BaseGeometry` YAML that drives the physics — wings, stabilator,
  vtail, rudder, ailerons, fuselage. Each section is a named `Object3D`
  so damage events target individual parts.
* **Trajectory trail.** Blue line growing along the inertial path as
  the animation advances.
* **HUD overlay (top-left).** Current time, α, β, ωx, ωz, and any
  damage event labels triggered in the past 1.5 s.
* **Camera presets (bottom).**
  - **Free** — full mouse orbit (default).
  - **Chase** — locks 25 m behind / 8 m above the aircraft, follows
    its attitude.
  - **Top-down** — orthogonal-style overhead view.
* **Timeline + speed.** Scrub anywhere through the episode; replay at
  0.25× / 0.5× / 1× / 2× / 4×.

## How damage shows up

The viewer reads `flight_log.damage_state_history` (binary search per
frame) and applies three classes of effect:

| Damage type | Visual |
|---|---|
| `section_loss` | Section colour lerps toward red, opacity fades to 0; the mesh hides at `f >= 1.0`. |
| `control_failure` (jam / efficiency_loss / lost / free_floating) | Yellow emissive outline on the affected surface. |
| `engine_failure` | Orange exhaust cone behind the fuselage scales with `thrust_factor`; `hard_failure=True` removes the cone. |

Symmetric loss preserves bilateral symmetry — both wing tips fade at
the same rate. Asymmetric loss (one side only) leaves the surviving
half intact while the other vanishes, which combined with the rolling
moment from the strip-theory aero correction gives a clear visual
explanation of the post-damage roll-rate behaviour.

## Programmatic control

The high-level entrypoint is `env.render()`, but you can also build
the HTML manually:

```python
from tensoraerospace.visualization.three_d import (
    build_flight_log, build_html, save_html, render,
)

log = build_flight_log(env)        # JSON-serializable dict
html = build_html(log)             # self-contained HTML string
path = save_html(log, "flight.html")   # write to disk
render(env, save_to="flight.html", open_in_browser=False)  # both
```

For Jupyter, force inline:

```python
from tensoraerospace.visualization.three_d import render
render(env, inline=True)   # → IPython.display.HTML
```

## Self-contained output

The generated HTML inlines:
* three.js v0.146 (UMD bundle, ~600 KB)
* `OrbitControls` (~26 KB)
* The viewer JS / CSS (~10 KB)
* The flight log JSON (~50–500 KB depending on episode length)

Total per file: ~700 KB to ~1 MB. **No CDN, no localhost server, no
network at runtime.** You can email the file or commit it to docs.

## Screenshot

![3D viewer rendering an F-16 episode with left-wingtip loss](img/16_3d_viewer_screenshot.png)

*Placeholder — screenshot will be added once the viewer has been rendered.*

## See also

- [Aircraft damage modeling](../model/aircraft-damage-modeling.md)
- [Recipe 15 — ET-DHP under in-flight damage](15_etdhp_damage.md)
- `example/visualization/example_3d_viewer_f16.py` — runnable demo
