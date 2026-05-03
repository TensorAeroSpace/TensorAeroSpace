# Aircraft Damage Modeling

The damage subsystem turns the F-16 from a fixed plant into a
**time-varying control object**. It lets you schedule failures during a
simulation — wing tip loss, jammed control surfaces, engine failure,
structural changes — and the env recomputes the aircraft's mass, inertia,
aerodynamic coefficients, and control-surface effectiveness in real time.
The agent under control then faces a different plant from the moment a
damage event fires.

Currently supported on the **nonlinear F-16** —
[longitudinal](f16_nonlinear_longitudinal.md) and
[6-DoF angular](f16_nonlinear_angular.md). The same `damage_profile` API
plugs into both env variants and into any controller / RL agent that uses
them. The linear F-16, B-747, and other plants are out of scope for now.

## Documentation structure

This topic is split across three pages:

- **This page** — overview, quick start, and limits.
- [Implementation and examples](aircraft-damage-modeling-code.md) — code
  layout (the three layers: section-based geometry, `DamageState` +
  events, runtime physics recompute), built-in scenarios, custom and
  random profiles, and worked examples of adaptive RL agents iADP and
  ET-DHP.
- [Damage modeling mathematics](aircraft-damage-modeling-math.md) — all
  formulas: mass / CG / inertia recompute via Huygens-Steiner, strip-theory
  deltas for the six aerodynamic coefficients, control-surface and engine
  failure models, and the resulting 6-DoF ODEs with damage.

## Quick start

```python
import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    WING_STRIKE_LEFT_TIP,
)
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

env = NonlinearAngularF16(
    initial_state=np.zeros(14),
    number_time_steps=2000,
    damage_profile=WING_STRIKE_LEFT_TIP,
    split_stab=True,
)
obs, _ = env.reset()
for _ in range(2000):
    obs, r, term, trunc, info = env.step(np.zeros(4))
    if info.get("damage_events_triggered"):
        print(info["damage_events_triggered"])
```

A runnable version of this snippet ships at `example/f16_damage_dogfight_demo.py`.

## What is modelled

| Failure class | What changes | Where described |
|---------------|-------------|-----------------|
| Section loss (wing, stabilator, vtail) | mass $m$, wing area $S$, span $b$, Mean Aerodynamic Chord (MAC), centre of gravity $\mathbf{r}_{cg}$, inertia tensor $\mathbf{J}$, aerodynamic coefficients | [code](aircraft-damage-modeling-code.md#layer-3--runtime-physics-recompute), [math](aircraft-damage-modeling-math.md) |
| Control-surface failure (jam / efficiency loss / lost) | Control vector $\mathbf{u}_{cmd} \to \mathbf{u}_{eff}$ | [code](aircraft-damage-modeling-code.md#control_failure), [math](aircraft-damage-modeling-math.md#control-surface-failures) |
| Engine failure (partial / total) | Effective thrust $T_{eff}$ | [code](aircraft-damage-modeling-code.md#engine_failure), [math](aircraft-damage-modeling-math.md#engine) |
| Structural changes (dropped stores, ice) | $\Delta$ mass / CG / inertia | [code](aircraft-damage-modeling-code.md#structural_change) |

## Limits

- The linear F-16, B-747, and other models are not yet supported.
- Aerodynamic corrections do not capture wake interference or stalled-flow
  perturbations beyond what the base lookup tables already encode.
- Wing flexibility / aeroelastic effects are not modelled.
- Cascading failures (one event triggering another) are not yet
  implemented; schedule them explicitly in the profile.
