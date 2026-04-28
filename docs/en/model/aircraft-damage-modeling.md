# Aircraft Damage Modeling

The damage subsystem allows you to schedule failures during a simulation —
wing tip loss, jammed control surfaces, engine failure, structural changes —
that update the aircraft's mass, inertia, aerodynamic coefficients, and
control-surface effectiveness in real time.

Currently supported on the **nonlinear F-16** (longitudinal and 6-DoF angular).

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

## Built-in scenarios

| Preset | Trigger | Effect |
|--------|---------|--------|
| `WING_STRIKE_LEFT_TIP` | t=10 s | Full loss of left wingtip |
| `WING_STRIKE_LEFT_HALF` | t=10 s | Left tip + 50% mid-section |
| `ELEVATOR_JAM_NEUTRAL` | t=5 s  | Both stabilator halves jammed at neutral |
| `ELEVATOR_JAM_PITCH_UP` | t=5 s | Both jammed at +10° |
| `RUDDER_LOST` | t=5 s | Rudder lost |
| `ENGINE_FLAMEOUT` | t=5 s | Engine flameout (thrust = 0) |
| `BIRDSTRIKE_COMPOUND` | t=5 s | 20% right wing + 70% engine loss |

Import them from `tensoraerospace.aerospacemodel.f16.nonlinear.damage`.

## Custom scenarios

```python
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageEvent, DamageProfile,
)

profile = DamageProfile(events=[
    DamageEvent(8.0, "section_loss",
                payload={"section": "right_mid", "loss_fraction": 0.4}),
    DamageEvent(15.0, "engine_failure",
                payload={"thrust_factor": 0.3}),
])
```

Available event types:

- `section_loss` — payload `{"section": str, "loss_fraction": float in [0,1]}`
- `control_failure` — payload `{"surface": str, "mode": str, ...mode-specific}`
  Modes: `"jam"` (with `jam_position_rad`), `"efficiency_loss"` (with `efficiency`), `"lost"`, `"free_floating"`
- `engine_failure` — payload `{"thrust_factor": float, "hard_failure": bool}`
- `structural_change` — payload `{"mass_delta_kg": float, "cg_shift_m": tuple, "inertia_delta": tuple}`

## Random profiles for RL

```python
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    RandomDamageProfileGenerator,
)

generator = RandomDamageProfileGenerator(
    event_types=["section_loss", "control_failure"],
    time_range=(5.0, 25.0),
    severity_range=(0.1, 1.0),
    num_events_range=(1, 2),
    seed=42,
)

profile = generator.sample()
obs, info = env.reset(options={"damage_profile": profile})
```

## Observable damage

By default, the agent does not observe the damage state — it must infer
deterioration from the dynamics. Pass `damage_observable=True` to extend
the observation vector with section-loss fractions and engine thrust
factor:

```python
env = NonlinearAngularF16(
    initial_state=np.zeros(14),
    number_time_steps=2000,
    damage_profile=profile,
    damage_observable=True,
    split_stab=True,
)
```

The observation grows from 14 to `14 + N_sections + 1` floats.

## Architecture and physical model

Implementation lives at
`tensoraerospace/aerospacemodel/f16/nonlinear/damage/`. The design
document is at
`docs/superpowers/specs/2026-04-28-aircraft-damage-modeling-design.md`.

Key points:

- **Parametric geometry recompute** — at each damage event, mass, wing
  area, span, MAC, centre of gravity, and the inertia tensor are
  recomputed from per-section contributions using Huygens-Steiner.
- **Strip-theory aerodynamic corrections** — each section contributes
  proportionally to lost lift/drag/moment when damaged. Approximate
  fidelity ~10–20 % vs full vortex-lattice methods.
- **Asymmetric damage requires the angular 6-DoF model** with
  `split_stab=True` (4-input action: `[stab_left, stab_right, ail, dir]`).
  Symmetric damage works in both longitudinal and angular envs.
- **Bit-identical baseline** — without a `damage_profile`, env behaviour
  is byte-for-byte identical to the un-damaged baseline. Existing tests,
  trained agents, and saved trajectories work unchanged.

## Resetting damage between episodes

`env.reset()` clears all damage and re-baselines parameters. To override
the profile per episode:

```python
obs, info = env.reset(options={"damage_profile": new_profile})
```

This is the standard pattern for RL training with randomised damage.

## Limits

- Linear F-16, B-747, and other models are not yet supported.
- Aerodynamic corrections do not capture wake interference or stalled-flow
  perturbations beyond what the base lookup tables already encode.
- Wing flexibility / aeroelastic effects are not modelled.
- Cascading failures (one event triggering another) are not yet
  implemented; schedule them explicitly in the profile.
