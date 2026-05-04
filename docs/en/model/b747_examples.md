# B-747: usage examples

Practical recipes for the
[`tensoraerospace.aerospacemodel.b747.nonlinear`](#nonlinear) and
[`tensoraerospace.aerospacemodel.b747.nonlinear.damage`](#damage)
modules. All snippets are self-contained. Theory:
[Boeing 747-100 (Nonlinear 6-DoF)](b747_nonlinear.md).

---

## Nonlinear 6-DoF { #nonlinear }

### 1. Trim cruise at any altitude / airspeed

`trim(h, V)` solves (u̇, ẇ, q̇) = 0 by Newton-Raphson and returns a
`TrimResult` with `(α, δ_e, δ_T)` plus a ready 12-state vector.

```python
import numpy as np
from tensoraerospace.aerospacemodel.b747.nonlinear import (
    NonlinearB747, B747Configuration, trim,
)

# Cruise at FL200 (20 000 ft), V = 400 KTAS ≈ 674 ft/s
result = trim(altitude_ft=20_000.0, V_ft_s=674.0)
print(f"α = {np.rad2deg(result.alpha_rad):+.2f}°")
print(f"δ_e = {np.rad2deg(result.elevator_rad):+.2f}°")
print(f"δ_T = {result.throttle:.3f}")

# Hand the trim straight to the model
m = NonlinearB747(x0=result.to_state(), dt=0.01, integrator="rk4")
u_trim = np.array([result.elevator_rad, 0.0, 0.0, result.throttle])
for _ in range(500):  # 5 s
    m.run_step(u_trim)
print(f"After 5 s: V={m.airspeed_ft_s:.1f} ft/s, h={m.altitude_ft:.1f} ft")
```

### 2. Initialise from one of the 10 published trim points

```python
from tensoraerospace.aerospacemodel.b747.nonlinear import (
    B747_FLIGHT_CONDITIONS, NonlinearB747, initial_state_from_fc,
)

fc6 = B747_FLIGHT_CONDITIONS[5]  # FC6: 20 kft × M=0.65, α₀=2.5°
m = NonlinearB747(x0=initial_state_from_fc(fc6), dt=0.01, integrator="rk4")
```

### 3. Pitch-up under elevator step

```python
import numpy as np
from tensoraerospace.aerospacemodel.b747.nonlinear import (
    NonlinearB747, B747_FLIGHT_CONDITIONS, initial_state_from_fc,
)

fc4 = B747_FLIGHT_CONDITIONS[3]
m = NonlinearB747(x0=initial_state_from_fc(fc4), dt=0.01, integrator="rk4")

u = np.array([np.deg2rad(-2.0), 0.0, 0.0, 0.4])  # δ_e<0 ⇒ nose up
for _ in range(300):  # 3 s
    m.run_step(u)

s = m.current_state
alpha = np.rad2deg(np.arctan2(s[2], s[0]))
print(f"After 3 s: α={alpha:+.2f}°, θ={np.rad2deg(s[7]):+.2f}°, "
      f"q={np.rad2deg(s[4]):+.2f} deg/s, climb={m.altitude_ft:+.0f} ft")
```

### 4. Coordinated turn (aileron + rudder)

```python
import numpy as np
from tensoraerospace.aerospacemodel.b747.nonlinear import (
    NonlinearB747, B747_FLIGHT_CONDITIONS, initial_state_from_fc,
)

fc4 = B747_FLIGHT_CONDITIONS[3]
m = NonlinearB747(x0=initial_state_from_fc(fc4), dt=0.01, integrator="rk4")

u = np.array([
    0.0,
    np.deg2rad(2.0),     # right roll
    np.deg2rad(-1.0),    # adverse-yaw compensation
    0.4,
])
for _ in range(500):
    m.run_step(u)
s = m.current_state
print(f"Bank φ = {np.rad2deg(s[6]):+.2f}°, heading ψ = {np.rad2deg(s[8]):+.2f}°")
```

### 5. State / control history with `get_state` / `get_control`

```python
import matplotlib.pyplot as plt
import numpy as np
from tensoraerospace.aerospacemodel.b747.nonlinear import (
    NonlinearB747, B747_FLIGHT_CONDITIONS, initial_state_from_fc,
)

fc4 = B747_FLIGHT_CONDITIONS[3]
m = NonlinearB747(x0=initial_state_from_fc(fc4), dt=0.01, integrator="rk4")

# 0.2-second elevator pulse
for k in range(500):
    de = np.deg2rad(-3.0) if 100 <= k < 120 else 0.0
    m.run_step(np.array([de, 0.0, 0.0, 0.4]))

theta_deg = m.get_state("theta", to_deg=True)
q_deg_s = m.get_state("q", to_deg=True)
de_log = m.get_control("de")

t = np.arange(len(theta_deg)) * m.dt
fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
axes[0].plot(t, np.rad2deg(de_log)); axes[0].set_ylabel(r"$\delta_e$, °")
axes[1].plot(t, q_deg_s);            axes[1].set_ylabel(r"$q$, °/s")
axes[2].plot(t, theta_deg);          axes[2].set_ylabel(r"$\theta$, °")
axes[2].set_xlabel("time, s")
plt.tight_layout(); plt.show()
```

### 6. Euler vs RK4 integrator comparison

```python
import numpy as np
from tensoraerospace.aerospacemodel.b747.nonlinear import (
    NonlinearB747, B747_FLIGHT_CONDITIONS, initial_state_from_fc,
)

fc = B747_FLIGHT_CONDITIONS[3]
results = {}
for integ in ("euler", "rk4"):
    m = NonlinearB747(x0=initial_state_from_fc(fc), dt=0.05, integrator=integ)
    u = np.array([np.deg2rad(-2.0), 0.0, 0.0, 0.4])
    for _ in range(200):
        m.run_step(u)
    results[integ] = m.current_state

print(f"Δθ_euler = {np.rad2deg(results['euler'][7]):.4f}°")
print(f"Δθ_rk4   = {np.rad2deg(results['rk4'][7]):.4f}°")
```

---

### 7. End-to-end smoke demo (`example/aircraft/example_b747_nonlinear.py`)

Working script demonstrating the full cycle: trim, open-loop step,
damage effect. Run:

```bash
poetry run python example/aircraft/example_b747_nonlinear.py
```

Output: trim at FL200, healthy elevator −1° step → Δθ = +1.98° in
10 s; same setup with 50% elevator loss after t=5 s — Δθ ≈ +1.10°
(exactly half the healthy response, since μ=0.5).

![B-747 step response: healthy vs. damaged](img/b747_step_response_demo.png)

---

## Damage subsystem { #damage }

### 1. Built-in preset — 50% elevator loss

```python
import gymnasium as gym
import numpy as np
import tensoraerospace  # registers env

from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    ELEVATOR_50PCT_LOSS,
)

env = gym.make(
    "NonlinearB747-v0",
    flight_condition_id=4,
    number_time_steps=1000,
    damage_profile=ELEVATOR_50PCT_LOSS,
)
obs, _ = env.reset()

trim_u = np.array([0.0, 0.0, 0.0, 0.32])
for k in range(1000):
    obs, _, _, trunc, info = env.step(trim_u)
    if "damage_events_triggered" in info:
        print(f"t = {k*0.01:.2f} s: {info['damage_events_triggered']}")
        print(f"  μ = {info['damage_state']['mu']}")
        break
    if trunc:
        break
```

### 2. Custom event — elevator hard-over at −2°

```python
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    DamageProfile, SurfaceJamEvent,
)

profile = DamageProfile(events=[
    SurfaceJamEvent(
        trigger_time=15.0, surface="elevator",
        jam_value=-0.0349,
        label="elevator_hardover",
    ),
])
```

### 3. Gradual rudder wear

```python
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    DamageProfile, SurfaceEffectivenessDecay,
)

leak = DamageProfile(events=[
    SurfaceEffectivenessDecay(
        trigger_time=2.0,
        surface="rudder",
        tau=8.0,
        mu_floor=0.3,
        label="rudder_hydraulic_leak",
    ),
])
```

### 4. Runtime event injection

```python
import numpy as np
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    SurfaceEffectivenessEvent,
)
from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env

env = NonlinearB747Env(flight_condition_id=4, number_time_steps=1000)
env.reset()

trim_u = np.array([0.0, 0.0, 0.0, 0.32])
for k in range(1000):
    if k == 300:
        env.damage_manager.inject_event(
            SurfaceEffectivenessEvent(
                trigger_time=3.0, surface="aileron", mu=0.0,
                label="aileron_failure",
            )
        )
    env.step(trim_u)
```

### 5. Event logging via callback

```python
import numpy as np
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    AILERON_TOTAL_LOSS,
)
from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env

events_log = []

def on_damage(event, state):
    events_log.append({
        "label": event.label,
        "surface": event.surface,
        "all_mu": dict(state.mu),
    })

env = NonlinearB747Env(
    flight_condition_id=4, number_time_steps=1000,
    damage_profile=AILERON_TOTAL_LOSS,
    damage_event_callback=on_damage,
)
env.reset()
trim_u = np.array([0.0, 0.0, 0.0, 0.32])
for _ in range(1000):
    env.step(trim_u)
print(events_log)
```

### 6. Profile swap via `reset(options=...)`

```python
import numpy as np
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    AILERON_TOTAL_LOSS, ELEVATOR_50PCT_LOSS, RUDDER_HYDRAULIC_LEAK,
)
from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env

env = NonlinearB747Env(flight_condition_id=4, number_time_steps=500)

scenarios = [ELEVATOR_50PCT_LOSS, AILERON_TOTAL_LOSS, RUDDER_HYDRAULIC_LEAK]
for sc in scenarios:
    env.reset(options={"damage_profile": sc})
    # ...one episode per scenario...
```

### 7. Engine flameout

```python
import numpy as np
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    ENGINE_FLAMEOUT,
)
from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env

env = NonlinearB747Env(
    flight_condition_id=4, number_time_steps=2000,
    damage_profile=ENGINE_FLAMEOUT,
)
env.reset()
for _ in range(2000):
    obs, _, _, trunc, info = env.step(np.array([0.0, 0.0, 0.0, 0.32]))
    if trunc:
        break
print(f"Final airspeed: {np.linalg.norm(obs[:3]):.1f} ft/s")
```

### 8. Single-engine failure with asymmetric thrust

```python
import numpy as np
from tensoraerospace.aerospacemodel.b747.nonlinear import trim
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    LEFT_OUTER_ENGINE_FAILURE,
)
from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env

r = trim(altitude_ft=20_000.0, V_ft_s=674.0)
trim_action = np.array([float(r.elevator_rad), 0., 0., float(r.throttle)])

env = NonlinearB747Env(
    trim_at=(20_000.0, 674.0), number_time_steps=3000, dt=0.01,
    damage_profile=LEFT_OUTER_ENGINE_FAILURE,
)
obs, _ = env.reset()
for _ in range(3000):
    obs, _, _, trunc, _ = env.step(trim_action)
    if trunc:
        break
print(f"Final psi  : {np.rad2deg(obs[8]):+.1f}°  (negative = nose left)")
print(f"Final phi  : {np.rad2deg(obs[6]):+.1f}°  (rolling toward dead engine)")
print(f"Final V    : {np.linalg.norm(obs[:3]):.1f} ft/s")
```

### 9. Two engines out on same wing — worst-case asymmetry

```python
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    LEFT_TWO_ENGINES_OUT,
)

env = NonlinearB747Env(
    trim_at=(20_000.0, 674.0), number_time_steps=2500, dt=0.01,
    damage_profile=LEFT_TWO_ENGINES_OUT,
)
# ...same loop as above; observe much larger yaw / roll excursion...
```

### 10. Custom engine failure — partial thrust loss

```python
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    DamageProfile, EngineFailureEvent,
)

# Engine 4 (right outer) drops to 30% thrust at t = 6 s, simulating
# a damaged compressor or partial bird strike.
profile = DamageProfile(events=[
    EngineFailureEvent(
        trigger_time=6.0, engine_id=4, thrust_fraction=0.3,
        label="engine_4_partial",
    ),
])
```

### 11. Flap jam during cruise

```python
import numpy as np
from tensoraerospace.aerospacemodel.b747.nonlinear import B747Configuration
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    FLAPS_JAMMED_LANDING,
)
from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env

# Flying clean cruise at FL200 — at t=5s flaps mechanically lock at 30°.
# The aerodynamic build switches to LANDING derivatives: large lift
# and drag jump, the trim throttle is no longer enough, the airframe
# pitches up and decelerates. The agent must apply forward stick + cut
# throttle to maintain control.
env = NonlinearB747Env(
    trim_at=(20_000.0, 674.0), number_time_steps=1500, dt=0.02,
    damage_profile=FLAPS_JAMMED_LANDING,
    config=B747Configuration.NOMINAL,
)
env.reset()
# ...closed-loop control loop here...
print(env.damage_manager.state.flap_jam_config)  # B747Configuration.LANDING after t=5s
```

### 12. Custom flap jam — stuck at approach setting

```python
from tensoraerospace.aerospacemodel.b747.nonlinear import B747Configuration
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    DamageProfile, FlapJamEvent,
)

# Pilot extended flaps for descent, but they refuse to go past 20°
# (POWER_APPROACH). Aircraft can land but with degraded handling.
profile = DamageProfile(events=[
    FlapJamEvent(
        trigger_time=4.0,
        jammed_config=B747Configuration.POWER_APPROACH,
        label="flaps_stuck_at_20deg_approach",
    ),
])
```

---

## See also

* [Boeing 747-100 (Nonlinear 6-DoF)](b747_nonlinear.md) — theory,
  CR-2144 reference, equations, validation.
* [Aircraft Damage Modeling](aircraft-damage-modeling.md) — general
  damage-subsystem concept (F-16 version).
* `example/reinforcement_learning/example_dsac_b747.ipynb` — DSAC
  pitch-tracking on the B-747 model.
