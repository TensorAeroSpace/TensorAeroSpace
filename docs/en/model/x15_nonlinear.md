# North American X-15 (Nonlinear 6-DoF Hypersonic Model)

`tensoraerospace.aerospacemodel.x15.nonlinear` — full nonlinear 6-DoF
model of the **X-15 hypersonic research aerospaceplane** with
Mach-tabulated aerodynamics, XLR99 rocket engine, and **variable
mass** dynamics covering the powered-flight envelope from drop
($M = 0.83$, $h = 45\,000$ ft) through hypersonic peak ($M = 6.7$,
$h = 102\,000$ ft) into post-burnout glide.

| Parameter | Value |
|---|---|
| Aerodynamic source | NASA TM X-1669 (Walker & Wolowicz 1968) + TM 2598 |
| Mach envelope | 0.4 — 6.7 |
| Altitude envelope | 0 — 250 000 ft |
| Configurations | BASIC (X-15-1/3), A2 (X-15A-2 with external tanks) |
| Engine | Reaction Motors XLR99 — 57 000 lbf, throttleable 30–100 % |
| Coordinates | NED, body axis, ZYX 321 Euler |
| State dimension | **13** (12-D rigid body + propellant mass) |
| Control surfaces | All-flying horizontal stabilizer, ailerons, vertical rudder, throttle |
| Damage subsystem | Hooks open (parity with B-747); not yet wired up |

## Geometry & mass (Walker/Wolowicz, Thompson 2000)

```
S    = 200 ft²        (planform reference area)
b    = 22.36 ft       (span)
c̄    = 10.27 ft       (mean aerodynamic chord)
c.g. = 0.22 c̄         (centre of gravity)
```

| Configuration | Empty W, lb | Propellant, lb | Gross W, lb | Iy, slug·ft² (full) |
|---|---:|---:|---:|---:|
| **BASIC** (X-15-1, X-15-3) | 14 600 | 17 900 | 32 500 | 88 × 10³ |
| **A2** (record airframe) | 16 050 | 30 900 | 46 950 | 110 × 10³ |

Mass and inertias are **time-varying**: the dynamics state carries a
`m_prop` channel (state index 12), and the parameters object linearly
interpolates inertias by propellant fraction:

$$
I_i(t) = I_i^{\text{empty}} + \frac{m_\text{prop}(t)}{m_\text{prop}^\text{full}} \bigl(I_i^{\text{full}} - I_i^{\text{empty}}\bigr).
$$

## Anchor flight conditions

Five published anchors span the powered-flight corridor, distilled
from NASA TM X-1669 Table 2 and Thompson 2000 mission timelines.

| FC | Label | h, ft | M | V, ft/s | α₀, deg | δ_e₀, deg | Propellant, lb |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | boost_start         | 45 000  | 0.83 | 797   | 4.5  | −2.5 | 17 900 |
| 2 | boost_climb         | 70 000  | 2.5  | 2 412 | 5.0  | −3.0 | 10 500 |
| 3 | cruise_M4           | 100 000 | 4.0  | 3 865 | 4.0  | −2.0 | 6 500  |
| 4 | coast_high          | 200 000 | 5.0  | 4 876 | 10.0 | −1.0 | 0      |
| 5 | hypersonic_record   | 102 000 | 6.7  | 6 525 | 4.5  | −2.0 | 2 000  |

These are *trajectory waypoints*, not steady-state trim points — see
[Trim envelope (or lack thereof)](#trim-envelope-or-lack-thereof) below.

## State and control

**State** (13-D, body axis, NED, ZYX 321 Euler):

```
[u, v, w,           # body velocity, ft/s
 p, q, r,           # body angular rates, rad/s
 φ, θ, ψ,           # Euler angles, rad
 x_e, y_e, z_e,     # NED position, ft  (z_e positive down ⇒ altitude = -z_e)
 m_prop]            # remaining propellant, lb
```

**Control** (4-D):

```
[δ_e,  δ_a,  δ_r,  δ_T]
   ↓     ↓     ↓     ↓
all-fly  ail  rud   throttle
 (rad)  (rad) (rad)   [0, 1]
```

The X-15 has an **all-flying horizontal tail** (no separate elevator
on a fixed stabilizer) — the entire surface rotates as one piece.
Limits: $|\delta_e| \le 15°$, $|\delta_a| \le 15°$, $|\delta_r| \le 8.5°$
(rudder authority is small because of the small wedge tail), all rate-limited
to $60\,°/s$.

## Hypersonic aerodynamic build

Coefficients are **Mach-interpolated** from a Walker/Wolowicz-derived
table at 8 anchor Mach numbers:

```
M_grid = [0.4, 0.8, 1.2, 2.0, 3.0, 4.0, 5.0, 6.7]
```

The lift slope $C_{L_\alpha}$ drops monotonically from
$\sim 3.5$ /rad at low Mach to $\sim 2.05$ /rad at $M = 6.7$ — close to
the Newtonian limit of 2 for a sharp-nosed hypersonic body. Drag
coefficient $C_{D_0}$ peaks transonic ($\sim 0.038$ at $M = 1.2$) and
drops at hypersonic Mach ($\sim 0.022$) once the shock structure is
fully attached.

Full coefficient tables live in
[`aero.py`](https://github.com/tensoraerospace/tensoraerospace/blob/main/tensoraerospace/aerospacemodel/x15/nonlinear/aero.py),
including:

| Symbol | Coverage |
|---|---|
| $C_{L_0}, C_{L_\alpha}, C_{L_q}, C_{L_M}, C_{L_{\delta_e}}$ | Longitudinal lift |
| $C_{D_0}, C_{D_\alpha}, C_{D_M}$ | Longitudinal drag |
| $C_{m_\alpha}, C_{m_q}, C_{m_M}, C_{m_{\delta_e}}, C_{m_{\dot\alpha}}$ | Pitching moment |
| $C_{Y_\beta}, C_{Y_p}, C_{Y_r}, C_{Y_{\delta_a}}, C_{Y_{\delta_r}}$ | Side force |
| $C_{l_\beta}, C_{l_p}, C_{l_r}, C_{l_{\delta_a}}, C_{l_{\delta_r}}$ | Rolling moment |
| $C_{n_\beta}, C_{n_p}, C_{n_r}, C_{n_{\delta_a}}, C_{n_{\delta_r}}$ | Yawing moment |

## XLR99 rocket engine

The Reaction Motors XLR99 (Thompson 2000, NASA SP-2000-4222):

* **Sea-level rated thrust** $T_{SLS} = 57\,000$ lbf.
* **Throttleable** from 30 % to 100 % — below 30 % the engine
  treats it as off.
* **Specific impulse** $I_{sp} = 254$ s (sea-level value, used
  throughout — vacuum correction ≤ 5 % is ignored).
* **Mass flow** at full throttle: $\dot m = T / I_{sp} \approx 224$ lb/s
  → 80 s burn time for the BASIC airframe.
* **Burnout** is automatic: when ``m_prop ≤ 0`` the engine returns
  zero thrust regardless of throttle command.

Unlike the B-747's air-breathing JT9D, **rocket thrust is independent
of Mach and altitude** — there is no inlet recovery, no ram effect.

## Equations of motion

Standard Newton-Euler in body axis, identical to the B-747 model
(see [Boeing 747-100 Nonlinear](b747_nonlinear.md#equations-of-motion)
for the full equations). The variable-mass effect adds a 13th state
equation:

$$
\dot m_{\text{prop}} = -\dot m_e\bigl(\delta_T,\, m_{\text{prop}}\bigr).
$$

For an on-axis exhaust the standard "constant-mass with current m"
form of $m\dot{\vec v} = \Sigma\vec F + \vec T$ is exact (the
exhaust's *velocity-of-mass-loss* term is already accounted for by
treating $T$ as the externally measured thrust). Mass and inertias
in the rotational equations are queried from the parameters object
*at every ODE evaluation*, so the integrator naturally sees the
correct values throughout the burn.

## Trim envelope (or lack thereof)

Unlike a transport aircraft, the X-15 **does not have a true
level-cruise envelope**. The XLR99 cannot scale its thrust to match
drag at every $(M, h)$ — at full throttle the rocket overpowers the
drag and the aircraft accelerates / climbs; below 30 % the engine is
off and the aircraft must descend.

Two trim modes are exposed:

* `trim(altitude, V, throttle)` — fixes throttle, solves for
  $(\alpha, \delta_e, \gamma)$. **Realistic X-15 behaviour**: at full
  throttle the aircraft climbs steeply, post-burnout it glides.
* `level_trim(altitude, V)` — fixes $\gamma = 0$, solves for
  $(\alpha, \delta_e, \delta_T)$. **Mostly fails** — flagged via
  `converged=False`. Useful only as a sanity check.

Glide trim (post-burnout) converges cleanly at low / mid altitude:

```python
from tensoraerospace.aerospacemodel.x15.nonlinear import trim

# 30 kft, M ≈ 0.7, no propellant left
result = trim(altitude_ft=30_000.0, V_ft_s=800.0,
              throttle=0.0, propellant_lb=0.0)
print(f"α = {math.degrees(result.alpha_rad):+.2f}°")    # ~ 4°
print(f"γ = {math.degrees(result.gamma_rad):+.2f}°")    # ~ -6° (descending)
print(f"converged = {result.converged}")                # True
```

## Gymnasium env

Registered as `"NonlinearX15-v0"`. Three initialisation modes:

```python
import gymnasium as gym
import tensoraerospace  # registers the env

# 1. By one of the 5 published trim points
env = gym.make("NonlinearX15-v0", flight_condition_id=2, number_time_steps=2000)

# 2. Trim-finder at any (h, V) and throttle setting
env = gym.make("NonlinearX15-v0",
    trim_at=(30_000.0, 800.0), trim_throttle=0.0, number_time_steps=2000)

# 3. Arbitrary 13-D initial state
import numpy as np
env = gym.make("NonlinearX15-v0",
    initial_state=np.array([2412, 0, 211, 0,0,0, 0, 0.087, 0,
                            0, 0, -70_000, 10_500]),
    number_time_steps=2000)
```

Action-space: either `"virtual"` (physical units) or `"normalized"`
(for RL: `[-1, 1]^4`).

The `info` dict from each step reports:

```python
{"propellant_lb": ..., "engine_running": True / False}
```

so RL agents can plan around burnout (e.g. choose throttle to extend
the powered phase, or switch into glide-control mode after flameout).

## Scope and limitations

This MVP focuses on the **aerodynamic-flight envelope**. Out of scope
for the initial release:

* **Reaction Control System (RCS)** — the X-15 had peroxide thrusters
  for attitude control above ~ 250 kft where aerodynamic surfaces lose
  authority (low $\bar q$). Modelling RCS adds 8 more thrusters with
  per-thruster fault modelling.
* **Ablative heat-shield mass loss** — X-15A-2 lost ~ 100 lb of
  ablative material per Mach-6.7 flight. Negligible for dynamics but
  noticeable for c.g. tracking.
* **Damage subsystem** — the parameters and model expose
  `damage_state` hooks (parity with the B-747 damage subsystem), but
  no events have been written yet. Engine flameout, surface jam, and
  control-effectiveness loss can be added when the use case arises.
* **Stratopause / mesosphere atmosphere** — above 200 kft the simple
  isothermal-stratosphere model used here begins to deviate from
  the US Std 1976 atmosphere by > 5 %. Densities are so low
  ($\rho \le 10^{-7}$ slug/ft³) that aerodynamic forces are
  vanishing — but trajectory accuracy past $h = 200$ kft would
  benefit from a more detailed atmosphere.
* **Six-DoF rocket-thrust offset moment** — the XLR99 thrust line is
  modelled as collinear with body x. In reality, the engine sits
  slightly below the c.g., creating a small nose-down moment that
  the trim-tab compensates. Negligible for control-design demos but
  worth adding for high-fidelity reentry studies.

## Related modules

* [Boeing 747-100 (Nonlinear 6-DoF)](b747_nonlinear.md) — heavy
  air-breather, complementary low-Mach high-mass model. Same code
  patterns as this one.
* [F-16 (Nonlinear longitudinal)](f16_nonlinear_longitudinal.md) —
  fighter, mid-Mach envelope with cubic-spline aero tables.
* [X-15 (Linear longitudinal)](x15.md) — legacy single-trim-point
  state-space model in FPS units. Backward-compatible import:
  `from tensoraerospace.aerospacemodel.x15 import LongitudinalX15`.

## References

* **NASA TM X-1669** — Walker H. J., Wolowicz C. H. *Stability and
  Control Derivatives of the X-15 Airplane*, NASA Flight Research
  Center, 1968.
* **NASA TN D-1402** — early X-15 stability data.
* **NASA SP-2000-4222** — Thompson M. O. *At the Edge of Space: The
  X-15 Flight Program*. Includes XLR99 propellant flow rates and
  flight envelope.
* **NASA TM 2598** — X-15A-2 advanced configuration reference.
* Stevens B. L., Lewis F. L., Johnson E. N. *Aircraft Control and
  Simulation*, Wiley, 3rd ed., 2015 — body-axis Newton-Euler form.
