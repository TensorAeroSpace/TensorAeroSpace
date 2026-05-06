# Boeing 737 (Nonlinear 6-DoF Model)

`tensoraerospace.aerospacemodel.b737.nonlinear` — full nonlinear 6-DoF
model of the Boeing 737 family, with two configurations covering
both the original 737-100/200 (twin JT8D-9) and the 737-NG / 737-800
(twin CFM56-7B27).

## Provenance

The 737 — unlike the 747, F-16 and X-15 — does **not** have a
single canonical NASA technical paper publishing its full set of
non-dimensional stability derivatives. The numerical values used here
are consolidated from the open sources widely accepted in academic
flight-mechanics work:

| Source | Role |
|---|---|
| **JSBSim 737 model** ([repo](https://github.com/JSBSim-Team/jsbsim/blob/master/aircraft/737/737.xml)) | Primary numerical source — geometry, masses, inertias, full coefficient set |
| **Roskam J.** *Airplane Flight Dynamics and Automatic Flight Controls* (1995), Vol VI Appendix B | Original 737-100 derivatives that JSBSim transcribes |
| **Hanke C. R.** *The Simulation of a Large Jet Transport Aircraft*, NASA CR-114494 (1971) | Wind-tunnel methodology behind Roskam's tables |
| **Cook M. V.** *Flight Dynamics Principles* (3rd ed., Elsevier, 2013), Ch. 11 | Cross-reference 737-100 example aircraft |
| **NASA TM-86821** *"Mach/CAS control law for the NASA TCV B737"* (1986) | Validation reference for nonlinear-simulation envelope |
| **FAA TCDS A16WE** | Boeing 737 family weights, geometry, certified engine ratings |
| **CFM International CFM56-3/-7B fact sheets** | Engine performance |

JSBSim is BSD-licensed and built explicitly on publicly available
information for educational use; we transcribe its coefficient
functional forms verbatim.

| Parameter | Value (737-100) |
|---|---|
| Aerodynamic source | JSBSim 737-100 / Roskam Vol VI |
| Configurations | B737_100, B737_800 |
| Engines | 2 × Pratt & Whitney JT8D-9 (737-100) or 2 × CFM56-7B27 (737-800) |
| Coordinates | NED, body axis, ZYX 321 Euler |
| Control surfaces | elevator, aileron, rudder + throttle |
| Damage subsystem | Hooks open (parity with B-747); not yet wired up |

## Geometry & mass

```
                      737-100         737-800
S  (wing area)        1171 ft²        1341 ft²
b  (span)             94.7 ft         117.5 ft
c̄  (MAC)              12.31 ft        12.97 ft
W  (mid-cruise)       100 000 lb      140 000 lb
T_SLS (cluster)       29 000 lbf      54 600 lbf
```

| Configuration | W, lb | Iₓ, slug·ft² | I_y | I_z | I_xz |
|---|---:|---:|---:|---:|---:|
| **B737_100** | 100 000 | 562 × 10³ | 1.473 × 10⁶ | 1.894 × 10⁶ | 8.0 × 10³ |
| **B737_800** | 140 000 | 820 × 10³ | 2.300 × 10⁶ | 3.000 × 10⁶ | 12.0 × 10³ |

## State and control

**State** (12-D, body axis, NED, ZYX 321 Euler):

```
[u, v, w,           # body velocity, ft/s
 p, q, r,           # body angular rates, rad/s
 φ, θ, ψ,           # Euler angles, rad
 x_e, y_e, z_e]     # NED position, ft
```

**Control** (4-D):

```
[δ_e,  δ_a,  δ_r,  δ_T]
   ↓     ↓     ↓     ↓
elevator ail   rud   throttle
 (rad)  (rad) (rad)   [0, 1]
```

Limits (JSBSim 737.xml): $|\delta_e| \le 17.2°$, $|\delta_a|, |\delta_r| \le 20.1°$,
all rate-limited at $40\,°/s$.

## Equations of motion

Standard Newton-Euler in body axis — identical to the
[B-747 nonlinear](b747_nonlinear.md#equations-of-motion). The two
differences from the 747 are:

1. **Lighter airframe with smaller inertias.** Short-period and
   Dutch-roll periods are shorter (~ 1–2 s vs the 747's 3–4 s).
2. **Twin-engine configuration.** Asymmetric-thrust events use the
   B-737 engine spanwise positions $y_1 = -16.5$ ft, $y_2 = +16.5$ ft
   (737-100 inboard pylons), tighter than the 747's outer engines
   at $\pm 71.7$ ft.

## Aerodynamic build (JSBSim functional forms)

Coefficients are computed at every ODE evaluation using the
**dimensional-decomposition approach** from JSBSim:

| Coefficient | Form |
|---|---|
| $C_L$ | α-table (peak $C_L \approx 1.45$ at $\alpha = 13°$) + linear $C_{L_{\delta_e}} = 0.20$ |
| $C_D$ | α-table + induced $0.043 C_L^2$ + Mach compressibility table + sideslip + elevator drag |
| $C_m$ | $C_{m_\alpha} = -0.6$/rad, Mach-dependent $C_{m_{\delta_e}}$ (-1.20 to -0.30), pitch + α̇ damping |
| $C_Y, C_l, C_n$ | Linear in (β, p̂, r̂, δa, δr); Mach-dependent $C_{l_{\delta_a}}$ |

The α-table gracefully degrades past stall (clamped to $C_L = 0.6$
at $\alpha = 35°$) so the integrator does not blow up if the
controller temporarily over-rotates. Ground / configuration effects
(flap, gear, speedbrake) are exposed in the source but not applied
in this MVP — the model is valid for the **clean cruise envelope**.

## Engine model

Two-engine cluster with Mach-altitude-derated installed thrust
following Mattingly §8.6.4 (same form as the JT9D model in the
B-747 module):

$$
T_{inst}(M, h, \delta_T) = T_{SLS} \cdot \sigma(h)^{n_h} \cdot \eta_{ram}(M) \cdot \mathrm{PLA}_{eff}
$$

with $\eta_{ram}(M) = 1 - 0.49\sqrt{M}$ (clamped to 0.05) and
$n_h = 0.7$ below tropopause / $1.0$ above. The same correlation
serves both JT8D-9 (737-100) and CFM56-7B (737-800) — cross-checked
against FAA TCDS A16WE certified ratings.

| Configuration | $T_{SLS}$, lbf | per-engine | Engine type |
|---|---:|---:|---|
| 737-100 | 29 000 | 14 500 | P&W JT8D-9 |
| 737-800 | 54 600 | 27 300 | CFM56-7B27 |

## Trim finder

`tensoraerospace.aerospacemodel.b737.nonlinear.trim(h, V)` solves
$\dot u = \dot w = \dot q = 0$ via Newton-Raphson, returning trimmed
$(\alpha, \delta_e, \delta_T)$. Unlike the X-15, the 737 is a
proper transport with air-breathing engines that scale with Mach
and altitude — **cruise trim converges across the entire normal
flight envelope**:

| Configuration | h, ft | M | V, ft/s | α | δ_e | δ_T |
|---|---:|---:|---:|---:|---:|---:|
| B737-100 | 25 000 | 0.74 | 738 | 1.0° | -0.6° | 0.92 |
| B737-800 | 35 000 | 0.83 | 820 | 2.4° | -1.6° | 0.91 |

Residual norms reach $10^{-13}$ — i.e. machine-precision trim.

## Gymnasium env

Registered as `"NonlinearB737-v0"`. Two initialisation modes:

```python
import gymnasium as gym
import tensoraerospace  # registers the env

# 1. Trim-finder at any (h, V)
env = gym.make("NonlinearB737-v0",
    trim_at=(25_000.0, 738.0), number_time_steps=2000)

# 2. Arbitrary initial state
import numpy as np
env = gym.make("NonlinearB737-v0",
    initial_state=np.array([738, 0, 13, 0,0,0, 0, 0.017, 0,
                            0, 0, -25_000]),
    number_time_steps=2000)
```

For the 737-NG / 737-800, pass `config=B737Configuration.B737_800`
through the constructor (or directly when you instantiate
`NonlinearB737Env(...)`).

Action-space: either `"virtual"` (physical units) or `"normalized"`
(for RL: `[-1, 1]^4`).

## Scope and limitations

* **High-lift devices not modelled** — flaps, slats, ground
  effect, gear, speedbrake aerodynamic increments are exposed in the
  source but not applied. The model is valid for clean cruise; for
  approach / landing scenarios you'd need to enable these.
* **Damage subsystem hooks open but no events** — parity with the
  B-747 architecture (`engines_mu`, `flap_jam_config`, etc.) is
  prepared, but no concrete events are wired up. Adding them is
  straightforward since the engine model already accepts an
  `engines_mu` dict.
* **737-NG aerodynamics use 737-100 derivatives scaled by geometry**
  — Roskam Vol VI does not separately publish 737-800 derivatives,
  so the dimensional CL/CD/Cm functions are evaluated at the new
  reference area / span / chord. Acceptable for control-design
  work; for performance studies a re-derivation is recommended.

## Related modules

* [Boeing 747-100 (Nonlinear 6-DoF)](b747_nonlinear.md) — heavier
  4-engine air-breather with the canonical NASA CR-2144 derivative
  bank. Same code patterns.
* [F-16 (Nonlinear longitudinal)](f16_nonlinear_longitudinal.md) —
  fighter, mid-Mach envelope.
* [X-15 (Nonlinear hypersonic)](x15_nonlinear.md) — research
  rocketplane, hypersonic envelope. Same code patterns, different
  engine model.

## References

* **JSBSim** — Berndt J. S. *"JSBSim: An Open Source Flight Dynamics
  Model in C++"*, AIAA Modeling and Simulation Technologies
  Conference, 2004 ([repo](https://github.com/JSBSim-Team/jsbsim)).
* **Roskam J.** *Airplane Flight Dynamics and Automatic Flight
  Controls*, Roskam Aviation, 1995. Vol VI Appendix B.
* **Hanke C. R.** *The Simulation of a Large Jet Transport
  Aircraft*, NASA CR-114494, 1971.
* **Cook M. V.** *Flight Dynamics Principles*, Elsevier 3rd ed.,
  2013, Chapter 11.
* **NASA TM-86821** — Bahm C. M., Sivolell P. *"Design and
  verification by nonlinear simulation of a Mach/CAS control law for
  the NASA TCV B737 aircraft"*, 1986
  ([NTRS 19870010857](https://ntrs.nasa.gov/citations/19870010857)).
* **FAA TCDS A16WE** — Boeing 737 type certificate data sheet.
* Mattingly J. D. *Aircraft Engine Design*, AIAA Education Series,
  2nd ed., 2002, §8.6.4 (installed-thrust lapse model).
