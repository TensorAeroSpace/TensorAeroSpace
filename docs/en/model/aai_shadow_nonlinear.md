# AAI RQ-7 Shadow — Class-II Tactical UAV (Nonlinear 6-DoF, SI units)

`tensoraerospace.aerospacemodel.aai_shadow.nonlinear` — full nonlinear
6-DoF model of the **AAI RQ-7 Shadow** (RQ-7B configuration), a
class-II tactical reconnaissance UAV operated by the US Army and
several allied forces.

The Shadow is a **larger, conventional-tail relative** of the
[Skywalker X8](skywalker_x8_nonlinear.md) flying-wing UAV — same
class of small fixed-wing platform but with a high-aspect-ratio
rectangular wing, twin tail booms, and an inverted V-tail. It serves
as the canonical class-II (50-300 kg) UAV in the tensoraerospace
roster.

| Parameter | Value |
|---|---|
| Aerodynamic source | Class-II UAV literature (Beard & McLain, NASA TM-2014-218686, Roskam Vol VI V-tail mixing) |
| Mass / span / area | 170 kg / 6.22 m / 4.42 m² |
| Aspect ratio | 8.75 |
| Engine | UEL AR-741 single-rotor Wankel rotary, 38 hp (28 kW), pusher |
| Coordinates | NED, body axis, ZYX 321 Euler |
| State | 12-D rigid-body |
| Controls | 4 channels: collective ruddervator (δ_e), aileron (δ_a), differential ruddervator (δ_r), throttle (δ_T) |
| Units | **SI** (kg, m, N, rad, s) — same as Skywalker X8 |

## Geometry & mass (FAS / AAI published spec, RQ-7B)

```
m   = 170 kg            (cruise weight, mid-fuel)
Ix  = 50 kg·m²
Iy  = 80 kg·m²
Iz  = 120 kg·m²
Ixz = 5 kg·m²
c̄  = 0.71 m            (mean aerodynamic chord)
b   = 6.22 m            (wingspan, RQ-7B; original RQ-7A was 4.27 m)
S   = 4.42 m²           (planform area, RQ-7B)
```

The geometry numbers are the **RQ-7B** ("Improved Tactical UAV")
configuration, which extended the original RQ-7A wingspan from 4.27 m
to 6.22 m. The published cruise speed of 36 m/s ≈ 70 kt is consistent
with the 4.42 m² wing area at 170 kg loading.

## State and control

**State** (12-D, body axis, NED, ZYX 321 Euler — SI):

```
[u, v, w,           # body velocity, m/s
 p, q, r,           # body angular rates, rad/s
 φ, θ, ψ,           # Euler angles, rad
 x_e, y_e, z_e]     # NED position, m
```

**Control** (4-D mixed V-tail convention):

```
[δ_e, δ_a, δ_r, δ_T]
```

The Shadow has an inverted V-tail with two **ruddervators** (combined
elevator + rudder surfaces). The standard mixing is

* `δ_e = (δ_l + δ_r) / 2` — collective deflection acts as elevator
* `δ_r = (δ_l - δ_r) / 2` — differential deflection acts as rudder

The agent always commands the mixed pair `(δ_e, δ_r)`; a mechanical
mixer would translate these into physical ruddervator deflections. The
aero coefficients are quoted in the mixed convention.

Limits: $|\delta_e|, |\delta_a| \le 20°$, $|\delta_r| \le 15°$, all
rate-limited at $120\,°/s$.

## Aerodynamic build

Coefficients are synthesised from class-II UAV literature with V-tail
effective-area scaling:

| Drag |  | Lift |  | Pitch |  |
|---|---:|---|---:|---|---:|
| $C_{D_0}$ | 0.030 | $C_{L_0}$ | 0.28 | $C_{m_0}$ | 0.0 |
| $C_{D_{k_2}}$ | 0.043 | $C_{L_\alpha}$ | 5.0 /rad | $C_{m_\alpha}$ | −1.50 /rad |
| | | $C_{L_q}$ | 7.95 | $C_{m_q}$ | −38.0 |
| | | $C_{L_{\delta_e}}$ | 0.43 | $C_{m_{\delta_e}}$ | −1.20 |
| | | | | $C_{m_{\dot\alpha}}$ | −7.0 |

| Side force |  | Roll |  | Yaw |  |
|---|---:|---|---:|---|---:|
| $C_{Y_\beta}$ | −0.83 | $C_{l_\beta}$ | −0.13 | $C_{n_\beta}$ | 0.073 |
| $C_{Y_p}$ | 0.0 | $C_{l_p}$ | −0.51 | $C_{n_p}$ | −0.069 |
| $C_{Y_r}$ | 0.30 | $C_{l_r}$ | 0.25 | $C_{n_r}$ | −0.095 |
| $C_{Y_{\delta_r}}$ | 0.18 | $C_{l_{\delta_a}}$ | 0.17 | $C_{n_{\delta_a}}$ | −0.011 |
| | | $C_{l_{\delta_r}}$ | 0.024 | $C_{n_{\delta_r}}$ | −0.069 |

Notable design points:

* **Lift slope** $C_{L_\alpha} \approx 5.0$/rad — consistent with
  AR = 8.75 and 2D-airfoil-corrected lifting-line theory.
* **Induced drag factor** $C_{D_{k_2}} = 1/(\pi\,AR\,e) \approx 0.043$
  with Oswald efficiency $e = 0.85$.
* **V-tail rudder authority** $C_{n_{\delta_r}} \approx -0.07$ —
  smaller than a conventional vertical-tail aircraft because the
  V-tail produces side force and yaw moment via differential
  deflection instead of a dedicated rudder.

## UEL AR-741 engine model

Single-rotor Wankel rotary, 38 hp (28 kW) at 7 600 rpm, pusher-mounted
on a 24" 2-blade carbon propeller. Calibrated quadratic thrust:

$$
T(\delta_T, V) = T_{\max} \cdot \delta_T^2 \cdot (1 - V / V_{\text{zero}})
$$

with $T_{\max} = 380$ N, $V_{\text{zero}} = 65$ m/s. Calibration
points:

| Condition | Thrust |
|---|---:|
| Static, full throttle | 380 N |
| 36 m/s, 70 % throttle | ~ 75 N |

The 70 % cruise throttle is consistent with published RQ-7 endurance
numbers (6-9 h at typical cruise weight).

## Trim finder

`tensoraerospace.aerospacemodel.aai_shadow.nonlinear.trim(h, V)`
solves $\dot u = \dot w = \dot q = 0$ via Newton-Raphson:

| Condition | h, m | V, m/s | α | δ_e | δ_T |
|---|---:|---:|---:|---:|---:|
| Typical loiter | 1000 | 36 | 3.10° | -3.87° | 0.93 |

Residual norm reaches **machine precision** ($10^{-15}$). Holding
the trimmed controls keeps the aircraft within ±0.000 m/s, ±0.000 m
altitude, ±0.000° pitch over 5 seconds.

The high trim throttle (0.93) reflects the small AR-741 engine's
limited margin at 170 kg gross weight — close to the published
service ceiling of ~ 4 600 m the trim solver fails (the engine cannot
sustain level flight there with this payload), which matches reality.

## Gymnasium env

Registered as `"NonlinearAAIShadow-v0"`:

```python
import gymnasium as gym
import tensoraerospace  # registers the env

# Trim-finder at any (altitude, airspeed) — note SI units!
env = gym.make("NonlinearAAIShadow-v0",
    trim_at=(1000.0, 36.0), number_time_steps=2000)

# Arbitrary 12-state initial condition (SI)
import numpy as np
env = gym.make("NonlinearAAIShadow-v0",
    initial_state=np.array([35.9, 0, 1.95, 0,0,0, 0, 0.054, 0,
                            0, 0, -1000.0]),
    number_time_steps=2000)
```

Action space: **4-channel** `[δ_e, δ_a, δ_r, δ_T]`. Use `"virtual"`
for raw rad / [0, 1] or `"normalized"` for `[-1, +1]^4`.

## Scope and limitations

* **Aerodynamic derivatives are synthesised**, not directly transcribed
  from a single canonical AAI / NASA paper. Magnitudes are
  cross-checked against NASA TM-2014-218686 and class-II UAV
  literature (Beard & McLain, Roskam Vol VI), but the model should be
  considered representative-class rather than tail-number-accurate.
* **Wing flexibility / catapult-launch transients** not modelled. Used
  for level flight envelope ~ 30-50 m/s, h = 0-4500 m only.
* **Damage subsystem hooks open** but no events wired up — parity
  with the rest of the family.

## Related modules

* [Skywalker X8](skywalker_x8_nonlinear.md) — flying-wing companion
  of the small UAV class. Same code patterns, different control
  layout (3-channel vs 4-channel).
* [Boeing 737 (Nonlinear 6-DoF)](b737_nonlinear.md) — large
  air-breathing transport using the same FPS modular pattern.

## References

* **Beard R. W., McLain T. W.** *Small Unmanned Aircraft: Theory and
  Practice*, Princeton Univ. Press (2012). Appendix E.1 — Aerosonde
  Mark 4.7 derivatives, used as a class-II baseline.
* **NASA TM-2014-218686** — RQ-7 Shadow aerodynamic database
  reference for sanity-checking $C_{L_\alpha}$, $C_{m_\alpha}$ and
  V-tail effective $C_{l_{\delta_r}}$ / $C_{n_{\delta_r}}$.
* **Roskam J.** *Airplane Flight Dynamics and Automatic Flight
  Controls*, Vol VI Appendix C — V-tail mixing relations and
  effective-area scaling.
* **Nelson R. C.** *Flight Stability and Automatic Control*,
  McGraw-Hill 2nd ed. (1998) — high-AR surveillance aircraft
  derivative ranges.
* **Federation of American Scientists (FAS)** RQ-7 fact sheet —
  geometry, weight, performance numbers.
