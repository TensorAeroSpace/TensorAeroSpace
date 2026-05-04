# Quadrotor / multirotor UAV — Nonlinear 6-DoF dynamics

Rigid-body quadrotor model in the full 6-DoF formulation: 12 states
(position, velocity, attitude, angular rates), 4 control inputs
(collective thrust + three body-frame torques). Implemented in pure
Python/NumPy with Euler and RK4 integrators. Suitable for PID
stabilisation, MPC trajectory tracking, and adaptive RL critics
(iADP, AIDI, AA-INDI) — especially for rotor-failure scenarios.

![Quadrotor X-configuration: top-down view showing body-frame axes, motor numbering M1–M4, rotation directions CCW/CW, per-motor thrusts $f_i$, body torques $\tau_x, \tau_y, \tau_z$, and the collective thrust $T$](img/quadrotor_top_down_view.png)

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Quick start**

    Run a 10-second hover in three lines.

    [:octicons-arrow-right-24: To example](#quick-start)

-   :material-cog-outline: **Model API**

    `NonlinearQuadrotor` — the 6-DoF dynamics class.

    [:octicons-arrow-right-24: To API](#python-api)

-   :material-book-open-variant: **Mathematics**

    Equations of motion, NED / body frame conventions.

    [:octicons-arrow-right-24: To theory](#mathematical-model)

-   :material-test-tube: **Validation tests**

    Hover, free fall, gyroscopic coupling.

    [:octicons-arrow-right-24: To checks](#validation)

</div>

## Reference platform parameters

The defaults match a typical 1.5 kg research quadrotor (AscTec
Hummingbird / Pelican class), arm length 22.5 cm. These are the
**defaults** from `QuadrotorParameters.default_parameters()`; override
them for other platforms (DJI F450, Crazyflie, X4-frame, etc.).

| Parameter | Value |
|-----------|-------|
| Mass $m$, kg | 1.5 |
| Inertia $J_x = J_y$, kg·m² | 0.0211 |
| Inertia $J_z$, kg·m² | 0.0366 |
| Arm length, m | 0.225 |
| Linear body-drag $k_{dx}=k_{dy}$, N·s/m | 0.10 |
| Linear body-drag $k_{dz}$, N·s/m | 0.20 |
| Maximum collective thrust, N | 30 (≈ 2:1 thrust-to-weight) |
| Maximum torque per axis, N·m | 1.5 |

## Coordinate systems and state

The model uses two right-handed orthogonal frames:

- **Body-fixed:** $x$ forward, $y$ right, $z$ down — coincides with
  NED when level.
- **Earth (NED, north-east-down):** $x$ north, $y$ east, $z$ down.
  Gravity is along $+z$.

Body-to-earth transformation is the standard ZYX (321) Euler rotation
matrix (yaw → pitch → roll).

State vector and control:

$$
\mathbf{x} = \begin{bmatrix}
x_e & y_e & z_e &
u_b & v_b & w_b &
\phi & \theta & \psi &
p & q & r
\end{bmatrix}^\top, \qquad
\mathbf{u} = \begin{bmatrix} T & \tau_x & \tau_y & \tau_z \end{bmatrix}^\top.
$$

=== "Variables"

    - **$x_e, y_e, z_e$** — earth-frame position (NED), m.
    - **$u_b, v_b, w_b$** — body-frame velocity, m/s.
    - **$\phi, \theta, \psi$** — roll, pitch, yaw (ZYX Euler), rad.
    - **$p, q, r$** — body-frame angular rates, rad/s.
    - **$T$** — collective thrust along body $-z$ (up when level), N.
    - **$\tau_x, \tau_y, \tau_z$** — body-frame torques (roll, pitch, yaw), N·m.

=== "Control abstraction level"

    The control vector $\mathbf{u}$ is **already the output** of the
    motor-mixing / allocation block. Mapping the 4 rotor speeds
    $\omega_i$ to $(T, \tau)$ depends on the configuration (X / +) and
    is **not** part of this model — it lives in a separate helper or
    inside the RL agent.

!!! note "Euler-angle singularity"
    At $|\theta| = \pi/2$ the kinematic mapping diverges through the
    $1/\cos\theta$ term (gimbal lock). Use a quaternion variant for
    aggressive acrobatic manoeuvres — currently not implemented.

## Mathematical model

The combined ODE has four blocks: position kinematics, velocity
dynamics, Euler-angle kinematics, angular-rate dynamics.

### 1. Position kinematics

$$
\dot{\mathbf{r}}_e = R_{eb}(\phi,\theta,\psi)\,\mathbf{v}_b,
$$

where $R_{eb}$ is the body-to-earth rotation (ZYX 321):

$$
R_{eb} = \begin{bmatrix}
c_\theta c_\psi & s_\phi s_\theta c_\psi - c_\phi s_\psi & c_\phi s_\theta c_\psi + s_\phi s_\psi \\
c_\theta s_\psi & s_\phi s_\theta s_\psi + c_\phi c_\psi & c_\phi s_\theta s_\psi - s_\phi c_\psi \\
-s_\theta & s_\phi c_\theta & c_\phi c_\theta
\end{bmatrix}
$$

(shorthand $c_\bullet = \cos\bullet$, $s_\bullet = \sin\bullet$).

### 2. Velocity dynamics (body frame)

Newton's equation in the body frame, including rotation, gravity,
thrust, and linear aerodynamic drag:

$$
m\,\dot{\mathbf{v}}_b
\;=\;
R_{be}\,\mathbf{F}_{\text{grav},e}
\;+\;
\mathbf{F}_{\text{thrust},b}
\;-\;
D\,\mathbf{v}_b
\;-\;
\boldsymbol\omega \times m\,\mathbf{v}_b,
$$

with:

- $\mathbf{F}_{\text{grav},e} = [0, 0, m g]^\top$ — gravity in NED (z-down),
- $\mathbf{F}_{\text{thrust},b} = [0, 0, -T]^\top$ — thrust along body $-z$,
- $D = \mathrm{diag}(k_{dx}, k_{dy}, k_{dz})$ — diagonal drag matrix,
- $\boldsymbol\omega = [p, q, r]^\top$.

### 3. Euler-angle kinematics (ZYX 321)

$$
\begin{bmatrix} \dot\phi \\ \dot\theta \\ \dot\psi \end{bmatrix}
=
\begin{bmatrix}
1 & s_\phi t_\theta & c_\phi t_\theta \\
0 & c_\phi & -s_\phi \\
0 & s_\phi / c_\theta & c_\phi / c_\theta
\end{bmatrix}
\begin{bmatrix} p \\ q \\ r \end{bmatrix},
$$

with $t_\theta = \tan\theta$.

### 4. Angular-rate dynamics (Newton-Euler)

With diagonal inertia $\mathbf{J} = \mathrm{diag}(J_x, J_y, J_z)$:

$$
\boxed{\;
\begin{aligned}
\dot p &= \bigl(\tau_x + (J_y - J_z)\,q\,r\bigr) / J_x, \\
\dot q &= \bigl(\tau_y + (J_z - J_x)\,p\,r\bigr) / J_y, \\
\dot r &= \bigl(\tau_z + (J_x - J_y)\,p\,q\bigr) / J_z.
\end{aligned}
\;}
$$

The $(J_i - J_j)\,\omega_i\,\omega_j$ terms are the gyroscopic
cross-axis coupling (Euler cross-product terms).

## Quick start

```python
import numpy as np

from tensoraerospace.aerospacemodel.quadrotor.nonlinear import NonlinearQuadrotor

# Hover at the origin
m = NonlinearQuadrotor(
    x0=np.zeros(12),
    dt=0.01,
    integrator="rk4",
)

# T = m·g holds the vehicle in equilibrium
u_hover = np.array([m.hover_thrust, 0.0, 0.0, 0.0])

for _ in range(1000):  # 10 seconds
    m.run_step(u_hover)

print(f"Final state (max |x|): {np.max(np.abs(m.current_state)):.2e}")
# → ≈ 0 (exact equilibrium with RK4)
```

### Hover with a perturbation and a small roll-torque

```python
import numpy as np

from tensoraerospace.aerospacemodel.quadrotor.nonlinear import (
    NonlinearQuadrotor,
    set_initial_state,
)

# Start with a slight 3-degree roll
m = NonlinearQuadrotor(
    x0=set_initial_state(phi=np.deg2rad(3.0)),
    dt=0.005,
    integrator="rk4",
)

# Hover thrust + small negative roll command (P-feedback)
T_hover = m.hover_thrust
for k in range(2000):
    tau_x = -0.05 * m.current_state[6]   # proportional feedback on phi
    u = np.array([T_hover, tau_x, 0.0, 0.0])
    m.run_step(u)

phi_final = np.rad2deg(m.current_state[6])
print(f"Final roll angle: {phi_final:.4f}°")
```

### Initial-state helper

```python
from tensoraerospace.aerospacemodel.quadrotor.nonlinear import set_initial_state

# Start: 5 m altitude (NED z = -5), 5° pitch
x0 = set_initial_state(z_e=-5.0, theta=np.deg2rad(5.0))
```

## Validation

ODE correctness is verified by five analytical tests in
`tests/aerospacemodel/quadrotor_test.py`:

| Test | Check | Tolerance |
|------|-------|-----------|
| Hover equilibrium | $T = m\,g$, zero torque → state unchanged over 10 s | $\max\|x\| < 10^{-9}$ |
| Free fall, no drag | $z_e(t) = \tfrac{1}{2}g\,t^2$ | $\delta < 10^{-3}$ |
| Free fall with drag | $z(t) = v_\infty t + v_\infty\tau(e^{-t/\tau} - 1)$ | $\delta < 10^{-3}$ |
| Single-step roll-torque | $p \approx \tau_x \cdot dt / J_x$ | $\delta < 10^{-9}$ |
| Gyroscopic coupling | $\dot p = (J_y-J_z)\,q\,r / J_x$ at $q=r=1$ | $\delta < 10^{-12}$ |

Plus 6 sanity checks: input dimensions, history accumulation, both
integrator backends (Euler + RK4), invalid-integrator-name handling.

## Allocation (mixer): bridge between virtual control and rotor speeds

The bare `NonlinearQuadrotor` model takes the virtual vector
$(T, \tau_x, \tau_y, \tau_z)$ — the level at which PID/MPC/RL
controllers naturally operate. To reproduce **rotor-level failures**
(needed for the FTC scenarios in Lu 2019, Wang 2019, Lanzon 2015), a
bidirectional `XConfigAllocator` is provided:

$$
\begin{bmatrix} T \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix}
=
\underbrace{
\begin{bmatrix}
k_T   &  k_T  &  k_T  &  k_T   \\
-k_T a &  k_T a&  k_T a& -k_T a \\
k_T a & -k_T a&  k_T a& -k_T a \\
k_M   &  k_M  & -k_M  & -k_M
\end{bmatrix}
}_{M\ \text{(X-config, PX4)}}
\begin{bmatrix} \omega_1^2 \\ \omega_2^2 \\ \omega_3^2 \\ \omega_4^2 \end{bmatrix}
$$

with $a = L/\sqrt 2$, $L$ the arm length, $k_T$ the thrust
coefficient, and $k_M$ the yaw-torque coefficient. The matrix is
full-rank, so the inverse is well-defined.

```python
from tensoraerospace.aerospacemodel.quadrotor import default_allocator

alloc = default_allocator()  # k_T=7.5e-6, k_M≈0.016·k_T, arm=0.225
v = alloc.mix(omega_squared)         # 4 ω² → [T, τ]
omega2 = alloc.unmix(v)              # [T, τ] → 4 ω² (may be < 0 for non-realisable v)
omega2 = alloc.saturate(omega2, 0, 1000)  # clip to physical bounds
```

The `NonlinearQuadrotor-v0` env supports **two action modes**:

```python
# Mode "virtual" (default): action = [T, τ_x, τ_y, τ_z]
env = gym.make("NonlinearQuadrotor-v0", initial_state=np.zeros(12),
               number_time_steps=1000, action_space="virtual")

# Mode "rotor": action = [ω₁², ω₂², ω₃², ω₄²]; env applies allocator internally
env = gym.make("NonlinearQuadrotor-v0", initial_state=np.zeros(12),
               number_time_steps=1000, action_space="rotor")
```

Without `damage_profile` both modes are equivalent (round-trip mix↔unmix).

## Damage subsystem (rotor-level events)

To reproduce the canonical FTC scenarios from the literature, the env
ships with an event-driven rotor-level failure system. Each rotor has
an effectiveness coefficient $\mu_i \in [0, 1]$, and the env applies
$\omega^2_{i,\text{eff}} = \mu_i \cdot \omega^2_{i,\text{cmd}}$ before
mixing back to $(T, \tau)$.

Three event types:

| Event | Semantics | Source |
|-------|-----------|--------|
| `RotorDamageEvent(rotor_id, mu)` | Instantaneous effectiveness loss on rotor $i$ | Lu et al. 2019 |
| `RotorLossEvent(rotor_id)` | Complete failure ($\mu = 0$) | Lanzon et al. 2015 |
| `MotorEfficiencyDecay(rotor_id, tau, mu_floor)` | Exponential wear $\dot\mu = -(1/\tau)(\mu - \mu_\text{floor})$ | gradual wear |

Three ready-made presets: `LANZON_M1_LOSS`, `LU_M1_50PCT_LOSS`,
`WEAR_DEGRADATION_M3`. Import from
`tensoraerospace.aerospacemodel.quadrotor.damage`.

```python
from tensoraerospace.aerospacemodel.quadrotor.damage import LANZON_M1_LOSS

env = gym.make("NonlinearQuadrotor-v0", initial_state=np.zeros(12),
               number_time_steps=2000, damage_profile=LANZON_M1_LOSS)
obs, _ = env.reset()
T_hover = 1.5 * 9.81
for k in range(2000):
    obs, r, term, trunc, info = env.step(np.array([T_hover, 0, 0, 0]))
    if "damage_events_triggered" in info:
        print(f"t={k*0.01}s: {info['damage_events_triggered']}")
```

Without `damage_profile` the env is bit-identical to the no-damage
baseline (rotor-effectiveness $\mu = 1$ on all four motors).

## Current limitations

1. **ZYX 321 Euler angles** — the model has gimbal lock at
   $|\theta| = \pi/2$. Acrobatic manoeuvres (loops, flips) need a
   quaternion variant — future extension.
2. **Linear drag only** — no quadratic-in-velocity term (significant
   above ~10 m/s cruise) and no blade-flap aerodynamics.
3. **Symmetric X-frame** in the allocator — asymmetric frames (Y, H,
   hex/octocopter) need their own allocator class.
4. **Element-wise saturation** only — no thrust-priority allocation
   (Faessler 2017) for control-saturation regimes where one channel
   should be preserved over another.

## References

### Dynamics and coordinate systems

- **Stevens, B. L., Lewis, F. L., Johnson, E. N.** (2015). *Aircraft
  Control and Simulation*, 3rd ed. Wiley. — methodological foundation
  for the 6-DoF rigid-body equations in the body-fixed frame (NED,
  ZYX Euler convention).
- **PX4 Airframe Reference**: [Quadrotor X](https://docs.px4.io/main/en/airframes/airframe_reference.html#quadrotor-x)
  — motor numbering and rotation-direction convention used in
  `XConfigAllocator`.

### Reference platform parameters

- **AscTec Hummingbird / Pelican research-class quadrotors** — typical
  values $m \approx 1.5$ kg, $J_x = J_y \approx 0.021$ kg·m²,
  $J_z \approx 0.037$ kg·m², 22.5 cm arm. Widely used in academic UAV
  FTC papers.

### FTC scenarios (sources for the damage subsystem and presets)

- **Lu, P., Yu, B., van Kampen, E.-J., Chu, Q. P.** (2019).
  *Quadrotor Fault Tolerant Incremental Sliding Mode Control driven by
  Sliding Mode Disturbance Observers*. Aerospace Science and Technology,
  87:417–430. [DOI: 10.1016/j.ast.2019.03.001](https://doi.org/10.1016/j.ast.2019.03.001)
  — multiplicative rotor effectiveness loss; basis for the
  `LU_M1_50PCT_LOSS` preset and the `RotorDamageEvent` class.
  **129 citations**.
- **Wang, X., van Kampen, E.-J., Chu, Q. P.** (2019). *Quadrotor
  fault-tolerant incremental nonsingular terminal sliding mode control*.
  Aerospace Science and Technology, 95:105514.
  [DOI: 10.1016/j.ast.2019.105514](https://doi.org/10.1016/j.ast.2019.105514)
  — parallel work using terminal sliding mode; same fault model.
- **Lanzon, A., Freddi, A., Longhi, S.** (2015). *Active fault-tolerant
  control for quadrotors subjected to a complete rotor failure*.
  IEEE/RSJ IROS 2015. [DOI: 10.1109/IROS.2015.7354046](https://doi.org/10.1109/IROS.2015.7354046)
  — extreme case of complete rotor failure (spin-mode recovery);
  basis for `LANZON_M1_LOSS` and the `RotorLossEvent` class.

### Related TensorAeroSpace topics

- [Aircraft damage modeling (F-16)](aircraft-damage-modeling.md) — a
  more developed damage subsystem for fixed-wing aircraft, with
  strip-theory and Huygens-Steiner inertia recompute.
- [AIDI](../agent/aidi.md) — adaptive INDI-style architecture
  applicable to this quadrotor model for fault-tolerant control.
- [iADP](../agent/iadp.md), [IM-GDHP](../agent/imgdhp.md) — online
  adaptive critics that recover from plant changes within tens of
  milliseconds.

## Python API

::: tensoraerospace.aerospacemodel.quadrotor.nonlinear.NonlinearQuadrotor

::: tensoraerospace.aerospacemodel.quadrotor.nonlinear.params.QuadrotorParameters

::: tensoraerospace.aerospacemodel.quadrotor.nonlinear.dynamics.quadrotor_ode_6dof

::: tensoraerospace.aerospacemodel.quadrotor.allocation.XConfigAllocator

::: tensoraerospace.aerospacemodel.quadrotor.damage.events.RotorDamageEvent

::: tensoraerospace.aerospacemodel.quadrotor.damage.events.RotorLossEvent

::: tensoraerospace.aerospacemodel.quadrotor.damage.events.MotorEfficiencyDecay

::: tensoraerospace.aerospacemodel.quadrotor.damage.manager.RotorDamageManager

::: tensoraerospace.envs.quadrotor.NonlinearQuadrotorEnv
