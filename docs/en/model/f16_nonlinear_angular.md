# F-16 Fighting Falcon — Nonlinear 6-DoF Angular Dynamics

This module provides a **nonlinear 6-DoF angular** model of the F-16 Fighting Falcon, implemented in pure Python/NumPy. It covers the full coupled dynamics: longitudinal, lateral, and directional channels. The aerodynamic coefficients (six force and moment components) are interpolated from wind-tunnel tables.

![F-16 Model](img/f-16_fighting_falcon.jpg){ width=800 }

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Quick start**

    Run the angular model in just a few lines of code.

    [:octicons-arrow-right-24: See example](#quick-start)

-   :material-cog-outline: **Model API**

    Python class documentation for the 6-DoF dynamics.

    [:octicons-arrow-right-24: Go to API](#python-api)

-   :material-book-open-variant: **Theory**

    Nonlinear 6-DoF equations of motion.

    [:octicons-arrow-right-24: Learn more](#mathematical-model)

</div>

## Control object structure

The model describes the full angular motion of the F-16 with three independent control surfaces: stabilator (elevator), ailerons, and rudder.

**State vector** (14 elements):

\[
x = \begin{bmatrix}
\alpha & \beta & \omega_x & \omega_y & \omega_z & \gamma & \psi & \theta &
\delta_{\text{stab}} & \dot{\delta}_{\text{stab}} &
\delta_{\text{ail}} & \dot{\delta}_{\text{ail}} &
\delta_{\text{dir}} & \dot{\delta}_{\text{dir}}
\end{bmatrix}^{\top}
\]

**Control vector** (3 elements):

\[
u = \begin{bmatrix}
\delta_{\text{stab,act}} & \delta_{\text{ail,act}} & \delta_{\text{dir,act}}
\end{bmatrix}^{\top}
\]

=== "State variables"

    | Variable | Symbol | Description |
    |----------|--------|-------------|
    | `alpha` | \(\alpha\) | Angle of attack, rad |
    | `beta` | \(\beta\) | Sideslip angle, rad |
    | `wx` | \(\omega_x\) | Roll rate, rad/s |
    | `wy` | \(\omega_y\) | Yaw rate, rad/s |
    | `wz` | \(\omega_z\) | Pitch rate, rad/s |
    | `gamma` | \(\gamma\) | Bank angle, rad |
    | `psi` | \(\psi\) | Heading angle, rad |
    | `theta` | \(\theta\) | Pitch angle, rad |
    | `stab` | \(\delta_{\text{stab}}\) | Stabilator position, rad |
    | `dstab` | \(\dot{\delta}_{\text{stab}}\) | Stabilator rate, rad/s |
    | `ail` | \(\delta_{\text{ail}}\) | Aileron position, rad |
    | `dail` | \(\dot{\delta}_{\text{ail}}\) | Aileron rate, rad/s |
    | `dir` | \(\delta_{\text{dir}}\) | Rudder position, rad |
    | `ddir` | \(\dot{\delta}_{\text{dir}}\) | Rudder rate, rad/s |

=== "Control variables"

    | Variable | Symbol | Description |
    |----------|--------|-------------|
    | `stab_act` | \(\delta_{\text{stab,act}}\) | Stabilator command, rad |
    | `ail_act` | \(\delta_{\text{ail,act}}\) | Aileron command, rad |
    | `dir_act` | \(\delta_{\text{dir,act}}\) | Rudder command, rad |

=== "Default parameters"

    | Parameter | Symbol | Value |
    |-----------|--------|-------|
    | Aircraft mass | \(m\) | 9295.44 kg |
    | Fuselage length | \(l\) | 9.144 m |
    | Wing area | \(S\) | 27.87 m² |
    | Wingspan | \(b_A\) | 3.45 m |
    | Roll inertia | \(J_x\) | 12874.8 kg m² |
    | Pitch inertia | \(J_y\) | 85552.1 kg m² |
    | Yaw inertia | \(J_z\) | 75673.6 kg m² |
    | Cross-product of inertia | \(J_{xy}\) | 1331.4 kg m² |
    | Altitude | \(H\) | 3000 m |
    | Airspeed | \(V\) | 120 m/s |

!!! note "Units"
    Inside the model, all angles and angular rates are in radians.

## Mathematical model {#mathematical-model}

### Aerodynamic forces and moments

Six aerodynamic coefficients are computed from tabular data at every time step:

\[
C_x, C_y, C_z, m_x, m_y, m_z = f(\alpha, \beta, \delta_{\text{stab}}, \delta_{\text{ail}}, \delta_{\text{dir}}, \delta_{\text{lef}}, \omega_x, \omega_y, \omega_z, V, \ldots)
\]

The body-frame forces and moments:

\[
X = -qSC_x, \quad Y = qSC_y, \quad Z = qSC_z
\]

\[
M_x = qSlm_x, \quad M_y = qSlm_y, \quad M_z = qSb_Am_z
\]

with CG offset corrections:

\[
M_{Ry} = M_y - x_{\text{cg}} Z, \qquad M_{Rz} = M_z + x_{\text{cg}} Y
\]

### Angular momentum equations

\[
\dot{\omega}_x = \frac{J_y M_{Rx} + J_{xy}(M_{Ry} - h_{Ex}\omega_z) + J_{xy}(J_z - J_x - J_y)\omega_x\omega_z + (J_{xy}^2 + J_y(J_y - J_z))\omega_y\omega_z}{\Gamma}
\]

\[
\dot{\omega}_z = \frac{M_{Rz} + h_{Ex}\omega_y + J_{xy}(\omega_x^2 - \omega_y^2) + (J_x - J_y)\omega_x\omega_y}{J_z}
\]

where \(\Gamma = J_x J_y - J_{xy}^2\).

### Angle-of-attack and sideslip

\[
\dot{\alpha} = \omega_z + (\omega_y \sin\alpha - \omega_x \cos\alpha)\tan\beta - \frac{Y_a + mg_{ay}}{mV\cos\beta}
\]

\[
\dot{\beta} = \omega_x \sin\alpha + \omega_y \cos\alpha + \frac{Z_a + mg_{az}}{mV}
\]

### Euler angle kinematics

\[
\dot{\gamma} = \omega_x - \cos\gamma \tan\theta \cdot \omega_y + \sin\gamma \tan\theta \cdot \omega_z
\]

\[
\dot{\theta} = \sin\gamma \cdot \omega_y + \cos\gamma \cdot \omega_z
\]

\[
\dot{\psi} = \frac{\cos\gamma}{\cos\theta} \omega_y - \frac{\sin\gamma}{\cos\theta} \omega_z
\]

### Actuator models

Each control surface is modelled as a second-order system with position and rate limiting:

| Surface | Time constant | Damping | Max deflection | Max rate |
|---------|---------------|---------|----------------|----------|
| Stabilator | 0.03 s | 0.707 | \(\pm 25°\) | \(\pm 60°/\text{s}\) |
| Aileron | 0.02 s | 0.707 | \(\pm 21.5°\) | \(\pm 80°/\text{s}\) |
| Rudder | 0.03 s | 0.707 | \(\pm 30°\) | \(\pm 120°/\text{s}\) |

### Integration methods

Two numerical integrators are available:

- **Euler** (default) — first-order forward Euler.
- **RK4** — fourth-order Runge-Kutta, higher accuracy at the same step.

## Data source

1. Stevens & Lewis, "Aircraft Control and Simulation".
2. Wind-tunnel aerodynamic tables for F-16A, stored as NumPy `.npz` archives.

## Quick start {#quick-start}

```python
import numpy as np
from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16

dt = 0.01
number_time_steps = 500

# State (14 elements): [alpha, beta, wx, wy, wz, gamma, psi, theta,
#                        stab, dstab, ail, dail, dir, ddir]
x0 = np.zeros(14)
x0[0] = np.radians(2.0)  # initial alpha = 2 deg

model = AngularF16(
    x0=x0,
    selected_state_output=["alpha", "beta", "wz"],
    dt=dt,
    integrator="rk4",
)

for t in range(number_time_steps - 1):
    u = np.array([np.radians(-2.0), 0.0, 0.0])  # [stab, ail, dir] (rad)
    state_next = model.run_step(u)
```

## Python API

=== "Model"

    ::: tensoraerospace.aerospacemodel.f16.nonlinear.angular.model.AngularF16

=== "Parameters"

    ::: tensoraerospace.aerospacemodel.f16.nonlinear.angular.params.F16AngularParameters
