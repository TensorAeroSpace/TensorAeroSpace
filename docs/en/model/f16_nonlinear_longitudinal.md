# F-16 Fighting Falcon — Nonlinear Longitudinal Dynamics

The General Dynamics F-16 Fighting Falcon is an American multirole lightweight 4th-generation fighter. This module provides a **nonlinear** longitudinal flight-channel model implemented in pure Python/NumPy. The aerodynamic coefficients are interpolated with cubic splines from wind-tunnel tables, providing high-fidelity dynamics across a wide range of angles of attack and control deflections. A Gymnasium environment is included for training control agents.

![F-16 Model](img/f-16_fighting_falcon.jpg){ width=800 }

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Quick start**

    Run the environment or the model in just a few lines of code.

    [:octicons-arrow-right-24: See example](#quick-start)

-   :material-cog-outline: **Model API**

    Python class documentation for the nonlinear F-16 longitudinal dynamics.

    [:octicons-arrow-right-24: Go to API](#python-api)

-   :material-gamepad-variant-outline: **Gymnasium environment**

    Ready-to-use environment for training control agents.

    [:octicons-arrow-right-24: Explore](#python-api)

-   :material-book-open-variant: **Theory**

    Nonlinear state equations and aerodynamic table structure.

    [:octicons-arrow-right-24: Learn more](#mathematical-model)

</div>

## Performance specs (reference)

| Parameter | Value |
|-------------|----------------|
| Variant | F-16A Block 10 |
| Wingspan, m | 9.45 |
| Aircraft length (with pitot boom), m | 15.03 |
| Aircraft height, m | 5.09 |
| Wing area, m² | 27.87 |
| Sweep angle, ° | 40.0 |
| Normal takeoff weight, kg | 11467 |

## Control object structure

Unlike the [linear F-16 model](f16.md), which uses constant-coefficient matrices \(A\) and \(B\), the nonlinear model computes aerodynamic forces and moments from tabular data at every time step. The equations of motion are integrated numerically (Euler or RK4).

The state vector and control input:

\[
 x = \begin{bmatrix} \alpha & \omega_z & \delta_{\text{stab}} & \dot{\delta}_{\text{stab}} \end{bmatrix}^{\top}, \quad
 u = \delta_{\text{stab,act}}
\]

=== "Variables"

    - **\(\alpha\)**: angle of attack, rad
    - **\(\omega_z\)**: pitch angular velocity, rad/s
    - **\(\delta_{\text{stab}}\)**: stabilator deflection, rad
    - **\(\dot{\delta}_{\text{stab}}\)**: stabilator deflection rate, rad/s
    - **\(\delta_{\text{stab,act}}\)**: stabilator command (control input), rad

=== "Default parameters"

    | Parameter | Symbol | Value |
    |-----------|--------|-------|
    | Aircraft mass | \(m\) | 9295.44 kg |
    | Wing area | \(S\) | 27.87 m² |
    | Mean aerodynamic chord | \(b_A\) | 3.45 m |
    | Pitch inertia | \(J_z\) | 75673.6 kg m² |
    | Stabilator time constant | \(T_{\text{stab}}\) | 0.03 s |
    | Stabilator damping ratio | \(\xi_{\text{stab}}\) | 0.707 |
    | Altitude | \(H\) | 3000 m |
    | Airspeed | \(V\) | 150 m/s |
    | Gravity | \(g\) | 9.80665 m/s² |

!!! note "Units"
    Inside the model, all angles and angular rates are in radians. The Gymnasium environment accepts actions in **degrees** (range \(\pm 25°\)) for compatibility with existing agents and converts to radians internally.

## Mathematical model {#mathematical-model}

The model solves the following system of nonlinear ODEs at each time step:

\[
\dot{\alpha} = \omega_z - \frac{R_y - mg}{mV}
\]

\[
\dot{\omega}_z = \frac{M_{Rz}}{J_z}
\]

where the aerodynamic force and moment are:

\[
R_y = qS \cdot C_y(\alpha, \beta, \delta_{\text{stab}}, \delta_{\text{lef}}, \omega_z, V, b_A, \delta_{\text{sb}})
\]

\[
M_{Rz} = qS b_A \cdot m_z(\alpha, \beta, \delta_{\text{stab}}, \delta_{\text{lef}}, \omega_z, V, b_A, \delta_{\text{sb}}) + x_{\text{cg}} \cdot R_y
\]

The dynamic pressure \(q\) is computed from the ISA atmosphere model:

\[
q = \frac{1}{2} \rho(H) V^2, \qquad \rho = \rho_0 \left(\frac{T_0 - LH}{T_0}\right)^{\frac{g}{LR} - 1}
\]

### Aerodynamic tables

The coefficients \(C_y\) and \(m_z\) are not constant — they are multi-dimensional functions interpolated from wind-tunnel data stored as `.npz` files. The tables are interpolated using cubic splines (`csaps`).

### Actuator model

The stabilator is modelled as a second-order system with position and rate limiting:

\[
\ddot{\delta}_{\text{stab}} = \frac{-2 T_{\text{stab}} \xi_{\text{stab}} \dot{\delta}_{\text{stab}} - \delta_{\text{stab}} + \delta_{\text{stab,act}}}{T_{\text{stab}}^2}
\]

!!! tip "Actuator limits"
    The model uses the following default control limits:

    - Maximum stabilator deflection: \(\pm 25°\)
    - Maximum stabilator rate: \(\pm 60°/\text{s}\)

### Integration methods

Two numerical integrators are available:

- **Euler** (default) — first-order forward Euler.
- **RK4** — fourth-order Runge-Kutta, provides higher accuracy at the same time step.

## Data source

1. Stevens & Lewis, "Aircraft Control and Simulation".
2. Wind-tunnel aerodynamic tables for F-16A, stored as NumPy `.npz` archives.

## Quick start {#quick-start}

=== "Gymnasium"

    ```python
    import gymnasium as gym
    import numpy as np

    from tensoraerospace.envs import NonlinearLongitudinalF16
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import sinusoid

    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)
    reference_signal = sinusoid(
        degree=3, tp=tp, frequency=0.1, output_rad=True
    ).reshape(1, -1)

    env = gym.make(
        'NonlinearLongitudinalF16-v0',
        number_time_steps=number_time_steps,
        initial_state=np.array([0.0, 0.0]),
        reference_signal=reference_signal,
        dt=dt,
        integrator="euler",
    )

    state, info = env.reset()
    for _ in range(number_time_steps - 1):
        action = np.array([0.0])  # degrees
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    ```

=== "Model only"

    ```python
    import numpy as np
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import LongitudinalF16

    dt = 0.01
    number_time_steps = 200

    # State: [alpha, wz, stab, dstab] (rad)
    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    model = LongitudinalF16(
        x0=x0,
        selected_state_output=["alpha", "wz"],
        dt=dt,
        integrator="rk4",
    )

    for t in range(number_time_steps - 1):
        u = np.array([np.radians(-2.0)])  # stabilator command (rad)
        state_next = model.run_step(u)
    ```

## Python API

=== "Model"

    ::: tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.model.LongitudinalF16

=== "Gymnasium environment"

    ::: tensoraerospace.envs.f16.nonlinear_longitudinal.NonlinearLongitudinalF16

=== "Parameters"

    ::: tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params.F16LongParameters
