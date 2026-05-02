# McDonnell Douglas F‑4C — Longitudinal Dynamics

The F‑4C Phantom II is an American 3rd‑generation fighter bomber. This page mirrors the ELV layout: quick start, math model, derivatives, and API.

![F4C Model](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/QF-4_Holloman_AFB.jpg/1024px-QF-4_Holloman_AFB.jpg){ width=800 }

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Quick start**

    Launch the environment or the model within minutes.

    [:octicons-arrow-right-24: See example](#quick-start)

-   :material-cog-outline: **Model API**

    Python class documentation for the F‑4C longitudinal dynamics.

    [:octicons-arrow-right-24: Go to API](#python-api)

-   :material-gamepad-variant-outline: **Gymnasium environment**

    Ready environment for RL agents.

    [:octicons-arrow-right-24: Explore](#python-api)

-   :material-book-open-variant: **Theory**

    State equations and numerical parameters.

    [:octicons-arrow-right-24: Learn more](#mathematical-model)

</div>

## Control object structure

The model is defined in the state space:

\[\dot{x} = A x + B u, \quad y = C x + D u\]

where:

\[
 x = \begin{bmatrix} u & w & q & \theta \end{bmatrix}^{\top}, \quad
 u_{in} = \delta_e
\]

The typical matrix structure is:

\[
\begin{bmatrix}
\dot{u} \\
\dot{w} \\
\dot{q} \\
\dot{\theta}
\end{bmatrix}
=
\begin{bmatrix}
x_u & x_w & x_q & x_{\theta} \\
z_u & z_w & z_q & z_{\theta} \\
m_u & m_w & m_q & m_{\theta} \\
0 & 0 & 1 & 0
\end{bmatrix}
\begin{bmatrix} u \\ w \\ q \\ \theta \end{bmatrix}
 +
\begin{bmatrix} x_{\delta_e} \\ z_{\delta_e} \\ m_{\delta_e} \\ 0 \end{bmatrix} \delta_e
\]

=== "Variables"

    - **u**: longitudinal speed, m/s
    - **w**: vertical speed, m/s
    - **q**: pitch rate, rad/s
    - **θ**: pitch angle, rad
    - **δₑ**: elevator deflection, rad

=== "Coefficients"

    - **x_u, x_w, x_q, x_θ** — partial derivatives of longitudinal force \(X\) with respect to \(u, w, q, \theta\)
    - **z_u, z_w, z_q, z_θ** — partial derivatives of normal force \(Z\)
    - **m_u, m_w, m_q, m_θ** — partial derivatives of pitch moment \(M\)
    - **x_δₑ, z_δₑ, m_δₑ** — derivatives with respect to the control \(\delta_e\)

!!! note "Units"
    Angles and angular rates are in radians. API methods can return values in degrees.

## Mathematical model {#mathematical-model}

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

The model is described by the standard state-space equation:

$$
\dot{x} = Ax + Bu
$$

where:

- **x** — state vector, \(x = [u, w, q, \theta]^{\top}\), representing deviations of longitudinal velocity, vertical velocity, pitch rate, and pitch angle.
- **u** — control vector, in this case \(u = [\delta_e]\), where \(\delta_e\) is the elevator deflection.
- **A** — state matrix (or system matrix).
- **B** — control matrix.

### Units

State vector \(x = [u, w, q, \theta]^{\top}\):

- **u, w**: m/s (velocities)
- **q**: rad/s (angular velocity)
- **θ**: rad (angle)

Control vector \(u = [\delta_e]\):

- **δₑ**: rad (elevator deflection)

### Flight Conditions

The computed matrices **A** and **B** for the F-4C are provided for the following flight conditions:

- **Mach number**: 0.6
- **Altitude**: 35,000 feet

### Numerical Matrices

\[
\begin{bmatrix}
\dot{u} \\
\dot{w} \\
\dot{q} \\
\dot{\theta}
\end{bmatrix}
=
\begin{bmatrix}
7.6180 \times 10^{-4} & 4.7612 \times 10^{-3} & 0 & -9.8100 \\
-6.6657 \times 10^{-2} & -2.8567 \times 10^{-1} & 1.8000 \times 10^{2} & 0 \\
1.5124 \times 10^{-3} & -1.0083 \times 10^{-2} & -1.6384 \times 10^{-1} & 0 \\
0 & 0 & 1 & 0 
\end{bmatrix}
\begin{bmatrix}
u \\
w \\
q \\
\theta 
\end{bmatrix}
 +
\begin{bmatrix}
2.6533 \times 10^{-3} \\
-6.8562 \\
-5.4446 \\
0
\end{bmatrix}
\delta_e
\]

### Derivatives (numerical values)

- **Matrix A (derivatives):**

  | Coefficient | Value |
  |-------------|----------|
  | x_u | 7.6180 × 10⁻⁴ |
  | x_w | 4.7612 × 10⁻³ |
  | x_q | 0.0 |
  | x_θ | -9.8100 |
  | z_u | -6.6657 × 10⁻² |
  | z_w | -2.8567 × 10⁻¹ |
  | z_q | 1.8000 × 10² |
  | z_θ | 0 |
  | m_u | 1.5124 × 10⁻³ |
  | m_w | -1.0083 × 10⁻² |
  | m_q | -1.6384 × 10⁻¹ |
  | m_θ | 0 |

- **Input δₑ (column B):**

  | Coefficient | Value |
  |-------------|----------|
  | x_δₑ | 2.6533 × 10⁻³ |
  | z_δₑ | -6.8562 |
  | m_δₑ | -5.4446 |

!!! tip "Actuator limits"
    The default control bounds are:

    - Maximum magnitude: \(\pm 25^\circ\)
    - Maximum rate: \(60^\circ/\text{s\)

    Internal computations use radians; the bounds are converted accordingly.

## Sources

1. Heffley R. K., Jewell W. F. Aircraft handling qualities data. – NASA, 1972. № AD‑A277031.
2. Etkin B., Reid L. D. Dynamics of flight. – New York : Wiley, 1959. – Vol. 2

## Reward

The default reward function returns the negative absolute tracking error for the pitch angle:

$$r_t = -|\theta(t) - \theta_{\text{ref}}(t)|$$

Higher reward (closer to 0) indicates better tracking performance. A custom reward function can be passed via the `reward_func` parameter.

## Quick start {#quick-start}

=== "Gymnasium"

    ```python
    import gymnasium as gym
    import numpy as np

    from tensoraerospace.envs import LinearLongitudinalF4C
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import unit_step

    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)
    reference_signals = unit_step(degree=5, tp=tp, time_step=10, output_rad=True).reshape(1, -1)

    env = gym.make(
        'LinearLongitudinalF4C-v0',
        number_time_steps=number_time_steps,
        initial_state=[[0],[0],[0],[0]],
        reference_signal=reference_signals,
    )
    state, info = env.reset()
    for _ in range(200):
        action = np.array([[0.1]])
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    ```

=== "Model only"

    ```python
    import numpy as np
    from tensoraerospace.aerospacemodel import LongitudinalF4C

    dt = 0.01
    number_time_steps = 200

    x0 = np.array([0.0, 0.0, 0.0, 0.0])  # [u, w, q, theta]

    model = LongitudinalF4C(
        x0=x0,
        number_time_steps=number_time_steps,
        selected_state_output=["u", "w", "q", "theta"],
        dt=dt,
    )

    for t in range(number_time_steps - 1):
        u = np.array([[0.05]])  # control (rad)
        x_next = model.run_step(u)
    ```

## Python API

=== "Model"

    ::: tensoraerospace.aerospacemodel.f4c.LongitudinalF4C

=== "Gymnasium environment"

    ::: tensoraerospace.envs.f4c.LinearLongitudinalF4C