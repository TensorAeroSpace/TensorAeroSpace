# McDonnell Douglas F‑4C — Longitudinal Dynamics

The F‑4C Phantom II is an American 3rd‑generation fighter bomber. This page mirrors the ELV layout: quick start, math model, derivatives, and API.

![Модель F4C](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/QF-4_Holloman_AFB.jpg/1024px-QF-4_Holloman_AFB.jpg){ width=800 }

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
 u_{in} = \eta
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
\begin{bmatrix} x_{\eta} \\ z_{\eta} \\ m_{\eta} \\ 0 \end{bmatrix} \eta
\]

=== "Variables"

    - **u**: longitudinal speed, m/s
    - **w**: vertical speed, m/s
    - **q**: pitch rate, rad/s
    - **θ**: pitch angle, rad
    - **η**: stabilator control deflection, rad

=== "Coefficients"

    - **x_u, x_w, x_q, x_θ** — partial derivatives of longitudinal force \(X\) with respect to \(u, w, q, \theta\)
    - **z_u, z_w, z_q, z_θ** — partial derivatives of normal force \(Z\)
    - **m_u, m_w, m_q, m_θ** — partial derivatives of pitch moment \(M\)
    - **x_η, z_η, m_η** — derivatives with respect to the control \(\eta\)

!!! note "Units"
    Angles and angular rates are in radians. API methods can return values in degrees.

## Mathematical model {#mathematical-model}

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

Numerical matrices (example linearization):

\[
\begin{bmatrix}
\dot{u} \\
\dot{w} \\
\dot{q} \\
\dot{\theta}
\end{bmatrix}
=
\begin{bmatrix}
-0.00679 & 0.00146 & 0 & -32.174 \\
0.0110 & -0.4940 & 1469.7600 & 0 \\
0.003410 & -0.019781184 & -0.4879811 & 0 \\
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
0.0027 \\
-0.0584 \\
-0.0001309 \\
0
\end{bmatrix}
\eta
\]

### Derivatives (numerical values)

- **Matrix A (derivatives):**

  | Коэффициент | Значение |
  |-------------|----------|
  | x_u | -0.00679 |
  | x_w | 0.00146 |
  | x_q | 0.0 |
  | x_θ | -32.174 |
  | z_u | 0.0110 |
  | z_w | -0.4940 |
  | z_q | 1469.7600 |
  | z_θ | 0 |
  | m_u | 0.003410 |
  | m_w | -0.019781184 |
  | m_q | -0.4879811 |
  | m_θ | 0 |

- **Input η (column B):**

  | Коэффициент | Значение |
  |-------------|----------|
  | x_η | 0.0027 |
  | z_η | -0.0584 |
  | m_η | -0.0001309 |

!!! tip "Actuator limits"
    The default control bounds are:

    - Maximum magnitude: \(\pm 25^\circ\)
    - Maximum rate: \(60^\circ/\text{s\)

    Internal computations use radians; the bounds are converted accordingly.

## Sources

1. Heffley R. K., Jewell W. F. Aircraft handling qualities data. – NASA, 1972. № AD‑A277031.
2. Etkin B., Reid L. D. Dynamics of flight. – New York : Wiley, 1959. – Т. 2

## Quick start {#quick-start}

=== "Gymnasium"

    ```python
    import gymnasium as gym
    import numpy as np

    from tensoraerospace.envs import LinearLongitudinalF4C
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import unit_step

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