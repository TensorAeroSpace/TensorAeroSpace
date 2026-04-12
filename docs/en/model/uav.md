# UAV — Longitudinal Dynamics

An unmanned aerial vehicle (UAV) is a remotely piloted or autonomous aircraft. This page mirrors the ELV layout: quick start, math model, derivatives, and API.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Quick start**

    Launch the environment or the model within minutes.

    [:octicons-arrow-right-24: See example](#quick-start)

-   :material-cog-outline: **Model API**

    Python class documentation for the UAV longitudinal dynamics.

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
    - **η**: stabilizer control deflection, rad

=== "Coefficients"

    - **x_u, x_w, x_q, x_θ** — partial derivatives of longitudinal force \(X\) with respect to \(u, w, q, \theta\)
    - **z_u, z_w, z_q, z_θ** — partial derivatives of normal force \(Z\)
    - **m_u, m_w, m_q, m_θ** — partial derivatives of pitch moment \(M\)
    - **x_η, z_η, m_η** — derivatives with respect to the control \(\eta\)

!!! note "Units"
    Angles and angular rates are in radians. API methods can produce values in degrees.

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
-0.1982 & 0.593 & 1.245 & -9.779 \\
-0.7239 & -3.9848 & 18.7028 & -0.6286 \\
0.3537 & -5.5023 & -5.4722 & 0 \\
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
0.2281 \\
-4.6830  \\
-36.1341 \\
0.0
\end{bmatrix}
\eta
\]

### Derivatives (numerical values)

- **Matrix A (derivatives):**

  | Coefficient | Value |
  |-------------|----------|
  | x_u | -0.1982 |
  | x_w | 0.593 |
  | x_q | 1.245 |
  | x_θ | -9.779 |
  | z_u | -0.7239 |
  | z_w | -3.9848 |
  | z_q | 18.7028 |
  | z_θ | -0.6286 |
  | m_u | 0.3537 |
  | m_w | -5.5023 |
  | m_q | -5.4722 |
  | m_θ | 0.0 |

- **Input η (column B):**

  | Coefficient | Value |
  |-------------|----------|
  | x_η | 0.2281 |
  | z_η | -4.6830 |
  | m_η | -36.1341 |

## Sources

1. A. Rauf, Muhammad Aamir Zafar, Z. Ashraf and H. Akhtar, "Aerodynamic modeling and state-space model extraction of a UAV using DATCOM and Simulink," 2011 3rd International Conference on Computer Research and Development, Shanghai, China, 2011, pp. 88-92, doi: 10.1109/ICCRD.2011.5763860.

## Reward

The default reward function returns the negative absolute tracking error for the pitch angle:

$$r_t = -|\theta(t) - \theta_{\text{ref}}(t)|$$

Higher reward (closer to 0) indicates better tracking performance. A custom reward function can be passed via the `reward_func` parameter.

## Quick start {#quick-start}

=== "Gymnasium"

    ```python
    import gymnasium as gym 
    import numpy as np

    from tensoraerospace.envs import LinearLongitudinalUAV
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import unit_step

    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)
    reference_signals = unit_step(degree=5, tp=tp, time_step=10, output_rad=True).reshape(1, -1)

    env = gym.make(
        'LinearLongitudinalUAV-v0',
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
    from tensoraerospace.aerospacemodel import LongitudinalUAV

    dt = 0.01
    number_time_steps = 200

    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    model = LongitudinalUAV(
        x0=x0,
        number_time_steps=number_time_steps,
        selected_state_output=["u", "w", "q", "theta"],
        dt=dt,
    )

    for t in range(number_time_steps - 1):
        u = np.array([[0.05]])
        x_next = model.run_step(u)
    ```

## Python API

=== "Model"

    ::: tensoraerospace.aerospacemodel.uav.LongitudinalUAV

=== "Gymnasium environment"

    ::: tensoraerospace.envs.uav.LinearLongitudinalUAV