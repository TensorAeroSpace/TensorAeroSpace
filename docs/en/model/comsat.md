# Communication Satellite (ComSat) — Longitudinal Dynamics

A communications satellite operates in orbit to relay and process radio signals. This page mirrors the ELV layout: quick start, math model, derivative tables, and API.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Quick start**

    Launch the environment or the model within minutes.

    [:octicons-arrow-right-24: See example](#quick-start)

-   :material-cog-outline: **Model API**

    Python class documentation for ComSat.

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
 x = \begin{bmatrix} \rho & \theta & \omega \end{bmatrix}^{\top}, \quad
 u_{in} = \eta
\]

The typical matrix structure is:

\[
\begin{bmatrix}
\dot{\rho} \\
\dot{\theta} \\
\dot{\omega}
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 & 0 \\
f_1(\rho, \omega) & 0 & f_2(\rho, \omega) \\
0 & f_3(\omega, r) & 0
\end{bmatrix}
\begin{bmatrix} \rho \\ \theta \\ \omega \end{bmatrix}
 +
\begin{bmatrix} 0 \\ 0 \\ g(r) \end{bmatrix} \eta
\]

=== "Variables"

    - **ρ**: altitude-to-Earth-radius ratio (dimensionless)
    - **θ**: satellite position relative to the Earth frame, rad
    - **ω**: angular velocity, rad/s
    - **η**: control input (thrust), N (normalized in the model)

=== "Coefficients"

    - **f1(ρ, ω) ≈ 0.01036** — derivative with respect to ρ (linearized term)
    - **f2(ρ, ω) ≈ 0.7753** — derivative with respect to ω in the θ̇ equation
    - **f3(ω, r) ≈ -0.1774** — derivative with respect to θ in the ω̇ equation
    - **g(r) ≈ 0.1512** — thrust influence on ω̇

!!! note "Units"
    Angles and angular rates are in radians. API methods can output in degrees.

## Mathematical model {#mathematical-model}

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

Numerical matrices (example linearization):

\[
\begin{bmatrix}
\dot{\rho} \\
\dot{\theta} \\
\dot{\omega}
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 & 0 \\
0.01036 & 0 & 0.7753 \\
0 & -0.1774 & 0 
\end{bmatrix}
\begin{bmatrix}
\rho \\
\theta \\
\omega 
\end{bmatrix}
 +
\begin{bmatrix}
0 \\
0 \\
0.1512
\end{bmatrix}
\eta
\]

### Derivatives (numerical values)

- **Matrix A (derivatives):**

  | Коэффициент | Значение |
  |-------------|----------|
  | a_ρθ (∂ρ̇/∂θ) | 1.0 |
  | a_θρ (∂θ̇/∂ρ) | 0.01036 |
  | a_θω (∂θ̇/∂ω) | 0.7753 |
  | a_ωθ (∂ω̇/∂θ) | -0.1774 |

- **Input η (column B):**

  | Коэффициент | Значение |
  |-------------|----------|
  | b_η→ω (∂ω̇/∂η) | 0.1512 |

!!! tip "Actuator limits"
    Default control limits inside the model (normalized):

    - Maximum magnitude: \(\pm 25^\circ\)
    - Maximum rate: \(60^\circ/\text{s\)

    Internal computations use radians; limits are converted accordingly.

## Sources

1. Santosh Kumar Choudhary (2015). Design and Analysis of an Optimal Orbit Control for a Communication Satellite. INTERNATIONAL JOURNAL OF COMMUNICATIONS. Volume 9, 2015

## Quick start {#quick-start}

=== "Gymnasium"

    ```python
    import gymnasium as gym
    import numpy as np

    from tensoraerospace.envs import ComSatEnv
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import unit_step

    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)
    reference_signals = unit_step(degree=5, tp=tp, time_step=10, output_rad=True).reshape(1, -1)

    env = gym.make(
        'ComSatEnv-v0',
        number_time_steps=number_time_steps,
        initial_state=[[0],[0],[0]],
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
    from tensoraerospace.aerospacemodel import ComSat

    dt = 0.01
    number_time_steps = 200

    x0 = np.array([0.0, 0.0, 0.0])  # [rho, theta, omega]

    model = ComSat(
        x0=x0,
        number_time_steps=number_time_steps,
        selected_state_output=["rho", "theta", "omega"],
        dt=dt,
    )

    for t in range(number_time_steps - 1):
        u = np.array([[0.05]])
        x_next = model.run_step(u)
    ```

## Python API

=== "Model"

    ::: tensoraerospace.aerospacemodel.comsat.ComSat

=== "Gymnasium environment"

    ::: tensoraerospace.envs.comsat.ComSatEnv