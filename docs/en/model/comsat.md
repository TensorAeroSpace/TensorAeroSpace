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
 x = \begin{bmatrix} x_1 \\ x_3 \\ x_4 \end{bmatrix} = \begin{bmatrix} \rho \\ \dot{\rho} \\ \dot{\theta} \end{bmatrix}, \quad
 u = u_2
\]

The linearized system:

\[
\begin{bmatrix}
\dot{x}_1 \\
\dot{x}_3 \\
\dot{x}_4
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 & 0 \\
0.01036 & 0 & 0.7753 \\
0 & -0.01775 & 0
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_3 \\ x_4 \end{bmatrix}
 +
\begin{bmatrix} 0 \\ 0 \\ 0.1513 \end{bmatrix} u_2
\]

=== "State Variables"

    - **x₁ = ρ**: radial position - distance from Earth center, km
    - **x₃ = ρ̇**: radial velocity, m/s
    - **x₄ = θ̇**: angular velocity, rad/s

=== "Control Input"

    - **u₂**: tangential thrust, N
        - u₂ > 0 — thrust in direction of motion (acceleration)
        - u₂ < 0 — thrust against direction of motion (deceleration)
        - u₂ = 0 — no thrust

=== "System Coefficients"

    - **a₁₃ = 1.0** — radial position changes with radial velocity
    - **a₃₁ = 0.01036** — radial acceleration component from position
    - **a₃₄ = 0.7753** — radial acceleration component from angular velocity
    - **a₄₃ = -0.01775** — angular acceleration component from radial velocity
    - **b₄ = 0.1513** — tangential thrust influence on angular acceleration

!!! note "Units"
    Angular rates are in radians. Position in km, velocity in m/s. API methods can convert units.

## Mathematical model {#mathematical-model}

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

Numerical matrices (linearized system):

\[
\begin{bmatrix}
\dot{x}_1 \\
\dot{x}_3 \\
\dot{x}_4
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 & 0 \\
0.01036 & 0 & 0.7753 \\
0 & -0.01775 & 0 
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_3 \\
x_4 
\end{bmatrix}
 +
\begin{bmatrix}
0 \\
0 \\
0.1513
\end{bmatrix}
u_2
\]

Expanded form:
\[
\begin{aligned}
\dot{x}_1 &= x_3 \\
\dot{x}_3 &= 0.01036 \cdot x_1 + 0.7753 \cdot x_4 \\
\dot{x}_4 &= -0.01775 \cdot x_3 + 0.1513 \cdot u_2
\end{aligned}
\]

### Derivatives (numerical values)

- **Matrix A (state derivatives):**

  | Coefficient | Value | Physical Meaning |
  |-------------|-------|------------------|
  | a₁₃ (∂ẋ₁/∂x₃) | 1.0 | Radial position rate = radial velocity |
  | a₃₁ (∂ẋ₃/∂x₁) | 0.01036 | Position effect on radial acceleration |
  | a₃₄ (∂ẋ₃/∂x₄) | 0.7753 | Angular velocity effect on radial acceleration |
  | a₄₃ (∂ẋ₄/∂x₃) | -0.01775 | Radial velocity effect on angular acceleration |

- **Matrix B (control input):**

  | Coefficient | Value | Physical Meaning |
  |-------------|-------|------------------|
  | b₄ (∂ẋ₄/∂u₂) | 0.1513 | Tangential thrust effect on angular acceleration |

!!! tip "Actuator limits"
    Default control limits inside the model (normalized):

    - Maximum magnitude: \(\pm 25^\circ\)
    - Maximum rate: \(60^\circ/\text{s\)

    Internal computations use radians; limits are converted accordingly.

## Sources

1. Santosh Kumar Choudhary (2015). Design and Analysis of an Optimal Orbit Control for a Communication Satellite. INTERNATIONAL JOURNAL OF COMMUNICATIONS. Volume 9, 2015

## Reward

The default reward function returns the negative absolute tracking error for the radial velocity:

$$r_t = -|\dot{\rho}(t) - \dot{\rho}_{\text{ref}}(t)|$$

Higher reward (closer to 0) indicates better tracking performance. A custom reward function can be passed via the `reward_func` parameter.

## Quick start {#quick-start}

=== "Gymnasium"

    ```python
    import gymnasium as gym
    import numpy as np

    from tensoraerospace.envs import ComSatEnv
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import unit_step

    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)
    # Reference signal for angular velocity control
    reference_signals = unit_step(degree=0.1, tp=tp, time_step=10, output_rad=True).reshape(1, -1)

    env = gym.make(
        'ComSatEnv-v0',
        number_time_steps=number_time_steps,
        initial_state=[[6371.0], [0.0], [0.001]],  # [rho (km), rho_dot (m/s), theta_dot (rad/s)]
        reference_signal=reference_signals,
    )
    state, info = env.reset()
    for _ in range(200):
        action = np.array([[0.1]])  # Tangential thrust u2
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

    # Initial state: [rho (km), rho_dot (m/s), theta_dot (rad/s)]
    x0 = np.array([6371.0, 0.0, 0.001])

    model = ComSat(
        x0=x0,
        number_time_steps=number_time_steps,
        selected_state_output=["rho", "rho_dot", "theta_dot"],
        dt=dt,
    )

    for t in range(number_time_steps - 1):
        u = np.array([[0.05]])  # Tangential thrust u2
        x_next = model.run_step(u)
    
    # Get state history
    rho_history = model.get_state('rho')
    rho_dot_history = model.get_state('rho_dot')
    theta_dot_history = model.get_state('theta_dot')
    ```

## Python API

=== "Model"

    ::: tensoraerospace.aerospacemodel.comsat.ComSat

=== "Gymnasium environment"

    ::: tensoraerospace.envs.comsat.ComSatEnv