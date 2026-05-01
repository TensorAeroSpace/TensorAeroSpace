# ELV Launch Vehicle — Longitudinal Dynamics

ELV (Expendable Launch Vehicle) is a carrier rocket for orbital payload delivery. Its longitudinal flight channel is implemented as a linear state-space model and a compatible Gymnasium environment.

![Expendable launch vehicle](img/evl.png){ width=800 }

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Quick start**

    Launch the environment or the model within minutes.

    [:octicons-arrow-right-24: See example](#quick-start)

-   :material-cog-outline: **Model API**

    Python class documentation for the ELV model.

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
 x = \begin{bmatrix} w & q & \theta \end{bmatrix}^{\top}, \quad
 u_{in} = \eta
\]

The typical matrix structure is:

\[
\begin{bmatrix}
\dot{w} \\
\dot{q} \\
\dot{\theta}
\end{bmatrix}
=
\begin{bmatrix}
z_w & z_q & z_{\theta} \\
m_w & m_q & m_{\theta} \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} w \\ q \\ \theta \end{bmatrix}
 +
\begin{bmatrix} z_{\eta} \\ m_{\eta} \\ 0 \end{bmatrix} \eta
\]

=== "Variables"

    - **w**: normal speed, m/s
    - **q**: pitch rate, rad/s
    - **θ**: pitch angle, rad
    - **η**: control input (pitch actuator), rad

=== "Coefficients"

    - **z_w, z_q, z_θ** — partial derivatives of normal force \(Z\) with respect to \(w, q, \theta\)
    - **m_w, m_q, m_θ** — partial derivatives of pitch moment \(M\) with respect to \(w, q, \theta\)
    - **z_η, m_η** — partial derivatives with respect to the control input \(\eta\)

!!! note "Units"
    Angles and angular rates are in radians. API methods can return values in degrees.

## Mathematical model {#mathematical-model}

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

Numerical matrices (example linearization):

\[
\begin{bmatrix}
\dot{w} \\
\dot{q} \\
\dot{\theta}
\end{bmatrix}
=
\begin{bmatrix}
-100.858 & 1 & -0.1256 \\
14.7805 & 0 & 0.01958 \\
0 & 1 & 0 
\end{bmatrix}
\begin{bmatrix}
w \\
q \\
\theta 
\end{bmatrix}
 +
\begin{bmatrix}
0 \\
3.4558 \\
20.42
\end{bmatrix}
\eta
\]

### Derivatives (numerical values)

- **Matrix A (derivatives):**

  | Coefficient | Value |
  |-------------|----------|
  | z_w | -100.858 |
  | z_q | 1.0 |
  | z_θ | -0.1256 |
  | m_w | 14.7805 |
  | m_q | 0.0 |
  | m_θ | 0.01958 |

- **Input η (column B):**

  | Coefficient | Value |
  |-------------|----------|
  | z_η | 0.0 |
  | m_η | 3.4558 |

!!! tip "Actuator limits"
    Default control limits:

    - Maximum magnitude: \(\pm 25^\circ\)
    - Maximum rate: \(60^\circ/\text{s\)

    Internal computations operate in radians; limits are converted accordingly.

## Sources

1. Aliyu, Bhar & Funmilayo, A. & Okwo, Odooh & Sholiyi, Olusegun. (2019). State‑Space Modelling of a Rocket for Optimal Control System Design. Journal of Aircraft and Spacecraft Technology. 3. 128‑137. 10.3844/jastsp.2019.128.137. [Link](https://www.researchgate.net/publication/335917723_State-Space_Modelling_of_a_Rocket_for_Optimal_Control_System_Design)
2. Aliyu, Bhar. (2011). Expendable Launch Vehicle Flight Control — Design & Simulation with Matlab/Simulink. [Link](https://www.researchgate.net/publication/301790480_Expendable_Launch_Vehicle_Flight_Control-Design_Simulation_with_MatlabSimulink)

## Reward

The default reward function returns the negative absolute tracking error for the pitch angle:

$$r_t = -|\theta(t) - \theta_{\text{ref}}(t)|$$

Higher reward (closer to 0) indicates better tracking performance. A custom reward function can be passed via the `reward_func` parameter.

## Quick start {#quick-start}

=== "Gymnasium"

    ```python
    import gymnasium as gym
    import numpy as np

    from tensoraerospace.envs import LinearLongitudinalELVRocket
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import unit_step

    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)
    reference_signals = unit_step(degree=5, tp=tp, time_step=10, output_rad=True).reshape(1, -1)

    env = gym.make(
        'LinearLongitudinalELVRocket-v0',
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
    from tensoraerospace.aerospacemodel import ELVRocket

    dt = 0.01
    number_time_steps = 200

    x0 = np.array([0.0, 0.0, 0.0])  # [w, q, theta]

    model = ELVRocket(
        x0=x0,
        number_time_steps=number_time_steps,
        selected_state_output=["w", "q", "theta"],
        dt=dt,
    )

    for t in range(number_time_steps - 1):
        u = np.array([[0.05]])  # control (rad)
        x_next = model.run_step(u)
    ```

## Python API

=== "Model"

    ::: tensoraerospace.aerospacemodel.elv.ELVRocket

=== "Gymnasium environment"

    ::: tensoraerospace.envs.elv.LinearLongitudinalELVRocket