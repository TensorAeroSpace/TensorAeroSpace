# Ultrastick‑25e — Longitudinal Dynamics

The Ultrastick‑25e UAV is a lightweight experimental platform. This page mirrors the ELV layout: quick start, math model, derivatives, and API.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Quick start**

    Launch the environment or the model within minutes.

    [:octicons-arrow-right-24: See example](#quick-start)

-   :material-cog-outline: **Model API**

    Python class documentation for the Ultrastick.

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
 x = \begin{bmatrix} u & w & \theta & q & h \end{bmatrix}^{\top}, \quad
 u_{in} = \begin{bmatrix} \eta & \delta_t \end{bmatrix}^{\top}
\]

=== "Variables"

- **u**: longitudinal speed, m/s
- **w**: vertical speed, m/s
- **θ**: pitch angle, rad
- **q**: pitch rate, rad/s
- **h**: altitude, m
- **η**: stabilizer deflection, rad
- **δ_t**: throttle deflection, rad

!!! note "Units"
    Angles and angular rates are in radians. API methods can expose values in degrees.

## Mathematical model {#mathematical-model}

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

Numerical matrices (example linearization):

\[
\begin{bmatrix}
\dot{u} \\
\dot{w} \\
\dot{\theta} \\
\dot{q} \\
\dot{h}
\end{bmatrix}
=
\begin{bmatrix}
-0.5944 & 0.8008 & -9.791 & -0.8747 & 5.077\times 10^{-5} \\
-0.744 & -7.56 & -0.5294 & 15.72 & -0.000939 \\
0 & 0 & 0 & 1 & 0 \\
1.041 & -7.406 & 0 & -15.81 & -7.284\times 10^{-18} \\
-0.05399 & 0.9985 & -17 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
u \\
w \\
\theta \\
q \\
 h
\end{bmatrix}
 +
\begin{bmatrix}
0.4669 & 0 \\
-2.703 & 0 \\
0 & 0 \\
-133.7 & 0 \\
0 & 0
\end{bmatrix}
\begin{bmatrix}
\eta \\
\delta_t
\end{bmatrix}
\]

### Key derivatives

- \(\dot{\theta} = q\) (element A[3,4] = 1)
- Influence of \(\eta\) on \(\dot{q}\): A[4,*], B[4,1] = −133.7
- Influence of \(\eta\) on \(\dot{u}\), \(\dot{w}\): B[1,1] = 0.4669, B[2,1] = −2.703

## Sources

1. Ahmed EA, Hafez A, Ouda AN, Ahmed HEH, Abd‑Elkader HM. Modelling of a Small Unmanned Aerial Vehicle. Adv Robot Autom 4:126, 2015.

## Reward

The default reward function returns the negative absolute tracking error for the pitch angle:

$$r_t = -|\theta(t) - \theta_{\text{ref}}(t)|$$

Higher reward (closer to 0) indicates better tracking performance. A custom reward function can be passed via the `reward_func` parameter.

## Quick start {#quick-start}

=== "Gymnasium"

```python
import gymnasium as gym 
import numpy as np

from tensoraerospace.envs import LinearLongitudinalUltrastick
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standard import unit_step

dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)
reference_signals = unit_step(degree=5, tp=tp, time_step=10, output_rad=True).reshape(1, -1)

env = gym.make(
    'LinearLongitudinalUltrastick-v0',
    number_time_steps=number_time_steps, 
    initial_state=[[0],[0],[0],[0],[0]],
    reference_signal=reference_signals,
)
state, info = env.reset()
for _ in range(200):
    action = np.array([[1.0, 0.1]])  # [η, δ_t]
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

=== "Model only"

```python
import numpy as np
from tensoraerospace.aerospacemodel import Ultrastick

dt = 0.01
number_time_steps = 200

x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

model = Ultrastick(
    x0=x0,
    number_time_steps=number_time_steps,
    selected_state_output=["u", "w", "q", "theta", "h"],
    dt=dt,
)

for t in range(number_time_steps - 1):
    u = np.array([1.0, 0.1])  # [η, δ_t]
    x_next = model.run_step(u)
```

## Python API

=== "Model"

::: tensoraerospace.aerospacemodel.ultrastick.Ultrastick

=== "Gymnasium environment"

::: tensoraerospace.envs.ultrastick.LinearLongitudinalUltrastick

