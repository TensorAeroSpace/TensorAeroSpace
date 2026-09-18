# LSU‑05 NG — Longitudinal Dynamics

The LAPAN Surveillance Aircraft (LSU)‑05 NG is a UAV for observation and research. This page mirrors the ELV layout: quick start, math model, derivatives, and API.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Quick start**

    Launch the environment or the model within minutes.

    [:octicons-arrow-right-24: See example](#quick-start)

-   :material-cog-outline: **Model API**

    Python class documentation for the LSU‑05 longitudinal dynamics.

    [:octicons-arrow-right-24: Go to API](#python-api)

-   :material-gamepad-variant-outline: **Gymnasium environment**

    Ready environment for RL agents.

    [:octicons-arrow-right-24: Explore](#python-api)

-   :material-book-open-variant: **Theory**

    State equations and numerical parameters.

    [:octicons-arrow-right-24: Learn more](#mathematical-model)

</div>

## Control object structure

\[\dot{x} = A x + B u, \quad y = C x + D u\]

\[
 x = \begin{bmatrix} u & w & q & \theta \end{bmatrix}^{\top}, \quad
 u_{in} = \eta
\]

The plant is modeled in the state space, consistent with other systems in the library. The state-space matrices are taken from the reference below. Because the system lacks internal disturbance processes, the output \(y\) is not used during simulation (\(C\) is diagonal, \(D\) is zero).

## Mathematical model {#mathematical-model}

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

\[
\begin{bmatrix}
\dot{u} \\
\dot{w} \\
\dot{q} \\
\dot{\theta}
\end{bmatrix}
=
\begin{bmatrix}
-0.00271615 & 0.248462 & 0 & -9.81 \\
-0.257616 & -11.3097 & 68.9497 & 0\\
0.0576336 & -7.23232 & -11.3237 & 0 \\
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
1.959083 \\
-73.99448 \\
-188.4752 \\
0.0
\end{bmatrix}
\eta
\]

### Derivatives (numerical values)

- **Matrix A (derivatives):**

  | Coefficient | Value |
  |-------------|----------|
  | x_u | -0.00271615 |
  | x_w | 0.248462 |
  | x_q | 0 |
  | x_θ | -9.81 |
  | z_u | -0.257616 |
  | z_w | -11.3097 |
  | z_q | 68.9497 |
  | z_θ | 0 |
  | m_u | 0.0576336 |
  | m_w | -7.23232 |
  | m_q | -11.3237 |
  | m_θ | 0 |

- **Input η (column B):**

  | Coefficient | Value |
  |-------------|----------|
  | x_η | 1.959083 |
  | z_η | -73.99448 |
  | m_η | -188.4752 |

where

- \(u\) — longitudinal speed [m/s]
- \(w\) — normal speed [m/s]
- \(q\) — pitch rate [rad/s]
- \(\theta\) — pitch angle [rad]
- \(\eta\) — stabilizer deflection angle [rad]
- \(x_u\) — partial derivative of longitudinal force with respect to longitudinal speed
- \(x_w\) — partial derivative of longitudinal force with respect to normal speed
- \(x_q\) — partial derivative of longitudinal force with respect to pitch rate
- \(x_{\theta}\) — partial derivative of longitudinal force with respect to pitch angle
- \(z_u\) — partial derivative of vertical force with respect to longitudinal speed
- \(z_w\) — partial derivative of vertical force with respect to normal speed
- \(z_q\) — partial derivative of vertical force with respect to pitch rate
- \(z_{\theta}\) — partial derivative of vertical force with respect to pitch angle
- \(m_u\) — partial derivative of pitch moment with respect to longitudinal speed
- \(m_w\) — partial derivative of pitch moment with respect to normal speed
- \(m_q\) — partial derivative of pitch moment with respect to pitch rate
- \(m_{\theta}\) — partial derivative of pitch moment with respect to pitch angle

## Sources

1. 2.	Lembaga, D.O., Antariksa, P.D., Septiyana, A., Hidayat, K., Rizaldi, A., Suseno, P.A., Jayanti, E.B., Atmasari, N., Ramadiansyah, M.L., Ramadhan, R.A., Suryo, V.N., Grüter, B., Diepolder, J., Holzapfel, F., Wijaya, Y.G., Dewan, S., Jurnal, P., Dirgantara, T., Wibowo, H., Panas, P., Septanto, H., Harno, A., Syah, N.A., Angkasa, R., Satelit, M.D., Irwanto, H.Y., Avionik, M.E., Hakim, A.N., Utama, A.B., Wahyudi, A.H., Kurniawati, F., Putro, I.E., & Astuti, R.A. STABILITY AND CONTROLLABILITY ANALYSIS ON LINEARIZED DYNAMIC SYSTEM EQUATION OF MOTION OF LSU 05-NG USING KALMAN RANK CONDITION METHOD. - Jurnal Teknologi Dirgantara Vol. 18 No. 2 Desember 2020 : hal 81 – 92 – 2020

## Reward

The default reward function returns the negative absolute tracking error for the pitch angle:

$$r_t = -|\theta(t) - \theta_{\text{ref}}(t)|$$

Higher reward (closer to 0) indicates better tracking performance. A custom reward function can be passed via the `reward_func` parameter.

## Quick start {#quick-start}

```python

import gymnasium as gym 
import numpy as np
from tqdm import tqdm

from tensoraerospace.envs import LinearLongitudinalLAPAN
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.signals.standard import unit_step

dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)
reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1])

env = gym.make(
    'LinearLongitudinalLAPAN-v0',
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

## Python API

=== "Model"

    ::: tensoraerospace.aerospacemodel.lapan.LAPAN

=== "Gymnasium environment"

    ::: tensoraerospace.envs.lapan.LinearLongitudinalLAPAN


## Units, actuator and environment contract

The model state `[u, w, q, theta]` contains perturbations about the reference
flight condition, in m/s, m/s, rad/s and rad. `LAPAN.run_step` takes elevator
**radians**. The A/B coefficients match page 86 of
[Septiyana et al. (2020)](https://ejournal.brin.go.id/ijoa/article/download/12529/9853).
This linear model is not a validation against flight measurements.

The plant retains its configured ±40° magnitude and 300°/s rate limits, including
the first command. `initial_control` sets the starting elevator position in
radians (default zero); `initialise_system` restores it. Invalid states, commands,
clocks and horizons are rejected. All recorded pre-transition outputs are
available through `get_output`; plotting velocity preserves m/s and stored data.

`LinearLongitudinalLAPAN` takes elevator commands in **degrees**, bounded by ±25°,
and converts them to radians for the model. Previously the declared ±25 range
was passed directly as radians. Existing controllers relying on that behavior
need their command scaling checked. Observations and reference angles are radians.
The default output is `[theta, q]`. `output_space` determines its order and shape,
while `tracking_states` independently selects the full model state for reward.
A single reference channel tracks the first selected state; multiple channels
use mean absolute error. Reference callables receive `i*dt` seconds; the time
horizon returns `truncated=True`. Initial arrays are copied for reproducible reset.
