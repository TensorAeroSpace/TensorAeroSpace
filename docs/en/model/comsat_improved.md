# ImprovedComSatEnv -- Communication satellite orbit control (normalized, RL-friendly)

This page documents the `ImprovedComSatEnv` environment used for training and evaluating reinforcement learning controllers for a communication satellite in the longitudinal control channel. The environment provides a normalized observation and action interface, an LQR-style shaped reward for precise angular velocity tracking and orbital stability, and realistic termination conditions.

The reference implementation lives in `tensoraerospace/envs/comsat.py` (class `ImprovedComSatEnv`).

## Summary

- Observation space: 4D, normalized to [-1, 1]
- Action space: 1D, normalized to [-1, 1] (tangential thrust command)
- Goal: track a time-varying angular velocity reference while maintaining orbital radius stability
- Reward: LQR-style quadratic costs on angular velocity tracking error, orbital radius deviation, radial velocity, absolute thrust, thrust rate and thrust jerk, plus a survival bonus
- Termination: exceeding safe orbital envelope or reaching time horizon

## Observation and Action

Let \(\dot\theta\) be the current angular velocity (rad/s), \(\rho\) the orbital radius (km), \(\dot\rho\) the radial velocity (m/s), and \(\dot\theta_{ref}(t)\) the target angular velocity.

The observation at step \(t\) is a 4-vector:

\[
\mathbf{o}_t = \big[\underbrace{\mathrm{clip}\!\Big(\frac{\dot\theta(t) - \dot\theta_{ref}(t)}{\dot\theta_{max}},\,-1,\,1\Big)}_{\text{[0] norm\_theta\_dot\_error}},\; \underbrace{\mathrm{clip}\!\Big(\frac{\rho(t) - \rho_0}{\Delta\rho_{max}},\,-1,\,1\Big)}_{\text{[1] norm\_rho\_error}},\; \underbrace{\mathrm{clip}\!\Big(\frac{\dot\rho(t)}{\dot\rho_{max}},\,-1,\,1\Big)}_{\text{[2] norm\_rho\_dot}},\; \underbrace{u_{t-1}}_{\text{[3] prev action}}\big]
\]

where

- \(\dot\theta_{max} = 0.1\) rad/s -- maximum angular velocity for normalization,
- \(\Delta\rho_{max} = 100.0\) km -- maximum radial position deviation from nominal,
- \(\dot\rho_{max} = 200.0\) m/s -- maximum radial velocity for normalization,
- \(\rho_0 = 6371.0\) km -- nominal orbital radius (Earth radius, configurable),
- \(u_{t-1} \in [-1, 1]\) is the previously applied normalized thrust command.

The action is a single normalized command \(u_t \in [-1, 1]\). It maps to a physical tangential thrust:

\[
F_{u2}(t) = u_t \cdot F_{max}, \quad F_{max} = 25.0.
\]

## Reward function

The environment uses an LQR-style shaped reward with six quadratic cost terms plus a survival bonus. The per-step reward is

\[
r_t = -\,s\,\Big(\,w_{\dot\theta}\,e_{\dot\theta}^2 \;+\; w_\rho\,e_\rho^2 \;+\; w_{\dot\rho}\,e_{\dot\rho}^2 \;+\; w_u\,u_t^2 \;+\; w_\Delta\,(\Delta u_t)^2 \;+\; w_{\Delta^2}\,(\Delta^2 u_t)^2\Big) \;+\; 0.1,
\]

with the default weights and scale:

\[w_{\dot\theta}=5.0,\quad w_\rho=0.01,\quad w_{\dot\rho}=0.2,\quad w_u=0.001,\quad w_\Delta=0.01,\quad w_{\Delta^2}=0.001,\quad s=0.1.\]

The error terms are defined as

\[
e_{\dot\theta} = \frac{\dot\theta(t) - \dot\theta_{ref}(t)}{\dot\theta_{max}},\qquad
e_\rho = \frac{\rho(t) - \rho_0}{\Delta\rho_{max}},\qquad
e_{\dot\rho} = \frac{\dot\rho(t)}{\dot\rho_{max}}.
\]

The input smoothness terms use

\[
\Delta u_t = u_t - u_{t-1},\qquad \Delta^2 u_t = u_t - 2 u_{t-1} + u_{t-2}.
\]

The constant \(+0.1\) is a survival bonus that encourages longer episodes.

Notes:

- A larger \(w_{\dot\theta}\) emphasizes precise angular velocity tracking (primary objective).
- \(w_\rho\) penalizes orbital radius drift from the nominal value (low weight for flexibility).
- \(w_{\dot\rho}\) damps radial velocity to stabilize the orbit.
- \(w_u, w_\Delta, w_{\Delta^2}\) regularize energy usage and command smoothness.
- The overall scale \(s\) keeps rewards in a compact numerical range for RL stability.

## Termination and truncation

- **Excessive angular velocity**: if \(|\dot\theta| > 50 \cdot \dot\theta_{max} = 5.0\) rad/s, the episode terminates early with a penalty of \(-10\).
- **Excessive radial deviation**: if \(|\rho - \rho_0| > 5 \cdot \Delta\rho_{max} = 500\) km, the episode terminates early with a penalty of \(-10\).
- **Excessive radial velocity**: if \(|\dot\rho| > 10 \cdot \dot\rho_{max} = 2000\) m/s, the episode terminates early with a penalty of \(-10\).
- **Truncation**: the episode truncates when the configured horizon (number of time steps) is reached.

## Episode dynamics

At each step:

1. The agent outputs \(u_t \in [-1, 1]\).
2. The environment clamps \(u_t\) to \([-1, 1]\), maps it to thrust \(F_{u2}\), and advances the internal satellite model.
3. The observation \(\mathbf{o}_{t+1}\) is built using normalized signals.
4. The reward \(r_t\) is computed as above.
5. Termination/truncation is checked.

## Usage example

```python
import numpy as np
from tensoraerospace.envs.comsat import ImprovedComSatEnv
from tensoraerospace.signals.standart import sinusoid_vertical_shift
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp

dt = 0.01
tn = 200
tp = generate_time_period(tn=tn, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)

# reference: smooth sinusoidal angular velocity in rad/s
reference_signal = np.reshape(
    sinusoid_vertical_shift(
        tp=np.asarray(tps), frequency=0.05, amplitude=0.01, vertical_shift=0.0
    ),
    (1, -1),
)

# Initial state: [rho (km), rho_dot (m/s), theta_dot (rad/s)]
initial_state = np.array([6371.0, 0.0, 0.0], dtype=np.float32)

env = ImprovedComSatEnv(
    initial_state=initial_state,
    reference_signal=reference_signal,
    number_time_steps=number_time_steps,
    dt=dt,
    initial_thrust=0.0,
    use_initial_action_on_first_step=True,
    nominal_rho=6371.0,
)

obs, info = env.reset()
done = False
while not done:
    # simple proportional control on normalized angular velocity error
    u = float(np.clip(2.0 * float(obs[0]), -1.0, 1.0))
    obs, reward, terminated, truncated, info = env.step(np.array([u], dtype=np.float32))
    done = bool(terminated or truncated)
```

## Implementation hints

- Observation normalization bounds (\(\dot\theta_{max}\), \(\Delta\rho_{max}\), \(\dot\rho_{max}\)) and thrust limits are defined inside the class and can be tuned if needed.
- Reward weights are exposed as instance attributes (`w_theta_dot`, `w_rho`, `w_rho_dot`, `w_action`, `w_smooth`, `w_jerk`, `reward_scale`) and can be changed to match specific control objectives.
- The internal state representation follows the order `[rho, rho_dot, theta_dot]` (km, m/s, rad/s), but observations are normalized to `[-1, 1]`.
- The nominal orbital radius `nominal_rho` defaults to 6371.0 km (Earth radius) and is configurable via the constructor.

## References

- `tensoraerospace/envs/comsat.py` -- full environment implementation
- Communication satellite model: `tensoraerospace/aerospacemodel/comsat.py`
