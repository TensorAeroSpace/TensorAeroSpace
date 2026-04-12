# ImprovedMissileEnv -- Guided missile pitch tracking (normalized, RL-friendly)

This page documents the `ImprovedMissileEnv` environment used for training and evaluating reinforcement learning controllers for the longitudinal channel of a guided missile. The environment provides a normalized observation and action interface, a shaped reward for precise and smooth pitch tracking, and realistic termination conditions.

The reference implementation lives in `tensoraerospace/envs/rocket.py` (class `ImprovedMissileEnv`).

## Summary

- Observation space: 4D, normalized to [-1, 1]
- Action space: 1D, normalized to [-1, 1] (stabilizer deflection command)
- Goal: track a time-varying pitch reference while keeping control smooth and economical
- Reward: quadratic costs on pitch tracking error, pitch-rate mismatch to reference dynamics, absolute input, input rate and input jerk (with scaling)
- Termination: exceeding safe pitch envelope or reaching time horizon

## Observation and Action

Let \(\theta\) be the current pitch angle (rad), \(q\) the pitch rate (rad/s), and \(\theta_{ref}(t)\) the target pitch.

The observation at step \(t\) is a 4-vector:

\[
\mathbf{o}_t = \big[\underbrace{\mathrm{clip}\!\Big(\frac{\theta_{ref}(t) - \theta(t)}{\theta_{max}},\,-1,\,1\Big)}_{\text{[0] norm\_pitch\_error}},\; \underbrace{\mathrm{clip}\!\Big(\frac{q(t)}{q_{max}},\,-1,\,1\Big)}_{\text{[1] norm\_q}},\; \underbrace{\mathrm{clip}\!\Big(\frac{\theta(t)}{\theta_{max}},\,-1,\,1\Big)}_{\text{[2] norm\_\(\theta\)}},\; \underbrace{u_{t-1}}_{\text{[3] prev action}}\big]
\]

where

- \(\theta_{max} = 20°\) (converted to rad internally),
- \(q_{max} = 5°/\text{s}\) (converted to rad/s internally),
- \(u_{t-1} \in [-1, 1]\) is the previously applied normalized stabilizer command.

**Important**: The observation vector is already normalized and ready to use. Index `[0]` contains the normalized pitch tracking error, which is directly usable for proportional control.

The action is a single normalized command \(u_t \in [-1, 1]\). It maps to a physical stabilizer deflection (deg), then converted to radians:

\[
\delta_{stab}^°(t) = u_t \cdot \delta_{stab,max}^°, \quad \delta_{stab,max}^° = 25°,
\]

which is converted to radians before passing to the underlying missile model.

## Reward function

The environment uses a shaped reward with five terms promoting accuracy and smoothness while penalizing excessive actuation. The per-step reward is

\[
r_t = -\,s\,\Big(\,w_\theta\,e_\theta^2\; +\; w_q\,e_q^2\; +\; w_u\,u_t^2\; +\; w_{\Delta}\,(\Delta u_t)^2\; +\; w_{\Delta^2}\,(\Delta^2 u_t)^2\Big),
\]

with the default weights and scale:

\[w_\theta=8.0,\quad w_q=0.2,\quad w_u=0.003,\quad w_\Delta=0.01,\quad w_{\Delta^2}=0.001,\quad s=0.1.\]

Note: the pitch tracking weight \(w_\theta = 8.0\) is higher than in most other environments (e.g., B747, ELV use 5.0), reflecting the need for tighter pitch control in guided missile applications.

The error terms are defined as

\[
e_\theta = \frac{\theta(t)-\theta_{ref}(t)}{\theta_{max}},\qquad
e_q = \frac{q(t)-\dot\theta_{ref}(t)}{q_{max}},
\]

where \(\dot\theta_{ref}(t)\) is a finite-difference derivative of the reference pitch computed with the simulation time step \(\Delta t\). The input smoothness terms use

\[
\Delta u_t = u_t - u_{t-1},\qquad \Delta^2 u_t = u_t - 2 u_{t-1} + u_{t-2}.
\]

Notes:

- The high \(w_\theta = 8.0\) emphasizes precise pitch tracking for guided missile applications.
- \(w_q\) damps response relative to the reference slope, reducing overshoot/oscillation.
- \(w_u, w_\Delta, w_{\Delta^2}\) regularize energy usage and command smoothness, suppressing chattering.
- The overall scale \(s\) keeps rewards in a compact numerical range for RL stability.

## Termination and truncation

- **Safety termination**: if \(|\theta| > \theta_{max}\), the episode terminates early and a large penalty (\(-100\)) is applied at that step.
- **Truncation**: the episode truncates when the configured horizon (number of time steps) is reached.

## Episode dynamics

At each step:

1. The agent outputs \(u_t \in [-1, 1]\).
2. The environment clamps \(u_t\) to \([-1, 1]\), maps it to \(\delta_{stab}^°\), converts to radians, and advances the internal missile model.
3. The observation \(\mathbf{o}_{t+1}\) is built using normalized signals.
4. The reward \(r_t\) is computed as above.
5. Termination/truncation is checked.

## Usage example

```python
import numpy as np
from tensoraerospace.envs.rocket import ImprovedMissileEnv
from tensoraerospace.signals.standart import sinusoid_vertical_shift
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp

dt = 0.01
tn = 200
tp = generate_time_period(tn=tn, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)

# reference: smooth sinusoidal pitch in radians (amplitude 1 deg)
reference_signal = np.reshape(
    sinusoid_vertical_shift(
        tp=np.asarray(tps), frequency=0.05, amplitude=np.deg2rad(1.0), vertical_shift=0.0
    ),
    (1, -1),
)

# Initial state: [u, w, q, theta] in SI units
initial_state = np.array([0, 0, 0, 0], dtype=np.float32)

env = ImprovedMissileEnv(
    initial_state=initial_state,
    reference_signal=reference_signal,
    number_time_steps=number_time_steps,
    initial_elevator_deg=0.0,
    use_initial_action_on_first_step=True,
    dt=dt,
)

obs, info = env.reset()
done = False
while not done:
    # simple proportional control on normalized pitch error in [-1, 1]
    u = float(np.clip(2.0 * float(obs[0]), -1.0, 1.0))
    obs, reward, terminated, truncated, info = env.step(np.array([u], dtype=np.float32))
    done = bool(terminated or truncated)
```

## Implementation hints

- Observation normalization bounds (\(\theta_{max}, q_{max}\)) and stabilizer limits are defined inside the class and can be tuned if needed.
- Reward weights are exposed as instance attributes (`w_pitch`, `w_q`, `w_action`, `w_smooth`, `w_jerk`, `reward_scale`) and can be changed to match specific control objectives.
- The internal state representation follows the order `[u, w, q, theta]`. The environment extracts \(q\) at index 2 and \(\theta\) at index 3. Observations are normalized to `[-1, 1]`.
- The missile model receives elevator input in radians; the environment converts from degrees internally (action -> degrees -> radians).

## References

- `tensoraerospace/envs/rocket.py` -- full environment implementation
- Guided missile dynamics model: `tensoraerospace/aerospacemodel/missile.py`
