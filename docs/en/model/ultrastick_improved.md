# ImprovedUltrastickEnv -- Ultrastick-25e UAV pitch tracking (normalized, RL-friendly)

This page documents the `ImprovedUltrastickEnv` environment used for training and evaluating reinforcement learning controllers for the longitudinal channel of the Ultrastick-25e UAV. Unlike most other improved environments in this library, this environment features a **2D action space** (elevator + throttle), a 5D observation vector, a progress-shaping bonus, and elevator rate limiting with low-pass filtering.

The reference implementation lives in `tensoraerospace/envs/ultrastick.py` (class `ImprovedUltrastickEnv`).

## Summary

- Observation space: 5D, normalized to [-1, 1]
- Action space: 2D, normalized to [-1, 1] (elevator deflection + throttle command)
- Goal: track a time-varying pitch reference while keeping control smooth and economical
- Reward: quadratic costs on pitch tracking error, pitch-rate mismatch, elevator and throttle magnitudes, rates, and jerks, plus a progress-shaping bonus
- Termination: exceeding safe pitch envelope or reaching time horizon

## Observation and Action

Let \(\theta\) be the current pitch angle (rad), \(q\) the pitch rate (rad/s), and \(\theta_{ref}(t)\) the target pitch.

The observation at step \(t\) is a 5-vector:

\[
\mathbf{o}_t = \big[\underbrace{\mathrm{clip}\!\Big(\frac{\theta_{ref}(t) - \theta(t)}{\theta_{max}},\,-1,\,1\Big)}_{\text{[0] norm\_pitch\_error}},\; \underbrace{\mathrm{clip}\!\Big(\frac{q(t)}{q_{max}},\,-1,\,1\Big)}_{\text{[1] norm\_q}},\; \underbrace{\mathrm{clip}\!\Big(\frac{\theta(t)}{\theta_{max}},\,-1,\,1\Big)}_{\text{[2] norm\_\(\theta\)}},\; \underbrace{u_{e,t-1}}_{\text{[3] prev elev}},\; \underbrace{u_{\delta_t,t-1}}_{\text{[4] prev throttle}}\big]
\]

where

- \(\theta_{max} = 30°\) (converted to rad internally),
- \(q_{max} = 30°/\text{s}\) (converted to rad/s internally),
- \(u_{e,t-1} \in [-1, 1]\) is the previously applied normalized elevator command,
- \(u_{\delta_t,t-1} \in [-1, 1]\) is the previously applied normalized throttle command.

**Note**: The Ultrastick model output order is `[Va, alpha, theta, q, h]`. The environment extracts \(\theta\) at index 2 and \(q\) at index 3.

The action is a 2D normalized command \(\mathbf{a}_t = [a_e, a_{\delta_t}] \in [-1, 1]^2\):

- **Elevator** (\(a_e\)): mapped to degrees with low-pass filtering and rate limiting:
  \[
  \delta_e^{pre} = \alpha \cdot a_e \cdot \delta_{e,max}^° + (1-\alpha) \cdot \delta_{e,t-1}^°, \quad \alpha = 0.5, \quad \delta_{e,max}^° = 15°,
  \]
  then rate-limited to \(\pm 300 \cdot \Delta t\) deg per step, and finally converted to radians for the model.
- **Throttle** (\(a_{\delta_t}\)): mapped from \([-1, 1]\) to physical throttle \([0, 1]\):
  \[
  \delta_t(t) = 0.5 \cdot (a_{\delta_t} + 1).
  \]

## Reward function

The environment uses a shaped reward with eight quadratic cost terms (separate for elevator and throttle) plus a progress-shaping bonus. The per-step reward is

\[
r_t = -s \cdot C_t + P_t,
\]

where the cost is

\[
C_t = w_\theta\,e_\theta^2 + w_q\,e_q^2 + w_{u_e}\,u_e^2 + w_{u_{\delta_t}}\,u_{\delta_t}^2 + w_{\Delta_e}\,(\Delta u_e)^2 + w_{\Delta_{\delta_t}}\,(\Delta u_{\delta_t})^2 + w_{\Delta^2_e}\,(\Delta^2 u_e)^2 + w_{\Delta^2_{\delta_t}}\,(\Delta^2 u_{\delta_t})^2,
\]

and the progress bonus is

\[
P_t = k_p \cdot \Big[\big(e_{\theta,t-1}^2 + e_{q,t-1}^2\big) - \big(e_\theta^2 + e_q^2\big)\Big].
\]

Default weights and scale:

\[
\begin{aligned}
&w_\theta=8.0,\quad w_q=0.5, \\
&w_{u_e}=0.003,\quad w_{u_{\delta_t}}=0.001, \\
&w_{\Delta_e}=0.003,\quad w_{\Delta_{\delta_t}}=0.002, \\
&w_{\Delta^2_e}=0.0005,\quad w_{\Delta^2_{\delta_t}}=0.0003, \\
&s=0.08,\quad k_p=0.05.
\end{aligned}
\]

The error terms are defined as

\[
e_\theta = \frac{\theta(t)-\theta_{ref}(t)}{\theta_{max}},\qquad
e_q = \frac{q(t)-\dot\theta_{ref}(t)}{q_{max}},
\]

where \(\dot\theta_{ref}(t)\) is a finite-difference derivative of the reference pitch computed with \(\Delta t\). The input smoothness terms use

\[
\Delta u_t = u_t - u_{t-1},\qquad \Delta^2 u_t = u_t - 2 u_{t-1} + u_{t-2},
\]

computed separately for elevator and throttle channels.

Notes:

- The progress bonus \(P_t\) provides a positive reward when tracking errors decrease and a negative reward when they increase, accelerating convergence.
- The elevator low-pass filter (\(\alpha=0.5\)) and rate limit (\(300°/\text{s}\)) smooth out abrupt commands, preventing high-frequency chattering.
- Separate weights for elevator and throttle allow fine-tuned penalization of each actuator channel.

## Termination and truncation

- **Safety termination**: if \(|\theta| > \theta_{max}\), the episode terminates immediately (no explicit penalty is added beyond the lost future reward; the terminated flag signals failure).
- **Truncation**: the episode truncates when the configured horizon (number of time steps) is reached.

## Episode dynamics

At each step:

1. The agent outputs \(\mathbf{a}_t = [a_e, a_{\delta_t}] \in [-1, 1]^2\).
2. The environment applies low-pass filtering and rate limiting to the elevator command, maps throttle to \([0, 1]\), and advances the internal Ultrastick model with \([\delta_e\text{ (rad)}, \delta_t]\).
3. The observation \(\mathbf{o}_{t+1}\) is built using normalized signals.
4. The reward \(r_t\) is computed as above.
5. Termination/truncation is checked.
6. The `info` dict contains diagnostic values: `elevator_deg`, `throttle`, `reward`, `cost`, `progress`, and per-term squared errors.

## Usage example

```python
import numpy as np
from tensoraerospace.envs.ultrastick import ImprovedUltrastickEnv
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

# Initial state: [u, w, q, theta, h] -- Ultrastick full state
initial_state = np.array([0, 0, 0, 0, 0], dtype=np.float32)

env = ImprovedUltrastickEnv(
    initial_state=initial_state,
    reference_signal=reference_signal,
    number_time_steps=number_time_steps,
    initial_elevator_deg=0.0,
    initial_throttle=0.0,
    use_initial_action_on_first_step=True,
    dt=dt,
)

obs, info = env.reset()
done = False
while not done:
    # simple proportional control: elevator from pitch error, throttle neutral
    elev_cmd = float(np.clip(2.0 * float(obs[0]), -1.0, 1.0))
    thr_cmd = 0.0  # neutral throttle
    action = np.array([elev_cmd, thr_cmd], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
```

## Implementation hints

- The environment robustly coerces incoming actions to shape `(2,)` via the internal `_to_norm_action` helper. Passing a scalar or 1D action will be padded with the previous throttle value.
- Observation normalization bounds (\(\theta_{max} = 30°\), \(q_{max} = 30°/\text{s}\)) are wider than in other environments to accommodate the UAV's dynamics range.
- Reward weights are exposed as instance attributes (`w_theta`, `w_q`, `w_action_elev`, `w_action_thr`, `w_smooth_elev`, `w_smooth_thr`, `w_jerk_elev`, `w_jerk_thr`, `reward_scale`, `k_progress`) and can be changed to match specific control objectives.
- The Ultrastick model expects a 2D input `[elevator (rad), throttle]` at each step; the environment handles all conversions internally.

## References

- `tensoraerospace/envs/ultrastick.py` -- full environment implementation
- Ultrastick-25e dynamics model: `tensoraerospace/aerospacemodel/ultrastick.py`
