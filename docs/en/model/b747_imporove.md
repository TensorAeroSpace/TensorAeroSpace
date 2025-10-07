# ImprovedB747Env – Boeing 747 longitudinal pitch tracking (normalized, RL‑friendly)

This page documents the ImprovedB747Env environment used for training and evaluating reinforcement learning controllers for the longitudinal channel of the Boeing 747. The environment provides a normalized observation and action interface, a shaped reward for precise and smooth pitch tracking, and realistic termination conditions.

The reference implementation lives in `tensoraerospace/envs/b747.py` (class `ImprovedB747Env`).

![](../example/agent/sac/img/sac-b747-impoved.jpg)



## Summary

- Observation space: 4D, normalized to [-1, 1]
- Action space: 1D, normalized to [-1, 1] (elevator deflection command)
- Goal: track a time‑varying pitch reference while keeping control smooth and economical
- Reward: quadratic costs on pitch tracking error, pitch‑rate mismatch to reference dynamics, absolute input, input rate and input jerk (with scaling)
- Termination: exceeding safe pitch envelope or reaching time horizon

## Observation and Action

Let \(\theta\) be the current pitch angle (rad), \(q\) the pitch rate (rad/s), and \(\theta_{ref}(t)\) the target pitch.

The observation at step \(t\) is a 4‑vector:

\[
\mathbf{o}_t = \big[\underbrace{\mathrm{clip}(\tfrac{\theta_{ref}(t) - \theta(t)}{\theta_{max}},-1,1)}_{\text{[0] norm\_pitch\_error}},\; \underbrace{\mathrm{clip}(\tfrac{q(t)}{q_{max}},-1,1)}_{\text{[1] norm\_q}},\; \underbrace{\mathrm{clip}(\tfrac{\theta(t)}{\theta_{max}},-1,1)}_{\text{[2] norm\_\theta}},\; \underbrace{u_{t-1}}_{\text{[3] prev action}}\big]
\]

where

- \(\theta_{max} = 20^{\circ}\) (converted to rad internally),
- \(q_{max} = 5^{\circ}/\text{s}\) (converted to rad/s internally),
- \(u_{t-1}\in[-1,1]\) is the previously applied normalized elevator command.

**Important**: The observation vector is already normalized and ready to use. Index `[0]` contains the normalized pitch tracking error, which is directly usable for proportional control.

The action is a single normalized command \(u_t \in [-1,1]\). It maps to a physical elevator deflection (deg):

\[
\delta_e^{\circ}(t) = u_t\,\delta_{e,\max}^{\circ}, \quad \delta_{e,\max}^{\circ} = 25^{\circ},
\]

which is converted to radians before passing to the underlying model.

## Reward function

The environment uses a shaped reward with five terms promoting accuracy and smoothness while penalizing excessive actuation. The per‑step reward is

\[
r_t = -\,s\,\Big(\,w_\theta\,e_\theta^2\; +\; w_q\,e_q^2\; +\; w_u\,u_t^2\; +\; w_{\Delta}\,(\Delta u_t)^2\; +\; w_{\Delta^2}\,(\Delta^2 u_t)^2\Big),
\]

with the default weights and scale:

\[w_\theta=5.0,\quad w_q=0.2,\quad w_{cross}=0.0,\quad w_u=0.003,\quad w_{\Delta}=0.01,\quad w_{\Delta^2}=0.001,\quad s=0.1.\]

Note: The implementation includes a cross-term weight \(w_{cross}\) (currently set to 0.0) for potential future use in coupling pitch and pitch-rate errors.

The error terms are defined as

\[
e_\theta = \frac{\theta(t)-\theta_{ref}(t)}{\theta_{max}},\qquad
e_q = \frac{q(t)-\dot\theta_{ref}(t)}{q_{max}},
\]

where \(\dot\theta_{ref}(t)\) is a finite‑difference derivative of the reference pitch computed with the simulation time step \(\Delta t\). The input smoothness terms use

\[
\Delta u_t = u_t - u_{t-1},\qquad \Delta^2 u_t = u_t - 2 u_{t-1} + u_{t-2}.
\]

Notes:

- A larger \(w_\theta\) emphasizes precise pitch tracking.
- \(w_q\) damps response relative to the reference slope, reducing overshoot/oscillation.
- \(w_u, w_{\Delta}, w_{\Delta^2}\) regularize energy usage and command smoothness, suppressing chattering.
- The overall scale \(s\) keeps rewards in a compact numerical range for RL stability.

## Termination and truncation

- Safety termination: if \(|\theta| > \theta_{max}\), the episode terminates early and a large penalty (\(-100\)) is applied at that step.
- Truncation: the episode truncates when the configured horizon (number of time steps) is reached.

## Episode dynamics

At each step:

1. The agent outputs \(u_t\in[-1,1]\).
2. The environment clamps \(u_t\) to \([-1,1]\), maps it to \(\delta_e^{\circ}\), converts to rad and advances the internal aircraft model.
3. The observation \(\mathbf{o}_{t+1}\) is built using normalized signals.
4. The reward \(r_t\) is computed as above.
5. Termination/truncation is checked.

## Usage example

```python
import numpy as np
from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standart import sinusoid_vertical_shift
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp

dt = 0.1
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

# Initial state: [u, w, q, theta]
initial_state = np.array([[0], [0], [0], [0]], dtype=np.float32)

env = ImprovedB747Env(
    initial_state=initial_state,
    reference_signal=reference_signal,
    number_time_steps=number_time_steps,
    initial_elevator_deg=0.0,
    use_initial_action_on_first_step=True,
    dt=dt,
)
# Ensure model discretization matches the environment step
env.unwrapped.model.discretisation_time = dt

obs, info = env.reset()
done = False
while not done:
    # simple proportional control on normalized pitch error in [-1, 1]
    u = float(np.clip(2.0 * float(obs[0]), -1.0, 1.0))
    obs, reward, terminated, truncated, info = env.step(np.array([u], dtype=np.float32))
    done = bool(terminated or truncated)
```

### Pretrained SAC agent example

Use a pretrained SAC agent from HuggingFace:

```python
import numpy as np
from tensoraerospace.agent.sac import SAC
from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standart import sinusoid_vertical_shift
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp

# Build env (short, consistent with example)
dt = 0.1
tn = 200
tp = generate_time_period(tn=tn, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
reference_signal = np.reshape(
    sinusoid_vertical_shift(
        tp=np.asarray(tps), frequency=0.05, amplitude=np.deg2rad(1.0), vertical_shift=0.0
    ),
    (1, -1),
)
initial_state = np.array([[0], [0], [0], [0]], dtype=np.float32)
env = ImprovedB747Env(
    initial_state=initial_state,
    reference_signal=reference_signal,
    number_time_steps=len(tp),
    dt=dt,
    initial_elevator_deg=0.0,
    use_initial_action_on_first_step=True,
)
env.unwrapped.model.discretisation_time = dt

# Load agent and run one episode
agent = SAC.from_pretrained("TensorAeroSpace/sac-b747")
obs, info = env.reset()
done = False
ret = 0.0
while not done:
    action = agent.select_action(obs, evaluate=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    ret += float(reward)
print(f"Return: {ret:.2f}")
```

## Implementation hints

- The class overlays a simple HUD and (optionally) a 2D sprite in the renderer; this has no effect on the physics or reward.
- Observation normalization bounds (\(\theta_{max}, q_{max}\)) and elevator limits are defined inside the class and can be tuned if needed.
- Reward weights are exposed as instance attributes (`w_pitch`, `w_q`, `w_cross`, `w_action`, `w_smooth`, `w_jerk`, `reward_scale`) and can be changed to match specific control objectives.
- The internal state representation follows the order `[u, w, q, theta]` (SI units: m/s, m/s, rad/s, rad), but observations are normalized to `[-1, 1]`.

## References

- `tensoraerospace/envs/b747.py` – full environment implementation
- Boeing 747 longitudinal dynamics model: `tensoraerospace/aerospacemodel/b747.py`


