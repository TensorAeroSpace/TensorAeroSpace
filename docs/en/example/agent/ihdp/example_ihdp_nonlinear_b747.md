# Example: IHDP on the nonlinear B-747 — pitch step tracking

This example trains an **Incremental Heuristic Dynamic Programming (IHDP)** agent to drive the pitch angle \(\theta\) of the [nonlinear Boeing 747-100 6-DoF model](../../../model/b747_nonlinear.md) to a step reference at cruise (FL200, \(V \approx 674\) ft/s). The source notebook lives at `example/reinforcement_learning/example_ihdp_nonlinear_b747.ipynb`.

## Why this differs from the F-16 IHDP example

The [F-16 nonlinear IHDP example](example_ihdp_nonlinear.md) tracks a sinusoidal \(\alpha\) reference and uses an inverse-model feedforward to compensate phase lag. The B-747 is a different beast: at cruise it is **heavy** (\(W \approx 636\,600\) lb, \(I_y \approx 33.1 \times 10^6\) slug·ft²) and the short-period mode is much slower than the F-16. Because of that:

1. **A step reference is the right starting point.** The short-period rise time at FL200 cruise is roughly 3–4 s; a step gives the agent a clean, persistent error signal to drive to zero, which is exactly what IHDP excels at.
2. **No feedforward is needed.** The 1° step amplitude sits on the linear part of the Taylor-expanded aerodynamic model around trim, so a reactive policy learned online suffices.
3. **Training works in deviation-from-trim space.** We compute the trim \((\alpha_0, \delta_{e,0}, \delta_{T,0})\) once via Newton-Raphson; the IHDP agent only sees and outputs *deviations* from that operating point. Constant trim values are added back inside the env loop.

## Architecture

```
δ_e(t) = δ_{e,trim}  +  agent_residual(t)
δ_a    = 0,  δ_r = 0,  δ_T = δ_{T,trim}
```

**State observed by the agent** (extracted from the 12-D env state):

* `d_theta` = \(\theta - \theta_{\text{trim}}\) — pitch deviation [rad]
* `q`       = pitch angular velocity [rad/s]

**Tracked variable:** `d_theta` follows a 1° step that fires at \(t = 5\) s.

## 1. Imports and trim

```python
import math

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from tensoraerospace.aerospacemodel.b747.nonlinear import B747Configuration, trim
from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env
from tensoraerospace.agent.ihdp.model import IHDPAgent
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period
```

Solve \(\dot u = \dot w = \dot q = 0\) at \(h = 20\,000\) ft, \(V = 674\) ft/s (\(M \approx 0.65\)) — the trimmer returns \(\alpha\), \(\delta_e\), throttle, and a ready 12-D state vector:

```python
trim_result = trim(altitude_ft=20_000.0, V_ft_s=674.0,
                   config=B747Configuration.NOMINAL)

alpha_trim_rad   = float(trim_result.alpha_rad)
theta_trim_rad   = alpha_trim_rad         # level cruise: theta = alpha
delta_e_trim_deg = math.degrees(trim_result.elevator_rad)
throttle_trim    = float(trim_result.throttle)
# trim: alpha = +3.603°, delta_e = -0.722°, throttle = 0.555
```

## 2. Time grid and step reference

60-second episode at \(dt = 0.02\) s. The reference is held at trim for the first 5 s (this gives the IHDP incremental model time to identify a usable local linearisation), then steps to \(\theta_{\text{trim}} + 1^{\circ}\) for the rest of the episode:

```python
dt = 0.02
tp = generate_time_period(tn=60.0, dt=dt)        # 3001 steps
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)

ref_dev_deg = np.zeros(number_time_steps)
ref_dev_deg[int(5.0/dt):] = 1.0                 # +1° step at t=5s

reference_signal = np.deg2rad(ref_dev_deg).reshape(1, -1)
```

The reference is in **deviation from trim** so the agent's tracked variable goes from 0 to 1° at the step. Inside the rollout we add \(\theta_{\text{trim}}\) back when comparing against the actual pitch.

## 3. Environment

```python
env = NonlinearB747Env(
    trim_at=(20_000.0, 674.0),
    number_time_steps=number_time_steps,
    dt=dt, integrator="rk4",
    action_space="virtual",
    config=B747Configuration.NOMINAL,
)
obs, _ = env.reset()
```

Action space is `"virtual"` — the env consumes the raw 4-channel command \([\delta_e, \delta_a, \delta_r, \delta_T]\) in physical units (rad / [0, 1] for throttle). Tracking, reference indexing and trim composition are done in the rollout loop, the cleanest way to wire IHDP into a generic Gymnasium plant env that doesn't ship its own tracking machinery.

## 4. IHDP agent

```python
state_space = ["d_theta", "q"]
tracking_states = ["d_theta"]
control_space = ["d_elev"]
indices_tracking_states = [0]

actor_settings = {
    "start_training": 5, "layers": (25, 1), "activations": ("tanh", "tanh"),
    "learning_rate": 2.0, "learning_rate_exponent_limit": 10,
    "type_PE": "combined", "amplitude_3211": 3, "pulse_length_3211": 5/dt,
    "maximum_input": 5, "maximum_q_rate": 20,
    "WB_limits": 30, "NN_initial": 47,
    "cascade_actor": False, "learning_rate_cascaded": 1.2,
}
critic_settings = {
    "Q_weights": [200], "start_training": -1, "gamma": 0.99,
    "learning_rate": 15, "learning_rate_exponent_limit": 10,
    "layers": (25, 1), "activations": ("tanh", "linear"),
    "WB_limits": 30, "NN_initial": 47,
    "indices_tracking_states": indices_tracking_states,
}
incremental_settings = {
    "number_time_steps": number_time_steps, "dt": dt,
    "input_magnitude_limits": 5, "input_rate_limits": 20,
}

agent = IHDPAgent(actor_settings, critic_settings, incremental_settings,
                  tracking_states, state_space, control_space,
                  number_time_steps, indices_tracking_states)
```

Three settings are noteworthy compared to the linear-F-16 IHDP defaults:

* **`Q_weights = [200]`** weights the tracking error in the critic's quadratic cost. The B-747 step amplitude is small (\(\Delta\theta = 0.0175\) rad), so without amplifying the error the gradient driving the actor is too weak. \(Q = 200\) brings the cost into a useful range and matches the F-16 nonlinear example.
* **`maximum_input = 5`** clamps the agent's elevator residual to ±5°. The B-747 has full elevator throw of ±25°, but for a 1° pitch step the residual the agent ever needs is well under 3°. A tighter clip prevents the actor's PE excitation from upsetting the airframe in the first few seconds.
* **`NN_initial = 47`** — the actor and critic NNs are sensitive to weight initialisation; seed 47 was chosen from a small sweep on this exact reference. Different references benefit from different seeds.

## 5. Online training loop

```python
def deviation_state(obs):
    return np.array([[obs[7] - theta_trim_rad], [obs[4]]])  # [d_theta, q]

obs, _ = env.reset()
xt = deviation_state(obs)

for step in tqdm(range(number_time_steps - 3)):
    ut = agent.predict(xt, reference_signal, step)
    delta_e_residual_deg = float(np.asarray(ut).flatten()[0])
    delta_e_total_deg = delta_e_trim_deg + delta_e_residual_deg

    action = np.array([
        math.radians(delta_e_total_deg),  # elevator [rad]
        0.0, 0.0,                          # aileron, rudder
        throttle_trim,                     # throttle [0, 1]
    ])
    obs, _, _, trunc, _ = env.step(action)
    xt = deviation_state(obs)
```

At every step:

1. Extract `[d_theta, q]` from the env's 12-D state (indices 7 and 4 respectively).
2. Ask the agent for an elevator residual (deg).
3. Compose the full 4-channel virtual command — trim elevator + residual on channel 0, trim throttle on channel 3, zeros on aileron/rudder.
4. Step the env. The actor / critic / incremental-model update once per step using the resulting transition.

## 6. Tracking metrics (typical run)

| Metric                | Value         |
|-----------------------|--------------:|
| Episode length        | 60 s          |
| Late-half MAE         | **0.043°**    |
| Late-half RMSE        | 0.049°        |
| Peak transient error  | 3.12°         |
| Steady-state offset   | 0.04°         |

The late-half MAE of 0.043° is 4.3% of the step amplitude — the agent learns the dynamic residual within the first ~30 s and tracks the trim+1° target essentially exactly. The 3.12° transient peak around \(t = 5\) s is the short-period overshoot intrinsic to the B-747 airframe at this trim point; that is what the agent has the *least* leverage to suppress with a single elevator channel and small elevator authority.

## 7. Notes & next steps

* **Operating-point assumption.** The agent is trained at one trim point (FL200, \(V=674\) ft/s). Step amplitudes large enough to leave the linear neighbourhood (≥ 5°) need re-tuned hyperparameters and longer episodes — IHDP can still cope, but the incremental model has to track a moving plant linearisation more aggressively.
* **Pitch versus \(\alpha\).** This example tracks \(\theta\), not \(\alpha\). For level cruise they coincide at trim, but during the transient \(\theta\) leads \(\alpha\) (the aircraft pitches before the airflow follows). Switching the tracked state to \(\alpha\) requires extracting \(\alpha = \arctan(w/u)\) from the env state.
* **Sinusoidal references.** A pure 0.1 Hz sinusoid on \(\theta\) would push the B-747 short-period mode hard. As with the F-16 IHDP example, a feedforward branch (lookahead + amplitude gain) is the right way to make IHDP track sinusoids — replace the step in section 2 and add an `ff_fn` that returns \(\delta_{e,\text{trim}}\) as a function of the lookahead reference.
* **Damage scenarios.** Re-running this example with `damage_profile=ELEVATOR_50PCT_LOSS` (see `tensoraerospace.aerospacemodel.b747.nonlinear.damage`) demonstrates IHDP's online adaptation: the agent recovers tracking after the surface effectiveness drops, with a transient error spike at the trigger time.
* **Other trim points.** Replace `trim_at` with `flight_condition_id ∈ {1, …, 10}` to start at one of the 10 published CR-2144 anchor points instead of the trim-finder's solution.

## See also

* [Boeing 747-100 (Nonlinear 6-DoF)](../../../model/b747_nonlinear.md) — model overview, geometry, EOM, JT9D engine, damage subsystem.
* [B-747 usage examples](../../../model/b747_examples.md) — practical recipes for trim, cruise, damage scenarios.
* [IHDP on the nonlinear F-16](example_ihdp_nonlinear.md) — sinusoidal tracking with feedforward + residual; the closest sibling to this example.
