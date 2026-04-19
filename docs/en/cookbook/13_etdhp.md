# Recipe 13 — ET-DHP on the nonlinear F-16

ET-DHP (Event-Triggered Dual Heuristic Programming) adds a **Lipschitz-based trigger** on top of IM-GDHP: the controller only pushes a new action when the state has drifted far enough from the last trigger point to justify the re-computation. On an embedded controller this cuts the per-tick compute by an order of magnitude.

**Agent docs.** [ET-DHP](../agent/et_dhp.md) · **Full notebook.** [`example_etdhp_nonlinear_f16.ipynb`](../example/agent/et_dhp/example_etdhp_nonlinear.md) · **Related recipe.** [Recipe 12 — IM-GDHP](12_imgdhp.md).

## When to use ET-DHP

- **Low-power / embedded deployments** where you can't afford a full NN forward pass every millisecond.
- **Bandwidth-limited command links** where each control update has a non-zero transmission cost (drones, satellites).

The trade-off: between triggers the actuator **holds the last value**. On a plant with fast dynamics this can look like a small sawtooth on the state trace. Tune the trigger threshold to the plant's bandwidth.

## Step 1 — Minimal config

```python
import numpy as np
from tensoraerospace.agent.et_dhp import ETDHPAgent, ETDHPConfig

cfg = ETDHPConfig(
    dt=0.01,
    # same actor / critic as IM-GDHP
    actor_hidden=(32, 32),
    critic_hidden=(64, 64),
    actor_lr=1e-3,
    critic_lr=5e-4,
    gamma=0.95,
    lambda_weight=0.5,
    # RLS identifier
    rls_forgetting=0.995,
    rls_cov_init=1e2,
    G_init=np.array([[-0.5]]),
    # event trigger: fire when ||x_t - x_last_trigger|| > trigger_threshold * Lipschitz_constant
    trigger_threshold=0.02,
    lipschitz_estimate=10.0,
    u_magnitude_limit=15.0,
    u_rate_limit=60.0,
    seed=0,
)
```

The **trigger threshold** is the key knob. Smaller → more frequent updates → tighter tracking at higher compute cost. Typical starting range: `0.01–0.05` scaled by your state's typical magnitude.

## Step 2 — Step loop

Same three-step pattern as the other online-adaptive agents:

```python
agent = ETDHPAgent(n_state=1, n_control=1, config=cfg)
env.reset()

for k in range(n_steps):
    obs = env.get_state()
    u = agent.predict(obs[controlled_channels], ref[:, k], k)
    obs, *_ = env.step(u)
    agent.learn(obs[controlled_channels], ref[:, k], k)
```

Internally, `predict()` either runs the full actor+critic forward (on trigger) or returns the cached action. `learn()` always runs the RLS update so the model stays fresh even between triggers.

## Step 3 — Expected behaviour on the nonlinear F-16

α-step tracking with an active trigger:

![ET-DHP on nonlinear F-16](img/13_etdhp_nonlinear.png)

You should see:

- Tracking that looks similar to IM-GDHP (same actor / critic structure).
- A sparse distribution of **trigger events** on one of the diagnostic traces — usually fewer than 10 % of the total ticks will actually fire the NN.
- A slightly blockier elevator command because the actuator holds between triggers.

See [`example_etdhp_nonlinear_f16.ipynb`](../example/agent/et_dhp/example_etdhp_nonlinear.md) for the trigger-count log and step-by-step analysis.

## Step 4 — Save / load / publish to HuggingFace

```python
run_dir = agent.save('./checkpoints')
restored = ETDHPAgent.from_pretrained(run_dir)
agent.publish_to_hub('me/my-etdhp', folder_path=run_dir, access_token='hf_…')
```

## Pitfalls

- **Trigger never fires.** `trigger_threshold` is too large for your state magnitudes — halve it until the tracking error bites.
- **Trigger fires every tick.** Threshold too small; may as well use IM-GDHP (no compute saving).
- **Tracking degrades under fast references.** Raise `lipschitz_estimate` to make the threshold tighter at high state-change rates.

## Where to go next

- **[Recipe 14 — AA-INDI](14_aaindi.md)** — non-neural fault-tolerant alternative.
- **[Recipe 09 — Fault-tolerance](09_fault_tolerance.md)** — head-to-head against iADP and AA-INDI.
- **[ET-DHP documentation](../agent/et_dhp.md)** — theory + full API.
