# Recipe 12 — IM-GDHP on the nonlinear F-16

IM-GDHP (Incremental Model Global Dual Heuristic Programming) upgrades IHDP with a **dual-head critic** that predicts both the value `J` and its gradient `λ = ∂J/∂x`. The extra gradient head gives the actor a cleaner training signal and — in the F-16 demo — faster convergence with less overshoot.

**Agent docs.** [IM-GDHP](../agent/imgdhp.md) · **Full notebook.** [`example_im_gdhp_nonlinear_f16.ipynb`](../example/agent/imgdhp/example_imgdhp_nonlinear.md) · **Related recipe.** [Recipe 11 — IHDP](11_ihdp.md).

## When to use IM-GDHP

Pick IM-GDHP when IHDP's single-head critic can't reliably track the cost-to-go gradient — usually that shows up as **jittery actor gradients** or **slow convergence on fast references**. The extra critic head is cheap: ~50 % more NN ops per tick, still dwarfed by the RLS step.

| Property | IM-GDHP vs IHDP |
|---|---|
| Critic output | `(J, λ)` vs `J` only |
| Actor gradient | Closed-form from `λ` | Finite-difference from `J` |
| Parameter count | ~2× critic | — |
| Convergence | Faster, smoother | Slower, noisier |

## Step 1 — Minimal config

```python
import numpy as np
from tensoraerospace.agent.im_gdhp import IMGDHPAgent, IMGDHPConfig

cfg = IMGDHPConfig(
    dt=0.01,
    actor_hidden=(32, 32),
    critic_hidden=(64, 64),
    actor_lr=1e-3,
    critic_lr=5e-4,
    gamma=0.95,
    # dual head weighting: loss = λ_weight·||λ_true - λ_pred||² + (1-λ_weight)·||J_true - J_pred||²
    lambda_weight=0.5,
    # RLS identifier
    rls_forgetting=0.995,
    rls_cov_init=1e2,
    G_init=np.array([[-0.5]]),
    u_magnitude_limit=15.0,
    u_rate_limit=60.0,
    seed=0,
)
```

## Step 2 — Step loop

Identical to IHDP:

```python
agent = IMGDHPAgent(n_state=1, n_control=1, config=cfg)
env.reset()

for k in range(n_steps):
    obs = env.get_state()
    u = agent.predict(obs[controlled_channels], ref[:, k], k)
    obs, *_ = env.step(u)
    agent.learn(obs[controlled_channels], ref[:, k], k)
```

Internally, `learn()` does one RLS step, then updates the dual-head critic using both `J` and the finite-difference `λ` estimate, then takes one actor gradient step in the direction suggested by the critic's `λ` head.

## Step 3 — Expected behaviour on the nonlinear F-16

Slow α tracking with an aggressive `actor_lr`:

![IM-GDHP on nonlinear F-16](img/12_imgdhp_nonlinear.png)

The commanded α is the stepped schedule; measured α follows with little overshoot. The elevator command stays smooth because the `λ`-head gives a cleaner gradient than the finite-difference alternative. Source: [`example_im_gdhp_nonlinear_f16.ipynb`](../example/agent/imgdhp/example_imgdhp_nonlinear.md).

## Step 4 — Save / load / publish to HuggingFace

```python
run_dir = agent.save('./checkpoints')
restored = IMGDHPAgent.from_pretrained(run_dir)
agent.publish_to_hub('me/my-imgdhp', folder_path=run_dir, access_token='hf_…')
```

Same contract as the other four — see [Recipe 08](08_huggingface.md).

## Pitfalls

- **`lambda_weight = 0` falls back to IHDP.** If you're getting IHDP behaviour despite using IM-GDHP, check this knob.
- **`lambda_weight = 1` ignores `J`.** The actor gradient becomes brittle if `λ`-head prediction is noisy. `0.3–0.7` is the useful range.
- **Critic loss oscillates.** Lower `critic_lr` or raise `rls_forgetting` (≥ 0.999) to smooth the model estimates the critic is chasing.

## Where to go next

- **[Recipe 13 — ET-DHP](13_etdhp.md)** — event-triggered cousin of IM-GDHP.
- **[Recipe 14 — AA-INDI](14_aaindi.md)** — non-neural alternative with stronger fault-tolerance guarantees.
- **[IM-GDHP documentation](../agent/imgdhp.md)** — theory + full API.
