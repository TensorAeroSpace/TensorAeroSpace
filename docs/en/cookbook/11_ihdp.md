# Recipe 11 — IHDP on the nonlinear F-16

IHDP (Incremental Heuristic Dynamic Programming) is the library's **neural actor + online-identified model** baseline. It's the historical starting point the other adaptive-critic agents (IM-GDHP, ET-DHP) extend.

**Agent docs.** [IHDP](../agent/ihdp.md) · **Full notebook.** [`example_ihdp_nonlinear_f16.ipynb`](../example/agent/ihdp/example_ihdp_nonlinear.md) · **Related recipe.** [Recipe 06 — Online-adaptive agents](06_online_adaptive.md).

## When to use IHDP

| Property | IHDP |
|---|---|
| Online adaptation | ✓ (RLS model + actor gradient step every tick) |
| Needs a model? | Warm-start `G_init`, rest is identified online |
| Neural net in the loop | Yes — actor & critic MLPs |
| Interpretability | Moderate (NN critic is opaque, RLS model is inspectable) |
| Fault tolerance | Natural — RLS re-identifies the plant under faults |

Pick IHDP when you want a **neural actor** that adapts online and you don't need the two-head critic of IM-GDHP or the event-trigger machinery of ET-DHP.

## Step 1 — Minimal config

```python
import numpy as np
from tensoraerospace.agent.ihdp import IHDPAgent, IHDPConfig

cfg = IHDPConfig(
    dt=0.01,
    # actor / critic
    actor_hidden=(32, 32),
    critic_hidden=(64, 64),
    actor_lr=1e-3,
    critic_lr=5e-4,
    gamma=0.95,
    # online model (RLS)
    rls_forgetting=0.995,
    rls_cov_init=1e2,
    G_init=np.array([[-0.5]]),         # from a linearisation or PE excitation
    # safety
    u_magnitude_limit=15.0,
    u_rate_limit=60.0,
    seed=0,
)
```

Like iADP / AA-INDI, IHDP needs a reasonable `G_init`. See [Recipe 06](06_online_adaptive.md) for the PE-excitation warm-start pattern.

## Step 2 — Step loop

Identical to the other online-adaptive agents:

```python
agent = IHDPAgent(n_state=1, n_control=1, config=cfg)
env.reset()

for k in range(n_steps):
    obs = env.get_state()
    u = agent.predict(obs[controlled_channels], ref[:, k], k)
    obs, *_ = env.step(u)
    agent.learn(obs[controlled_channels], ref[:, k], k)
```

`predict` runs the actor NN forward; `learn` does the RLS update + one SGD step on actor and critic.

## Step 3 — Expected behaviour on the nonlinear F-16

Tracking a smooth α schedule on `NonlinearLongitudinalF16-v0`:

![IHDP on nonlinear F-16](img/11_ihdp_nonlinear.png)

The plot is taken directly from the full example notebook — see [`example_ihdp_nonlinear_f16.ipynb`](../example/agent/ihdp/example_ihdp_nonlinear.md) for the exact env construction, reference signal and warm-start PE.

## Step 4 — Save / load / publish to HuggingFace

```python
run_dir = agent.save('./checkpoints')
restored = IHDPAgent.from_pretrained(run_dir)
agent.publish_to_hub('me/my-ihdp', folder_path=run_dir, access_token='hf_…')
```

Same contract as all five online-adaptive agents — see [Recipe 08](08_huggingface.md).

## Pitfalls

- **Actor collapses to zero.** Check `G_init` sign. Wrong sign makes every gradient step push the actor the wrong way.
- **Critic diverges.** Lower `critic_lr` (5e-4 is already conservative), or increase `rls_forgetting` closer to 1 for a less noisy model signal.
- **Unstable early ticks.** Increase `policy_eval_warmup_updates` (in agents that expose it) so the critic sees a settled model before it starts training.

## Where to go next

- **[Recipe 12 — IM-GDHP](12_imgdhp.md)** — dual-head critic, one level up in capability.
- **[Recipe 13 — ET-DHP](13_etdhp.md)** — event-triggered variant for low-compute platforms.
- **[IHDP documentation](../agent/ihdp.md)** — theory + full API.
