# Recipe 14 — AA-INDI on the nonlinear F-16

AA-INDI (Active-Adaptive Incremental Nonlinear Dynamic Inversion) pairs the classical **INDI control law** with **VFF-RLS** identification of the control-effectiveness matrix `G` and a lightweight sensor-filter stand-in. It's the library's strongest **fault-tolerance** baseline: the variable forgetting factor contracts under large residuals, so the identifier catches up with a halved elevator within a few hundred ms.

**Agent docs.** [AA-INDI](../agent/aa_indi.md) · **Full notebook.** [`example_aaindi_nonlinear_f16.ipynb`](../example/agent/aa_indi/example_aaindi_nonlinear.md) · **Related recipe.** [Recipe 09 — Fault-tolerance](09_fault_tolerance.md).

## When to use AA-INDI

- **Actuator-fault tolerance is a primary requirement** (damaged surfaces, icing, jamming).
- You want an **inverse-dynamics controller** (INDI) without writing the full nonlinear model — just a warm-start `G`.
- You prefer a **non-neural** online controller that's easy to audit.

| Property | AA-INDI vs iADP |
|---|---|
| Control law | Pseudo-inverse of `G̃`, classical INDI | Closed-form LQT minimiser (paper eq. (11)) |
| Adaptation trigger | Variable forgetting factor λ (contracts on fault) | Constant `γ_RLS` |
| Outer loop | PI on reference tracking error (optional) | Built into the LQT cost |
| Hyperparameters | ~10 knobs | 4 knobs (Q, R, γ, γ_RLS) |

## Step 1 — Minimal config

```python
import numpy as np
from tensoraerospace.agent.aa_indi import AAINDIAgent, AAINDIConfig

cfg = AAINDIConfig(
    dt=0.01,
    # reference model — shapes the commanded rate before INDI inverts
    ref_wn=2.5,
    ref_zeta=0.9,
    # actuator limits (hard safety)
    u_magnitude_limit=15.0,
    u_rate_limit=60.0,
    # VFF-RLS: forgetting factor contracts under large residuals
    vff_forgetting_min=0.97,
    vff_forgetting_max=0.9999,
    vff_eps_sensitivity=0.1,
    vff_cov_init=1.0,
    # sensor filter (low-pass differentiator + bias estimator)
    sensor_cutoff_hz=15.0,
    bias_forgetting=0.995,
    enable_bias_correction=False,
    # warm-start of G; sign and order of magnitude matter
    G_init=np.array([[-0.5]]),
    # optional outer PI on reference tracking error
    ref_error_kp=0.6,
    ref_error_ki=0.0,
    seed=0,
)
```

The `ref_error_kp` / `ref_error_ki` knobs add an outer PI loop that closes steady-state tracking error — useful for rate commands that have non-zero steady-state.

## Step 2 — Step loop

Identical to the other online-adaptive agents:

```python
agent = AAINDIAgent(n_state=1, n_control=1, config=cfg)
env.reset()

for k in range(n_steps):
    obs = env.get_state()
    u = agent.predict(obs[controlled_channels], ref[:, k], k)
    obs, *_ = env.step(u)
    agent.learn(obs[controlled_channels], ref[:, k], k)
```

`predict()` applies the reference filter + INDI inversion + outer PI. `learn()` does the VFF-RLS update on `(Δu, Δẏ)` and the bias-estimator step.

## Step 3 — Expected behaviour under a 50 % elevator fault

Classical AA-INDI demo: pitch-rate tracking with a 50 % elevator effectiveness loss injected at `t = 10 s`.

![AA-INDI with elevator fault](img/14_aaindi_fault.png)

Key signatures to look for:

- **ω_z keeps tracking** through the fault event; small transient bump, then back on the command.
- **λ (forgetting factor) contracts** toward `vff_forgetting_min` right after the fault — visible on the diagnostic trace of the full notebook.
- **G̃ re-identifies** to the halved value within a few hundred ms.

Source: [`example_aaindi_nonlinear_f16.ipynb`](../example/agent/aa_indi/example_aaindi_nonlinear.md). Recipe 09 runs AA-INDI head-to-head against iADP on the same fault profile.

## Step 4 — Save / load / publish to HuggingFace

```python
run_dir = agent.save('./checkpoints')
restored = AAINDIAgent.from_pretrained(run_dir)
agent.publish_to_hub('me/my-aaindi', folder_path=run_dir, access_token='hf_…')
```

Mid-episode saves are bit-identical on reload — see [Recipe 08](08_huggingface.md).

## Pitfalls

- **Wrong sign of `G_init`.** INDI inverts `G`, so a sign flip drives the actuator the wrong way and the plant diverges. Check the PE-excitation warm-start gives you the sign you expect.
- **Persistent steady-state error on rate commands.** Raise `ref_error_kp` (0.3 → 0.8 is the typical range); add a bit of `ref_error_ki` for integral action.
- **Bias correction destabilises on noisy envs.** `enable_bias_correction=True` is off by default for the F-16 sim; turn on only if your sensor has a measurable DC bias.

## Where to go next

- **[Recipe 09 — Fault-tolerance](09_fault_tolerance.md)** — head-to-head against iADP on the same fault profile.
- **[Recipe 11 — IHDP](11_ihdp.md)** — neural-actor alternative with simpler hyperparameters.
- **[AA-INDI documentation](../agent/aa_indi.md)** — theory + full API.
