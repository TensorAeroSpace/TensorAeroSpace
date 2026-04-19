# Recipe 08 — Save / load / publish to HuggingFace

**Goal.** Checkpoint an adaptive-critic agent, reload it bit-identically, and push it to the [Hugging Face Hub](https://huggingface.co/). Five agents share this contract: IHDP, IM-GDHP, ET-DHP, AA-INDI, iADP.

**Related.** [Recipe 06](06_online_adaptive.md) for the agent lifecycle these checkpoints fit into.

## The contract

Every compliant agent implements four methods:

| Method | Purpose |
|---|---|
| `agent.save(path)` | Write a date-stamped directory under `path` containing everything needed to resume. Returns the absolute path. |
| `AgentClass.from_pretrained(loc, access_token=None, version=None)` | Load from a local directory **or** from a `namespace/repo` on the Hub. |
| `agent.publish_to_hub(repo_name, folder_path, access_token=None)` | Upload a previously-saved folder to the Hub. |
| `agent.get_param_env()` | Build the JSON-serialisable config dict used by `save()`. |

These are in-place contract methods — no subclass gymnastics. Identical across the five agents.

## Minimal round-trip

```python
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig

agent = IADPAgent(n_state=1, n_control=1, config=IADPConfig(seed=0))
# ... run some env steps so there's non-trivial state to save ...

# 1. Save locally
run_dir = agent.save('./checkpoints')     # './checkpoints/Apr19_12-34-56_IADPAgent'

# 2. Reload from the same directory
restored = IADPAgent.from_pretrained(run_dir)

# 3. The next predict() must produce the bit-identical command
np.testing.assert_allclose(agent.predict(x, ref, k), restored.predict(x, ref, k), atol=1e-12)
```

## What lives on disk

A saved directory contains:

| File | Content |
|---|---|
| `config.json` | Full dataclass config + constructor args; arrays (Q, R, G_init, …) serialised as lists. |
| `rls.npz` / `vff_rls.npz` | RLS parameter matrix `theta`, covariance, update counter, last residual. |
| `value.npz` | Kernel matrix `P̃` (iADP), critic weights (IHDP/IMGDHP/ET-DHP), etc. |
| `weights.npz` | Active Q, R matrices (after any default fill-in). |
| `loop_state.npz` | Rolling state — `X_prev`, `delta_prev`, integrator state, step counter. |
| `window.npz` | Policy-evaluation transition buffer (iADP). |
| `deriv_state.npz`, `bias_state.npz` | Sensor-filter states (AA-INDI). |

**Mid-episode saves are bit-identical on reload** — a guarantee exercised by the library's test suite. This is the key difference from naive `pickle` of the agent object: the contract persists *only what's needed*, and the reload reconstructs from `config.json`.

## HuggingFace Hub round-trip

### Upload

```python
run_dir = agent.save('./checkpoints')

agent.publish_to_hub(
    repo_name='your-username/iadp-f16-v1',
    folder_path=run_dir,
    access_token='hf_...',        # from https://huggingface.co/settings/tokens
)
```

If `repo_name` doesn't exist on the Hub, it's created. The whole `run_dir` is uploaded as-is — so add a `README.md` with training context before calling `publish_to_hub` if you want a proper model card.

### Download

```python
restored = IADPAgent.from_pretrained(
    repo_name='your-username/iadp-f16-v1',
    access_token='hf_...',        # required only for private repos
    version='main',               # optional branch / tag / commit
)
```

Under the hood `from_pretrained` calls `huggingface_hub.snapshot_download(repo_name)` when the path isn't a local directory.

## Writing a minimal model card

Add `README.md` to your `run_dir` before `publish_to_hub`:

```markdown
---
tags:
  - tensoraerospace
  - flight-control
  - iadp
library_name: tensoraerospace
---
# iADP on nonlinear F-16 (v1)

Pitch-rate tracking agent trained on `NonlinearLongitudinalF16-v0` at `dt = 0.01` s,
with the DARE-based `P_init` and `policy_eval_blend = 0.1` (see the
[cookbook recipe](https://...)).

## Intended use
Online-adaptive ω_z tracking at the F-16 trim point.

## Hyperparameters
- Q = 30 000, R = 0.1, γ = 0.9
- γ_RLS = 0.9999, φ_init = 1.0
- policy_eval_blend = 0.1, policy_eval_every = 5

## Reproduce
`IADPAgent.from_pretrained('your-username/iadp-f16-v1')`
```

Any YAML frontmatter is rendered natively by the Hub.

## Pitfalls

- **Path starting with `./`, `../`, `/`, or `~` is treated as local.** If the local path doesn't exist, `from_pretrained` raises `FileNotFoundError` instead of falling through to Hub download.
- **`config.json` arrays must be JSON-serialisable.** The library converts NumPy arrays to lists in `get_param_env()`; if you subclass, preserve this.
- **Older checkpoints without `loop_state.npz`** simply get a freshly-constructed zero state on reload. This means ep-level reload is safe across library versions; mid-episode reload requires the same library minor version that wrote it.
- **Uploading large window buffers.** For agents that store a policy-evaluation window (iADP), the `window.npz` can grow with `policy_eval_window`. At `policy_eval_window=300` on a 2-D augmented state this is ~4 kB — negligible. At `policy_eval_window=10 000` for a 6-DoF problem, it's worth checking.

## Public example

The iADP agent's [nonlinear-F-16 example notebook](../example/agent/iadp/example_iadp_nonlinear.md) exercises the full round-trip as part of its tests.

## Where to go next

- **[Recipe 09 — Fault-tolerance](09_fault_tolerance.md)** — saving an agent under fault conditions and reloading it safely.
- **[Recipe 10 — Adding your own plant model](10_custom_plant.md)** — extending the persistence contract to new agents you write.
