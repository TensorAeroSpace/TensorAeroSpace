# Recipe 08 — Save, resume and share adaptive controllers

**Goal:** save an agent with nontrivial learning history, reload it, and verify
that the next 50 physical transitions agree. Then package the checkpoint for an
optional Hugging Face upload. Run the local Python blocks in order with this version of `tensoraerospace` installed. No network operation runs automatically in this recipe.

A checkpoint is useful only if you know what it contains. The controller and the
simulation own different state, and AA-INDI and iADP use different file formats.

## 1. Understand the persistence boundary

| Component | Saved by the agent? | What the experiment must retain |
|---|---|---|
| Learned model / critic | Yes, for the corresponding algorithm | Same implementation and compatible configuration |
| Controller history and filters | Included in the current iADP and AA-INDI checkpoints | Same next observation and call order |
| Aircraft / servo / engine state | No | Physical states, actuator states and model parameters |
| Simulation clock and event progress | No | Current time/index and already applied plant events |
| Reference generator / random generators | Not external generators | Their state or full reproducible schedules |

`reset()` is not a restore operation. For these adaptive agents it clears loop
history while retaining learned parameters; use a new agent for an independent
nominal run and `from_pretrained` to continue a saved run.

## 2. Create an iADP checkpoint with actual learning history

This small deterministic plant follows `x_next = a*x + b*u`. A nominal DARE seed
keeps the example focused on persistence; it is not an aircraft controller.


```python
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_discrete_are
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig

dt = 0.01
a = np.exp(-2 * dt)
b = -np.expm1(-2 * dt) / 2
F = np.diag([a, 1.0])
G = np.array([[b], [0.0]])
gamma = 0.9
R = np.array([[0.01]])
Q_aug = np.array([[1.0, -1.0], [-1.0, 1.0]])
P = solve_discrete_are(np.sqrt(gamma) * F, np.sqrt(gamma) * G, Q_aug, R)
agent = IADPAgent(1, 1, IADPConfig(
    dt=dt, Q=np.eye(1), R=R, gamma=gamma,
    F_init=F, G_init=G, P_init=P,
    learning_mode="continuous", policy_eval_window=100,
    policy_eval_min_samples=100, policy_eval_every=20,
    u_magnitude_limit=0.5, u_rate_limit=2.0,
))
x, reference = np.zeros(1), np.array([0.05])
for k in range(200):
    u = agent.predict(x, reference, k)
    x = a * x + b * u
    agent.learn(x, reference, k, applied_action=u)
```


The controller has now seen 200 transitions. Save immediately after `learn`, so
there is no ambiguity about whether an action still needs to be applied.


```python
run_dir = Path(agent.save("./checkpoints/recipe08-iadp"))
# The controller checkpoint does not contain the external plant.
(run_dir / "simulation.json").write_text(json.dumps({
    "state": x.tolist(), "next_step": 200,
    "reference": reference.tolist(), "dt": dt, "a": a, "b": b,
}, indent=2))
print("Saved files:", sorted(path.name for path in run_dir.iterdir()))

restored = IADPAgent.from_pretrained(str(run_dir))
simulation = json.loads((run_dir / "simulation.json").read_text())
x_restored = np.array(simulation["state"])
reference_restored = np.array(simulation["reference"])
for k in range(simulation["next_step"], simulation["next_step"] + 50):
    u = agent.predict(x, reference, k)
    u_restored = restored.predict(x_restored, reference_restored, k)
    np.testing.assert_array_equal(u, u_restored)
    x = a * x + b * u
    x_restored = simulation["a"] * x_restored + simulation["b"] * u_restored
    agent.learn(x, reference, k, applied_action=u)
    restored.learn(x_restored, reference_restored, k, applied_action=u_restored)
    np.testing.assert_array_equal(x, x_restored)
    np.testing.assert_array_equal(agent.P, restored.P)
    np.testing.assert_array_equal(agent.rls.theta, restored.rls.theta)
print("50 continued transitions match exactly in this process")
```


The assertion checks actions, plant states, RLS coefficients and critic matrices
through 50 further transitions, including learning. Exact equality here applies
to this deterministic continuation in the same software environment. Different
BLAS implementations or versions can change floating-point results.

`simulation.json` belongs to the example harness. A complete aircraft continuation
requires more than this scalar state: reconstruct the environment with the same
parameters, clock, fault progress, actuator state and sensor generators.

## 3. Know which files belong to which agent

| Agent | Current files and contents |
|---|---|
| iADP | `config.json`: constructor/configuration and output maps; `rls.npz`: incremental model/covariance/counters; `value.npz`: critic; `weights.npz`: cost weights; `loop_state.npz`: command and transition history; `window.npz`: critic samples. |
| AA-INDI | `paper_aaindi.json`: geometry/configuration, moment estimators, OTSEKF covariance state, HOSM state, filtered measurements and pending command. |

Use the directory returned by `save`, rather than guessing its time-stamped name.
It can be a relative path when the supplied parent is relative. Save independent
runs under separate parents; the directory names have second-level timestamps.

For IHDP, IM-GDHP and ET-DHP, follow the persistence sections in their
[agent documentation](../agent/ihdp.md), [IM-GDHP documentation](../agent/imgdhp.md)
and [ET-DHP documentation](../agent/et_dhp.md). Do not assume the same file layout
or auxiliary constructor arguments for every algorithm.

## 4. Save AA-INDI through its local interface

AA-INDI's loader accepts a **local folder**. This standalone example uses the
nominal B747 initializer and a valid SI sensor packet; it checks a command at the
saved timestamp. A complete flight loop is in [Recipe 14](14_aaindi.md).


```python
from tensoraerospace.agent.aa_indi import AAINDIAgent
from tensoraerospace.benchmark import B747EngineFailureBenchmark

from tensoraerospace.agent.aa_indi import FlightMeasurement

benchmark = B747EngineFailureBenchmark(dt=0.02)
aa = benchmark.make_aaindi()
trim = benchmark.nominal_trim()
state = trim.to_state()
action = np.array([trim.elevator_rad, 0.0, 0.0, trim.throttle])
sample = FlightMeasurement.from_model(benchmark.nominal_model(), applied_action=action, surface_indices=(1, 2))
rate_command = np.array([0.001, 0.0, 0.0])
aa.predict(sample, rate_command)
aa_dir = aa.save("./checkpoints/recipe08-aaindi")
aa_restored = AAINDIAgent.from_pretrained(aa_dir)
np.testing.assert_array_equal(
    aa.predict(sample, rate_command), aa_restored.predict(sample, rate_command),
)
print("AA-INDI local checkpoint:", aa_dir)
```


For a checkpoint between `predict` and `learn`, the pending command is retained.
Continue by applying that command once and calling `learn` with the next sensor
packet. For a checkpoint after `learn`, continue with the next `predict` using
that same packet. Do not invent a different observation at an existing timestamp.

## 5. Add enough context to reproduce the run

Before sharing, place a `README.md` alongside the checkpoint. Record:

- Repository commit, Python/dependency versions and the agent class.
- Aircraft/model configuration, trim, sampling interval and integrator.
- State order, SI/US conversions, absolute vs trim-relative commands, actuator limits.
- Reference schedule, seed, controller gains and nominal initialization.
- Whether the checkpoint is before or after `learn`, and how to restore the plant.
- Healthy/faulty metrics with time windows and any incomplete or divergent runs.

For the B747 example, link the [comparison protocol](../comparison/aaindi_vs_pid_lqr_lqi_b747.md).
A saved model alone does not encode its evaluation assumptions or prove robustness.

## 6. Upload or download explicitly

The following functions use the installed `huggingface_hub` package. Define them
locally; call the commented examples only when you want to publish or download.
Set `HF_TOKEN` in your environment instead of writing a token into a notebook.


```python
import os
from huggingface_hub import HfApi, snapshot_download

def upload_checkpoint(folder, repo_id):
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    return api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(folder))

def download_aaindi(repo_id, revision):
    folder = snapshot_download(
        repo_id=repo_id, revision=revision, token=os.environ.get("HF_TOKEN"),
    )
    return AAINDIAgent.from_pretrained(folder)

# Explicit optional network operations, after replacing the repository and revision:
# upload_checkpoint(aa_dir, "your-username/aaindi-b747")
# downloaded = download_aaindi("your-username/aaindi-b747", "COMMIT_SHA")
# downloaded_iadp = IADPAgent.from_pretrained(
#     "your-username/iadp-example", version="COMMIT_SHA",
#     access_token=os.environ.get("HF_TOKEN"),
# )
```


`create_repo` explicitly creates the destination if needed. The helper defaults
to a private new repository. Use a commit revision for reproducible downloads.
The current iADP class also has `publish_to_hub`, which uploads to an existing
repository, and a Hub-aware `from_pretrained`. AA-INDI has a local loader and no
`publish_to_hub` method; download its files first, then call its loader.

## Troubleshooting

| Symptom | Likely cause and next check |
|---|---|
| First command differs after reload | The plant observation, reference, pending transition or time differs. Compare these before changing model weights. |
| Commands agree once, then diverge | Restore the simulator/servo/RNG state as well, and compare multiple learning steps. |
| `paper_aaindi.json` is missing | The folder is not a checkpoint from the current AA-INDI implementation. |
| Old iADP config is rejected | Removed critic regularization/blending options cannot represent the current update law; regenerate the checkpoint. |
| AA-INDI cannot load `username/repo` | Download to a local directory first. |
| Upload reports repository not found | Create the repository explicitly and check access to it. |

Old rate-only AA-INDI checkpoints lack physical geometry and independent
navigation state. Do not describe loading them as an equivalent continuation.
Keep the source revision with each experiment and regenerate incompatible files.

**Next:** [Recipe 09 — Fault tolerance](09_fault_tolerance.md) ·
[Recipe 14 — AA-INDI on B737](14_aaindi.md).
