# DSAC B747 Evaluation Notebook

This notebook demonstrates how to **load a pretrained DSAC agent** and evaluate its performance on the Boeing 747 longitudinal pitch control task. It produces transient response plots and computes standard control-quality metrics.

## Overview

| Aspect | Description |
|--------|-------------|
| **Task** | Step response tracking (θ = 0° → 1°) |
| **Environment** | `ImprovedB747Env` (normalized state-space) |
| **Agent** | Pretrained DSAC checkpoint |
| **Output** | Transient plots, benchmark metrics table |

## What this notebook does

1. **Loads a trained DSAC checkpoint** using `DSAC.from_pretrained()` — supports local paths or HuggingFace Hub repositories.
2. **Configures the evaluation environment** with a unit step reference (1° pitch step at t=5s, dt=0.1s, T=20s).
3. **Runs a single evaluation episode** without exploration noise (`evaluate=True`).
4. **Visualizes the transient process** using the built-in `plot_transient_process()` and `plot_control()` methods.
5. **Computes quality metrics** with `ControlBenchmark` — overshoot, settling time, rise time, IAE, ISE, ITAE, etc.

## Run it

Open the notebook in Jupyter or VS Code:

```bash
jupyter notebook example/reinforcement_learning/deep_rl/eval_dsac_b747.ipynb
```

Or run directly if you have `nbconvert` installed:

```bash
jupyter nbconvert --execute --to notebook example/reinforcement_learning/deep_rl/eval_dsac_b747.ipynb
```

## Core code

```python
from tensoraerospace.agent import DSAC
from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standard import unit_step
from tensoraerospace.benchmark.bench import ControlBenchmark
import numpy as np

# 1. Setup evaluation environment
dt, tn, step_deg, step_time = 0.1, 20.0, 1.0, 5.0
t = np.arange(0.0, tn + dt, dt, dtype=np.float32)
ref = unit_step(t, degree=step_deg, time_step=step_time, output_rad=True).reshape(1, -1)

env = ImprovedB747Env(
    initial_state=np.zeros(4, dtype=float),
    reference_signal=ref,
    number_time_steps=ref.shape[1],
    dt=dt,
    include_reference_in_obs=True,
    reward_mode="step_response",
)

# 2. Load pretrained agent
# You can load from Hugging Face Hub:
agent = DSAC.from_pretrained("TensorAeroSpace/dsac-b747-step-response")
agent.env = env
agent.to_device("cpu")  # or "cuda" / "mps"
agent.eval()

# 3. Run evaluation episode
obs, _ = env.reset()
done = False
while not done:
    action = agent.select_action(obs, evaluate=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# 4. Plot transient response
env.unwrapped.model.plot_transient_process(
    "theta", t, ref[0], to_deg=True, figsize=(15, 4)
)

# 5. Compute benchmark metrics
benchmark = ControlBenchmark()
benchmark.plot(ref[0], system_signal, tps=t, signal_val=0.0, dt=dt)
print(benchmark.generate_report(ref[0], system_signal, signal_val=0.0, dt=dt))
```

## Output example

The notebook produces:

1. **Transient response plot** — θ(t) vs reference signal with settling zone highlighted
2. **Control signal plot** — elevator deflection δe(t)
3. **Metrics table** with:
   - Overshoot (%)
   - Settling time (s) — time to enter and stay within ±5% band
   - Rise time (s) — time to reach 90% of final value
   - Peak time (s)
   - Decay ratio
   - Steady-state error
   - IAE, ISE, ITAE integrals
   - Quality index (weighted combination)

## Tips

!!! tip "Device selection"
    The notebook auto-detects CUDA/MPS/CPU. For Apple Silicon Macs, MPS gives a significant speedup over CPU.

!!! tip "Checkpoint paths"
    You can load from:
    
    - **Local path**: `/path/to/checkpoint/folder`
    - **HuggingFace Hub**: `TensorAeroSpace/dsac-b747-step-response`

!!! tip "Customizing the reference"
    Change `step_deg`, `step_time`, `tn`, and `dt` to test different scenarios (e.g., larger steps, faster dynamics).

## See also

- [DSAC Algorithm](../../../agent/dsac.md) — architecture and hyperparameters
- [DSAC Training (Step Response)](train-dsac-b747-step-response.md) — how to train from scratch
- [DSAC Training (Sine Tracking)](train-dsac-b747-tracking.md) — training for continuous tracking
- [DSAC vs PID Comparison](../../../comparison/dsac_vs_pid_b747.md) — benchmark comparison

