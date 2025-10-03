# SAC B747 — Evaluate a pretrained agent (ImprovedB747Env)

A clean, minimal example that runs a pretrained SAC policy on the normalized Boeing 747 environment.

![b747](img/sac-b747-impoved.jpg)

---

## Prerequisites

```bash
pip install -U tensoraerospace pygame torch
```

Note: Rendering uses Pygame; run on a machine with a display.

## Quick run

```bash
python example/reinforcement_learning/sac-b747-render.py --render --dt 0.1 --tn 200 --repo TensorAeroSpace/sac-b747
```

## Minimal example

```python
import numpy as np
from tensoraerospace.agent.sac import SAC
from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standart import sinusoid_vertical_shift
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp

# Build env (short, consistent with repository example)
dt = 0.1
tn = 200

# Time base and reference signal (1 deg sinusoid in radians)
tp = generate_time_period(tn=tn, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
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
    number_time_steps=len(tp),
    dt=dt,
    initial_elevator_deg=0.0,
    use_initial_action_on_first_step=True,
)
# Ensure model discretization matches the environment step
env.unwrapped.model.discretisation_time = dt

# Load agent from the Hub and run one episode
agent = SAC.from_pretrained("TensorAeroSpace/sac-b747")
obs, info = env.reset()
done = False
ret = 0.0
while not done:
    action = agent.select_action(obs, evaluate=True)
    obs, reward, terminated, truncated, info = env.step(action)
    # Render visualization each step
    env.render(mode="human")
    done = bool(terminated or truncated)
    ret += float(reward)
print(f"Return: {ret:.2f}")
```

### Notes

- Actions and observations in `ImprovedB747Env` are normalized to [-1, 1].
- The reference signal here is a smooth 1° sinusoid in radians.
- `discretisation_time` is set explicitly to keep the model and env in sync.
- Initial state ordering is `[u, w, q, theta]` in SI units.

See also the organization page on the Hub: [TensorAeroSpace on Hugging Face](https://huggingface.co/TensorAeroSpace).
