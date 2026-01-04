# Distributional Soft Actor-Critic (DSAC)

DSAC extends SAC with distributional (quantile) critics and the CAPS smoothness regularizer. Each critic is an Implicit Quantile Network (IQN) that predicts a set of quantiles instead of a single Q-value, giving richer uncertainty estimates and better robustness on noisy aerospace dynamics.

## Key differences vs SAC

- IQN critics: twin quantile heads (`QuantileTwin`) with cosine embeddings
- Quantile Huber loss: learns full return distribution per action
- CAPS regularization: spatial and temporal smoothness penalties on the policy output
- Same replay buffer, entropy tuning, and target updates as SAC

## Quick start

```python
import numpy as np
import torch
from tensoraerospace.agent import DSAC
from tensoraerospace.envs.b747 import ImprovedB747Env

def step_reference(steps: int, deg: float = 5.0) -> np.ndarray:
    ref = np.zeros((1, steps), dtype=np.float32)
    ref[:, steps // 5 :] = np.deg2rad(deg)
    return ref

device = "cuda" if torch.cuda.is_available() else "cpu"
num_steps = 800

env = ImprovedB747Env(
    initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
    reference_signal=step_reference(num_steps, deg=5.0),
    number_time_steps=num_steps,
    dt=0.02,
    reward_mode="step_response",
)

agent = DSAC(
    env,
    batch_size=128,
    memory_capacity=200_000,
    learning_starts=1_000,
    updates_per_step=1,
    num_quantiles=32,
    embedding_dim=32,
    hidden_layers=[64, 64],
    huber_threshold=1.0,
    lr=3e-4,
    policy_lr=3e-4,
    device=device,
    log_every_updates=50,
    automatic_entropy_tuning=True,
)

agent.train(num_episodes=5, save_best=False)
agent.save("./runs")
agent.close()
```

!!! tip
    Keep `num_quantiles` between 16–64 and `huber_threshold` near 1.0 for stable training. CAPS smoothness is built in; reduce `updates_per_step` if actions become overly smooth.

## API reference

:::: tensoraerospace.agent.dsac.dsac.DSAC

:::: tensoraerospace.agent.dsac.model.QuantileTwin

:::: tensoraerospace.agent.dsac.model.IQNCritic

