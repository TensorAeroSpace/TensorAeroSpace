# DSAC on Boeing 747 (step response)

Minimal DSAC training run for longitudinal pitch control of the normalized `ImprovedB747Env`. The script uses IQN-based twin critics with CAPS smoothness regularization to track a 5° elevator step reference.

## Run it

```bash
python example/reinforcement_learning/deep_rl/example_dsac_b747.py
```

What happens:

- Builds a 800-step reference signal (0 → 5° step at 20% of the episode)
- Creates `ImprovedB747Env` with normalized observations/actions
- Instantiates `DSAC` with 32 quantiles, CAPS enabled, and auto-entropy tuning
- Trains for 5 episodes and writes TensorBoard logs

!!! note
    The script auto-selects GPU if available: `device = "cuda" if torch.cuda.is_available() else "cpu"`.

## Core snippet

```python
import numpy as np
from tensoraerospace.agent import DSAC
from tensoraerospace.envs.b747 import ImprovedB747Env

def make_reference(steps: int, step_deg: float = 5.0) -> np.ndarray:
    ref = np.zeros((1, steps), dtype=np.float32)
    ref[:, steps // 5 :] = np.deg2rad(step_deg)
    return ref

env = ImprovedB747Env(
    initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
    reference_signal=make_reference(800, step_deg=5.0),
    number_time_steps=800,
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
agent.close()
```

## Tips

- Increase `num_episodes` for meaningful performance; the default 5 is only for a quick smoke test.
- Watch TensorBoard (`tensorboard --logdir runs`) to monitor critic/policy losses and entropy.
- If actions look too smooth, lower `updates_per_step` or the smoothness lambdas in `DSAC.update_parameters`.

