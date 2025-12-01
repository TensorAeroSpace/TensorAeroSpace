# SAC with Unity ML-Agents

![Unity Demo](../../../model/img/example_run.jpg){ width=800 }

## Overview

This example demonstrates training a **Soft Actor-Critic (SAC)** agent from TensorAeroSpace in a Unity ML-Agents environment for continuous control.

!!! info "Unity Environment Repository"
    The Unity environment source code is available at: [TensorAeroSpace/UnityAirplaneEnvironment](https://github.com/TensorAeroSpace/UnityAirplaneEnvironment)

## Goals and Requirements

**What you will do:**

- Connect Unity environment to TensorAeroSpace (Editor or build)
- Train SAC agent on continuous actions
- Evaluate the trained policy quality
- Save and load the model

**Requirements:**

- Unity + ML-Agents (Python package `mlagents==1.1.0`)
- Python 3.8+
- TensorAeroSpace
- GPU recommended for training

## Installation

```bash
pip install mlagents==1.1.0
```

## Imports

```python
import gymnasium as gym
import numpy as np
from tensoraerospace.agent import SAC

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
```

## Gymnasium API Wrapper

Unity ML-Agents uses the old `gym` API. We need a wrapper to convert to the modern `gymnasium` API:

```python
class UnityToGymnasiumWrapper(gym.Wrapper):
    """Wrapper to convert Unity ML-Agents gym to Gymnasium API.
    
    Args:
        env: Unity environment wrapped with UnityToGymWrapper
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        """Execute one step in the environment."""
        result = self.env.step(action)
        obs, reward, done, info = result
        # Gymnasium uses 5-tuple instead of 4-tuple
        return obs, reward, done, False, info

    def close(self):
        """Close the Unity environment."""
        self.env.close()
```

## Connecting to Unity

=== "Unity Editor"

    ```python
    unity_env = UnityEnvironment(
        file_name=None,  # None to connect to Editor
        log_folder="./logs/",
        additional_args=["-logfile", "unity.log"]
    )
    ```

=== "Standalone Build"

    ```python
    # Linux
    unity_env = UnityEnvironment("/path/to/build.x86_64")
    
    # Windows
    unity_env = UnityEnvironment("C:\\path\\to\\build.exe")
    ```

```python
# Wrapper for compatibility
gym_env = UnityToGymWrapper(unity_env, uint8_visual=True)
env = UnityToGymnasiumWrapper(gym_env)

print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
```

!!! note "Connection Port"
    Default port is 5004. If busy, close other Unity processes.

## SAC Agent Configuration

```python
agent = SAC(
    env=env,
    # Training dynamics
    updates_per_step=1,
    batch_size=256,
    memory_capacity=1_000_000,
    
    # Learning rates
    lr=3e-4,          # Critic
    policy_lr=3e-4,   # Actor
    
    # RL hyperparameters
    gamma=0.99,       # Discount factor
    tau=0.005,        # Soft update coefficient
    alpha=0.2,        # Initial entropy coefficient
    
    # Policy configuration
    policy_type="Gaussian",
    target_update_interval=1,
    automatic_entropy_tuning=True,
    
    # Network architecture
    hidden_size=256,
    
    # Device and logging
    device="cuda",  # Use "cpu" if no GPU available
    verbose_histogram=False,
    seed=42,
)
```

### Hyperparameters Table

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 256 | Mini-batch size |
| `memory_capacity` | 1,000,000 | Replay buffer size |
| `lr` | 3e-4 | Critic learning rate |
| `policy_lr` | 3e-4 | Actor learning rate |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Soft update coefficient |
| `automatic_entropy_tuning` | True | Auto-adjust exploration |
| `hidden_size` | 256 | Hidden layer neurons |

## Training

```python
NUM_EPISODES = 1000
agent.train(num_episodes=NUM_EPISODES)
```

!!! tip "TensorBoard"
    Training logs are saved for TensorBoard:
    ```bash
    tensorboard --logdir runs/
    ```

## Evaluation

```python
def evaluate_agent(agent, env, num_episodes=5):
    """Evaluate trained agent."""
    rewards = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")
    
    print(f"\nMean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    return np.mean(rewards), np.std(rewards), rewards

# Run evaluation
mean_reward, std_reward, all_rewards = evaluate_agent(agent, env, num_episodes=5)
```

## Save and Load Model

```python
# Save
agent.save(path="./checkpoints")

# Load
loaded_agent = SAC.from_pretrained("./checkpoints/Nov24_11-01-27_SAC")
```

## Close Environment

```python
env.close()
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Port 5004 is busy` | Close other Unity processes |
| `allow_multiple_obs` warning | Set `allow_multiple_obs=True` or ignore |
| Connection timeout | Start Unity scene before Python script |
| CUDA out of memory | Reduce `batch_size` or `hidden_size` |

## Typical Connection Log

```text
[INFO] Listening on port 5004...
[INFO] Connected to Unity environment with package version X.X.X
[INFO] Connected new brain: BehaviorName?team=0
```

## References

- [SAC Paper](https://arxiv.org/abs/1801.01290)
- [Unity ML-Agents Documentation](https://unity-technologies.github.io/ml-agents/)
- [TensorAeroSpace SAC API](../../../agent/sac.md)

