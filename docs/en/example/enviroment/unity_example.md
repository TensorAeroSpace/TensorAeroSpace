# Unity Environment with a DQN Agent

![Unity Demo](../../model/img/example_run.jpg){ width=800 }

## Lesson Overview

Quick path: connect the Unity environment (Editor or standalone build), train a DQN agent, and interact via a random policy. For Unity setup follow "Unity Environment" — see [Unity Environment](../../guide/unity_env.md).

!!! info "Unity Environment Repository"
    The Unity environment source code is available at: [TensorAeroSpace/UnityAirplaneEnvironment](https://github.com/TensorAeroSpace/UnityAirplaneEnvironment)

## Goals and Requirements

- What you will do:
  - Connect the Unity environment to TensorAeroSpace (Editor or build).
  - Launch DQN training and evaluation.
  - Test interaction with a random agent.
- Requirements:
  - Unity + ML-Agents (`mlagents==1.1.0`), Python 3.8+, `tensoraerospace`.

## Imports

```python
from tensoraerospace.agent.dqn.model import Model, DQNAgent
from tensoraerospace.envs.unity_env import get_plane_env, unity_discrete_env
```

## Connect the Environment

=== "Editor"

    ```python
    # Connect to the Unity Editor (press Play after starting the script)
    env = unity_discrete_env()
    ```

=== "Standalone build"

    ```python
    # Provide the path to the compiled Unity build
    build_path = "/abs/path/to/build.x86_64"   # Linux
    # build_path = "C:\\path\\to\\build.exe"  # Windows
    env = get_plane_env(build_path, server=True)
    ```

!!! note "Port and connection"
    Default port is 5004. If it is busy, stop conflicting processes or change it in the ML-Agents/environment settings.

## Train and Evaluate DQN

```python
num_actions = env.action_space.n
model = Model(num_actions)
target_model = Model(num_actions)

agent = DQNAgent(model, target_model, env, train_nums=100)
agent.train()

# Evaluation after training
rewards_sum = agent.evaluation(env)
print("After Training: %d out of 200" % rewards_sum)
```

## Typical Connection Logs

```text
[INFO] Listening on port 5004. Start training by pressing the Play button in the Unity Editor.
[INFO] Connected to Unity environment with package version 2.2.1-exp.1 and communication version 1.5.0
[INFO] Connected new brain: My Behavior?team=0
[WARNING] uint8_visual was set to true, but visual observations are not in use. This setting will not have any effect.
[WARNING] The environment contains multiple observations. You must define allow_multiple_obs=True to receive them all.
```

## Launch Screen

![Unity Interface](../../guide/img/6.png){ width=800 }

_Figure 2.1. Training visualization and interaction (Unity Environment)_

## Random Agent Interaction

=== "Plane env"

    ```python
    env = get_plane_env()
    env.reset()

    print(env.action_space)       # Action space size
    print(env.observation_space)  # Observation dimension

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()
    ```

=== "Discrete env"

    ```python
    env = unity_discrete_env()
    env.reset()

    print(env.action_space)

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()
    ```

## Running in Docker on Multiple GPU/CPU

Training on multiple GPUs speeds up experiments, allows richer models, and parallel experience collection.

```bash
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN pip install mlagents==1.1.0 scipy==1.5.4 tensorboard==2.17.0
RUN mkdir /workspace/logs
COPY a3c_example.py /workspace

ENTRYPOINT tensorboard --logdir /workspace/logs --port 8889 --host 0.0.0.0 & python /workspace/a3c_example.py
```

## Open sources (documentation)

- Unity ML-Agents: https://github.com/Unity-Technologies/ml-agents
- TensorBoard documentation: https://www.tensorflow.org/tensorboard
- Docker: https://docs.docker.com/

### A3C Launch Script Inside Docker

```python
from tensoraerospace.envs.unity_env import get_plane_env
from tensoraerospace.agent.a3c import Agent, setup_global_params

def env_function(worker_id):
    # /tf/linux_build/build.x86_64 — path to the Unity executable
    return get_plane_env("/tf/linux_build/build.x86_64", server=True, worker=worker_id)

actor_lr = 0.0005
critic_lr = 0.001
gamma = 0.99
hidden_size = 128
update_interval = 1
max_episodes = 100

setup_global_params(actor_lr, critic_lr, gamma, hidden_size, update_interval, max_episodes)

agent = Agent(env_function, gamma)
agent.train()
```

### Run the Container

```bash
docker run \
  -v ./tensoraerospace:/tf/tensoraerospace \
  -v ./linux_build:/tf/linux_build \
  -p 8889:8889 unity_docker
```

## Troubleshooting

- Port 5004 busy: change it in the configuration or stop the conflicting process.
- `allow_multiple_obs=True` warning: enable the flag or use the first observation stream.
- `mlagents` version mismatch: ensure version aligns with ML-Agents (`mlagents==1.1.0`).
- Build does not start: verify `build_path` and execution permissions (Linux: `chmod +x`).

## Training Showcase

![Training Example](../../model/img/example_run.jpg){ width=800 }

## Related Examples

- [Unity with SAC](../agent/sac/example-sac-unity.md) — SAC agent for continuous control
- [Unity Environment Setup](../../guide/unity_env.md) — complete setup guide
- [Unity Environment Model](../../model/unity_env.md) — environment details and scenes
