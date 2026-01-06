# DSAC Training: Boeing 747 Step Response

This production-ready script trains a DSAC agent for **step response control** of the Boeing 747 longitudinal dynamics. It uses a **curriculum learning** approach to progressively increase task difficulty.

## Overview

| Aspect | Description |
|--------|-------------|
| **Task** | Step response tracking (pitch angle θ) |
| **Environment** | `ImprovedB747VecEnvTorch` (vectorized, GPU-accelerated) |
| **Training paradigm** | Multi-stage curriculum learning |
| **Checkpointing** | Best checkpoint saved based on eval return |

## Why curriculum learning?

Step response control with strict quality metrics (low overshoot, fast settling) creates a **sparse reward landscape**. Training directly on the final task often fails because:

1. The agent learns to terminate early to avoid accumulating negative rewards
2. Large reward discontinuities destabilize critic learning
3. The agent never experiences successful trajectories to learn from

The curriculum addresses these issues by:

1. **Stage 0 (Bootstrap)**: Train with `tracking` reward — dense, smooth feedback
2. **Stage 1a**: Switch to `step_response` with relaxed thresholds (20% overshoot, 5% settle band)
3. **Stage 1b**: Tighten thresholds to target values (5% overshoot, 1% settle band)
4. **Stage 2**: Remove completion/early-termination bonuses for robustness

## Run it

```bash
python example/reinforcement_learning/train_dsac_b747_step_response.py
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs
```

## Script structure

```
train_dsac_b747_step_response.py
├── load_dsac_checkpoint()     # Resume from previous run
├── find_latest_metrics()      # Auto-find best checkpoint
├── make_reference()           # Generate step reference signal
├── eval_one_episode()         # Single-episode evaluation
├── save_eval_best()           # Checkpoint management
└── main()                     # Training curriculum
```

## Key hyperparameters

### Vectorized environment

```python
env_train = ImprovedB747VecEnvTorch(
    num_envs=128,              # Parallel environments
    dt=0.1,                    # Time step (s)
    tn=20.0,                   # Episode duration (s)
    auto_reset=True,           # Auto-reset on termination
    include_reference_in_obs=True,  # Reference visible to agent
    reward_mode="tracking",    # Initial mode (switched later)
    step_randomization={       # Domain randomization
        "amplitude_deg_range": (-5.0, 5.0),
        "step_time_sec_range": (1, 15),
    },
)
```

### DSAC agent

```python
agent = DSAC(
    env_train,
    batch_size=256,
    memory_capacity=1_000_000,
    learning_starts=100_000,   # Large buffer before training
    updates_per_step=4,        # Multiple gradient steps per env step
    lr=4.4e-4,
    gamma=0.995,               # High discount for long episodes
    tau=0.005,
    num_quantiles=8,           # IQN quantiles
    embedding_dim=64,
    hidden_layers=[64, 64],
    automatic_entropy_tuning=True,
    reward_clip=50.0,          # Clip large penalties
)
```

## Curriculum stages explained

### Stage 0: Bootstrap with tracking reward

```python
env_train.reward_mode = "tracking"
agent.train_vector(total_steps=10_000, ...)
```

**Why**: Dense reward signal helps the agent learn basic control authority before facing sparse step-response metrics.

### Stage 1a: Relaxed step response

```python
env_train.reward_mode = "step_response"
env_train.overshoot_limit_ratio = 0.2   # 20% allowed
env_train.settle_band_ratio = 0.05      # 5% band
env_train.w_overshoot = 50.0            # Mild penalty
```

**Why**: Gradual transition to target task without overwhelming the agent with penalties.

### Stage 1b: Full step response

```python
env_train.overshoot_limit_ratio = 0.05  # 5% allowed
env_train.settle_band_ratio = 0.01      # 1% band
env_train.w_overshoot = 300.0           # Strong penalty
```

**Why**: Final task difficulty with strict quality requirements.

### Stage 2: Remove shaping

```python
env_train.completion_bonus = 0.0
env_train.early_termination_penalty_per_step = 0.0
```

**Why**: Ensure learned policy works without artificial shaping rewards.

## Preventing early termination hacks

The agent might learn to "game" the reward by terminating early. We counter this with:

1. **Completion bonus**: +5.0 reward for surviving the full episode
2. **Early termination penalty**: -1.0 per remaining step if episode ends early
3. **Reward clipping**: Limit extreme penalties to stabilize learning

```python
completion_bonus = 5.0
early_term_penalty_per_step = 1.0
agent.reward_clip = 20.0
```

## Checkpointing

The script maintains two checkpoint directories:

1. **`best_checkpoints/`**: Best during training (based on rolling reward)
2. **`best_eval/`**: Best on fixed evaluation episode

```python
# After each stage
m = eval_one_episode(agent, env_eval_step)
if m["return_sum"] > eval_best:
    save_eval_best(agent, eval_best_dir, metrics=m)
```

## Resume training

The script automatically finds and resumes from the latest `best_eval` checkpoint:

```python
resume_metrics = find_latest_metrics(Path("runs"))
if resume_checkpoint_dir is not None:
    agent = load_dsac_checkpoint(resume_checkpoint_dir, env_train, device=device)
```

## Tips

!!! tip "GPU acceleration"
    Set `device="cuda"` for significant speedup. The vectorized environment runs on GPU.

!!! tip "Replay buffer reset"
    When switching from `tracking` to `step_response`, the replay buffer is reset to avoid mixing incompatible reward scales.

!!! warning "Memory usage"
    With `memory_capacity=1_000_000` and `num_envs=128`, expect ~4-8 GB GPU memory.

## See also

- [DSAC Algorithm](../../../agent/dsac.md) — architecture and hyperparameters
- [DSAC Training (Sine Tracking)](train-dsac-b747-tracking.md) — alternative task
- [DSAC Evaluation](eval-dsac-b747.md) — visualize trained agent
- [ImprovedB747Env](../../../model/b747_imporove.md) — environment details

