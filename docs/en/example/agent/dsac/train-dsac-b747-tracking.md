# DSAC Training: Boeing 747 Sine Tracking

This script trains a DSAC agent for **continuous sinusoidal tracking** of the Boeing 747 pitch angle. Unlike step response, tracking requires smooth, continuous control to follow a time-varying reference.

## Overview

| Aspect | Description |
|--------|-------------|
| **Task** | Sine tracking (θ follows ±1° sinusoid at 0.05 Hz) |
| **Environment** | `ImprovedB747VecEnvTorch` (vectorized, GPU-accelerated) |
| **Training paradigm** | Multi-stage amplitude curriculum |
| **Reference** | Fixed sinusoid: ±1°, f=0.05 Hz, T=20s |

## Why sine tracking?

Sine tracking tests the agent's ability to:

1. **Follow dynamic references** — not just settle to a constant
2. **Produce smooth control** — jerky actions cause phase lag
3. **Handle direction changes** — requires anticipatory control
4. **Maintain precision** — small amplitude requires fine-grained actuation

## Run it

```bash
python example/reinforcement_learning/train_dsac_b747_tracking.py
```

The script displays a preview of the reference signal and waits for you to close the window before training starts.

Monitor training:

```bash
tensorboard --logdir runs
```

## Script structure

```
train_dsac_b747_tracking.py
├── load_dsac_checkpoint()     # Resume from previous run
├── find_latest_metrics()      # Auto-find best checkpoint
├── make_sine_reference()      # Generate sinusoidal reference
├── save_reference_plot()      # Save reference preview PNG
├── show_reference_plot()      # Display reference (blocking)
├── eval_one_episode()         # Single-episode evaluation
├── save_eval_best()           # Checkpoint management
└── main()                     # Training curriculum
```

## Key hyperparameters

### Reference signal

```python
dt = 0.1                    # Time step (s)
tn = 20.0                   # Episode duration (s) — 1 full period
sine_amplitude_deg = 1.0    # ±1° amplitude
sine_frequency_hz = 0.05    # 0.05 Hz = 20s period
```

### Vectorized environment

```python
env_train = ImprovedB747VecEnvTorch(
    num_envs=128,
    dt=dt,
    tn=tn,
    include_reference_in_obs=True,
    reward_mode="tracking",
    step_randomization={
        "signal_type": "sine",
        "amplitude_deg_range": (-1.0, 1.0),
        "frequency_hz_range": (0.05, 0.05),  # Fixed frequency
    },
)
```

### DSAC agent

```python
agent = DSAC(
    env_train,
    batch_size=256,
    memory_capacity=500_000,
    learning_starts=256,       # Start quickly (tracking is dense)
    updates_per_step=4,
    lr=4.4e-4,
    gamma=0.99,
    tau=0.005,
    num_quantiles=8,
    embedding_dim=64,
    hidden_layers=[64, 64],
    automatic_entropy_tuning=True,
    exploration_noise_std=0.0,  # No extra noise for tracking
    risk_distortion="neutral",
)
```

## Curriculum stages explained

### Stage 1: Moderate amplitude

```python
train_amp_stage1_deg = 1.0
env_train.step_rand.amplitude_deg_range = (-1.0, 1.0)
agent.train_vector(total_steps=20_000, ...)
```

**Why**: Start with the target amplitude to learn basic tracking behavior.

### Stage 2: Matched amplitude

```python
train_amp_stage2_deg = 1.0
env_train.step_rand.amplitude_deg_range = (-1.0, 1.0)
agent.train_vector(total_steps=40_000, ...)
```

**Why**: Extended training at target difficulty for policy refinement.

### Stage 3: Fine-tune without exploration

```python
agent.exploration_noise_std = 0.0
agent.train_vector(total_steps=20_000, ...)
```

**Why**: Remove any residual exploration noise for precise tracking.

## Tracking reward mode

The `tracking` reward mode provides dense feedback at every step:

$$
r_t = -|θ_t - θ_{ref}| - λ_u |u_t| - λ_{du} |Δu_t|
$$

Where:
- $θ_t$ — current pitch angle
- $θ_{ref}$ — reference at time t
- $u_t$ — control input (elevator)
- $Δu_t$ — control rate (smoothness penalty)

This dense signal is critical for tracking tasks where the reference constantly changes.

## Differences from step response training

| Aspect | Step Response | Sine Tracking |
|--------|---------------|---------------|
| Reference | Constant after step | Continuously varying |
| Reward mode | `step_response` | `tracking` |
| Curriculum focus | Overshoot/settling | Phase lag/amplitude |
| Learning starts | 100,000 (sparse) | 256 (dense) |
| Memory capacity | 1,000,000 | 500,000 |

## Checkpointing

Same as step response — two directories:

- **`best_checkpoints/`**: Rolling best during training
- **`best_eval/`**: Best on fixed sine evaluation

## Resume training

Automatic resume from latest `best_eval/metrics.json`:

```python
resume_metrics = find_latest_metrics(Path("runs"))
# Looks for: runs/dsac_b747_sine_tracking_*/best_eval/metrics.json
```

## Tips

!!! tip "Frequency matters"
    The 0.05 Hz frequency (20s period) is slow enough for the B747 dynamics. Higher frequencies require smaller dt and may exceed actuator bandwidth.

!!! tip "Reference preview"
    The script blocks to show the reference signal. Close the matplotlib window to proceed with training.

!!! tip "Evaluation during training"
    After each stage, the script evaluates on a fixed sine and saves if improved. Watch `best_eval/metrics.json` for progress.

!!! warning "CAPS regularization"
    For tracking, CAPS is disabled (`caps_lambda_smoothness=0.0`) because the environment already penalizes control rates. Enable it if you see jerky behavior.

## See also

- [DSAC Algorithm](../../../agent/dsac.md) — architecture and hyperparameters
- [DSAC Training (Step Response)](train-dsac-b747-step-response.md) — alternative task
- [DSAC Evaluation](eval-dsac-b747.md) — visualize trained agent
- [Signal Types](../../../signals/signals.md) — available reference signals

