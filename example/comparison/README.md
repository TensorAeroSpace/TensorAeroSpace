# 🔬 Control Method Comparison: ML vs PID

This folder contains examples and experiments comparing machine learning methods (SAC, PPO, MPC, DSAC) with a classical PID controller.

## 📋 Contents

### Baseline Notebooks

| Method | Plant | File | Description |
|--------|-------|------|-------------|
| **PID** | F-16 | `pid_f16_baseline.ipynb` | PID controller for F-16 longitudinal control |
| **SAC** | F-16 | `sac_f16_baseline.ipynb` | Soft Actor-Critic for F-16 |
| **PPO** | B747 | `ppo_b747_baseline.ipynb` | Proximal Policy Optimization for Boeing 747 |
| **MPC** | B747 | `mpc_b747_baseline.ipynb` | Model Predictive Control for B747 |

### Comparison Experiments

| Experiment | File | Description |
|------------|------|-------------|
| **All vs PID (B747)** | `comparison_all_vs_pid_b747.ipynb` | DSAC vs PPO vs MPC vs PID on B747 |
| **DSAC vs PID (B747)** | `comparison_dsac_vs_pid_b747.ipynb` | DSAC vs PID on B747 |
| **PPO vs PID (B747)** | `comparison_ppo_vs_pid_b747.ipynb` | PPO vs PID on B747 |
| **MPC vs PID (B747)** | `comparison_mpc_vs_pid_b747.ipynb` | MPC vs PID on B747 |
| **SAC vs PID (F-16)** | `comparison_sac_vs_pid_f16.ipynb` | SAC vs PID on F-16 |

## 📊 Comparison Metrics

The following transient response quality metrics are measured for each experiment:

- **Settling Time** — time to reach steady state
- **Overshoot** — maximum percentage above the setpoint
- **Static Error** — difference between the setpoint and the steady-state value

## 🎯 Experiment Objective

Demonstrate that machine learning methods achieve **30%+ faster transient response** compared to a classical PID controller with comparable or better control quality.

## 🚀 Getting Started

```bash
# Launch Jupyter
cd example/comparison
jupyter lab
```

## 📚 Key Imports

```python
from tensoraerospace.benchmark.function import overshoot, settling_time, static_error
from tensoraerospace.agent.pid import PID
from tensoraerospace.agent.sac import SAC
from tensoraerospace.agent.ppo.model import PPO
```

## 📈 Results

- Summary comparisons on B747 are available in `comparison_all_vs_pid_b747.ipynb`.
- For individual scenarios, use the corresponding `comparison_*_vs_pid_*.ipynb` notebooks.
