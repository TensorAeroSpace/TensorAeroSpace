# 🚀 TensorAeroSpace

<div align="center">

[![en](https://img.shields.io/badge/lang-en-red.svg)](./readme.md)
[![ru](https://img.shields.io/badge/lang-ru-green.svg)](./README.ru-ru.md)
[![Documentation Status](https://readthedocs.org/projects/tensoraerospace/badge/?version=latest)](https://tensoraerospace.readthedocs.io/en/latest/?badge=latest)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TensorAeroSpace-FFD21E)](https://huggingface.co/TensorAeroSpace)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TensorAeroSpace/TensorAeroSpace)
[![Python](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/tensoraerospace/tensoraerospace.svg)](https://github.com/tensoraerospace/tensoraerospace/stargazers)
[![Coverage Status](https://coveralls.io/repos/github/TensorAeroSpace/TensorAeroSpace/badge.svg?branch=main)](https://coveralls.io/github/TensorAeroSpace/TensorAeroSpace?branch=main)

![TensorAeroSpace Logo](./img/logo-no-background.png)

**Advanced Aerospace Control Systems & Reinforcement Learning Framework**

*A comprehensive Python library for aerospace simulation, control algorithms, and reinforcement learning implementations*

[📖 Documentation](https://tensoraerospace.readthedocs.io/) • [🚀 Quick Start](#-quick-start) • [💡 Examples](./example/) • [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## 🌟 Overview

**TensorAeroSpace** is a cutting-edge Python framework that combines aerospace engineering with modern machine learning. It provides:

- 🎯 **Control Systems**: Advanced control algorithms including PID, MPC, and modern RL approaches
- ✈️ **Aerospace Models**: High-fidelity aircraft and spacecraft simulation models — including a **fully nonlinear F-16** (longitudinal + 6-DoF angular)
- 💥 **In-flight Damage Simulation**: Schedule wing-tip loss, jammed control surfaces, engine flameout — the env recomputes mass, inertia, aerodynamics on-the-fly
- 🎮 **OpenAI Gym Integration**: Ready-to-use environments for reinforcement learning
- 🧠 **RL Algorithms**: State-of-the-art reinforcement learning implementations, including online-adaptive critics (iADP, IM-GDHP, ET-DHP, AIDI, AA-INDI) for fault-tolerant control
- 🔧 **Extensible Architecture**: Easy to extend and customize for your specific needs

## 🧭 Applied Use Cases

1. **Autonomous Flight Vehicle Control** — stabilization, trajectory tracking, attitude control for aircraft, UAVs, and experimental vehicles.
2. **Rocket & Spacecraft Systems Control** — modeling and control of launch vehicles, satellites in various orbital classes, trajectory optimization.
3. **Hybrid Control Systems** — design and tuning of controllers combining classical and intelligent control methods.
4. **Algorithm Optimization & Benchmarking** — automated hyperparameter tuning, comparative analysis of control algorithms, quality metrics visualization.
5. **Simulation Platform Integration** — interfacing with game engines, CAD/CAE systems, model import/export between environments.
6. **Reliability Analysis & Diagnostics** — failure mode investigation, control system robustness assessment, training data preparation.

## 🚀 Quick Start

### ✅ Minimum Technical Requirements

| Component | Minimum | Recommended |
| --- | --- | --- |
| **OS** | Linux x86_64, Windows 10, macOS 13 | Ubuntu 22.04 LTS / Windows 11 |
| **CPU** | 4 cores, AVX | 8+ cores, AVX2/FMA |
| **RAM** | 8 GB | 16–32 GB for RL/Simulink |
| **GPU** | Optional | NVIDIA RTX with ≥8 GB VRAM for SAC/DSAC/PPO, CUDA 12.2 support |
| **Python** | 3.10–3.13 | 3.11/3.12 |
| **Additional** | Git, Poetry or pip, Docker (optional) | MATLAB/Simulink R2022b+ (for simulink-example), Unity 2021.3.5f1/2023.2.20f1 |

### 📦 Installation

#### Using Poetry (Recommended)
```bash
git clone https://github.com/tensoraerospace/tensoraerospace.git
cd tensoraerospace
poetry install
```

#### Using pip
```bash
pip install tensoraerospace
```

#### 🐳 Docker
The image starts **JupyterLab by default** (see `Dockerfile` CMD). The runtime image is built from the repository sources, installs TensorAeroSpace as a wheel, and includes examples in `/workspace/examples`.

**Ubuntu / Linux (bash):**

```bash
docker pull ghcr.io/tensoraerospace/tensoraerospace:latest
docker run --rm -it -p 8888:8888 \
  -v "$(pwd)/projects:/workspace/projects" \
  ghcr.io/tensoraerospace/tensoraerospace:latest

# Or build the same image locally from source
docker build -t tensoraerospace:local . --platform=linux/amd64
docker run --rm -it -p 8888:8888 \
  -v "$(pwd)/projects:/workspace/projects" \
  tensoraerospace:local

# Optional: enable NVIDIA GPU inside the container
docker run --rm -it --gpus all -p 8888:8888 \
  -v "$(pwd)/projects:/workspace/projects" \
  ghcr.io/tensoraerospace/tensoraerospace:latest
```

**Windows (PowerShell):**

```powershell
docker pull ghcr.io/tensoraerospace/tensoraerospace:latest
docker run --rm -it -p 8888:8888 `
  -v "${PWD}\projects:/workspace/projects" `
  ghcr.io/tensoraerospace/tensoraerospace:latest

# Or build the same image locally from source
docker build -t tensoraerospace:local . --platform=linux/amd64
docker run --rm -it -p 8888:8888 `
  -v "${PWD}\projects:/workspace/projects" `
  tensoraerospace:local

# Optional: enable NVIDIA GPU inside the container
docker run --rm -it --gpus all -p 8888:8888 `
  -v "${PWD}\projects:/workspace/projects" `
  ghcr.io/tensoraerospace/tensoraerospace:latest
```
> Open the printed URL (default `http://127.0.0.1:8888`) and navigate to `examples/quickstart.ipynb` to run the SAC walkthrough inside Docker.

### 🏃‍♂️ Quick Examples

> 💡 **Interactive walkthrough**: open the [Quickstart notebook](./example/quickstart.ipynb) to run the SAC B747 benchmark flow end-to-end inside Jupyter/VS Code.

#### 🚀 Pretrained SAC Agent (Boeing 747)

Run a pretrained Soft Actor-Critic agent on Boeing 747 pitch control:

<div align="center">

![SAC B747](./docs/en/example/agent/sac/img/sac-b747-impoved.jpg)

</div>

**Command line:**
```bash
python example/reinforcement_learning/sac-b747-render.py \
    --render \
    --dt 0.1 \
    --tn 200 \
    --repo TensorAeroSpace/sac-b747 \
    --device cuda  # Optional: 'cuda', 'mps', or 'cpu' (auto-detects if not specified)
```

> 📖 **See full tutorial**: [SAC B747 Documentation](https://tensoraerospace.readthedocs.io/en/latest/example/agent/sac/example-sac-b747/)

---

#### 🎛️ PID Controller (F-16)

```python
import gymnasium as gym
import numpy as np

from tensoraerospace.agent.pid import PID
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standard import unit_step

# Simulation setup
dt = 0.01
tp = generate_time_period(tn=10, dt=dt)  # 10 seconds
N = len(tp)

# Reference signal for alpha tracking (5 deg step in radians)
reference = unit_step(
    degree=5, tp=tp, time_step=100, output_rad=True
).reshape(1, -1)

# Create F-16 longitudinal environment
env = gym.make(
    'LinearLongitudinalF16-v0',
    number_time_steps=N,
    initial_state=[[0], [0]],
    reference_signal=reference,
    use_reward=False,
)

# PID controller with tuned coefficients
pid = PID(
    env,
    kp=-14.290139135229715,
    ki=-8.240470780203491,
    kd=-1.2991634935096958,
    dt=dt
)

obs, info = env.reset()
for t in range(N - 1):
    setpoint = reference[0, t]
    alpha = float(obs[0])
    u = pid.select_action(setpoint, alpha)
    action = np.array([[float(u)]], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

#### 💥 In-flight Damage Modeling (Nonlinear F-16)

Schedule failures during a simulation — wingtip loss, jammed control surfaces, engine flameout, structural changes — and the env recomputes mass, inertia, aerodynamic coefficients, and control-surface effectiveness in real time. The control agent then faces a different plant from the moment the damage event fires.

```python
import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    WING_STRIKE_LEFT_TIP,  # ready-made: full loss of left wingtip at t=10s
)
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

env = NonlinearAngularF16(
    initial_state=np.zeros(14),
    number_time_steps=2000,
    damage_profile=WING_STRIKE_LEFT_TIP,
    split_stab=True,
)
obs, _ = env.reset()
for _ in range(2000):
    obs, r, term, trunc, info = env.step(np.zeros(4))
    if info.get("damage_events_triggered"):
        print(info["damage_events_triggered"])  # → ['left_tip_full_loss']
```

What's modelled:

- **Section loss** (wing/stabilator/vtail): mass *m*, wing area *S*, span *b*, MAC, CG, inertia tensor **J**, aerodynamic coefficients all recomputed from per-section contributions via Huygens-Steiner.
- **Control-surface failure** (`jam` / `efficiency_loss` / `lost`): commanded vector **u**<sub>cmd</sub> → **u**<sub>eff</sub> before the integrator.
- **Engine failure** (partial / full): effective thrust scaled or zeroed.
- **Structural changes** (dropped stores, ice accretion): Δ on mass / CG / inertia.

7 ready-made presets (`WING_STRIKE_LEFT_TIP`, `ELEVATOR_JAM_NEUTRAL`, `RUDDER_LOST`, `ENGINE_FLAMEOUT`, `BIRDSTRIKE_COMPOUND`, …) plus a `RandomDamageProfileGenerator` for RL curricula. Without `damage_profile` the env is byte-for-byte identical to the un-damaged baseline.

> 📖 **Full reference**: [Aircraft Damage Modeling docs](https://tensoraerospace.readthedocs.io/en/latest/model/aircraft-damage-modeling/) — overview, code/examples, mathematics.

## 🤖 Supported Algorithms

### Classical control

| Algorithm | Description |
|-----------|-------------|
| **PID** | Proportional-Integral-Derivative regulator with anti-windup and MATLAB-style auto-tuning. Strong baseline for state-space tasks; the auto-tuner extracts the env's `(A, B, C, D)` matrices and runs differential evolution on a step-response cost. |
| **MPC** | Model Predictive Control with three pluggable plant-model variants — MLP, NARX, and Transformer — driving a receding-horizon QP/optimisation. Use it when an explicit model is known or learnable from data. |

### Deep RL — on-policy

| Algorithm | Description |
|-----------|-------------|
| **PPO** | Proximal Policy Optimization — clipped-surrogate on-policy actor-critic. Stable and easy to tune; the standard «just works» starting point for continuous and discrete control. |
| **A2C** | Advantage Actor-Critic — synchronous on-policy actor-critic baseline; simpler than PPO, useful when you need a clean reference implementation. |
| **A2C-NARX** | A2C with a NARX-network critic — captures temporal structure better than an MLP critic; good for tasks where state alone doesn't expose phase / lag. |
| **A3C** | Asynchronous Advantage Actor-Critic — many workers update a shared global net in parallel. Best for CPU-parallel distributed training (e.g. Unity environments). |

### Deep RL — off-policy

| Algorithm | Description |
|-----------|-------------|
| **SAC** | Soft Actor-Critic — off-policy stochastic actor-critic with maximum entropy. Sample-efficient continuous-control default; ships with `from_pretrained` / `publish_to_hub` HuggingFace integration. |
| **DSAC** | Distributional Soft Actor-Critic — SAC with quantile (IQN-style) twin critics + CAPS regularisation. Better tracking dynamics than vanilla SAC, especially under sensor noise or when the cost surface is multi-modal. |
| **DDPG** | Deep Deterministic Policy Gradient — off-policy deterministic actor-critic. Foundational; SAC supersedes it in most cases, but DDPG remains useful for low-noise low-bandwidth tasks. |
| **DQN** | Deep Q-Learning — off-policy value-based learning for discrete action spaces; used here for Unity environments with discrete action sets. |

### Imitation learning

| Algorithm | Description |
|-----------|-------------|
| **GAIL** | Generative Adversarial Imitation Learning — learn from expert demonstrations without an explicit reward signal. Useful for cloning a known PID/MPC trajectory before fine-tuning with an RL critic. |

### Adaptive Dynamic Programming (model-based critics)

| Algorithm | Description |
|-----------|-------------|
| **HDP** | Heuristic Dynamic Programming — actor-critic with an offline-trained plant-model network providing the policy gradient via `∂f/∂u`. |
| **ADHDP** | Action-Dependent HDP — value function depends on `(state, action)`; bypasses the explicit model gradient at the cost of larger critic input. |
| **ADP** | Generic Adaptive Dynamic Programming — value-iteration-style adaptive controller without an explicit plant model. |
| **IHDP** | Incremental HDP — actor-critic with **online incremental linearisation** of the plant; adaptive without needing a pre-trained plant network. Strong baseline for online flight control. |
| **NARX** | Nonlinear Autoregressive Network — used both as a plant model for MPC and as a critic for `A2C-NARX`. |

### Online-adaptive critics for fault-tolerant flight (new)

| Algorithm | Description |
|-----------|-------------|
| **iADP** | Incremental Approximate Dynamic Programming — online RLS identification of the local incremental model `(F̃, G̃)` plus a closed-form quadratic policy. Recovers from in-flight plant changes in tens of milliseconds; no fault detection needed. |
| **IM-GDHP** | Incremental-Model GDHP — online RLS plant identifier coupled to a GDHP critic. Lightweight (no neural plant network), interpretable, with explicit `(F, G)`. |
| **ET-DHP** | Event-Triggered Dual HDP — Lipschitz event trigger fires actor/critic updates only when the tracking error breaches a threshold. Bandwidth-aware execution useful for embedded deployments. |
| **AIDI** | Adaptive Incremental Dynamic Inversion — INDI with a per-row VFF-RLS that adapts the multiplicative scaling `Θ` of the onboard control-effectiveness matrix. Fault-tolerant and model-agnostic. |
| **AA-INDI** | Adaptive Augmented INDI — incremental nonlinear dynamic inversion with online RLS adaptation; designed for asymmetric actuator failures and flying-wing-style coupled control surfaces. |

## ✈️ Aircraft & Spacecraft Models

<details>
<summary><b>🛩️ Fixed-Wing Aircraft</b></summary>

- **General Dynamics F-16 Fighting Falcon** — high-fidelity fighter jet, available in **three forms**:
  - linear longitudinal (state-space, fast),
  - **nonlinear longitudinal** (NumPy ODE, full lookup tables),
  - **nonlinear 6-DoF angular** (full body-frame angular dynamics, with optional **split-stab** asymmetric control).
- **Boeing 747** — commercial airliner dynamics (linear + normalised `ImprovedB747Env`)
- **McDonnell Douglas F-4C Phantom II** — military aircraft model
- **North American X-15** — hypersonic research aircraft

</details>

<details>
<summary><b>🚁 UAVs & Drones</b></summary>

- **LAPAN Surveillance Aircraft (LSU)-05** - Indonesian surveillance UAV
- **Ultrastick-25e** - RC aircraft model
- **Generic UAV State Space** - Configurable UAV dynamics

</details>

<details>
<summary><b>🚀 Rockets & Satellites</b></summary>

- **ELV (Expendable Launch Vehicle)** - Launch vehicle dynamics
- **Generic Rocket Model** - Customizable rocket simulation
- **Geostationary Satellite** - Orbital mechanics simulation
- **Communication Satellite** - ComSat dynamics and control

</details>

## 🎮 Simulation Environments

### 🎯 Unity ML-Agents Integration

<div align="center">

![Unity Demo](docs/en/model/img/img_demo_unity.gif)

</div>

TensorAeroSpace seamlessly integrates with Unity ML-Agents for immersive 3D simulations:

- 🎮 **3D Visualization**: Real-time 3D aircraft simulation
- 🔄 **Real-time Training**: Train agents in realistic environments
- 📊 **Rich Sensors**: Camera, LiDAR, and physics-based sensors
- 🌍 **Custom Environments**: Build your own aerospace scenarios

> 📁 **Example Environment**: [UnityAirplaneEnvironment](https://github.com/TensorAeroSpace/UnityAirplaneEnvironment)

### 🔧 MATLAB Simulink Support

![Simulink Model](docs/en/example/simulink/img/model.png)

- 📐 **Model Import**: Convert Simulink models to Python
- ⚡ **High Performance**: Compiled C++ integration
- 🔄 **Bidirectional**: MATLAB ↔ Python workflow
- 📊 **Validation**: Cross-platform model validation

### 📊 State Space Matrices

Mathematical foundation for control system design:

- 🧮 **Linear Models**: State-space representation
- 🎛️ **Control Design**: Modern control theory implementation
- 📈 **Analysis Tools**: Stability, controllability, observability
- 🔄 **Linearization**: Nonlinear model linearization

## 📚 Examples & Tutorials

Explore our comprehensive example collection in the [`./example`](./example/) directory:

| Category | Description | Notebooks |
|----------|-------------|-----------|
| 🚀 **Quick Start** | Basic usage and concepts | [`quickstart.ipynb`](./example/quickstart.ipynb) |
| 🤖 **Reinforcement Learning** | RL algorithm implementations | [`reinforcement_learning/`](./example/reinforcement_learning/) |
| 🎛️ **Control Systems** | PID, MPC controllers | [`pid_controllers/`](./example/pid_controllers/), [`mpc_controllers/`](./example/mpc_controllers/) |
| ✈️ **Aircraft Models** | Environment examples | [`environments/`](./example/environments/) |
| 🔧 **Optimization** | Hyperparameter tuning | [`optimization/`](./example/optimization/) |

## 🛠️ Development & Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### 🏗️ Development Setup

```bash
git clone https://github.com/tensoraerospace/tensoraerospace.git
cd tensoraerospace
poetry install --with dev
poetry run pytest  # Run tests
```

### 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run specific test category
poetry run pytest tests/envs/
poetry run pytest tests/agents/
```

## 📖 Documentation

- 📚 **Full Documentation**: [tensoraerospace.readthedocs.io](https://tensoraerospace.readthedocs.io/)
- 🚀 **API Reference**: Detailed API documentation
- 📝 **Tutorials**: Step-by-step guides
- 💡 **Examples**: Practical use cases

## 🤝 Community & Support

- 💬 **Discussions**: [GitHub Discussions](https://github.com/tensoraerospace/tensoraerospace/discussions)
- 🐛 **Issues**: [Bug Reports](https://github.com/tensoraerospace/tensoraerospace/issues)
- 📧 **Contact**: [Email Support](mailto:support@tensoraerospace.org)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Gym team for the excellent RL framework
- Unity ML-Agents team for 3D simulation capabilities
- The aerospace engineering community for domain expertise
- All contributors who make this project possible

---

<div align="center">

**⭐ Star us on GitHub if you find TensorAeroSpace useful! ⭐**

Made with ❤️ by the TensorAeroSpace team

</div>
