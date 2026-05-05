# 🚀 TensorAeroSpace

<div align="center">

[![en](https://img.shields.io/badge/lang-en-red.svg)](./readme.md)
[![ru](https://img.shields.io/badge/lang-ru-green.svg)](./README.ru-ru.md)
[![Documentation Status](https://readthedocs.org/projects/tensoraerospace/badge/?version=latest)](https://tensoraerospace.readthedocs.io/en/latest/?badge=latest)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TensorAeroSpace/TensorAeroSpace)
[![Python](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/tensoraerospace/tensoraerospace.svg)](https://github.com/tensoraerospace/tensoraerospace/stargazers)

![TensorAeroSpace logo](./img/logo-no-background.png)

**Open-source aerospace simulation toolkit + adaptive control catalogue**

*Pure-NumPy 6-DoF dynamics · Gymnasium-native envs · Classical / ADP / Deep RL agents · 894 tests*

[📖 Documentation](https://tensoraerospace.readthedocs.io/) • [🚀 Quickstart](#-quickstart) • [💡 Examples](./example/) • [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## 🌟 Overview

**TensorAeroSpace** ships **12+ aircraft and spacecraft models** (7 of them as full nonlinear 6-DoF airframes with peer-reviewed source data), **20 control algorithms** spanning classical PID/MPC through the full incremental-ADP family to modern deep RL, and **101 runnable example notebooks** covering trim, cruise, coordinated turns, in-flight damage and fault recovery — all glued together by the standard Gymnasium API.

Why it stands out:

- 🎯 **Real airframes, not toys.** B-747 transcribed from NASA CR-2144, X-15 from NASA TM X-1669, Skywalker X8 from CEAS Aeronautical Journal 2025, B-737 from JSBSim + Roskam, RQ-7 Shadow from Beard & McLain. Trim-points reach machine precision.
- ⚡ **Pure NumPy core.** No proprietary simulators, no MATLAB licence, no compiled binaries. Order-of-magnitude faster than JSBSim for control-synthesis sweeps.
- 🧠 **Unique adaptive-control catalogue.** Standard RL stack (PPO, SAC, DDPG, DQN, A2C, A3C, GAIL) **plus** the full incremental-ADP family (IHDP, IM-GDHP, ET-DHP, iADP, AA-INDI, AIDI) — rarely co-located in a single OSS package.
- 💥 **Damage subsystem built-in.** Per-surface effectiveness loss, hard-overs, jam events, asymmetric-thrust engine-out, flap-jam configuration override — all composable into `DamageProfile` instances.
- 🧪 **894 unit tests.** Trim convergence, surface-deflection sign conventions, propellant burnout times all locked down by regression coverage.

## 🧭 Application areas

1. **Automatic flight control** — stabilisation, trajectory tracking, attitude control for aircraft, UAVs, experimental vehicles.
2. **Rocket / spacecraft control** — launch vehicles, satellites in different orbital classes, ascent-trajectory optimisation.
3. **Hybrid control synthesis** — designing and tuning loops that combine classical and intelligent methods.
4. **Algorithm benchmarking** — automated hyperparameter search, head-to-head comparisons, metric visualisation.
5. **Simulator integration** — game engines (Unity ML-Agents), CAD/CAE (Simulink, SimInTech), bidirectional model exchange.
6. **Reliability and FTC research** — failure-mode studies, controller-reconfiguration assessment, dataset preparation.

Each area has working examples and documentation (see [📚 Examples & guides](#-examples--guides)).

## 🚀 Quickstart

> 💡 **Interactive walkthrough**: open [`quickstart.ipynb`](./example/quickstart.ipynb) to run a SAC benchmark on the B-747 end-to-end in Jupyter / VS Code.

### ✅ System requirements

| Component | Minimum | Recommended |
| --- | --- | --- |
| **OS** | Linux x86_64, Windows 10, macOS 13 | Ubuntu 22.04 LTS / Windows 11 |
| **CPU** | 4 cores, AVX | 8+ cores, AVX2/FMA |
| **RAM** | 8 GB | 16–32 GB for RL/Simulink |
| **GPU** | Optional | NVIDIA RTX with ≥8 GB VRAM for SAC/DSAC/PPO, CUDA 12.2 |
| **Python** | 3.10–3.13 | 3.11/3.12 |
| **Optional** | Git, Poetry or pip, Docker | MATLAB/Simulink R2022b+ (Simulink examples), Unity 2021.3.5f1/2023.2.20f1 |

### 📦 Installation

#### Poetry (recommended)

```bash
git clone https://github.com/tensoraerospace/tensoraerospace.git
cd tensoraerospace
poetry install
poetry shell        # activate the venv
poetry run pytest   # quick smoke test
```

#### pip

```bash
pip install tensoraerospace
```

#### 🐳 Docker

Image **launches JupyterLab by default** (see `Dockerfile`).

```bash
docker pull ghcr.io/tensoraerospace/tensoraerospace:latest
docker run --rm -it -p 8888:8888 \
  -v "$(pwd)/projects:/workspace/projects" \
  ghcr.io/tensoraerospace/tensoraerospace:latest

# GPU (NVIDIA Container Toolkit)
docker run --rm -it --gpus all -p 8888:8888 \
  -v "$(pwd)/projects:/workspace/projects" \
  ghcr.io/tensoraerospace/tensoraerospace:latest
```

### 🏃‍♂️ Quick example — PID + linear F-16

```python
import gymnasium as gym
import numpy as np
from tensoraerospace.agent.pid import PID
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standard import unit_step

dt = 0.01
tp = generate_time_period(tn=10, dt=dt)
N = len(tp)
reference = unit_step(degree=5, tp=tp, time_step=100, output_rad=True).reshape(1, -1)

env = gym.make('LinearLongitudinalF16-v0',
               number_time_steps=N, initial_state=[[0], [0]],
               reference_signal=reference, use_reward=False)
pid = PID(env, kp=-14.290, ki=-8.240, kd=-1.299, dt=dt)

obs, _ = env.reset()
for t in range(N - 1):
    u = pid.select_action(reference[0, t], float(obs[0]))
    obs, *_ = env.step(np.array([[float(u)]], dtype=np.float32))
```

### 💥 In-flight damage modelling

Compose damage events declaratively, run any controller — agent sees a different plant the moment the event fires.

```python
import numpy as np
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import WING_STRIKE_LEFT_TIP
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

env = NonlinearAngularF16(
    initial_state=np.zeros(14),
    number_time_steps=2000,
    damage_profile=WING_STRIKE_LEFT_TIP,  # left-tip loss at t=10s
    split_stab=True,
)
obs, _ = env.reset()
for _ in range(2000):
    obs, r, term, trunc, info = env.step(np.zeros(4))
    if info.get("damage_events_triggered"):
        print(info["damage_events_triggered"])  # → ['left_tip_full_loss']
```

What's modelled: **section loss** (mass / S / b / MAC / c.g. / inertia tensor / aero coefficients all recomputed via Huygens-Steiner), **surface failure** (jam / efficiency loss / lost), **engine flameout** (partial / full thrust scaling), **structural events** (payload drop, icing, Δm / Δc.g. / ΔJ).

7 ready presets (`WING_STRIKE_LEFT_TIP`, `ELEVATOR_JAM_NEUTRAL`, `RUDDER_LOST`, `ENGINE_FLAMEOUT`, `BIRDSTRIKE_COMPOUND`, etc.) + `RandomDamageProfileGenerator` for RL curricula. With no `damage_profile`, the env is bit-identical to the undamaged baseline.

📖 [Aircraft damage modelling guide](https://tensoraerospace.readthedocs.io/en/latest/model/aircraft-damage-modeling/)

## 🤖 Supported algorithms

### Classical control

| Algorithm | Description |
|---|---|
| **PID** | Proportional-Integral-Derivative with anti-windup and MATLAB-style autotuning. Strong baseline for state-space tasks; the autotuner extracts the env's `(A,B,C,D)` matrices and optimises gains via differential evolution against the step-response criterion. |
| **MPC** | Model-Predictive Control with three swappable plant-model variants — MLP, NARX, Transformer — over a QP / numerical receding-horizon optimisation. Use when the plant is known or can be learned from data. |

### Deep RL — on-policy

| Algorithm | Description |
|---|---|
| **PPO** | Proximal Policy Optimization. Stable and easy to tune; the standard "just works" starting point for continuous and discrete tasks. |
| **A2C** | Advantage Actor-Critic — synchronous on-policy. Simpler than PPO, useful as a clean reference. |
| **A2C-NARX** | A2C with a NARX critic instead of MLP — better captures temporal structure. |
| **A3C** | Asynchronous Advantage Actor-Critic — multiple workers, shared global network. Great for CPU-parallel distributed training (Unity envs). |

### Deep RL — off-policy

| Algorithm | Description |
|---|---|
| **SAC** | Soft Actor-Critic — off-policy, maximum-entropy. Data-efficient default for continuous control; ships with `from_pretrained` / `publish_to_hub` Hugging Face integration. |
| **DSAC** | Distributional SAC with quantile (IQN-style) twin critics + CAPS regularisation. Better tracking dynamics than vanilla SAC under sensor noise / multi-modal cost. |
| **DDPG** | Deep Deterministic Policy Gradient — foundational; SAC outperforms it in most cases but DDPG remains useful for quiet low-frequency tasks. |
| **DQN** | Deep Q-Learning — value-based for discrete action spaces; used here for Unity envs with discrete control. |

### Imitation learning

| Algorithm | Description |
|---|---|
| **GAIL** | Generative Adversarial Imitation Learning — learn from expert demonstrations without explicit reward. Useful for cloning a pre-built PID/MPC trajectory before fine-tuning with an RL critic. |

### Adaptive Dynamic Programming (model-based critics)

| Algorithm | Description |
|---|---|
| **HDP** | Heuristic Dynamic Programming — actor-critic with a pre-trained offline plant network providing the policy gradient via `∂f/∂u`. |
| **ADHDP** | Action-Dependent HDP — value function depends on `(state, action)`; no explicit model gradient required. |
| **ADP** | Base Adaptive Dynamic Programming — value-iteration-style adaptive control without an explicit plant model. |
| **IHDP** | Incremental HDP — actor-critic with **online incremental linearisation** of the plant. Adaptive without a pre-trained plant network. Strong baseline for online flight control. |
| **NARX** | Nonlinear AutoRegressive network — used both as MPC plant model and as the critic in `A2C-NARX`. |

### Online adaptive critics for fault-tolerant flight

| Algorithm | Description |
|---|---|
| **iADP** | Incremental Approximate Dynamic Programming — online RLS identification of a local incremental model `(F̃, G̃)` plus closed-form quadratic policy. Recovers from plant change in tens of milliseconds; no fault-detector required. |
| **IM-GDHP** | Incremental-Model GDHP — online RLS plant identifier paired with a GDHP critic. Lightweight (no plant network), interpretable, with explicit `(F, G)` matrices. |
| **ET-DHP** | Event-Triggered Dual HDP — Lipschitz event-trigger fires actor/critic updates only when the tracking error crosses a threshold. Bandwidth-aware, embedded-friendly. |
| **AIDI** | Adaptive Incremental Dynamic Inversion — INDI with per-channel VFF-RLS adapting a multiplicative scaling `Θ` of a known on-board control-effectiveness matrix. Fault-tolerant, model-agnostic. |
| **AA-INDI** | Adaptive Augmented INDI — incremental nonlinear dynamic inversion with online RLS adaptation; designed for asymmetric actuator faults and flying-wing layouts with coupled control surfaces. |

## ✈️ Aircraft & spacecraft library

### 🛩️ Fixed-wing (nonlinear 6-DoF, peer-reviewed source data)

| Airframe | Class | Aerodynamic source | Speciality |
|---|---|---|---|
| **F-16 Fighting Falcon** | Fighter | NASA / Stevens-Lewis | Cubic-spline aero, full damage subsystem (linear longitudinal · nonlinear longitudinal · 6-DoF angular) |
| **Boeing 747-100** | Heavy transport | NASA CR-2144 (Heffley & Jewell) | Per-engine asymmetric thrust + flap jam (3 configurations: NOMINAL, POWER_APPROACH, LANDING) |
| **Boeing 737-100/800** | Mid-size transport | JSBSim + Roskam Vol VI | Coordinated-turn benchmarks, JT8D / CFM56-7B engines |
| **X-15** | Hypersonic research | NASA TM X-1669 + Thompson 2000 | Mach 0.4–6.7 tabulated, XLR99 rocket, variable mass |
| **Skywalker X8** | Small UAV (3.4 kg) | CEAS Aeronautical Journal 2025 | Peer-reviewed flight-test ID, flying-wing |
| **AAI RQ-7 Shadow** | Class-II UAV (170 kg) | Beard & McLain + NASA TM-2014-218686 | V-tail mixed control, 4-channel |
| **F-4C Phantom II** | Military fighter-bomber | Roskam | Linear longitudinal + improved env |

### 🚁 UAVs and drones

- **LAPAN LSU-05** — Indonesian surveillance UAV
- **Ultrastick-25e** — RC airplane model
- **Generic UAV** — customisable state-space dynamics
- **Quadrotor** — full nonlinear 6-DoF + per-rotor damage subsystem + X-config allocator

### 🚀 Rockets and satellites

- **ELV (Expendable Launch Vehicle)** — booster dynamics
- **Generic missile** — customisable simulation
- **GeoSat** — geostationary orbital mechanics
- **ComSat** — communication-satellite dynamics and control

## 🎮 Simulation environments

### 🎯 Unity ML-Agents integration

<div align="center">

![Unity demo](./docs/ru/example/environment/img/img_demo_unity.gif)

</div>

- 🎮 **3D visualisation** in real time
- 🔄 **Realistic training** — agents learn in physics-rich scenes
- 📊 **Sensor suite** — camera, LiDAR, physical sensors
- 🌍 **Custom scenarios** — author your own aerospace tasks

> 📁 Example environment: [UnityAirplaneEnvironment](https://github.com/TensorAeroSpace/UnityAirplaneEnvironment)

### 🔧 MATLAB Simulink support

![Simulink model](./docs/ru/example/simulink/img/model.png)

- 📐 **Model import** — convert Simulink models to Python
- ⚡ **High performance** via compiled C++
- 🔄 **Bidirectional** MATLAB ↔ Python workflow
- 📊 **Cross-platform validation**

### 📊 State-space matrices

Mathematical foundation for control-system design:

- 🧮 **Linear models** — state-space representation
- 🎛️ **Control synthesis** — modern control theory
- 📈 **Analysis tools** — stability, controllability, observability
- 🔄 **Linearisation** — from nonlinear models

## 📚 Examples & guides

The [`example/`](./example/) directory ships **101 runnable notebooks**, organised by controller class. The folder was recently restructured for predictable navigation — see [`example/README.md`](./example/README.md) for the full map.

| Category | Folder | Highlights |
|---|---|---|
| 🚀 **Quickstart** | [`quickstart.ipynb`](./example/quickstart.ipynb) | Minimal end-to-end pipeline |
| 🎮 **Environments** | [`environments/`](./example/environments/) | All bundled aircraft envs (no agent) |
| 🎛️ **Classical** | [`pid_controllers/`](./example/pid_controllers/), [`mpc_controllers/`](./example/mpc_controllers/) | PID + MPC (MLP / NARX / Transformer) |
| 🧠 **Classical ADP** | [`dynamic_programming/`](./example/dynamic_programming/) | HDP, DHP, GDHP, AD-HDP, AD-GDHP, AD-DHP |
| 🔄 **Online ADP** | [`reinforcement_learning/incremental_adp/`](./example/reinforcement_learning/incremental_adp/) | IHDP, IM-GDHP, ET-DHP, iADP, AA-INDI, AIDI |
| 🤖 **Deep RL** | [`reinforcement_learning/deep_rl/`](./example/reinforcement_learning/deep_rl/) | A2C, A3C, PPO, DQN, SAC, DSAC, DDPG, GAIL |
| 📊 **Comparison** | [`comparison/`](./example/comparison/) | PID vs RL head-to-head benchmarks |
| 💥 **Failure demos** | [`failure_demos/`](./example/failure_demos/) | F-16 dogfight with damage, IHDP failure recovery |
| 📖 **Cookbook** | [`cookbook/`](./example/cookbook/) | Step-by-step recipes from "hello world" to FTC |
| 🔧 **Optimization** | [`optimization/`](./example/optimization/) | Optuna hyperparameter search |

### 🆕 Featured new examples

| Example | Aircraft | Result |
|---|---|---|
| [**ET-DHP heading hold under engine flameout**](./example/reinforcement_learning/incremental_adp/example_etdhp_b747_engine_failure.ipynb) | B-747 | ψ-error **0.28°** vs open-loop −85.5° |
| [**MIMO IHDP 90° coordinated turn**](./example/reinforcement_learning/incremental_adp/example_ihdp_nonlinear_b737_turn.ipynb) | B-737 | Final ψ-error **0.98°**, max sideslip 0.11° |
| [**IHDP θ-step tracking on nonlinear B-747**](./example/reinforcement_learning/incremental_adp/example_ihdp_nonlinear_b747.ipynb) | B-747 | Late-half MAE **0.043°** |
| [**X-15 hypersonic boost-burnout demo**](./example/aircraft/example_b747_nonlinear.py) | X-15 | Burnout 79.8 s vs Thompson 2000: 80 s |

### Quick run commands

```bash
# Run pretrained SAC on B-747
poetry run python example/reinforcement_learning/deep_rl/sac-b747-render.py --render --dt 0.1

# Run pretrained DDPG
poetry run python example/reinforcement_learning/deep_rl/ddpg-b747-render.py --repo TensorAeroSpace/ddpg-b747

# Train DSAC step-response
poetry run python example/reinforcement_learning/deep_rl/train_dsac_b747_step_response.py
```

📖 Detailed walkthroughs:
- [Example SAC F-16](https://tensoraerospace.readthedocs.io/en/latest/example/agent/sac/example-sac-f16.html)
- [B-737 coordinated turn (IHDP)](https://tensoraerospace.readthedocs.io/en/latest/example/agent/ihdp/example_ihdp_nonlinear_b737_turn.html)
- [B-747 engine-out heading hold (ET-DHP)](https://tensoraerospace.readthedocs.io/en/latest/example/agent/et_dhp/example_etdhp_b747_engine_failure.html)
- [Optuna optimisation](https://tensoraerospace.readthedocs.io/en/latest/example/optimization/example_optimization.html)
- [Unity guide](https://tensoraerospace.readthedocs.io/en/latest/guide/unity_env.html)

## 🛠️ Development

```bash
git clone https://github.com/tensoraerospace/tensoraerospace.git
cd tensoraerospace
poetry install --with dev
poetry run pytest                      # all 894 tests
poetry run pytest tests/aerospacemodel # specific category
poetry run mkdocs serve -a 0.0.0.0:8000 # docs preview
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📖 Documentation

- 📚 **Full docs**: [tensoraerospace.readthedocs.io](https://tensoraerospace.readthedocs.io/)
- 🚀 **API reference**: detailed module-by-module
- 📝 **16-recipe cookbook**: from hello-world to FTC under damage
- 💡 **11-lesson tutorial**: state-space → controllability → RL fundamentals → XFLR5 / Simulink hands-on
- ❓ **Q&A**: [DeepWiki AI assistant](https://deepwiki.com/TensorAeroSpace/TensorAeroSpace)

## 🤝 Community & support

- 💬 [GitHub Discussions](https://github.com/tensoraerospace/tensoraerospace/discussions)
- 🐛 [Issue tracker](https://github.com/tensoraerospace/tensoraerospace/issues)
- 📧 [Email support](mailto:support@tensoraerospace.org)

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🙏 Acknowledgements

- The Gymnasium / OpenAI Gym team for the canonical RL environment API
- The Unity ML-Agents team for 3D simulation infrastructure
- The aerospace research community for decades of open published derivative data — NASA CR-2144 (Heffley & Jewell), NASA TM X-1669 (Walker & Wolowicz), CEAS Aeronautical Journal 2025 (Løw-Hansen et al.), JSBSim, Roskam, Beard & McLain, Mattingly
- Every contributor who has made this project possible

---

<div align="center">

**⭐ Star us on GitHub if TensorAeroSpace helps your work! ⭐**

Made with ❤️ by the TensorAeroSpace team

</div>
