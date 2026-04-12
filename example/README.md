# 📚 TensorAeroSpace Examples

<div align="center">

![TensorAeroSpace Logo](../img/logo-no-background.png)

**Comprehensive collection of examples and tutorials**

*Explore TensorAeroSpace capabilities through hands-on examples*

[🏠 Home](../) • [📖 Documentation](https://tensoraerospace.readthedocs.io/) • [🚀 Quick Start](../README.md)

</div>

---

## 🌟 Overview

This folder contains an extensive collection of TensorAeroSpace usage examples, organized by category for easy navigation and learning. Each example includes detailed explanations and ready-to-run code.

## 📁 Example Structure

### ✈️ Aerospace Environments
> **Folder:** [`environments/`](./environments/)

Examples of various aerospace environments with high-fidelity models:

| Model | Description | Notebook |
|-------|-------------|----------|
| **🛩️ Boeing 747** | Longitudinal control of a commercial airliner | [`example-env-LinearLongitudinalB747.ipynb`](./environments/example-env-LinearLongitudinalB747.ipynb) |
| **⚡ F-16 Fighting Falcon** | Highly maneuverable fighter jet | [`example-env-LinearLongitudinalF16.ipynb`](./environments/example-env-LinearLongitudinalF16.ipynb) |
| **🚀 F-4C Phantom II** | Military fighter-bomber | [`example-env-f4c.ipynb`](./environments/example-env-f4c.ipynb) |
| **🚀 ELV Rocket** | Expendable launch vehicle | [`example-env-LinearLongitudinalELVRocket.ipynb`](./environments/example-env-LinearLongitudinalELVRocket.ipynb) |
| **🛸 UAV** | Unmanned aerial vehicle | [`example-env-LinearLongitudinalUAV.ipynb`](./environments/example-env-LinearLongitudinalUAV.ipynb) |
| **🎯 Missile Model** | Guided missile model | [`example-env-LinearLongitudinalMissileModel.ipynb`](./environments/example-env-LinearLongitudinalMissileModel.ipynb) |
| **🛰️ ComSat** | Communication satellite | [`example-env-comsat.ipynb`](./environments/example-env-comsat.ipynb) |
| **🌍 GeoSat** | Geostationary satellite | [`example-env-geosat.ipynb`](./environments/example-env-geosat.ipynb) |
| **🎮 Unity** | Unity ML-Agents integration | [`example-env-unity.ipynb`](./environments/example-env-unity.ipynb) |
| **🚁 X-15** | Experimental hypersonic aircraft | [`example-env-x15.ipynb`](./environments/example-env-x15.ipynb) |

### 🤖 Reinforcement Learning
> **Folder:** [`reinforcement_learning/`](./reinforcement_learning/)

State-of-the-art RL algorithms for aerospace tasks:

| Algorithm | Description | Examples |
|-----------|-------------|----------|
| **🎯 A3C** | Asynchronous Advantage Actor-Critic | [`example-a3c.ipynb`](./reinforcement_learning/example-a3c.ipynb) |
| **🎭 SAC** | Soft Actor-Critic | [`example-sac.ipynb`](./reinforcement_learning/example-sac.ipynb), [`example-sac-f16.ipynb`](./reinforcement_learning/example-sac-f16.ipynb) |
| **🎪 A2C** | Advantage Actor-Critic | [`example_a2c.ipynb`](./reinforcement_learning/example_a2c.ipynb) |
| **🧠 DQN** | Deep Q-Network | [`example_dqn_b747_improved.ipynb`](./reinforcement_learning/example_dqn_b747_improved.ipynb) |
| **🚀 PPO** | Proximal Policy Optimization | [`example_ppo.ipynb`](./reinforcement_learning/example_ppo.ipynb) |
| **🎨 GAIL** | Generative Adversarial Imitation Learning | [`create_dataset_for_gail.ipynb`](./reinforcement_learning/create_dataset_for_gail.ipynb) |

### 🎛️ Control Systems
> **Folders:** [`mpc_controllers/`](./mpc_controllers/), [`pid_controllers/`](./pid_controllers/)

#### 🔮 Model Predictive Control (MPC)
- **📊 MPC with MLP dynamics**: [`example-mpc-b747-torch-mpc-mlp.ipynb`](./mpc_controllers/example-mpc-b747-torch-mpc-mlp.ipynb)
- **🤖 MPC with Transformer dynamics**: [`example-mpc-b747-torch-mpc-transformer.ipynb`](./mpc_controllers/example-mpc-b747-torch-mpc-transformer.ipynb)
- **📈 MPC with NARX dynamics**: [`example-mpc-b747-torch-mpc-narx.ipynb`](./mpc_controllers/example-mpc-b747-torch-mpc-narx.ipynb)

#### ⚙️ PID Controllers
- **🎯 PID Tuning**: [`tune_pid.ipynb`](./pid_controllers/tune_pid.ipynb)
- **🔧 MATLAB-style PID Tuning**: [`pid_matlab_tuning.ipynb`](./pid_controllers/pid_matlab_tuning.ipynb)
- **📊 Tuning Methods Comparison**: [`pid_tuning_methods.ipynb`](./pid_controllers/pid_tuning_methods.ipynb)
- **💼 Practical Usage**: [`pid_use.ipynb`](./pid_controllers/pid_use.ipynb)

### 🛠️ Utilities and Tools
> **Folder:** [`utilities/`](./utilities/)

Helper tools for analysis and development:

| Tool | Description | Notebook |
|------|-------------|----------|
| **📡 Signal Generation** | Creating test signals | [`signals.ipynb`](./utilities/signals.ipynb) |
| **🔄 Simulink Conversion** | Converting models to Python | [`example_sim_model_to_python.ipynb`](./utilities/example_sim_model_to_python.ipynb) |
| **🔍 Exploration** | Data analysis and visualization | [`example_explarotaion.ipynb`](./utilities/example_explarotaion.ipynb) |
| **⚡ Hyperparameter Optimization** | Algorithm parameter tuning | [`hyperparam_optimization.ipynb`](./utilities/hyperparam_optimization.ipynb) |

### 📚 General Examples
> **Folder:** [`general_examples/`](./general_examples/)

Core concepts and classic tasks:

| Example | Description | Notebook |
|---------|-------------|----------|
| **🎯 Classic Example** | Library usage fundamentals | [`classic_example.ipynb`](./general_examples/classic_example.ipynb) |
| **🧮 IHDP** | Infinite-Horizon Dynamic Programming | [`example_ihdp.ipynb`](./general_examples/example_ihdp.ipynb), [`example_ihdp_beautiful.ipynb`](./general_examples/example_ihdp_beautiful.ipynb) |
| **📈 NARX** | Nonlinear Autoregressive with Exogenous Inputs | [`example-narx.ipynb`](./general_examples/example-narx.ipynb) |
| **⚠️ Failure Handling** | Working with system failures | [`example-ihdp-failure.ipynb`](./general_examples/example-ihdp-failure.ipynb) |

### 🔧 Optimization
> **Folder:** [`optimization/`](./optimization/)

Optimization algorithms and methods:
- **📊 General Optimization**: [`example_optimization.ipynb`](./optimization/example_optimization.ipynb)

### 🔬 Comparison
> **Folder:** [`comparison/`](./comparison/)

Controller comparison experiments (ML vs PID):
- **📊 All Methods vs PID (B747)**: [`comparison_all_vs_pid_b747.ipynb`](./comparison/comparison_all_vs_pid_b747.ipynb)
- **🎭 DSAC vs PID (B747)**: [`comparison_dsac_vs_pid_b747.ipynb`](./comparison/comparison_dsac_vs_pid_b747.ipynb)
- **🚀 PPO vs PID (B747)**: [`comparison_ppo_vs_pid_b747.ipynb`](./comparison/comparison_ppo_vs_pid_b747.ipynb)
- **🔮 MPC vs PID (B747)**: [`comparison_mpc_vs_pid_b747.ipynb`](./comparison/comparison_mpc_vs_pid_b747.ipynb)

## 🚀 Quick Start

### 1. 📋 Prerequisites

Make sure all required dependencies are installed:

```bash
# Install the main library
pip install tensoraerospace

# Or using Poetry
poetry install
```

### 2. 🔀 Choose an Example Format

- **CLI scripts** — suitable for quickly reproducing results or integrating examples into your own pipeline. Scripts are located in `example/**` and can be run directly via `python` or `poetry run python`.
- **Jupyter Notebooks** — use for interactive learning, step-by-step explanations, and experiments. Each section above links to the corresponding notebook.

### 3. 🏃‍♂️ CLI Commands for Typical Scenarios

Run scripts from the repository root (after installing dependencies):

- **F-16 baseline (general_examples/example.py)** — basic environment check for `LinearLongitudinalF16-v0`.
  ```bash
  poetry run python example/general_examples/example.py
  # or, if using pip/venv
  python example/general_examples/example.py
  ```

- **DDPG Boeing 747 (reinforcement_learning/ddpg-b747-render.py)** — replay a trained agent and optionally visualize the trajectory.
  ```bash
  poetry run python example/reinforcement_learning/ddpg-b747-render.py \
    --repo TensorAeroSpace/ddpg-b747 --dt 0.1 --tn 200 --render
  ```
  Flags `--repo`, `--dt`, `--tn`, `--render/--no-render` configure weights source, discretization, and visualization.

### 4. 📓 Running Notebooks

```bash
# Launch Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

### 5. ⚡️ GPU and "FAST_PRESET" in Notebooks

Some notebooks in `example/` can be computationally heavy.
To speed up execution and make results reproducible on a typical laptop/CI:

- **GPU**: if CUDA (or MPS on macOS) is available, examples automatically select the device and print `Using device: ...`.
- **FAST_PRESET**: many notebooks have a `FAST_PRESET = True` toggle that reduces horizon/episodes/epochs so the example completes in minutes.

## 📖 Recommended Learning Path

### 🌱 For Beginners:
1. 📚 [`quickstart.ipynb`](./quickstart.ipynb) — Library basics
2. 🎯 [`classic_example.ipynb`](./general_examples/classic_example.ipynb) — Classic control tasks
3. ✈️ [`example-env-LinearLongitudinalF16.ipynb`](./environments/example-env-LinearLongitudinalF16.ipynb) — Simple aircraft model

### 🚀 For Advanced Users:
1. 🤖 [`example_ppo.ipynb`](./reinforcement_learning/example_ppo.ipynb) — Reinforcement learning
2. 🔮 [`example-mpc-b747-torch-mpc-mlp.ipynb`](./mpc_controllers/example-mpc-b747-torch-mpc-mlp.ipynb) — Model predictive control
3. 🎮 [`example-env-unity.ipynb`](./environments/example-env-unity.ipynb) — Unity integration

## 🔧 Additional Dependencies

Some examples may require additional libraries:

```bash
# For Unity integration
pip install mlagents

# For advanced visualization
pip install plotly seaborn

# For optimization
pip install optuna

# For PyTorch
pip install torch torchvision
```

## 🤝 Contributing

Want to add your own example? We welcome your contributions!

1. 🍴 **Fork** the repository
2. 🌿 **Create a branch** for your example
3. 📝 **Add documentation** and comments
4. 🧪 **Test** your code
5. 📤 **Create a Pull Request**

## 📞 Support

Need help with the examples?

- 💬 **GitHub Discussions**: [Discussions](https://github.com/tensoraerospace/tensoraerospace/discussions)
- 🐛 **Issues**: [Report an issue](https://github.com/tensoraerospace/tensoraerospace/issues)

---

<div align="center">

**🌟 Explore, experiment, and build amazing aerospace systems! 🌟**

[⬆️ Back to top](#-tensoraerospace-examples) • [🏠 Home](../) • [📖 Documentation](https://tensoraerospace.readthedocs.io/)

</div>