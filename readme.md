# 🚀 TensorAeroSpace

<div align="center">

[![en](https://img.shields.io/badge/lang-en-red.svg)](./README.md)
[![ru](https://img.shields.io/badge/lang-ru-green.svg)](./README.ru-ru.md)
[![Documentation Status](https://readthedocs.org/projects/tensoraerospace/badge/?version=latest)](https://tensoraerospace.readthedocs.io/en/latest/?badge=latest)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/tensoraerospace/tensoraerospace.svg)](https://github.com/tensoraerospace/tensoraerospace/stargazers)
[![Coverage Status](https://coveralls.io/repos/github/TensorAeroSpace/TensorAeroSpace/badge.svg?branch=develop)](https://coveralls.io/github/TensorAeroSpace/TensorAeroSpace?branch=develop)
![TensorAeroSpace Logo](./img/logo-no-background.png)

**Advanced Aerospace Control Systems & Reinforcement Learning Framework**

*A comprehensive Python library for aerospace simulation, control algorithms, and reinforcement learning implementations*

[📖 Documentation](https://tensoraerospace.readthedocs.io/) • [🚀 Quick Start](#-quick-start) • [💡 Examples](./example/) • [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## 🌟 Overview

**TensorAeroSpace** is a cutting-edge Python framework that combines aerospace engineering with modern machine learning. It provides:

- 🎯 **Control Systems**: Advanced control algorithms including PID, MPC, and modern RL approaches
- ✈️ **Aerospace Models**: High-fidelity aircraft and spacecraft simulation models
- 🎮 **OpenAI Gym Integration**: Ready-to-use environments for reinforcement learning
- 🧠 **RL Algorithms**: State-of-the-art reinforcement learning implementations
- 🔧 **Extensible Architecture**: Easy to extend and customize for your specific needs

## 🚀 Quick Start

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
```bash
docker build -t tensoraerospace . --platform=linux/amd64
docker run -v $(pwd)/example:/app/example -p 8888:8888 -it tensoraerospace
```

### 🏃‍♂️ Quick Example

```python
import tensoraerospace as tas

# Create an F-16 environment
env = tas.envs.F16Env()

# Initialize a PPO agent
agent = tas.agent.PPO(env.observation_space, env.action_space)

# Train the agent
for episode in range(1000):
    obs = env.reset()
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
```

## 🤖 Supported Algorithms

| Algorithm | Type | HuggingFace Export | Status |
|-----------|------|:------------------:|:------:|
| **IHDP** | Incremental Heuristic Dynamic Programming | ❌ | ✅ |
| **DQN** | Deep Q-Learning | ❌ | ✅ |
| **DDPG** | Deep Deterministic Policy Gradient | ❌ | ✅ |
| **SAC** | Soft Actor-Critic | ✅ | ✅ |
| **A3C** | Asynchronous Advantage Actor-Critic | ❌ | ✅ |
| **PPO** | Proximal Policy Optimization | ✅ | ✅ |
| **GAIL** | Imitation Learning (Adversarial) | ❌ | ✅ |
| **MPC** | Model Predictive Control | ✅ | ✅ |
| **A2C** | Advantage Actor-Critic | ✅ | ✅ |
| **A2C-NARX** | A2C with NARX Critic | ❌ | ✅ |
| **PID** | Proportional-Integral-Derivative | ✅ | ✅ |

## ✈️ Aircraft & Spacecraft Models

<details>
<summary><b>🛩️ Fixed-Wing Aircraft</b></summary>

- **General Dynamics F-16 Fighting Falcon** - High-fidelity fighter jet model
- **Boeing 747** - Commercial airliner dynamics
- **McDonnell Douglas F-4C Phantom II** - Military aircraft model
- **North American X-15** - Hypersonic research aircraft

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

![Unity Demo](./docs/example/env/img/img_demo_unity.gif)

</div>

TensorAeroSpace seamlessly integrates with Unity ML-Agents for immersive 3D simulations:

- 🎮 **3D Visualization**: Real-time 3D aircraft simulation
- 🔄 **Real-time Training**: Train agents in realistic environments
- 📊 **Rich Sensors**: Camera, LiDAR, and physics-based sensors
- 🌍 **Custom Environments**: Build your own aerospace scenarios

> 📁 **Example Environment**: [UnityAirplaneEnvironment](https://github.com/TensorAeroSpace/UnityAirplaneEnvironment)

### 🔧 MATLAB Simulink Support

![Simulink Model](docs/example/simulink/img/model.png)

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
