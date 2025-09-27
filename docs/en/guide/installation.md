# Installation

> :material-rocket-launch-outline: Install TensorAeroSpace in 10 seconds and start experimenting.

!!! note
    Supported Python versions: 3.9 — 3.11. Python 3.12 support is planned.

| :material-python: Python | Status |
|-------------------------:|:------:|
| 3.9                      | ✅ |
| 3.10                     | ✅ |
| 3.11                     | ✅ |
| 3.12                     | ⏳ |

## Quick install (PyPI)

=== "pip"

    ```bash
    pip install -U pip setuptools wheel
    pip install tensoraerospace
    ```

=== "poetry"

    ```bash
    poetry add tensoraerospace
    ```

=== "conda"

    ```bash
    conda create -n tas python=3.10 -y
    conda activate tas
    pip install -U pip setuptools wheel
    pip install tensoraerospace
    ```

## Verify the installation

Quick version check and minimal example:

```bash
python -c "import tensoraerospace as tas; print(tas.__version__)"
```

```python
import gymnasium as gym
import numpy as np

from tensoraerospace.agent.pid import PID
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standart import unit_step

# Simulation setup
dt = 0.01
tp = generate_time_period(tn=10, dt=dt)  # 10 seconds
N = len(tp)

# Reference signal for alpha tracking (5 deg step in radians)
reference = unit_step(degree=5, tp=tp, time_step=100, output_rad=True).reshape(1, -1)

# Create F-16 longitudinal environment (state order here: [alpha, q])
env = gym.make(
    'LinearLongitudinalF16-v0',
    number_time_steps=N,
    initial_state=[[0], [0]],
    reference_signal=reference,
    use_reward=False,
)

# PID controller (coefficients from PID example)
pid = PID(env, kp=-14.290139135229715, ki=-8.240470780203491, kd=-1.2991634935096958, dt=dt)

obs, info = env.reset()
for t in range(N - 1):
    setpoint = reference[0, t]
    alpha = float(obs[0])  # env returns [alpha, q]
    u = pid.select_action(setpoint, alpha)
    action = np.array([[float(u)]], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

## Platforms and environments

<div class="grid cards" markdown>

-   :material-linux: **Linux**

    Recommended platform for training. `pip` and compatible wheels are enough.

-   :material-microsoft-windows: **Windows**

    Runs natively; for advanced scenarios consider WSL2.

-   :material-apple: **macOS (Intel/M‑series)**

    On Apple Silicon use compatible Python versions (3.9–3.11) and `tensorflow-macos`.

</div>

## Install from source (Dev)

=== "Poetry"

    ```bash
    git clone https://github.com/TensorAeroSpace/TensorAeroSpace.git
    cd TensorAeroSpace
    poetry install
    ```

=== "pip + venv"

    ```bash
    git clone https://github.com/TensorAeroSpace/TensorAeroSpace.git
    cd TensorAeroSpace

    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

    pip install -U pip setuptools wheel
    pip install -r requirements.txt
    pip install -e .
    ```

!!! tip
    Always isolate dependencies with a virtual environment (`venv`/`conda`/`poetry`) to avoid conflicts with global packages.

## CPU/GPU tips

- The project uses both TensorFlow and PyTorch. If you need GPU support, install the frameworks for your platform:
  - PyTorch with CUDA: follow the official installation guide for the version compatible with your CUDA stack.
  - macOS (Apple Silicon): the `tensorflow-macos` package installs automatically (see project dependencies). Make sure you use a compatible Python version (3.9–3.11).
- If GPU acceleration is not required, the default PyPI wheels are usually enough.

## Run with Docker

!!! info
    :material-docker: Docker is the recommended way to get a unified environment on Linux/Windows/macOS.

Сборка образа:

```bash
docker build -t tensoraerospace .
```

Run the container exposing the examples directory and the Jupyter port (if needed):

```bash
docker run --rm -it \
  -p 8888:8888 \
  -v "$(pwd)/example:/app/example" \
  --name tas tensoraerospace
```

!!! tip
    Mount any required directories with `-v <host>:<container>` to keep results outside the container.

## Common issues and fixes

???+ question "Unable to resolve dependencies"
    Isolation and up-to-date build tooling usually help.

    === "venv"

        ```bash
        python -m venv .venv
        source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
        pip install -U pip setuptools wheel
        pip install tensoraerospace
        ```

    === "conda"

        ```bash
        conda create -n tas python=3.10 -y
        conda activate tas
        pip install -U pip setuptools wheel
        pip install tensoraerospace
        ```

??? question "TensorFlow/PyTorch version conflicts"
    Install framework builds that match your platform (CUDA/CPU/macOS), then reinstall `tensoraerospace`.

??? question "macOS (M series)"
    Use Python 3.9–3.11. Reinstall `tensorflow-macos` with a compatible version if needed.

??? question "Permissions/network issues"
    Try a clean virtual environment or run inside Docker. For corporate proxies configure the `PIP_*` environment variables.

## Next steps

[:material-play-circle-outline: Examples](../example/enviroment/gymnasium.md){ .md-button .md-button--primary }
[:material-airplane-takeoff: Models](../model/f16.md){ .md-button }
[:material-robot-outline: Algorithms](../agent/sac.md){ .md-button }
[:material-book-open-variant: Tutorials](../lesson/0intro.md){ .md-button }
