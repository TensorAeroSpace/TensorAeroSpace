# Installation

> :material-rocket-launch-outline: Get TensorAeroSpace running in 60 seconds — from `pip install` to a converged trim point.

!!! tip "TL;DR"
    ```bash
    pip install tensoraerospace
    python -c "import tensoraerospace; from tensoraerospace.aerospacemodel.b747.nonlinear import trim; r = trim(altitude_ft=20_000.0, V_ft_s=674.0); print(f'B-747 trim residual = {r.residual:.2e}, converged = {r.converged}')"
    # → B-747 trim residual = 8.66e-14, converged = True
    ```

---

## System requirements

| Component | Minimum | Recommended |
|---|---|---|
| **OS** | Linux x86_64, Windows 10, macOS 13 | Ubuntu 22.04 LTS / Windows 11 / macOS 14 |
| **Python** | 3.10 | 3.11 or 3.12 |
| **CPU** | 4 cores, AVX | 8+ cores, AVX2 / FMA |
| **RAM** | 8 GB | 16–32 GB for RL training |
| **Disk** | 2 GB | 5 GB (PyTorch wheels + checkpoints) |
| **GPU** (optional) | — | NVIDIA RTX with ≥ 8 GB VRAM, CUDA 12.2 |
| **MATLAB** (optional) | — | R2022b+ for the Simulink-bridge example |
| **Unity** (optional) | — | 2021.3.5f1 / 2023.2.20f1 for the Unity ML-Agents bridge |

**Why these constraints?** PyTorch wheels on PyPI bundle compiled CUDA kernels and require AVX. The published nonlinear airframe models (B-747, X-15, B-737) integrate at `dt = 0.01 s` with an RK4 integrator — a single 60 s episode runs in under 1 second on a modern laptop, so heavy hardware is only needed for RL training, not for control synthesis.

| :material-python: Python | Status | Notes |
|---:|:---:|---|
| 3.10 | :material-check-circle: | Minimum supported |
| 3.11 | :material-check-circle: | **Recommended** |
| 3.12 | :material-check-circle: | Recommended for latest PyTorch |
| 3.13 | :material-flask-outline: | Experimental — some optional deps may lag |
| ≤ 3.9 | :material-close-circle: | Unsupported (uses 3.10+ syntax) |

---

## Quick install (PyPI)

=== "pip"

    ```bash
    pip install -U pip setuptools wheel
    pip install tensoraerospace
    ```

=== "poetry"

    Recommended for projects that want lockfile-pinned dependencies:

    ```bash
    poetry add tensoraerospace
    ```

=== "conda"

    ```bash
    conda create -n tas python=3.11 -y
    conda activate tas
    pip install -U pip setuptools wheel
    pip install tensoraerospace
    ```

=== "uv"

    Fast modern alternative to pip:

    ```bash
    uv venv --python 3.11
    source .venv/bin/activate     # Windows: .venv\Scripts\activate
    uv pip install tensoraerospace
    ```

!!! info "Wheel size"
    The wheel itself is small (~ 5 MB) but pulls in PyTorch (~ 800 MB), Gymnasium, NumPy, SciPy, matplotlib. First install takes 1–3 minutes depending on bandwidth.

---

## Verify the installation

Run the three checks below — they cover (1) Python imports, (2) Gymnasium env registration, and (3) numerical correctness of the trim solver.

### 1. Module import

```bash
python -c "import tensoraerospace as tas; print('TensorAeroSpace', tas.__version__, 'OK')"
```

Expected: `TensorAeroSpace 0.3.x OK` (no traceback).

### 2. Env registry

```python
import gymnasium as gym
import tensoraerospace  # registers ~ 30 envs

# All nonlinear 6-DoF aircraft envs:
for env_id in [
    "NonlinearLongitudinalF16-v0",
    "NonlinearAngularF16-v0",
    "NonlinearB747-v0",
    "NonlinearB737-v0",
    "NonlinearX15-v0",
    "NonlinearSkywalkerX8-v0",
    "NonlinearAAIShadow-v0",
]:
    env = gym.make(env_id, trim_at=(20_000.0, 674.0)
                   if env_id == "NonlinearB747-v0" else None,
                   number_time_steps=10)
    print(f"  ✓ {env_id}")
```

### 3. Numerical sanity (trim convergence)

```python
from tensoraerospace.aerospacemodel.b747.nonlinear import trim

result = trim(altitude_ft=20_000.0, V_ft_s=674.0)
assert result.converged, "trim solver should converge"
assert result.residual < 1e-6, f"residual too high: {result.residual}"
print(f"B-747 trim @ FL200, V=674 ft/s: residual = {result.residual:.2e} ✓")
# → B-747 trim @ FL200, V=674 ft/s: residual = 8.66e-14 ✓
```

If all three pass, the install is **fully functional** — both Python wiring and numerical computation are correct.

### Optional: full test sweep (dev install only)

```bash
poetry run pytest tests/aerospacemodel/ tests/envs/ -q
# → 894 passed in ~ 19 s
```

---

## GPU acceleration

=== "NVIDIA CUDA"

    PyTorch wheels on PyPI bundle CUDA 12.x by default. To verify:

    ```python
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("CUDA version:", torch.version.cuda)
    ```

    For specific CUDA versions, install matching PyTorch wheels first:

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install tensoraerospace
    ```

    See the [official PyTorch install matrix](https://pytorch.org/get-started/locally/).

=== "Apple Silicon (M1/M2/M3)"

    Native Metal Performance Shaders (MPS) backend:

    ```python
    import torch
    print("MPS available:", torch.backends.mps.is_available())
    # → MPS available: True
    ```

    Pretrained agents auto-detect MPS:

    ```python
    from tensoraerospace.agent.sac import SAC
    agent = SAC.from_pretrained("TensorAeroSpace/sac-b747")
    # device picked automatically: "mps" on Apple Silicon
    ```

=== "CPU only"

    No GPU? The default wheels work fine:

    - **Control synthesis** (PID, MPC, classical ADP, IHDP) — CPU is plenty.
    - **Deep RL training** (SAC, PPO, DDPG) — slower but possible (10–50× slower than GPU).
    - **Trim / simulation** — CPU only, no GPU benefit.

    To install CPU-only PyTorch (smaller download):

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install tensoraerospace
    ```

---

## Optional dependencies

| Feature | Install command | When you need it |
|---|---|---|
| **Hugging Face Hub** integration | bundled — already installed | Loading pretrained agents (`from_pretrained`), publishing models |
| **Unity ML-Agents** | `pip install mlagents-envs==0.30.0` | Running the [Unity airplane environment](../guide/unity_env.md) |
| **MATLAB / Simulink bridge** | MATLAB R2022b+ + `python -m matlab.engine.install` | Running [Simulink interop examples](../example/simulink/sim_pyth.md) |
| **3D flight viewer** | bundled (uses three.js) | `env.render()` returning interactive 3D scene |
| **Optuna hyperparameter search** | `pip install optuna` | [Hyperparameter optimisation cookbook](../cookbook/07_optuna_search.md) |
| **TensorBoard logging** | `pip install tensorboard` | Real-time metrics during agent training |

---

## Install from source (development)

When you want to **modify the library**, run the test suite, or build the documentation locally:

=== "Poetry (recommended)"

    ```bash
    git clone https://github.com/TensorAeroSpace/TensorAeroSpace.git
    cd TensorAeroSpace

    poetry install --with dev   # main + dev dependencies (pytest, mkdocs, etc.)
    eval $(poetry env activate)  # activate venv

    # Run tests
    poetry run pytest tests/aerospacemodel/ tests/envs/ -q

    # Build docs locally
    poetry run mkdocs serve -a 0.0.0.0:8000
    ```

=== "pip + venv"

    ```bash
    git clone https://github.com/TensorAeroSpace/TensorAeroSpace.git
    cd TensorAeroSpace

    python -m venv .venv
    source .venv/bin/activate    # Windows: .venv\Scripts\activate

    pip install -U pip setuptools wheel
    pip install -e ".[dev]"

    pytest tests/ -q
    ```

After cloning, run a quick smoke test:

```bash
poetry run python example/aircraft/example_b747_nonlinear.py
# → Trim @ FL200, V=674 ft/s: alpha=+3.603°, delta_e=-0.722°, throttle=0.555
# → Healthy step response, damaged step response, trajectory plot saved
```

---

## Run with Docker

!!! info "Recommended for reproducible environments"
    The official image bundles JupyterLab with all 101 example notebooks ready to run. No host-side Python setup needed.

=== "Pull the published image"

    ```bash
    docker pull ghcr.io/tensoraerospace/tensoraerospace:latest

    docker run --rm -it -p 8888:8888 \
      -v "$(pwd)/projects:/workspace/projects" \
      ghcr.io/tensoraerospace/tensoraerospace:latest
    ```

    Open the URL printed in the terminal (usually `http://127.0.0.1:8888`) and navigate to `/workspace/example/quickstart.ipynb`.

=== "Build locally"

    ```bash
    git clone https://github.com/TensorAeroSpace/TensorAeroSpace.git
    cd TensorAeroSpace
    docker build -t tas:local . --platform=linux/amd64

    docker run --rm -it -p 8888:8888 \
      -v "$(pwd)/projects:/workspace/projects" \
      tas:local
    ```

=== "GPU passthrough"

    Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):

    ```bash
    docker run --rm -it --gpus all -p 8888:8888 \
      -v "$(pwd)/projects:/workspace/projects" \
      ghcr.io/tensoraerospace/tensoraerospace:latest
    ```

---

## Update or uninstall

=== "Update"

    ```bash
    # pip
    pip install -U tensoraerospace

    # poetry
    poetry update tensoraerospace
    ```

    To get the latest unreleased changes from `main`:

    ```bash
    pip install -U git+https://github.com/TensorAeroSpace/TensorAeroSpace.git@main
    ```

=== "Uninstall"

    ```bash
    pip uninstall tensoraerospace
    # or
    poetry remove tensoraerospace
    ```

    To also free disk space taken by PyTorch and other deps:

    ```bash
    pip uninstall tensoraerospace torch numpy scipy gymnasium matplotlib
    ```

---

## Troubleshooting

???+ question "ImportError: No module named tensoraerospace"
    The package is not visible to the active Python. Check:

    ```bash
    which python && python -c "import sys; print(sys.executable)"
    pip show tensoraerospace
    ```

    Both should point to the same environment. If they don't, activate the right venv (`source .venv/bin/activate` or `poetry env activate` or `conda activate tas`).

??? question "PyTorch version conflicts (`undefined symbol`, `RuntimeError: CUDA error`)"
    Likely your installed PyTorch CUDA version doesn't match the system driver, or a stale wheel cache is in use.

    ```bash
    pip uninstall torch torchvision torchaudio -y
    pip cache purge
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install --upgrade --force-reinstall tensoraerospace
    ```

??? question "macOS (Apple Silicon) — torch can't find OpenMP"
    Apple's Xcode CLT ships clang without OpenMP. Either install via brew (`brew install libomp`) or rely on PyTorch's bundled MPS backend (no OpenMP needed for inference).

??? question "Trim solver fails to converge for my (h, V) point"
    For most aircraft this means the operating point is outside the cruise envelope. Check:

    - **Below stall**: `V` too low for the required lift at this weight.
    - **Above ceiling**: density too low for the engine's installed thrust.
    - **X-15** specifically: the rocket-engine model has no level-cruise envelope — use `gamma_rad=…` for climbing trim or `level_trim()` for the unphysical case. See [X-15 trim docs](../model/x15_nonlinear.md#trim-envelope-or-lack-thereof).

??? question "Permissions / corporate proxy"
    Use `pip install --user` to install into the user site (no admin needed), or run inside Docker. For corporate proxies set `PIP_INDEX_URL` / `HTTP_PROXY`:

    ```bash
    export PIP_INDEX_URL=https://your-mirror.example.com/simple
    export HTTP_PROXY=http://proxy:8080
    pip install tensoraerospace
    ```

??? question "CUDA available but agent runs on CPU"
    Some pretrained agents have a hardcoded device in their checkpoint. Force-move:

    ```python
    import torch
    agent = SAC.from_pretrained("TensorAeroSpace/sac-b747")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.policy = agent.policy.to(device)
    agent.critic = agent.critic.to(device)
    agent.critic_target = agent.critic_target.to(device)
    agent.device = device
    ```

??? question "`Could not find platform-independent libraries` on Windows"
    Usually a corrupted Python install. Reinstall Python 3.11 from [python.org](https://www.python.org/downloads/) (don't use the Windows Store version), then recreate your venv.

---

## What next?

You're ready to build.

[:material-rocket: 30-second quickstart](../cookbook/01_hello.md){ .md-button .md-button--primary }
[:material-airplane-takeoff: Models](../model/f16.md){ .md-button }
[:material-robot-outline: Algorithms](../agent/sac.md){ .md-button }
[:material-book-open-variant: 16-recipe cookbook](../cookbook/01_hello.md){ .md-button }
[:material-school-outline: Lessons](../lesson/base/tutor_1.md){ .md-button }

If you hit a snag the troubleshooting section didn't cover, ping us on [GitHub Discussions](https://github.com/TensorAeroSpace/TensorAeroSpace/discussions) or open an [issue](https://github.com/TensorAeroSpace/TensorAeroSpace/issues).
