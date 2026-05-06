---
hide:
  - navigation
  - toc
---

<div class="hero">
  <h1>TensorAeroSpace</h1>
  <p class="tagline">Open-source aerospace simulation + adaptive control toolkit. Trim, fly, fail, recover — all in pure NumPy.</p>
  <p>
    <a href="guide/installation.md" class="md-button md-button--primary">Install</a>
    <a href="cookbook/01_hello.md" class="md-button">Quickstart</a>
    <a href="model/b747_nonlinear.md" class="md-button">Models</a>
    <a href="agent/ihdp.md" class="md-button">Algorithms</a>
  </p>
  <p>
    <a href="https://github.com/TensorAeroSpace/TensorAeroSpace"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-TensorAeroSpace-000?logo=github"></a>
    <a href="https://pypi.org/project/tensoraerospace/"><img alt="PyPI" src="https://img.shields.io/pypi/v/tensoraerospace?color=3775A9&logo=pypi&label=PyPI"></a>
    <a href="https://huggingface.co/TensorAeroSpace"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TensorAeroSpace-FFD21E"></a>
    <a href="https://pypi.org/project/tensoraerospace/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/tensoraerospace?logo=python&label=Python"></a>
    <a href="https://pypi.org/project/tensoraerospace/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/tensoraerospace?label=Downloads"></a>
    <a href="https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
    <a href="https://deepwiki.com/TensorAeroSpace/TensorAeroSpace"><img alt="Ask DeepWiki" src="https://deepwiki.com/badge.svg"></a>
  </p>
</div>

<style>
.hero {
  text-align: center;
  margin: 2rem 0 2.5rem 0;
  padding: 2.2rem 1rem;
  background: linear-gradient(120deg, rgba(59,130,246,.18), rgba(59,130,246,0) 50%),
              radial-gradient(60rem 60rem at 10% -20%, rgba(59,130,246,.25), transparent 40%),
              radial-gradient(50rem 50rem at 90% 120%, rgba(59,130,246,.18), transparent 40%),
              linear-gradient(135deg, rgba(59,130,246,.08), rgba(59,130,246,0));
  background-size: 200% 200%, auto, auto, auto;
  animation: gradientShift 12s ease-in-out infinite alternate;
  border-radius: 16px;
}
@keyframes gradientShift {
  0% { background-position: 0% 50%, 0 0, 0 0, 0 0; }
  100% { background-position: 100% 50%, 0 0, 0 0, 0 0; }
}
.hero .tagline {
  font-size: 1.08rem;
  color: var(--md-default-fg-color--light);
  margin-top: .3rem;
}
.hero .md-button { margin: .25rem .25rem; }
.hero a img { vertical-align: middle; margin: 0 .22rem; }
.cards .card-icon { font-size: 1.6rem; }
.stats { text-align: center; margin: 1.5rem 0 0; color: var(--md-default-fg-color--light); }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin: 1.5rem 0; text-align: center; }
.stat-grid .stat { padding: 1rem; border-radius: 12px; background: var(--md-code-bg-color); }
.stat-grid .stat-num { font-size: 2.1rem; font-weight: 600; color: var(--md-primary-fg-color); line-height: 1; }
.stat-grid .stat-lbl { font-size: 0.85rem; margin-top: .35rem; color: var(--md-default-fg-color--light); }
.logos { display: flex; gap: 1.2rem; align-items: center; justify-content: center; flex-wrap: wrap; margin: 1rem 0; }
.logos img { height: 42px; opacity: .9; filter: saturate(0) contrast(1.1); }
</style>

## Why TensorAeroSpace?

<div class="grid cards" markdown>

-   :material-airplane-takeoff: **Real airframes, not toys**

    Every model is transcribed from peer-reviewed source data: NASA CR-2144 (B-747), NASA TM X-1669 (X-15), CEAS Aeronautical Journal 2025 (Skywalker X8), JSBSim (B-737), Roskam Vol VI + class-II UAV literature (RQ-7 Shadow). Trim points reach machine precision; cross-validated against published derivatives.

-   :material-flash: **Pure NumPy core**

    No proprietary simulators, no MATLAB licence, no compiled binaries. The 6-DoF Newton-Euler ODE is an order of magnitude faster than JSBSim for control-synthesis sweeps, and trivially differentiable for adjoint-based methods.

-   :material-brain: **Unique adaptive-control catalogue**

    The standard RL stack (PPO, SAC, DDPG, DQN, A2C, A3C, GAIL) **plus** the full incremental-ADP family (IHDP, IM-GDHP, ET-DHP, iADP, AA-INDI, AIDI) — rarely co-located in a single open-source package. All wrapped in the same Gymnasium API.

-   :material-shield-check: **Damage subsystem built-in**

    Per-surface effectiveness loss, hard-overs, jam events, asymmetric-thrust engine-out, flap-jam configuration override — all composable into `DamageProfile` instances. Runs out of the box on B-747 and F-16; parity hooks on every other airframe.

-   :material-puzzle-outline: **Gymnasium-native**

    Every env implements the standard `reset()` / `step()` / `action_space` / `observation_space` contract. Drop-in replaceable in any Gymnasium / Stable Baselines3 / CleanRL pipeline.

-   :material-test-tube: **Continuously verified**

    Trim convergence, surface-deflection sign conventions, propellant burnout times — all locked down by 894 unit tests with regression coverage on every push.

</div>

---

## 30-second start

=== ":fontawesome-solid-plane: Run a flight simulation"

    ```python
    import gymnasium as gym
    import tensoraerospace  # registers all envs
    import numpy as np

    env = gym.make("NonlinearB747-v0", trim_at=(20_000.0, 674.0),
                   number_time_steps=2000, dt=0.01)
    obs, _ = env.reset()
    trim_action = np.array([-0.0126, 0.0, 0.0, 0.555])  # rad / [0, 1]
    for _ in range(2000):
        obs, _, _, trunc, _ = env.step(trim_action)
        if trunc: break

    print(f"V={float(np.linalg.norm(obs[:3])):.1f} ft/s, "
          f"alt={-float(obs[11]):.0f} ft")
    # → V=674.0 ft/s, alt=20000 ft  (perfect trim hold)
    ```

=== ":material-cog: Classical PID"

    ```python
    import gymnasium as gym
    import numpy as np
    from tensoraerospace.agent.pid import PID
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import unit_step

    dt = 0.01
    tp = generate_time_period(tn=10, dt=dt)
    ref = unit_step(degree=5, tp=tp, time_step=2.0,
                    output_rad=True).reshape(1, -1)

    env = gym.make('LinearLongitudinalF16-v0',
                   number_time_steps=len(tp),
                   initial_state=[[0], [0]],
                   reference_signal=ref, use_reward=False)
    pid = PID(env, kp=-14.29, ki=-8.24, kd=-1.30, dt=dt)

    obs, _ = env.reset()
    for t in range(len(tp) - 1):
        u = pid.select_action(ref[0, t], float(obs[0]))
        obs, _, term, trunc, _ = env.step(np.array([[float(u)]],
                                                    dtype=np.float32))
        if term or trunc: break
    ```

=== ":material-brain: Online ADP (IHDP)"

    ```python
    from tensoraerospace.aerospacemodel.b747.nonlinear import trim
    from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env
    from tensoraerospace.agent.ihdp.model import IHDPAgent
    import numpy as np, math

    # Trim at FL200 cruise
    r = trim(altitude_ft=20_000.0, V_ft_s=674.0)
    env = NonlinearB747Env(trim_at=(20_000.0, 674.0),
                            number_time_steps=3000, dt=0.02)

    agent = IHDPAgent(actor_settings={...}, critic_settings={...},
                      incremental_settings={...},
                      tracking_states=["d_theta"],
                      selected_states=["d_theta", "q"],
                      selected_input=["d_elev"],
                      number_time_steps=3000,
                      indices_tracking_states=[0])

    # Single-pass online learning — no offline pre-training
    obs, _ = env.reset()
    # ... run rollout ...
    # → Late-half MAE θ = 0.043° on a 1° step (4.3% of amplitude)
    ```

=== ":material-rocket: Pretrained SAC"

    ```bash
    python example/reinforcement_learning/deep_rl/sac-b747-render.py \
        --render --dt 0.1 --tn 200 \
        --repo TensorAeroSpace/sac-b747 --device cuda
    ```

    Or via Python:

    ```python
    from tensoraerospace.agent.sac import SAC
    from tensoraerospace.envs.b747 import ImprovedB747Env

    agent = SAC.from_pretrained("TensorAeroSpace/sac-b747")
    env = ImprovedB747Env(dt=0.1, number_time_steps=200)
    obs, _ = env.reset()
    while True:
        action = agent.select_action(obs, evaluate=True)
        obs, _, term, trunc, _ = env.step(action)
        env.render(mode="human")
        if term or trunc: break
    ```

---

## Aircraft library

Every model has a Gymnasium env, a trim solver, full 6-DoF dynamics, and peer-reviewed source data.

| Airframe | Class | Configurations | Aerodynamic source | Speciality |
|---|---|---|---|---|
| **F-16 Fighting Falcon** | Fighter | longitudinal · 6-DoF angular · damage | NASA / Stevens-Lewis tables | Cubic-spline aero, full damage subsystem |
| **Boeing 747-100** | Heavy transport | NOMINAL · POWER_APPROACH · LANDING | NASA CR-2144 (Heffley & Jewell) | Per-engine asymmetric thrust + flap jam |
| **Boeing 737-100/800** | Mid-size transport | 737-100 · 737-800 (NG) | JSBSim + Roskam Vol VI | Coordinated-turn benchmarks |
| **X-15** | Hypersonic research | BASIC · A2 record | NASA TM X-1669 + Thompson 2000 | Mach-tabulated 0.4–6.7, XLR99 rocket, variable mass |
| **Skywalker X8** | Small UAV (3.4 kg) | flying-wing | CEAS Aeronautical Journal 2025 | Peer-reviewed flight-test ID |
| **AAI RQ-7 Shadow** | Class-II UAV (170 kg) | RQ-7B | Beard & McLain + NASA TM-2014-218686 | V-tail mixed control |
| **Quadrotor** | Multirotor | nonlinear 6-DoF + damage | Standard quad-X derivation | Per-rotor failure, X-config allocator |
| **F-4C, ELV, ComSat, GeoSat, LSU, Ultrastick, UAV, Missile** | Linear / improved | various | Roskam, AIAA conf | Classical state-space + RL-friendly wrapper |

[See full model gallery →](model/f16.md)

---

## Control catalogue

20 algorithms, organised by family:

=== "Classical"

    | Algorithm | Description |
    |---|---|
    | **PID** | Classical PID with multiple tuning workflows |
    | **MPC** | Model-Predictive Control with MLP / NARX / Transformer plant models |

=== "Adaptive Dynamic Programming (offline)"

    Classical batch-trained ADP family:

    | Algorithm | Notebook |
    |---|---|
    | **HDP** (Heuristic Dynamic Programming) | `acd_hdp_b747.ipynb` |
    | **DHP** (Dual Heuristic Programming) | `acd_dhp_b747.ipynb` |
    | **GDHP** (Globalized Dual HP) | `acd_gdhp_b747.ipynb` |
    | **AD-HDP** | `acd_adhdp_b747.ipynb` |
    | **AD-GDHP** | `acd_adgdhp_b747.ipynb` |
    | **AD-DHP** | `acd_addhp_b747.ipynb` |

=== "Incremental ADP (online)"

    Online single-pass adaptive critic — the unique part of the catalogue:

    | Algorithm | Pitch-step / cruise tracking | Damage scenarios |
    |---|---|---|
    | **IHDP** | F-16 sin-α, B-747 θ-step, B-737 90° turn, quadrotor | failure recovery |
    | **IM-GDHP** | F-16 nonlinear | — |
    | **ET-DHP** | F-16 sinusoid | B-747 engine-out (0.28° ψ-error), F-16 damage |
    | **iADP** | F-16 nonlinear | F-16 damage |
    | **AA-INDI** | F-16 nonlinear | — |
    | **AIDI** | — | F-16 damage |

=== "Deep RL"

    Standard model-free RL stack:

    | Algorithm | Type | Pretrained checkpoint |
    |---|---|---|
    | **SAC** | Off-policy actor-critic | [HF: TensorAeroSpace/sac-b747](https://huggingface.co/TensorAeroSpace/sac-b747) |
    | **DSAC** | Distributional SAC | step-response & tracking variants |
    | **PPO** | On-policy clipped objective | 8 airframes |
    | **DDPG** | Deterministic policy gradient | B-747 |
    | **DQN** | Discrete value iteration | B-747, Unity |
    | **A2C** | Synchronous actor-critic | B-747 + NARX critic |
    | **A3C** | Asynchronous A-C | B-747 |
    | **GAIL** | Generative imitation | F-16 dataset |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        tensoraerospace package                         │
├──────────────────┬──────────────────────┬───────────────────────────────┤
│  aerospacemodel  │         envs         │            agent              │
├──────────────────┼──────────────────────┼───────────────────────────────┤
│  pure-NumPy      │   Gymnasium-spec     │   Classical · ADP · Deep RL   │
│  6-DoF dynamics  │   wrappers           │   PID, IHDP, ET-DHP, SAC, ... │
│  Trim solvers    │   "virtual" /        │   All consume Gym env API     │
│  Damage subsys.  │   "normalized"       │                               │
│                  │   action modes       │                               │
└──────────────────┴──────────────────────┴───────────────────────────────┘
                                │
                                ▼
                         Gymnasium / SB3 / CleanRL  ← any RL pipeline plugs in
```

The three packages are weakly coupled. You can use the dynamics standalone (no Gym), the env without an agent (just a wrapper), or build your own controller against any model. Same Gymnasium API across linear / nonlinear / damaged models means the controller code is portable.

---

## Use cases

=== ":material-account-cog: Control engineer"

    "I need a faithful 6-DoF aircraft model to design a controller against."

    1. Pick an airframe from the [model gallery](model/f16.md).
    2. Use the **trim solver** (`trim(altitude, V)`) to find your operating point.
    3. Linearise around trim or run nonlinear simulation directly.
    4. Validate against the published cruise / loiter / hypersonic conditions in the docs.

    Example: [Boeing 747 nonlinear](model/b747_nonlinear.md), [B-737 coordinated turn](example/agent/ihdp/example_ihdp_nonlinear_b737_turn.md).

=== ":material-school-outline: RL researcher"

    "I want to benchmark a new agent on aerospace tasks."

    1. Drop-in any of the 20+ envs registered with `gym.make("...-v0")`.
    2. Use the included PID / IHDP / SAC baselines for fair comparison.
    3. Compare metrics with the published [comparison studies](comparison/all_vs_pid_b747.md).
    4. Push your trained model to Hugging Face under the same wrapper.

    Example: [SAC on B-747](example/agent/sac/example-sac-b747.md), [IHDP vs PID on F-16](comparison/ihdp_imgdhp_vs_pid_f16_nonlinear.md).

=== ":material-shield-alert: FTC / fault-tolerant control"

    "I'm researching control reconfiguration after damage."

    1. Use the [damage subsystem](model/aircraft-damage-modeling.md) — surface jam, effectiveness loss, engine flameout, flap jam, RCS thruster failure.
    2. Compose `DamageProfile` instances declaratively.
    3. Run an online ADP agent (ET-DHP, iADP, AIDI) that adapts to the change in real time.

    Example: [ET-DHP B-747 engine-out yaw hold (0.28°)](example/agent/et_dhp/example_etdhp_b747_engine_failure.md).

=== ":material-book-open: Student / educator"

    "I'm learning flight dynamics or adaptive control."

    1. Start with the [cookbook](cookbook/01_hello.md) — 16 step-by-step recipes from "hello world" to FTC under damage.
    2. Read the [11 lessons](lesson/base/tutor_1.md) on state-space, controllability, RL fundamentals, and a hands-on practical (XFLR5 → Simulink → Python).
    3. Open any notebook from the [example gallery](https://github.com/TensorAeroSpace/TensorAeroSpace/tree/main/example) and run it locally.

    Example: [Lesson 1 — State-Space Intro](lesson/base/tutor_1.md), [Cookbook — Online-adaptive agents](cookbook/06_online_adaptive.md).

---

## What's new

<div class="grid cards" markdown>

-   :material-airplane: **AAI RQ-7 Shadow nonlinear**

    Class-II tactical UAV, 170 kg, V-tail mixed convention, 4-channel control. Synthesised from Beard & McLain + NASA TM-2014-218686 + Roskam Vol VI.

    [:octicons-arrow-right-24: model/aai_shadow_nonlinear](model/aai_shadow_nonlinear.md)

-   :material-airplane-takeoff: **Boeing 737 nonlinear**

    737-100 / 737-800 with JSBSim aerodynamic data, JT8D / CFM56-7B engine models, machine-precision trim, MIMO IHDP coordinated turn example.

    [:octicons-arrow-right-24: model/b737_nonlinear](model/b737_nonlinear.md)

-   :material-rocket-launch: **X-15 hypersonic**

    M = 0.4 → 6.7, XLR99 rocket engine, variable mass, 13-state vector with propellant channel. Burnout time 79.8 s — matches Thompson 2000 to 0.2 %.

    [:octicons-arrow-right-24: model/x15_nonlinear](model/x15_nonlinear.md)

-   :material-quadcopter: **Skywalker X8 small UAV**

    Peer-reviewed flight-test identification (CEAS Aeronautical Journal 2025). 3.4 kg flying-wing, 3-channel control, propeller-airframe drag coupling.

    [:octicons-arrow-right-24: model/skywalker_x8_nonlinear](model/skywalker_x8_nonlinear.md)

-   :material-target: **B-747 damage subsystem v2**

    Per-engine asymmetric thrust + flap-jam configuration override + 5 ready-made presets including `LEFT_OUTER_ENGINE_FAILURE`.

    [:octicons-arrow-right-24: model/b747_nonlinear](model/b747_nonlinear.md#damage-subsystem)

-   :material-format-list-bulleted-square: **Restructured `example/`**

    Top-level grouped by controller class; `reinforcement_learning/` split into `incremental_adp/` and `deep_rl/`. 101 notebooks, all paths updated in docs.

    [:octicons-arrow-right-24: example README](https://github.com/TensorAeroSpace/TensorAeroSpace/tree/main/example)

</div>

---

## Featured benchmarks

Real numbers from runnable example notebooks:

| Scenario | Result | Source |
|---|---|---|
| **B-747 ET-DHP heading hold under engine flameout** | ψ-error **0.28°** vs open-loop −85.5° | [notebook](example/agent/et_dhp/example_etdhp_b747_engine_failure.md) |
| **B-737 MIMO IHDP 90° coordinated turn** | final ψ-error **0.98°**, max sideslip **0.11°** | [notebook](example/agent/ihdp/example_ihdp_nonlinear_b737_turn.md) |
| **B-747 IHDP θ step (1°)** | late-half MAE **0.043°** | [notebook](example/agent/ihdp/example_ihdp_nonlinear_b747.md) |
| **X-15 boost-burnout (full throttle)** | burnout **79.8 s** vs Thompson 2000: 80 s | [model](model/x15_nonlinear.md) |
| **Skywalker X8 cruise trim** | machine precision (residual 1e-15) | [model](model/skywalker_x8_nonlinear.md) |

---

## Installation

=== "pip"

    ```bash
    pip install tensoraerospace
    ```

=== "poetry"

    ```bash
    poetry add tensoraerospace
    ```

=== "from source"

    ```bash
    git clone https://github.com/TensorAeroSpace/TensorAeroSpace.git
    cd TensorAeroSpace
    poetry install
    ```

=== "Docker"

    ```bash
    docker run --rm -p 8888:8888 ghcr.io/tensoraerospace/tas-jupyter
    ```

Python 3.10–3.12, no MATLAB required, no proprietary code paths.

---

## Resources

| | |
|---|---|
| 📦 **Package** | [PyPI](https://pypi.org/project/tensoraerospace/) · [GitHub](https://github.com/TensorAeroSpace/TensorAeroSpace) · [Hugging Face](https://huggingface.co/TensorAeroSpace) |
| 📚 **Documentation** | [Models](model/f16.md) · [Algorithms](agent/sac.md) · [Cookbook](cookbook/01_hello.md) · [Lessons](lesson/base/tutor_1.md) |
| 🧪 **Examples** | [101 notebooks on GitHub](https://github.com/TensorAeroSpace/TensorAeroSpace/tree/main/example) |
| 📊 **Benchmarks** | [Comparison studies](comparison/all_vs_pid_b747.md) · [Metrics](benchmark/metrics.md) |
| 💬 **Community** | [Issues](https://github.com/TensorAeroSpace/TensorAeroSpace/issues) · [DeepWiki Q&A](https://deepwiki.com/TensorAeroSpace/TensorAeroSpace) |

---

## Citation

If you use TensorAeroSpace in your research, please cite:

```bibtex
@software{tensoraerospace,
  title  = {TensorAeroSpace: An Open-Source Aerospace Simulation and Adaptive Control Toolkit},
  author = {Mazaev, A. and contributors},
  year   = {2026},
  url    = {https://github.com/TensorAeroSpace/TensorAeroSpace},
  note   = {Pure-NumPy 6-DoF aerospace dynamics + Gymnasium envs + classical/adaptive/deep RL agents}
}
```

For the underlying aerodynamic data sources — NASA CR-2144 (B-747), NASA TM X-1669 (X-15), CEAS Aeronautical Journal 2025 (Skywalker X8), JSBSim (B-737), Beard & McLain (Aerosonde / Shadow), Roskam Vol VI — please also cite the original references listed in each model's documentation page.

---

<div style="text-align:center; margin: 1.6rem 0 0.5rem;">
  <a href="guide/installation.md" class="md-button md-button--primary">Get started</a>
  <a href="cookbook/01_hello.md" class="md-button">Open the cookbook</a>
  <a href="https://github.com/TensorAeroSpace/TensorAeroSpace" class="md-button">Star on GitHub</a>
</div>

<p style="text-align:center; color: var(--md-default-fg-color--light); margin-top: 1rem;">
MIT licensed · built with NumPy, PyTorch, Gymnasium · powered by the aerospace research community.
</p>
