# 📚 TensorAeroSpace Examples

<div align="center">

![TensorAeroSpace Logo](../img/logo-no-background.png)

**Comprehensive collection of examples and tutorials**

*Explore TensorAeroSpace capabilities through hands-on examples*

[🏠 Home](../) • [📖 Documentation](https://tensoraerospace.readthedocs.io/) • [🚀 Quick Start](../README.md)

</div>

---

## 🌟 Overview

This folder contains an extensive collection of TensorAeroSpace usage examples,
organised by **controller class** for predictable navigation. Inside each folder,
notebooks are named `example_<algorithm>_<aircraft>_<scenario>.ipynb` so they
sort alphabetically by algorithm.

The two large catch-all folders that previously held mixed content
(`reinforcement_learning/` and `general_examples/`) have been split into
agent-family subfolders — see [`reinforcement_learning/`](#-reinforcement-learning)
below.

---

## 📁 Folder map

```
example/
├── quickstart.ipynb              ← start here
├── environments/                 ← env-only demos (no agent)
├── pid_controllers/              ← classical PID
├── mpc_controllers/              ← Model Predictive Control
├── dynamic_programming/          ← classical ADP family (HDP, DHP, GDHP, ADHDP, ADGDHP)
├── reinforcement_learning/
│   ├── incremental_adp/          ← IHDP, IM-GDHP, ET-DHP, iADP, AA-INDI, AIDI
│   └── deep_rl/                  ← A2C, A3C, PPO, DQN, SAC, DSAC, DDPG, GAIL
├── comparison/                   ← head-to-head benchmarks (PID vs RL etc.)
├── failure_demos/                ← in-flight damage / dogfight scenarios
├── cookbook/                     ← step-by-step recipes
├── optimization/                 ← Optuna hyperparameter search
├── visualization/                ← 3D viewer, F-16 visuals
├── utilities/                    ← signal generators, sim-to-Python
├── aircraft/                     ← standalone aircraft demo scripts
└── README.md                     ← this file
```

---

## ✈️ Aerospace Environments

> **Folder:** [`environments/`](./environments/)

Examples of all aerospace environments shipped with TensorAeroSpace, with no
control agent. Use them to explore plant dynamics, validate trim, or as the
starting point for a new controller.

| Aircraft | Notebook |
|---|---|
| 🛩️ Boeing 747 (linear)        | [`example-env-LinearLongitudinalB747.ipynb`](./environments/example-env-LinearLongitudinalB747.ipynb) |
| 🛩️ Boeing 747 (improved)      | [`example-env-ImprovedB747.ipynb`](./environments/example-env-ImprovedB747.ipynb) |
| ⚡ F-16 Fighting Falcon        | [`example-env-LinearLongitudinalF16.ipynb`](./environments/example-env-LinearLongitudinalF16.ipynb) |
| 🚀 F-4C Phantom II             | [`example-env-f4c.ipynb`](./environments/example-env-f4c.ipynb) |
| 🚀 ELV Rocket                  | [`example-env-LinearLongitudinalELVRocket.ipynb`](./environments/example-env-LinearLongitudinalELVRocket.ipynb) |
| 🛸 UAV / Ultrastick / LAPAN    | [`example-env-LinearLongitudinalUAV.ipynb`](./environments/example-env-LinearLongitudinalUAV.ipynb) |
| 🎯 Missile                     | [`example-env-LinearLongitudinalMissileModel.ipynb`](./environments/example-env-LinearLongitudinalMissileModel.ipynb) |
| 🛰️ ComSat / GeoSat            | [`example-env-comsat.ipynb`](./environments/example-env-comsat.ipynb) |
| 🚁 X-15 (linear / improved)    | [`example-env-x15.ipynb`](./environments/example-env-x15.ipynb) |
| 🎮 Unity                       | [`example-env-unity.ipynb`](./environments/example-env-unity.ipynb) |

The newer **nonlinear** 6-DoF models (B-737, B-747, X-15, Skywalker X8,
AAI Shadow) are demonstrated in their own example notebooks under
`reinforcement_learning/` rather than here.

---

## 🔧 Classical control

### PID
> **Folder:** [`pid_controllers/`](./pid_controllers/)

| Notebook | Description |
|---|---|
| `pid_use.ipynb` | Basic PID controller usage |
| `pid_tuning_methods.ipynb` | Ziegler-Nichols and other tuning methods |
| `pid_matlab_tuning.ipynb` | MATLAB-style tuning workflow |

### MPC (Model Predictive Control)
> **Folder:** [`mpc_controllers/`](./mpc_controllers/)

| Notebook | Aircraft + model |
|---|---|
| `example-mpc-b747-torch-mpc-mlp.ipynb` | Boeing 747 + MLP plant model |
| `example-mpc-b747-torch-mpc-narx.ipynb` | Boeing 747 + NARX plant model |
| `example-mpc-b747-torch-mpc-transformer.ipynb` | Boeing 747 + transformer plant |

---

## 🧠 Adaptive Dynamic Programming (ADP)

### Classical ADP
> **Folder:** [`dynamic_programming/`](./dynamic_programming/)

The classical ADP family (offline / batch-trained variants):

| Algorithm | Notebook |
|---|---|
| HDP | `example_acd_hdp_b747.ipynb` |
| DHP | `example_acd_dhp_b747.ipynb` |
| GDHP | `example_acd_gdhp_b747.ipynb` |
| ADHDP | `example_acd_adhdp_b747.ipynb` |
| ADGDHP | `example_acd_adgdhp_b747.ipynb` |
| ADDHP | `example_acd_addhp_b747.ipynb` |
| ADP improved | `example_adp_b747_improved.ipynb` |
| Common helpers | `acd_b747_common.py` |

### Incremental ADP
> **Folder:** [`reinforcement_learning/incremental_adp/`](./reinforcement_learning/incremental_adp/)

Online incremental ADP variants — fast adaptation, single-pass learning:

| Algorithm | Notebooks |
|---|---|
| **IHDP** | `example_ihdp_linear_f16.ipynb`, `example_ihdp_nonlinear_f16.ipynb`, `example_ihdp_nonlinear_b747.ipynb`, `example_ihdp_nonlinear_b737.ipynb`, `example_ihdp_nonlinear_b737_turn.ipynb`, `example_ihdp_quadrotor.ipynb`, `example_ihdp_beautiful_demo.ipynb` |
| **IM-GDHP** | `example_im_gdhp_nonlinear_f16.ipynb` |
| **ET-DHP** | `example_etdhp_nonlinear_f16.ipynb`, `example_etdhp_b747_engine_failure.ipynb`, `example_etdhp_damage_f16.{ipynb,py}`, `example_etdhp_quadrotor_damage.ipynb` |
| **iADP** | `example_iadp_nonlinear_f16.ipynb`, `example_iadp_damage_f16.py` |
| **AA-INDI** | `example_aaindi_nonlinear_f16.ipynb` |
| **AIDI** | `example_aidi_damage_f16.ipynb` |

---

## 🎮 Reinforcement Learning

> **Folder:** [`reinforcement_learning/deep_rl/`](./reinforcement_learning/deep_rl/)

Deep RL agents trained on the various aircraft envs:

| Algorithm | Notebooks |
|---|---|
| **A2C** | `example_a2c.ipynb`, `example_a2c_b747_improved.ipynb`, `example_a2c_b747_narx_critic.ipynb` |
| **A3C** | `example-a3c.ipynb`, `example_a3c_b747_improved.ipynb` |
| **PPO** | `example_ppo_b747_improved.{ipynb,py}`, `example_ppo_comsat_improved.{ipynb,py}`, `example_ppo_elv_improved.ipynb`, `example_ppo_f4c_improved.ipynb`, `example_ppo_lsu_improved.ipynb`, `example_ppo_rocketmissle_improved.ipynb`, `example_ppo_ultrastick_improved.ipynb`, `example_ppo_x15_improved.ipynb` |
| **DQN** | `example_dqn_b747_improved.ipynb`, `example_dqn_unity.ipynb` |
| **SAC** | `example-sac.ipynb`, `example-sac-f16.ipynb`, `example_sac_b747_improved.ipynb`, `example_sac_unity.ipynb`, `sac-b747-render.py` |
| **DSAC** | `example_dsac_b747.{ipynb,py}`, `eval_dsac_b747.ipynb`, `eval_dsac_b747_step_response.ipynb`, `train_dsac_b747_step_response.py`, `train_dsac_b747_tracking.py` |
| **DDPG** | `example_ddpg_b747_improved.ipynb`, `ddpg-b747-render.py` |
| **GAIL** | `example_gail.ipynb`, `create_dataset_for_gail.ipynb` |
| **NARX baseline** | `example_narx.ipynb` |

---

## 📊 Comparison studies

> **Folder:** [`comparison/`](./comparison/)

Head-to-head benchmarks of different controllers on the same aircraft:

| Notebook | Comparison |
|---|---|
| `comparison_all_vs_pid_b747.ipynb` | All RL agents vs. PID baseline (B-747) |
| `comparison_dsac_vs_pid_b747.ipynb` | DSAC vs. PID (B-747) |
| `comparison_mpc_vs_pid_b747.ipynb` | MPC vs. PID (B-747) |
| `comparison_ppo_vs_pid_b747.ipynb` | PPO vs. PID (B-747) |
| `comparison_sac_vs_pid_f16.ipynb` | SAC vs. PID (F-16) |
| `comparison_f16_nonlinear_ml_vs_pid.ipynb` | ML-based vs. PID on nonlinear F-16 |
| `pid_f16_baseline.ipynb`, `sac_f16_baseline.ipynb` | F-16 reference baselines |
| `mpc_b747_baseline.ipynb`, `ppo_b747_baseline.ipynb`, `sac_b747_vec.ipynb` | B-747 reference baselines |

---

## 💥 Failure demos

> **Folder:** [`failure_demos/`](./failure_demos/)

Damage / fault scenarios:

| File | Scenario |
|---|---|
| `f16_damage_dogfight_demo.py` | F-16 dogfight with airframe damage |
| `example_ihdp_failure.ipynb` | IHDP recovering from in-flight failure |

Algorithm-specific failure demos remain in their respective folders (e.g.
`reinforcement_learning/incremental_adp/example_etdhp_damage_f16.ipynb`).

---

## 📖 Cookbook recipes

> **Folder:** [`cookbook/`](./cookbook/)

Step-by-step tutorials, structured to be read in order:

| Recipe | Topic |
|---|---|
| `recipe_01_hello.ipynb` | Hello, TensorAeroSpace |
| `recipe_03_reference_signals.ipynb` | Crafting reference signals |
| `recipe_07_optuna.ipynb` | Optuna hyperparameter search |
| `recipe_09_fault_tolerance.ipynb` | Fault-tolerant control |
| `classic_example.ipynb` | Classic adaptive-control workflow |
| `example.py` | Plain-Python example skeleton |

---

## ⚙️ Other folders

| Folder | Contents |
|---|---|
| [`optimization/`](./optimization/) | Optuna-based hyperparameter optimisation |
| [`visualization/`](./visualization/) | 3D flight viewer, F-16 longitudinal/angular visuals |
| [`utilities/`](./utilities/) | Signal generators, Simulink-to-Python, benchmark usage |
| [`aircraft/`](./aircraft/) | Standalone aircraft demo scripts |

---

## 🚀 Where to start

1. **First time?** Open [`quickstart.ipynb`](./quickstart.ipynb) — minimal end-to-end pipeline.
2. **New plant?** Check [`environments/`](./environments/) to see how to instantiate any of the bundled aircraft.
3. **New controller?** Pick from `pid_controllers/`, `mpc_controllers/`, `dynamic_programming/` (classical), `reinforcement_learning/incremental_adp/` (online ADP), or `reinforcement_learning/deep_rl/` (deep RL).
4. **Comparison study?** Check `comparison/`.
5. **Damage / fault tolerance?** See `failure_demos/` and the damage notebooks under `reinforcement_learning/incremental_adp/`.

---

<div align="center">

**Need help?** [Open an issue](https://github.com/TensorAeroSpace/TensorAeroSpace/issues) • **Star the repo** if these examples are useful ⭐

</div>
