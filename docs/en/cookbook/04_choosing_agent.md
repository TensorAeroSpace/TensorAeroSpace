# Recipe 04 — Choosing an agent

**Goal.** Pick the right agent for your task in 60 seconds. TensorAeroSpace ships 17 agents across four families; this page compresses the decision into a short tree and one trade-off table.

**Related.** [Recipe 05](05_deep_rl_training.md) for deep-RL workflow, [Recipe 06](06_online_adaptive.md) for online-adaptive agents.

## The 3-question decision tree

### Q1. Do you have a reliable plant model?

- **Yes, and it's accurate** → **MPC** (`tensoraerospace.agent.mpc`). Uses the model for lookahead optimisation. Best when the model is closed-form or quickly identified.
- **Yes, but it drifts / deforms** → skip to Q2 under "online adaptive".
- **No, you only have a simulator to roll out in** → skip to Q2 under "deep-RL".

### Q2a. Online-adaptive branch (model drifts or unknown at deployment)

- **You know you'll face actuator / sensor faults** → **AA-INDI** (`agent.aa_indi`). INDI-based, with VFF-RLS that contracts the forgetting factor under large residuals — fastest abrupt-fault recovery in the library.
- **You want a fully parametric RL-style controller, model-free at deployment** → **iADP** (`agent.iadp`). Online RLS + batch-LS policy evaluation + closed-form LQT policy. Few knobs, interpretable.
- **You want a neural actor-critic with online identification** → **IHDP / IM-GDHP / ET-DHP** (`agent.ihdp`, `agent.im_gdhp`, `agent.et_dhp`). Adaptive Dynamic Programming with online RLS model; ET-DHP adds event-triggered updates for low-compute regimes.

### Q2b. Deep-RL branch (no analytical model, enough sim compute)

- **Fastest to train, discrete actions** → **DQN** (`agent.dqn`). Rarely the right choice for flight control (continuous actions), but baseline for educational contexts.
- **Continuous actions, stability is the main concern** → **PPO** (`agent.ppo`). Safest default in the library.
- **Continuous actions, sample efficiency matters** → **SAC** or **DSAC** (`agent.sac`, `agent.dsac`). DSAC (distributional SAC) tends to be tighter on tail-risk tasks (gust rejection, outer-loop speed control).
- **Continuous actions, you have expert demos** → **GAIL** (`agent.gail`). Imitates a demo dataset instead of learning from scratch.
- **Continuous actions, asynchronous training fits your compute** → **A2C-NARX** or **A3C** (`agent.a2c_narx`, `agent.a3c`). The NARX variant adds a recurrent critic, useful for partially observable settings.

### Q3. Do you need a classical baseline?

- **Yes** → **PID** (`agent.pid`). One-line controller, anti-windup built in. Every comparison in the library benchmarks against it.

## The trade-off matrix

| Agent | Needs model? | Online adapt? | Learns over episodes? | Continuous actions? | Fault tolerant? | Interpretable? |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| PID | — | — | — | ✓ | — | ✓✓✓ |
| MPC | ✓ | — | — | ✓ | partial | ✓✓ |
| IHDP | partial | ✓ | — | ✓ | ✓ | ✓ |
| IM-GDHP | partial | ✓ | — | ✓ | ✓ | ✓ |
| ET-DHP | partial | ✓ (triggered) | — | ✓ | ✓ | ✓ |
| AA-INDI | warm-start G | ✓✓ | — | ✓ | ✓✓✓ | ✓✓ |
| iADP | warm-start F,G | ✓ | — | ✓ | ✓ | ✓✓ |
| HDP / ADHDP | — | partial | ✓ | ✓ | — | ✓ |
| DQN | — | — | ✓ | ✗ | — | — |
| PPO | — | — | ✓ | ✓ | — | — |
| SAC / DSAC | — | — | ✓ | ✓ | — | — |
| DDPG | — | — | ✓ | ✓ | — | — |
| A3C / A2C-NARX | — | — | ✓ | ✓ | — | — |
| GAIL | — | — | ✓ (from demos) | ✓ | — | — |

Legend: ✓✓✓ = first-class, ✓ = supported, — = not a design goal.

## Typical combinations

- **"I have an F-16 model and want to track pitch angle."** → PID for baseline, then MPC for the headroom.
- **"I want robust rate tracking under elevator damage."** → AA-INDI. If you also want interpretable LQT cost, iADP.
- **"I want the best long-term tracking on a B747, no analytic model."** → SAC or DSAC after hyperparameter search (Recipe 07).
- **"I have pilot demonstrations."** → GAIL.
- **"I need event-triggered control for an embedded platform."** → ET-DHP.

## Where to go next

- **[Recipe 05 — Deep-RL training end-to-end](05_deep_rl_training.md)** — the train-eval-save cycle.
- **[Recipe 06 — Online-adaptive agents](06_online_adaptive.md)** — warm-start patterns and when to use which.
