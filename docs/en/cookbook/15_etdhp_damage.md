# Recipe 15 — ET-DHP under in-flight damage

This recipe walks through `example_etdhp_damage_f16.ipynb` end-to-end: training the **ET-DHP** agent on a healthy F-16 longitudinal model, then evaluating it for 60 s with a **real damage event injected at t = 20 s**. The damage plumbing is the [aircraft damage modeling](../model/aircraft-damage-modeling.md) subsystem — the env recomputes mass, area, and inertias on the fly, and the longitudinal ODE picks up the strip-theory `ΔCy / ΔMy` corrections, so the agent really does fly a different plant from t = 20 s onward.

**Agent docs.** [ET-DHP](../agent/et_dhp.md) · **Notebook.** `example/reinforcement_learning/example_etdhp_damage_f16.ipynb` · **Companion script (iADP).** `example/reinforcement_learning/example_iadp_damage_f16.py` · **Related recipe.** [Recipe 13 — ET-DHP on the nonlinear F-16](13_etdhp.md).

## Why this recipe

ET-DHP is well-suited to embedded deployments thanks to its event-triggered actor/critic updates. But how does it behave when the **plant itself changes** mid-flight? This recipe answers two questions:

- Can ET-DHP keep flying when 30% of both wing tips are lost at t = 20 s?
- Does the Lipschitz event trigger correctly react to the changed dynamics?

The companion iADP example (linked above) tackles the same scenario with online RLS plant identification — comparing the two illustrates the trade-off between offline-fit plant NN (ET-DHP) and online RLS (iADP).

## The damage event

A symmetric `DamageProfile` of two `section_loss` events fires at t = 20 s:

```python
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageEvent, DamageProfile,
)

damage_profile = DamageProfile(events=[
    DamageEvent(20.0, "section_loss",
                payload={"section": "left_tip", "loss_fraction": 0.30}),
    DamageEvent(20.0, "section_loss",
                payload={"section": "right_tip", "loss_fraction": 0.30}),
])
```

At firing the env:

1. Marks `state.section_loss["left_tip"] = state.section_loss["right_tip"] = 0.30`.
2. Recomputes effective `m, S, b, bA, J*` from per-section contributions (Huygens-Steiner for the inertia tensor; mass-weighted centroid for CG).
3. The longitudinal ODE adds `Δcy = -Σ cl_α_s · α · f_s · area_s/S_base` from strip theory at every step.

Symmetric loss leaves the CG centred but reduces the effective lift-curve slope and shifts inertias — the plant feels different to the controller.

## Pipeline

The notebook (and equivalent script) follows ET-DHP's standard pattern with one extra piece: a `damage_profile` is passed to the env in the final eval episode.

```python
import gymnasium as gym
from tensoraerospace.agent.et_dhp import ETDHPAgent, ETDHPConfig

# 1. Trim + feedforward elevator curve δₑ_trim(α)
#    (same as recipe 13)

# 2. Plant-NN pre-training on 30 s of multi-sine PE around trim
#    (healthy aircraft only — plant NN sees no damaged data)
agent.fit_plant_model(states_arr, actions_arr, next_states_arr)

# 3. Train the actor & critic for 6 episodes on the healthy F-16
for ep in range(NUM_TRAIN_EPISODES):
    log = run_episode(agent, healthy_env, learn=True)

# 4. Final evaluation — baseline (no damage)
baseline = run_episode(agent, healthy_env, learn=True)

# 5. Final evaluation — same agent, env now wraps the damage profile
def damaged_env():
    return gym.make(
        "NonlinearLongitudinalF16-v0",
        ..., damage_profile=damage_profile,
    ).unwrapped

damaged = run_episode(agent, damaged_env, learn=True)
```

Online actor/critic learning stays on during both eval episodes, so the trigger fires whenever tracking error crosses the Lipschitz threshold. The plant NN is **frozen** at its offline-pre-trained weights — that's the structural limitation we'll see below.

## Training curve (6 episodes, healthy aircraft)

A representative run from the script:

```
ep 1/6: RMSE_late=12.5554°  triggers=266
ep 2/6: RMSE_late=0.6101°   triggers=927
ep 3/6: RMSE_late=0.2439°   triggers=530
ep 4/6: RMSE_late=0.1061°   triggers=201
ep 5/6: RMSE_late=0.1142°   triggers=160
ep 6/6: RMSE_late=0.1096°   triggers=158
```

The agent goes from `12.5°` (random init) to `~0.11°` late-window RMSE in 4–6 episodes, with trigger count converging to ~160/episode as the closed loop stabilises. Same trend as recipe 13 — adding a damage hook to the env doesn't slow down healthy training.

## Final eval results

After training, two 60 s episodes:

| | Baseline (no damage) | With damage (30 % bilateral tip loss @ t=20 s) |
|---|---|---|
| Pre-damage MAE  (5 – 20 s) | 0.094 ° | 0.210 ° |
| Pre-damage RMSE | 0.114 ° | 0.268 ° |
| Post-damage MAE (22 – 60 s) | 0.166 ° | 0.702 ° |
| Post-damage RMSE | 0.235 ° | 0.913 ° |
| Triggers pre  / post | 56 / 261 | 219 / 547 |
| Damage events triggered | — | t=19.99 s : left_tip_30pct_loss<br>t=19.99 s : right_tip_30pct_loss |

The headlines:

- **The post-damage tracking error roughly quadruples** (RMSE 0.235 ° → 0.913 °). Not catastrophic, but visibly degraded — this is the price of holding the plant NN constant across the damage event.
- **The Lipschitz event trigger correctly responds**: trigger count after t = 20 s **doubles** versus the no-damage baseline (547 vs 261). The supervisor is doing its job — it just can't get the actor/critic to a stable post-damage policy because the plant NN's `F = ∂f/∂x`, `G = ∂f/∂u` no longer match the damaged dynamics.

## Tracking traces

![ET-DHP under damage — α tracking, residual elevator, ω_z](img/15_etdhp_damage.png)

Top panel: α tracking. Both runs hit the reference cleanly until t = 20 s; after the red dashed line the damaged trace shows a visible bias and reduced amplitude (the agent is undercommanding because it's using a stale plant gain). Middle: the residual elevator goes saturation-friendly more often after damage. Bottom: pitch rate `ω_z` shows phase shift after damage as the actuator timing falls out of sync.

## Why ET-DHP degrades — and how to recover

ET-DHP's closed-form policy is

$$u^*_t \;=\; u_b \cdot \tanh\!\Bigl(-\tfrac{1}{2}\,\gamma\,R^{-1} G^{T} \lambda(x_{t+1}) \,/\, u_b\Bigr)$$

so the action depends on `G = ∂f/∂u` from the **plant NN**. The actor and critic can adjust their weights through the event-trigger updates, but they cannot fix a stale `G`. Three ways to recover:

1. **Online plant-NN updates.** Periodically call `agent.fit_plant_model(...)` on a sliding window of the most recent `(x, u, x_next)` transitions. Practically: a 5-second window every 200 steps lets the plant NN catch the damaged dynamics within a couple of seconds.
2. **Damage-conditioned actor.** Set `damage_observable=True` on the env and feed the section-loss vector into the agent's state. The actor can then learn a control law indexed on damage state.
3. **Increase `u_bound`.** From `±2°` to `±5°` so the residual can absorb a larger plant change.

The companion **iADP** example (linked at the top) skips this problem entirely: its RLS identifier tracks `G̃` online and the closed-form policy adapts within milliseconds. iADP's post-damage RMSE ends up indistinguishable from the no-damage baseline — at the cost of a bias-prone PE warm-start phase.

## Try it

```bash
# Notebook — full narrative with plots:
jupyter lab example/reinforcement_learning/example_etdhp_damage_f16.ipynb

# Or the script — same logic, faster to iterate:
poetry run python example/reinforcement_learning/example_etdhp_damage_f16.py
```

Total runtime is ~3-5 minutes (dominated by the 6 training episodes; each is 60 s of simulation at dt = 10 ms with online actor/critic gradient steps).

## See also

- [Aircraft damage modeling](../model/aircraft-damage-modeling.md) — the damage subsystem in detail.
- [Recipe 13 — ET-DHP on the nonlinear F-16](13_etdhp.md) — same agent, no damage.
- [Recipe 09 — Fault tolerance](09_fault_tolerance.md) — broader context on adaptive RL under failures.
- iADP example with damage: `example/reinforcement_learning/example_iadp_damage_f16.py` — same scenario, online plant identification.
