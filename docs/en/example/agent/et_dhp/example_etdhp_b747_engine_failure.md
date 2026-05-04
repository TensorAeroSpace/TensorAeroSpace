# Example: ET-DHP holding initial heading after a single-engine flameout (B-747)

This example trains an **Event-Triggered Dual Heuristic Programming (ET-DHP)** agent to keep the [nonlinear Boeing 747-100](../../../model/b747_nonlinear.md) on its initial heading $\psi_0 = 0$ after the **left outer engine** flames out at $t = 10$ s. Source notebook: `example/reinforcement_learning/example_etdhp_b747_engine_failure.ipynb`.

## What this demonstrates

* **Per-engine failure** modelled by the [`LEFT_OUTER_ENGINE_FAILURE`](../../../model/b747_nonlinear.md#damage-subsystem) preset — engine #1 effectiveness goes to 0 and the engine model returns an asymmetric-thrust yaw moment (computed from the engine's spanwise position $y_1 = -71.7$ ft).
* **Persistent-disturbance regulation** — unlike sinusoidal tracking, the post-damage scenario has a constant yaw moment that the actor must compensate with a non-zero rudder offset. ET-DHP's actor learns this offset within ~2 episodes.
* **Open-loop vs. closed-loop ablation** — without the agent, $\psi$ drifts to $-85°$ in 50 s; with ET-DHP, $\psi$ stays within $\pm 0.3°$.

## Architecture

```
agent input  : x̃ = [ψ_deg, r_deg/s, φ_deg, p_deg/s]   (regulation state)
agent output : u  = [δ_a_deg, δ_r_deg]                  (aileron + rudder, deg)
env input    : [δ_e_trim, δ_a, δ_r, δ_T_trim]           (4-D virtual command)
```

Pitch (elevator) and total throttle are held at the cruise trim values and **never** updated by the agent. The agent only fights the lateral-directional disturbance via aileron + rudder.

## Why this is non-trivial

* The dead engine produces a **persistent yaw disturbance** ($N_\text{thrust} = -y_1 \cdot T_1 \approx -593\,000$ ft·lb) that pushes the nose toward the dead side.
* B-747 is heavy ($I_z = 49.7 \times 10^6$ slug·ft²); the divergence is slow but **unbounded** — open-loop $\psi$ exceeds $80°$ in 50 s.
* The required steady-state rudder is small but non-zero (~1.8° opposite-side rudder); the actor must learn a constant offset, not just a transient response.

## Hyperparameters

| Setting | Value | Why |
|---|---:|---|
| `actor_hidden`, `critic_hidden` | (32, 32) | Larger than the F-16 default — the post-damage policy must learn a non-zero offset, not just a small linear gain. |
| `Q` | `[100, 1, 5, 0.2]` | $\psi$ is the primary objective ⇒ heaviest weight. $\varphi$ matters less but should not run away. Rates have small weights. |
| `R` | `[0.5, 0.5]` | Light penalty on control effort — under engine-out, large rudder offsets are physically necessary. |
| `u_bound` | 8 deg | Actor saturates at $\pm 8°$. The B-747 has $\pm 25°$ rudder authority but the steady-state value is $\sim$ 1.8°. |
| `rho` | 0.05 | Lipschitz constant of the event trigger — lower than F-16 examples because the trigger should fire often during the long persistent-disturbance phase. |
| `trigger_floor` | 0.5 deg | Floor on the trigger threshold so $\tilde x \to 0$ doesn't lock out updates. |
| `num_epochs_per_trigger` | 10 | More inner-loop iterations per trigger ⇒ faster learning of the steady-state offset. |

## Pipeline

### 1. Plant identification (healthy aircraft, short bursts)

40 short 3-second bursts of multi-sine excitation on aileron + rudder, fresh env reset between each. This keeps the state bounded near the linearisation point and gives a clean MSE around $10^{-5}$ (per-component, in degree units).

```python
N_BURST, N_BURSTS = 60, 40
for burst in range(N_BURSTS):
    env_id = make_env(damage_profile=None, n_steps=N_BURST + 5)
    obs, _ = env_id.reset()
    for t in range(N_BURST):
        # multi-sine PE on da, dr (deg)
        da_deg = 1.5 * np.sin(2*np.pi*f_a*t*DT + phase_a) + 0.4 * rng.normal()
        dr_deg = 1.5 * np.sin(2*np.pi*f_r*t*DT + phase_r) + 0.4 * rng.normal()
        ...
```

### 2. Closed-loop training under damage (8 episodes)

The agent trains on the *damaged* plant from episode 1. Healthy training would see no disturbance and the trigger would not fire — the actor must have a persistent error signal to learn the steady-state offset.

```python
agent = ETDHPAgent(n_state=4, n_control=2, state_transform=state_transform, config=cfg)
agent.fit_plant_model(states_arr, actions_arr, next_states_arr,
                      batch_size=128, verbose=False)

for ep in range(8):
    log = run_episode(LEFT_OUTER_ENGINE_FAILURE, learn=True)
    # ...
```

Typical training curve:

| Episode | Late-half RMSE ψ | max \|ψ\| post-damage | Final rudder |
|---:|---:|---:|---:|
| 1 | 26.7° | 43.2° | −0.0° |
| 2 | 0.65° | 0.74° | −1.75° |
| 3 | 0.44° | 0.48° | −1.72° |
| ... | ... | ... | ... |
| 8 | 0.26° | 0.28° | −1.80° |

By episode 2 the actor has discovered the engine-out compensation.

### 3. Final evaluation

| Metric | Value |
|---|---:|
| Late-half MAE ψ | **0.26°** |
| Late-half RMSE ψ | 0.26° |
| Peak \|ψ\| post-damage | 0.28° |
| Steady-state rudder | −1.80° (right rudder per CR-2144 sign) |
| Steady-state aileron | −0.22° |

## Open-loop vs. ET-DHP

The same scenario with the agent **disabled** (zero aileron/rudder, just trim elevator + trim throttle):

| Time | Open-loop ψ | ET-DHP ψ |
|---:|---:|---:|
| t = 10 s (damage) | 0° | 0° |
| t = 30 s | −20° | −0.3° |
| t = 60 s | **−85.5°** | **−0.28°** |

The aircraft yaws unchecked into the dead-engine side without the agent; with ET-DHP, the heading deviation stays under $0.3°$ for the entire post-damage phase.

## What the agent learned

The actor's converged steady-state output (~ −1.8° rudder, small aileron) closely matches the analytical estimate from $\Delta N_\text{thrust} / (q_\text{dyn}\, S\, b\, C_{n_{\delta_r}}) \approx 1.6°$ at FL200 cruise. The agent re-discovered the canonical engine-out compensation by minimising a quadratic cost — without an explicit controller-design step.

**Sign of the steady-state rudder.** The CR-2144 derivative bank uses $C_{n_{\delta_r}} < 0$ — positive rudder produces *negative* yaw moment. After engine #1 fails ($N_\text{thrust} < 0$, nose-left), the rudder must add positive aerodynamic $N$ to cancel the thrust moment, so $\delta_r < 0$. The agent's negative steady-state rudder is consistent with that algebra.

## Notes & extensions

* **Pitch and altitude.** Holding throttle at the all-engines-healthy trim with one engine out gives a $\sim$ 25 % thrust deficit; the aircraft slowly descends. A practical FTC stack would close pitch and throttle channels too — for example by coupling this lateral controller with the [IHDP pitch tracker](../ihdp/example_ihdp_nonlinear_b747.md) and a simple PI on airspeed.
* **Other engine scenarios.** Replace `LEFT_OUTER_ENGINE_FAILURE` with `LEFT_TWO_ENGINES_OUT` for the maximum-asymmetry case (≈ 50 % thrust). The steady-state rudder will roughly double; you may need a higher `u_bound` (e.g. 12°) and to reduce `Q[2]` to let the aircraft fly with a small natural bank toward the dead engines.
* **No-rudder ablation.** Removing the rudder channel forces yaw control via the dihedral effect of bank, which requires several seconds of bank build-up. Useful as a stress test.

## See also

* [Boeing 747-100 (Nonlinear 6-DoF)](../../../model/b747_nonlinear.md) — model overview, damage subsystem, asymmetric-thrust formula.
* [B-747 usage examples](../../../model/b747_examples.md) — recipes for engine-out and flap-jam scenarios (#8–#12).
* [IHDP on the nonlinear B-747](../ihdp/example_ihdp_nonlinear_b747.md) — pitch step tracking; complements this lateral-directional example.
* [ET-DHP on the nonlinear F-16](example_etdhp_nonlinear.md) — sinusoidal $\alpha$ tracking with the same agent.
