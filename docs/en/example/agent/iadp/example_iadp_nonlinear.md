# Example: iADP on the nonlinear F-16 — pitch-rate tracking with fault injection

This example runs the [**iADP**](../../../agent/iadp.md) agent on a sinusoidal pitch-rate command using the [nonlinear F-16 longitudinal model](../../../model/f16_nonlinear_longitudinal.md), and shows the controller recovering after a mid-episode 50 % elevator effectiveness loss. Source notebook: `example/reinforcement_learning/example_iadp_nonlinear_f16.ipynb`.

## Idea

iADP combines an **online RLS identifier** for the incremental plant model \(\Delta X_{t+1} \approx \tilde{F} \Delta X_t + \tilde{G} \Delta\delta_t\) with a **batch least-squares policy evaluator** of the quadratic cost-to-go \(V_\pi(X_t) = X_t^T \tilde{P} X_t\). The policy is the closed-form minimiser of the Bellman right-hand side (paper eq. (11)):

\[
\Delta\delta_t = -(R + \gamma\,\tilde{G}^T \tilde{P} \tilde{G})^{-1}\bigl[R\,\delta_{t-1} + \gamma\,\tilde{G}^T \tilde{P} X_t + \gamma\,\tilde{G}^T \tilde{P} \tilde{F} \Delta X_t\bigr].
\]

Because the policy is evaluated against the *identified* model, a sudden halving of the real actuator effectiveness is absorbed by RLS updating \(\tilde{G}\) and the LS re-fitting \(\tilde{P}\) — no explicit fault-detection logic, no re-tuning.

## 1. Imports and trim

```python
import math
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import fsolve

import tensoraerospace
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import f16_ode_long
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params import default_parameters
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig

dt = 0.01
params = default_parameters()

def trim_residual(z):
    alpha, stab = z
    x = np.array([alpha, 0.0, stab, 0.0])
    return list(f16_ode_long(x, np.array([stab]), 0.0, params)[:2])

sol, *_ = fsolve(trim_residual, x0=[math.radians(2.0), math.radians(-2.0)], full_output=True)
alpha_trim_rad, stab_trim_rad = float(sol[0]), float(sol[1])
# global trim: α = +4.918°, stab = -4.447°
```

## 2. Env builder

```python
def make_env(n_steps):
    env = gym.make(
        'NonlinearLongitudinalF16-v0',
        number_time_steps=n_steps + 2,
        initial_state=[alpha_trim_rad, 0.0, stab_trim_rad, 0.0],
        reference_signal=np.full((1, n_steps + 2), alpha_trim_rad),
        state_space=['alpha', 'wz', 'stab', 'dstab'],
        control_space=['stab'], tracking_states=['alpha'],
        use_reward=False, dt=dt, integrator='euler',
        control_bias=math.degrees(stab_trim_rad),
    ).unwrapped
    env.reset()
    return env
```

## 3. Warm-start `F_init`, `G_init` via a short PE excitation

iADP's policy (eq. (11)) is ill-defined until \(\tilde{G}\) has a non-trivial estimate — with \(\tilde{G} \approx 0\) the control law collapses to \(\Delta\delta_t \approx -\delta_{t-1}\), pinning the actuator and starving the RLS of Δ-signal. We bootstrap with a short multi-sine excitation and fit a discrete-time scalar form \(\Delta\omega_{z,t+1} \approx F \Delta\omega_{z,t} + G \Delta\delta_t\).

```python
N_PE = 300
env_pe = make_env(N_PE); obs, _ = env_pe.reset()
wz_hist, u_hist = [float(obs[1])], [0.0]
for t in range(N_PE):
    u = 2.0*math.sin(2*math.pi*0.7*t*dt) + 1.0*math.sin(2*math.pi*1.5*t*dt)
    obs, *_ = env_pe.step(np.array([u]))
    wz_hist.append(float(obs[1])); u_hist.append(float(u))

wz = np.asarray(wz_hist); us = np.asarray(u_hist)
dwz, du = np.diff(wz), np.diff(us)
A_pe = np.column_stack([dwz[:-1], du[:-1]])
F_wz, G_wz = np.linalg.lstsq(A_pe, dwz[1:], rcond=None)[0]

F_init = np.array([[F_wz, 0.0], [0.0, 1.0]])       # reference row is stationary
G_init = np.array([[G_wz], [0.0]])                 # elevator drives only ω_z row
# PE: F_wz ≈ 1.00, G_wz ≈ -0.0014 (per °, discrete-time)
```

## 4. Warm-starting `P_init` from the discrete-LQT DARE

The batch-LS policy evaluator fits \(\tilde{P}\) incrementally from windows of transitions. On clean synthetic data it converges to the infinite-horizon LQT solution, but on a real flight trajectory the LS is biased by the finite window and starved of cross-information between \(x\) and \(x_r\) when the reference stays near a single level — so the policy works but stays consistently a few percent under the optimal gain.

We sidestep this by computing the analytical **Discrete Algebraic Riccati Equation (DARE)** solution once, offline, from the warm-start model, and using it as `P_init`. The discount \(\gamma\) enters by the standard substitution \(\bar{F} = \sqrt{\gamma}F,\, \bar{G} = \sqrt{\gamma}G\) — so `scipy.linalg.solve_discrete_are` applies directly:

```python
Q_val = 30_000.0       # tracking-error weight
R_val = 0.1            # control weight
gamma = 0.9            # Bellman discount

# Cost (wz - wz_ref)^2 · Q  =  X' · Q_aug · X on the augmented state.
Q_aug = Q_val * np.array([[1.0, -1.0], [-1.0, 1.0]])
R_aug = np.array([[R_val]])

P_dare = solve_discrete_are(
    np.sqrt(gamma) * F_init, np.sqrt(gamma) * G_init,
    Q_aug, R_aug,
)
```

With a well-seeded \(\tilde{P}\) the online LS only has to *keep* \(\tilde{P}\) current as the model drifts, not find it from scratch — baseline RMSE on this plant drops from ≈ 0.13 °/s (ad-hoc block initial) to ≈ 0.09 °/s.

## 5. Soft update for \(\tilde{P}\): `policy_eval_blend`

By default the batch LS *replaces* \(\tilde{P}\) every `policy_eval_every` ticks. On a real plant this is a small **step change**, and because the policy depends linearly on \(\tilde{P}\), the commanded elevator shows a visible **sawtooth** at every update tick. On this demo the peak per-tick jump without blending is ≈ 0.14° — very visible on the elevator trace.

`policy_eval_blend` mixes the fresh LS solution into the incumbent via an exponential moving average,

\[
\tilde{P} \leftarrow \beta\,\tilde{P}_\text{solved} + (1-\beta)\,\tilde{P}_\text{prev}.
\]

With a small \(\beta\) and a shorter `policy_eval_every`, \(\tilde{P}\) becomes a smoothed filter of the LS solution instead of a series of step changes — analogous to the *soft-target* trick used elsewhere in RL. Two immediate benefits on this plant:

- **Control smoothness.** Peak per-tick \(|\Delta\delta|\) drops by ~14× (from 0.14° to 0.01°) — no visible sawtooth on the elevator trace.
- **Higher usable gain.** The same `Q` that previously triggered the "replace then saturate" instability feedback becomes safe. We can push `Q` from 10 000 up to 30 000, which tightens tracking by another ~25 %.

## 6. iADP harness

```python
def run_iadp(wz_cmd, n_steps, *, fault_gain=1.0, fault_at_step=None):
    cfg = IADPConfig(
        dt=dt,
        Q=np.array([[Q_val]]), R=np.array([[R_val]]),
        gamma=gamma, gamma_rls=0.9999, phi_init=1.0,
        policy_eval_window=300, policy_eval_every=5,
        policy_eval_warmup_updates=20,
        policy_eval_regularization=1e-10,
        policy_eval_blend=0.10,        # EMA soft-update of P̃
        F_init=F_init, G_init=G_init,
        P_init=P_dare,
        u_magnitude_limit=8.0, u_rate_limit=200.0,
        seed=0,
    )
    agent = IADPAgent(n_state=1, n_control=1, config=cfg)
    env = make_env(n_steps); obs, _ = env.reset()
    logs = {k: [] for k in ('wz', 'u_agent', 'u_applied', 'G_est', 'residual')}
    current_gain = 1.0
    for k in range(n_steps):
        if fault_at_step is not None and k >= fault_at_step:
            current_gain = fault_gain
        u_agent = agent.predict(np.array([float(obs[1])]),
                                np.array([wz_cmd[k]]), k)
        u_applied = u_agent * current_gain
        obs, *_ = env.step(u_applied)
        m = agent.learn(np.array([float(obs[1])]),
                        np.array([wz_cmd[k]]), k)
        logs['wz'].append(float(obs[1]))
        logs['u_agent'].append(float(u_agent[0]))
        logs['u_applied'].append(float(u_applied[0]))
        logs['G_est'].append(float(agent.G[0, 0]))
        logs['residual'].append(m['rls_pred_error_norm'])
    return {k: np.asarray(v) for k, v in logs.items()}
```

!!! warning "Scale `policy_eval_regularization` to the state magnitude"
    For angular rates in rad/s (≈ 0.01), the outer-product regressor entries are \(\sim 10^{-4}\). The default ridge `1e-4` would dominate the normal equations and collapse \(\tilde{P}\) toward zero — the policy then outputs almost nothing. We use `1e-10` here; the `enforce_psd` projection keeps \(\tilde{P}\) well-behaved.

## 7. Baseline — sinusoidal pitch-rate tracking

```python
N = 1800
t_arr = np.arange(N) * dt
wz_cmd = math.radians(0.8) * np.sin(2*math.pi*0.12*t_arr)  # 0.8 °/s amplitude, T ≈ 8.3 s
baseline = run_iadp(wz_cmd, N)
# Late-window RMSE (t ≥ 5 s):  ≈ 0.072 °/s  (9.0 % of amplitude)
# Peak per-tick |Δδₑ|:         ≈ 0.010 °
# Final G̃:                     ≈ -0.00013
```

The policy tracks a \(\pm 0.8\) °/s sinusoid at the F-16 trim point with a late-window RMSE around **0.07 °/s** (≈ 9 % of amplitude). The commanded elevator stays well inside \(\pm 0.5\)° and moves by at most **0.01° per tick** — the sawtooth from the pre-blend version is gone.

## 8. Fault injection — 50 % elevator loss at t = 10 s

```python
fault_step = int(10.0 / dt)
fault = run_iadp(wz_cmd, N, fault_gain=0.5, fault_at_step=fault_step)
# Pre-fault  RMSE (5 s ≤ t < 10 s):  ≈ 0.090 °/s
# Post-fault RMSE (11 s ≤ t ≤ 18 s): ≈ 0.098 °/s
```

At \(t = 10\,\text{s}\) the env multiplies the agent's command by `0.5` before the plant sees it. iADP's policy consults the *identified* \(\tilde{G}\); as the RLS residual rises and \(\tilde{G}\) shifts, the closed-form eq. (11) scales up \(\Delta\delta\) to compensate. Tracking degrades only mildly during the transient and holds after RLS settles.

| Metric | Baseline | With fault |
|---|---|---|
| Late-window RMSE (ω_z) | ≈ 0.07 °/s | ≈ 0.10 °/s (post) |
| Peak \|Δδ\| (agent) | ≈ 0.5 ° | ≈ 0.5 ° |
| Peak per-tick \|Δδ\| | ≈ 0.01 ° | ≈ 0.01 ° |
| Behaviour at fault | n/a | RLS residual spike, G̃ shift, tracking resumes |

## Notes

- **Warm-start matters.** A random \(\tilde{G}\) would either produce the \(\Delta\delta \to -\delta_{t-1}\) collapse or drive the actuator into a limit cycle. A 2–3 s offline PE excitation (as here) or an onboard linearisation in a real deployment is sufficient.
- **DARE warm-start for `P_init`** is the single biggest accuracy win after the basic (Q, R, γ) tune. It puts the policy near-optimal from step 0 instead of waiting for the online LS to settle.
- **Soft update (`policy_eval_blend`)** removes the sawtooth that would otherwise appear on the commanded elevator at every batch-LS tick, and lets us push the tracking weight `Q` 3× higher without destabilising the control loop. Both the RMSE and the control smoothness improve together.
- **Policy-eval regularisation scales with state.** At airspeeds or angles in the O(1) range the library default `1e-4` is fine; at angular rates in rad/s it must be reduced several orders of magnitude.
- **No outer-loop PI.** Unlike AA-INDI (which closes a PI around the INDI inner loop), iADP's quadratic cost-to-go *is* the outer loop — the `(Q, R, γ)` knobs set the tracking vs control-effort trade-off directly.
- **Residual ~9 %** of amplitude is the intrinsic phase-lag floor of iADP's proportional-plus-\(\delta^2\)-penalty policy against this 0.12 Hz sinusoid. Going lower requires extensions of the paper's formulation (feedforward, reference-derivative augmentation, integral terms).

## See also

- [iADP documentation](../../../agent/iadp.md) — theory, full API, hyperparameter reference.
- [AA-INDI on the nonlinear F-16](../aa_indi/example_aaindi_nonlinear.md) — INDI alternative, same plant + fault profile.
- [IM-GDHP on the nonlinear F-16](../imgdhp/example_imgdhp_nonlinear.md) — GDHP-based incremental approach.
- [ET-DHP on the nonlinear F-16](../et_dhp/example_etdhp_nonlinear.md) — event-triggered DHP.
