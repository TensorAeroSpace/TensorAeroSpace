# Active-Adaptive Incremental Nonlinear Dynamic Inversion (AA-INDI)

AA-INDI is a **fault-tolerant flight controller** built on top of Incremental Nonlinear Dynamic Inversion (INDI). It combines a classical INDI control law with online **Variable-Forgetting-Factor RLS** identification of the control-effectiveness matrix so that the controller adapts quickly to actuator faults, and a lightweight sensor-filter surrogate that mimics the OTSEKF-HOSM branch of the reference paper. See also the nonlinear F-16 model: [NonlinearLongitudinalF16](../model/f16_nonlinear_longitudinal.md).

**Reference**: Sun et al., *"Active Incremental Nonlinear Dynamic Inversion for Sensor and Actuator Fault Diagnosis and Fault-Tolerant Flight Control"*, TU Delft Aerospace, [research.tudelft.nl](https://research.tudelft.nl/en/publications/active-incremental-nonlinear-dynamic-inversion-for-sensor-and-act/).

## Key ideas

- **INDI control law**: the applied control increment \(\Delta u = G^+ \cdot (\nu_{\text{des}} - \dot{\omega}_{\text{meas}})\) requires only the control-effectiveness matrix \(G\), not the full nonlinear dynamics \(f\). This eliminates model-uncertainty sensitivity.
- **Reference model**: a second-order filter shapes the commanded angular rate into a smooth desired rate and its derivative \(\nu_{\text{des}} = \dot{\omega}_{\text{ref}}\).
- **VFF-RLS**: the forgetting factor \(\lambda_k\) contracts toward a lower bound when the prediction residual grows (fast adaptation during faults/manoeuvres) and relaxes toward the upper bound in quiet operation (noise rejection).
- **Sensor-filter surrogate**: a low-pass differentiator produces \(\dot{\omega}\) from raw \(\omega\), and a residual-based bias estimator yields a coarse IMU bias that the agent subtracts from measurements — a minimal stand-in for the paper's OTSEKF-HOSM stack.

## Differences from related methods

| Aspect | INDI | Adaptive INDI | **AA-INDI** |
| --- | --- | --- | --- |
| Control-effectiveness \(G\) | Offline / fixed | Online (basic RLS) | Online VFF-RLS |
| Sensor fault handling | None | None | Bias estimator (OTSEKF-HOSM surrogate) |
| Reaction to abrupt faults | Poor | Moderate | Fast (λ contracts under large residuals) |
| Noise rejection in nominal flight | Good | Moderate | Good (λ relaxes to max) |

## AA-INDI components

| Component | Role | Implementation |
| --- | --- | --- |
| VFFRLSEstimator | Online identification of \(G = \partial \dot{\omega}/\partial u\) with variable forgetting | `tensoraerospace.agent.aa_indi.VFFRLSEstimator` |
| LowPassDerivative | Causal differentiator (HOSM surrogate) | `tensoraerospace.agent.aa_indi.LowPassDerivative` |
| BiasEstimator | Exponential-forgetting IMU-bias estimator | `tensoraerospace.agent.aa_indi.BiasEstimator` |
| Reference model | 2nd-order filter for \(\nu_{\text{des}}\) | Inline in `AAINDIAgent` |
| AAINDIAgent | Orchestrates INDI law, estimators, filter | `tensoraerospace.agent.aa_indi.AAINDIAgent` |

## Algorithm

On each control tick \(k\), given the measurement \(\omega_k\) and command \(r_k\):

1. **Measurement conditioning.** Subtract the current bias estimate (if enabled): \(\omega_k^c = \omega_k - \hat{b}\). The low-pass differentiator yields \(\dot{\omega}_k^{\text{meas}}\) (advanced inside `learn()` to avoid double-stepping).
2. **Reference model.** Second-order filter:

\[
\ddot{r} = -2\zeta\omega_n \dot{r} + \omega_n^2 (r_{\text{cmd}} - r), \qquad \nu_{\text{des}} = \dot{r}.
\]

3. **INDI law.**

\[
\Delta u = G^{+} \cdot (\nu_{\text{des}} - \dot{\omega}^{\text{meas}}), \qquad
u = \mathrm{clip}(u_{\text{prev}} + \Delta u,\ \pm u_{\max}),
\]

   with \(\Delta u\) first rate-limited to \(\pm\dot{u}_{\max} \cdot dt\).
4. **VFF-RLS update.** From \((\Delta u_k, \Delta \dot{\omega}_k)\):

\[
\varepsilon = \Delta \dot{\omega} - \theta^{\top} \Delta u,\qquad
\lambda_k = \mathrm{clip}\bigl(e^{-\|\varepsilon\|^2/\sigma_\varepsilon^2},\ \lambda_{\min},\ \lambda_{\max}\bigr),
\]

   followed by the usual RLS gain / covariance recursion with forgetting factor \(\lambda_k\).
5. **Bias update.** Exponential moving average of the residual between \(\omega\) and its reintegration from \(\dot{\omega}\).

## Quick start

```python
import numpy as np
from tensoraerospace.agent.aa_indi import AAINDIAgent, AAINDIConfig

# Onboard model snapshot of the control-effectiveness matrix at design trim.
G_init = np.array([[-2.0, 0.1, 0.0],
                   [0.05, -1.5, 0.2],
                   [0.0,  0.05, -0.9]])

cfg = AAINDIConfig(
    dt=0.01,
    ref_wn=5.0,
    ref_zeta=0.7,
    u_magnitude_limit=25.0,
    u_rate_limit=200.0,
    vff_forgetting_min=0.9,
    vff_forgetting_max=0.999,
    vff_eps_sensitivity=2.0,
    sensor_cutoff_hz=50.0,
    enable_bias_correction=True,
    G_init=G_init,
    seed=0,
)
agent = AAINDIAgent(n_state=3, n_control=3, config=cfg)

omega = np.zeros(3)
ref = np.array([0.2, -0.1, 0.05])  # rad/s targets for roll/pitch/yaw rates

for k in range(500):
    u = agent.predict(omega, ref, k)
    # Plant step (placeholder — plug your environment here)
    omega = omega + cfg.dt * (G_init @ u)
    metrics = agent.learn(omega, ref, k)
```

!!! tip "Warm-start `G_init` matters"
    INDI needs a reasonable \(G\) on the first few ticks — with the default random init, the pseudo-inverse explodes and the actuator saturates before VFF-RLS has converged. Provide `G_init` from a linearised on-board model.

## Hyperparameters

### Reference model

| Parameter | Default | Description |
| --- | --- | --- |
| `ref_wn` | 10.0 | Natural frequency of the reference filter (rad/s). Higher → faster tracking, larger Δu. |
| `ref_zeta` | 0.7 | Damping ratio. 0.7 gives a critically-damped-ish response. |

### Actuator bounds

| Parameter | Default | Description |
| --- | --- | --- |
| `dt` | 0.01 | Control step (s) |
| `u_magnitude_limit` | 25.0 | Hard magnitude clamp per channel (same units as env action) |
| `u_rate_limit` | 60.0 | Max Δu per second per channel |
| `pinv_rcond` | 1e-6 | Cutoff for `np.linalg.pinv(G)` |
| `G_init` | None | Warm-start of shape `(n_state, n_control)` |

### VFF-RLS

| Parameter | Default | Description |
| --- | --- | --- |
| `vff_forgetting_min` | 0.7 | Lower bound on λ — fast-adaptation regime |
| `vff_forgetting_max` | 0.999 | Upper bound on λ — noise-rejection regime |
| `vff_eps_sensitivity` | 1.0 | Residual norm at which λ drops ~1/e |
| `vff_cov_init` | 1e2 | Initial covariance scale |

### Sensor filter

| Parameter | Default | Description |
| --- | --- | --- |
| `sensor_cutoff_hz` | 10.0 | Low-pass cutoff of the differentiator |
| `bias_forgetting` | 0.99 | EMA retention of the bias estimator |
| `enable_bias_correction` | True | Subtract bias estimate from ω before forming the INDI residual |

## Supported environments

- Any Gymnasium env whose observation vector contains measurable angular rates (e.g. `[alpha, wz]` in `NonlinearLongitudinalF16-v0` after light shaping, or a full `[p, q, r]` vector from a 6-DoF plant).

## Persistence

Same API as the other adaptive-critic agents:

```python
run_dir = agent.save("./checkpoints")        # creates <date>_AAINDIAgent/
restored = AAINDIAgent.from_pretrained(run_dir)
agent.publish_to_hub("me/my-aaindi", folder_path=run_dir, access_token="hf_...")
```

Saved artefacts:

- `config.json` — full `AAINDIConfig` + `n_state` / `n_control`.
- `vff_rls.npz` — RLS `θ`, covariance `P`, last forgetting factor `λ`, update counter.
- `bias_state.npz` — exponential bias estimate.
- `deriv_state.npz` — low-pass differentiator state.
- `loop_state.npz` — reference-model state, PI integrator, last applied control, cached `ω̇`. Persisting these means a mid-episode save resumes bit-identically on reload (essential when `ref_error_kp` / `ref_error_ki` are non-zero).

## API reference

::: tensoraerospace.agent.aa_indi.model.AAINDIAgent

::: tensoraerospace.agent.aa_indi.model.AAINDIConfig

::: tensoraerospace.agent.aa_indi.vff_rls.VFFRLSEstimator

::: tensoraerospace.agent.aa_indi.sensor_filter.LowPassDerivative

::: tensoraerospace.agent.aa_indi.sensor_filter.BiasEstimator

## Sources

- Sun et al. *"Active Incremental Nonlinear Dynamic Inversion for Sensor and Actuator Fault Diagnosis and Fault-Tolerant Flight Control"*, TU Delft Aerospace, [research.tudelft.nl](https://research.tudelft.nl/en/publications/active-incremental-nonlinear-dynamic-inversion-for-sensor-and-act/).
- Smeur, Chu, de Croon. *"Adaptive Incremental Nonlinear Dynamic Inversion for Attitude Control of Micro Air Vehicles"*, J. Guid. Control Dyn., 2016.
- Fortescue, Kershenbaum, Ydstie. *"Implementation of Self-Tuning Regulators with Variable Forgetting Factors"*, Automatica, 1981.
