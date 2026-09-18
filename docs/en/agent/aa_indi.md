# Active-Adaptive Incremental Nonlinear Dynamic Inversion (AA-INDI)

AA-INDI combines incremental dynamic inversion with online identification of control effectiveness. This implementation uses VFF-RLS and a first-order filtered differentiator. Its residual smoother is a heuristic, not the paper's OTSEKF-HOSM sensor-fault estimator. Tracking and fault recovery depend on excitation, tuning, actuator dynamics and the initial estimate of effectiveness. See [NonlinearLongitudinalF16](../model/f16_nonlinear_longitudinal.md).

**Reference**: Atmaca, de Visser, van Kampen (2026), *"Active Incremental Nonlinear Dynamic Inversion for Sensor and Actuator Fault-Tolerant Control"*, TU Delft Aerospace, [research.tudelft.nl](https://research.tudelft.nl/en/publications/active-incremental-nonlinear-dynamic-inversion-for-sensor-and-act/).

## Key ideas

- **INDI control law**: the applied control increment \(\Delta u = G^+ \cdot (\nu_{\text{des}} - \dot{\omega}_{\text{meas}})\) requires only the control-effectiveness matrix \(G\), not the full nonlinear dynamics \(f\). This reduces dependence on the full model; effectiveness errors and delays still affect tracking.
- **Reference model**: a second-order filter shapes the commanded angular rate into a smooth desired rate and its derivative \(\nu_{\text{des}} = \dot{\omega}_{\text{ref}}\).
- **VFF-RLS**: the forgetting factor \(\lambda_k\) contracts toward a lower bound when the prediction residual grows (fast adaptation during faults/manoeuvres) and relaxes toward the upper bound in quiet operation (noise rejection).
- **Sensor-filter surrogate**: a low-pass differentiator produces \(\dot{\omega}\) from raw \(\omega\), and a residual smoother supplies an optional heuristic correction. Constant sensor bias is not observable from this reintegration residual alone.

## Differences from related methods

| Aspect | INDI | Adaptive INDI | **AA-INDI** |
| --- | --- | --- | --- |
| Control-effectiveness \(G\) | Offline / fixed | Online (basic RLS) | Online VFF-RLS |
| Sensor fault handling | None | None | Residual heuristic; constant bias is unobservable without an independent reference |
| Adaptation after faults | Fixed effectiveness | RLS updates | Variable forgetting; recovery must be measured |
| Noise handling | Measurement filtering | Filtering and RLS tuning | Matched input/output filters and VFF tuning |

## AA-INDI components

| Component | Role | Implementation |
| --- | --- | --- |
| VFFRLSEstimator | Online identification of \(G = \partial \dot{\omega}/\partial u\) with variable forgetting | `tensoraerospace.agent.aa_indi.VFFRLSEstimator` |
| LowPassDerivative | Causal differentiator (HOSM surrogate) | `tensoraerospace.agent.aa_indi.LowPassDerivative` |
| BiasEstimator | Exponential mean of a supplied innovation | `tensoraerospace.agent.aa_indi.BiasEstimator` |
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
u = \mathrm{clip}(u_{\text{filtered}} + \Delta u,\ \pm u_{\max}),
\]

   Here the baseline is filtered actuator feedback. Rate limiting is applied to the candidate command relative to the previous actual input, with a limit of \(\dot{u}_{\max} dt\).
4. **VFF-RLS update.** From \((\Delta u_k, \Delta \dot{\omega}_k)\):

\[
\varepsilon = \Delta \dot{\omega} - \theta^{\top} \Delta u,\qquad
\varphi_k = \Delta u_k,\qquad K_k = \frac{P_k\varphi_k}{1+\varphi_k^T P_k\varphi_k},\qquad
\lambda_k = \mathrm{clip}\left(1-\frac{\|\varepsilon\|^2}{\sigma_\varepsilon^2(1+\varphi_k^T P_k\varphi_k)},\lambda_{\min},\lambda_{\max}\right),
\]

   followed by the usual RLS gain / covariance recursion with forgetting factor \(\lambda_k\).
5. **Bias update.** Exponential moving average of the residual between \(\omega\) and its reintegration from \(\dot{\omega}\).

## Relation to the original paper

The VFF gain and forgetting rule follow Eqs. (54)–(57) of
[Atmaca et al., AIAA 2026-1743](https://repository.tudelft.nl/file/File_ee9931f5-cf45-45a5-b5a3-0225b0f35da2).
`vff_eps_sensitivity**2` corresponds to Σ₀. The upper limit
`vff_forgetting_max < 1` is a library extension; set it to 1 to allow the
paper's maximum. Covariance uses an algebraically equivalent Joseph update
to avoid cancellation. The previous exponential rule was not Eq. (55).
Existing checkpoints load, but their adaptation tuning should be revalidated.

This agent uses filtered increments of acceleration and actuator position.
The paper instead reconstructs aerodynamic moments and fits surface derivatives;
it also includes OTSEKF-HOSM. These subsystems are not reproduced here.
An experiment on this class therefore does not establish the performance of
the complete published AA-INDI architecture.

Nonfinite samples and numerical overflow are rejected before changing RLS
parameters. Unexcited directions still follow the configured forgetting law:
this numerical guard does not solve covariance windup or guarantee stability.

## Measurement and actuator timing

Call `predict(measurement, reference, k)`, step the plant, then call
`learn(next_measurement, reference, k, applied_action=actual_input)` once.
The next measurement must also be the next call's current measurement.
Both commands and feedback use the same control units; remove the same trim
bias from both. Omitting `applied_action` assumes exact command tracking.

`LinearLongitudinalB747` and `LinearLongitudinalLAPAN` expose the actual,
rate-limited elevator in **degrees** through `info["applied_action"]`.
For a continuous servo, use measured surface motion over the transition;
the requested command is not the surface position.

The first prediction primes the differentiator with the initial measurement.
The actuator feedback uses the same low-pass filter as the acceleration;
RLS operates on their filtered increments. New checkpoints preserve both
filters, previous measurements and a pending command. Older checkpoints load,
but their missing filter/history state requires identification warm-up.

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
| `ref_zeta` | 0.7 | Damping ratio. 0.7 is underdamped; 1 is critically damped. |

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
| `vff_eps_sensitivity` | 1.0 | Square root of Σ₀ in Eq. (55) |
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

- Atmaca, de Visser, van Kampen (2026). *"Active Incremental Nonlinear Dynamic Inversion for Sensor and Actuator Fault-Tolerant Control"*, TU Delft Aerospace, [research.tudelft.nl](https://research.tudelft.nl/en/publications/active-incremental-nonlinear-dynamic-inversion-for-sensor-and-act/).
- Smeur, Chu, de Croon. *"Adaptive Incremental Nonlinear Dynamic Inversion for Attitude Control of Micro Air Vehicles"*, J. Guid. Control Dyn., 2016.
- Fortescue, Kershenbaum, Ydstie. *"Implementation of Self-Tuning Regulators with Variable Forgetting Factors"*, Automatica, 1981.
