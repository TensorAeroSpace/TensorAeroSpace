# Incremental Approximate Dynamic Programming (iADP)

iADP is an **online adaptive reinforcement-learning flight-control law** built on top of an incremental plant model. It combines

- **online incremental plant identification** via recursive least squares (RLS),
- an **approximate quadratic cost-to-go** \(V_\pi(X_t) = X_t^T \tilde{P} X_t\) fitted by batch least-squares to Bellman residuals, and
- a **closed-form policy improvement** derived from Bellman's optimality principle.

Because the model is identified online, the controller is **model-free** at deployment and tolerates vehicle configuration changes, actuator degradation, and unmodelled aerodynamics. Flight-tested on the Cessna Citation II (PH-LAB) at TU Delft / DLR; see the F-16 nonlinear model: [NonlinearLongitudinalF16](../model/f16_nonlinear_longitudinal.md).

**Reference**: Konatala, Milz, Weiser, Looye, van Kampen, *"Flight Testing Reinforcement Learning based Online Adaptive Flight Control Laws on CS-25 Class Aircraft"*, AIAA SCITECH 2024, [DOI 10.2514/6.2024-2402](https://doi.org/10.2514/6.2024-2402).

## Key ideas

- **Incremental model.** Linearising the nonlinear plant around the latest operating point gives \(\Delta X_{t+1} \approx \tilde{F}_t \Delta X_t + \tilde{G}_t \Delta\delta_t\). Stacking \(\tilde{\Theta}_t = [\tilde{F}_t; \tilde{G}_t]^T\) and \(W_t = [\Delta X_t; \Delta \delta_t]\), the recursive update is a standard fixed-forgetting RLS driving \(\tilde{\Theta}^T W \to \Delta X_{t+1}\).
- **Quadratic value function.** Following the linear-quadratic tracking (LQT) relaxation, \(V_\pi(X) = X^T \tilde{P} X\) with the augmented state \(X_t = [x_t; x_t^r]\) stacking system and reference dynamics — this makes the value function applicable across the entire reachable set rather than only visited samples.
- **Batch LS policy evaluation.** On a sliding window of transitions, every sample contributes one scalar Bellman equation \((X_t \otimes X_t)^T \mathrm{vec}(\tilde{P}^{j+1}) = c_t + \gamma X_{t+1}^T \tilde{P}^{j} X_{t+1}\); stacking and solving gives a new kernel matrix, which is symmetrised and optionally projected to the PSD cone for stability.
- **Closed-form improved policy.** Minimising the Bellman RHS with respect to \(\Delta \delta_t\) gives paper eq. (11) — no neural network, no gradient descent: \(\Delta \delta_t = -(R + \gamma \tilde{G}^T \tilde{P} \tilde{G})^{-1} [R \delta_{t-1} + \gamma \tilde{G}^T \tilde{P} X_t + \gamma \tilde{G}^T \tilde{P} \tilde{F} \Delta X_t]\).

## Differences from related methods

| Aspect | IHDP | IM-GDHP | ET-DHP | **iADP** |
| --- | --- | --- | --- | --- |
| Model | Online LS on actor/critic gradients | Online RLS (dual) | Online RLS (triggered) | **Online RLS** (paper eq. (9)) |
| Critic | Neural network | Two heads \(J, \lambda\) | \(\lambda\)-critic | **Parametric quadratic** \(X^T P X\) |
| Actor | NN + gradient updates | NN + GDHP gradients | NN + triggered updates | **Closed form** eq. (11) |
| Update schedule | Every step | Every step | Event-triggered | Model 1 kHz / policy 20 Hz |
| Hyperparameter count | Many (layers, lrs, …) | Many | Many | Few: \(Q\), \(R\), \(\gamma\), \(\gamma_{RLS}\) |

iADP's chief attraction is **interpretability** — LQT-style cost, a matrix that's PSD by construction, and a policy you can write on a napkin. It also has no training-data distribution, no replay buffer and no stochastic gradient descent.

## iADP components

| Component | Role | Implementation |
| --- | --- | --- |
| IncrementalRLS | Identify \(\tilde{\Theta} = [\tilde{F}; \tilde{G}]^T\) online with fixed forgetting | `tensoraerospace.agent.iadp.IncrementalRLS` |
| Policy-evaluation LS | Fit the kernel matrix \(\tilde{P}\) by batch LS on a sliding window | Inline in `IADPAgent._policy_evaluation` |
| Policy improvement | Closed-form eq. (11) | `IADPAgent._compute_policy_increment` |
| IADPAgent | Orchestrates identification, evaluation, improvement | `tensoraerospace.agent.iadp.IADPAgent` |

## Algorithm

On each control tick \(k\), given the measurement \(x_k\) and reference \(r_k\):

1. **Augment the state.**
\[
X_k = \begin{pmatrix} x_k \\ r_k \end{pmatrix}, \qquad \Delta X_k = X_k - X_{k-1}.
\]

2. **Policy improvement (eq. (11)).** Using the current model and kernel:
\[
\Delta \delta_k = -\bigl(R + \gamma \tilde{G}^T \tilde{P} \tilde{G}\bigr)^{-1}
\bigl[R \delta_{k-1} + \gamma \tilde{G}^T \tilde{P} X_k + \gamma \tilde{G}^T \tilde{P} \tilde{F} \Delta X_k\bigr].
\]
Rate-limit \(\Delta \delta_k\) to \(\pm \dot{u}_{\max} \cdot dt\), then \(\delta_k = \mathrm{clip}(\delta_{k-1} + \Delta \delta_k, \pm u_{\max})\).

3. **Apply to the plant and observe** \(x_{k+1}\).

4. **Model update (RLS).** Build \(W = [\Delta X_{k-1}; \Delta \delta_{k-1}]\) and target \(\Delta X_k\). Then
\[
\varepsilon = \Delta X_k - \tilde{\Theta}^T W, \quad
K = \frac{\Phi W}{\gamma_{RLS} + W^T \Phi W}, \quad
\tilde{\Theta} \leftarrow \tilde{\Theta} + K \varepsilon^T, \quad
\Phi \leftarrow \tfrac{1}{\gamma_{RLS}}(\Phi - K W^T \Phi).
\]

5. **Cost accumulation.** Append \((X_k, X_{k+1}, c_k)\) to the window with
\(c_k = (x_k - r_k)^T Q (x_k - r_k) + \delta_k^T R \delta_k\).

6. **Policy evaluation (every `policy_eval_every` steps).** With the current \(\tilde{P}^j\), solve the ridge-regularised LS
\[
A \mathrm{vec}(\tilde{P}^{j+1}) = b, \quad
A_i = \mathrm{vec}(X_i X_i^T)^T, \quad
b_i = c_i + \gamma X_{i+1}^T \tilde{P}^j X_{i+1}.
\]
Symmetrise and optionally project to PSD.

## Quick start

```python
import numpy as np
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig

# Warm-start from an onboard linearisation (trim-point model).
F_init = np.eye(2)                              # reference dynamics + integrator
G_init = np.array([[-0.5], [0.0]])              # pitch-rate / elevator sensitivity

cfg = IADPConfig(
    dt=0.01,
    Q=np.array([[10.0]]),                       # tracking-error weight
    R=np.array([[0.1]]),                        # control weight
    gamma=0.8,                                  # Bellman discount
    gamma_rls=0.995,                            # RLS forgetting factor
    policy_eval_window=200,
    policy_eval_every=50,                       # ≈ 20 Hz at dt = 0.01 s
    policy_eval_warmup_updates=30,
    F_init=F_init, G_init=G_init,
    u_magnitude_limit=15.0,
    u_rate_limit=60.0,
    seed=0,
)
agent = IADPAgent(n_state=1, n_control=1, config=cfg)

x = np.zeros(1)
for k in range(2000):
    ref = np.array([0.1 if k > 100 else 0.0])
    u = agent.predict(x, ref, k)
    # Plug your environment step here.
    x = x + cfg.dt * (-0.5 * u - 2.0 * x)
    agent.learn(x, ref, k)
```

!!! tip "Warm-start `G_init` matters"
    Policy eq. (11) needs a reasonable \(\tilde{G}\) on the first few ticks, otherwise \((R + \gamma \tilde{G}^T \tilde{P} \tilde{G}) \approx R\) gives only the control-regularisation response. Seed `G_init` from a linearised onboard model; the RLS will refine it online.

!!! tip "Warm-start `P_init` from the DARE for faster convergence"
    The online batch-LS fits \(\tilde{P}\) from transition windows, but its finite-window bias can leave the policy a few percent below optimal. Seed `P_init` from the analytical LQT DARE computed off the warm-start model:

    ```python
    from scipy.linalg import solve_discrete_are
    Q_aug = Q_val * np.block([[np.eye(n_state), -np.eye(n_state)],
                              [-np.eye(n_state), np.eye(n_state)]])
    P_init = solve_discrete_are(
        np.sqrt(gamma) * F_init, np.sqrt(gamma) * G_init,
        Q_aug, R,
    )
    ```

    Discount enters via \(\bar{F} = \sqrt{\gamma}F,\, \bar{G} = \sqrt{\gamma}G\), so `solve_discrete_are` applies directly. See the [nonlinear-F-16 example](../example/agent/iadp/example_iadp_nonlinear.md).

!!! note "Sequential vs Continuous learning"
    Set `model_learning_only_steps > 0` with `excitation_signal` to replicate the paper's **Sequential Learning Approach** — the first *N* steps run in open loop with a user-supplied multi-sine so the RLS converges before the policy engages. By default both phases run concurrently (**Continuous Learning Approach**).

## Hyperparameters

### Cost & discount

| Parameter | Default | Description |
| --- | --- | --- |
| `Q` | `I` | Tracking-error weight, shape `(n_state, n_state)` |
| `R` | `I` | Control weight, shape `(n_control, n_control)` |
| `gamma` | 0.8 | Bellman discount \(\gamma \in (0, 1)\) |

### RLS identifier

| Parameter | Default | Description |
| --- | --- | --- |
| `gamma_rls` | 0.995 | Constant forgetting factor, closer to 1 ⇒ longer memory |
| `phi_init` | 1e2 | Initial covariance scale |
| `F_init` | None | Warm-start for \(\tilde{F}\), shape `(n_aug, n_aug)` |
| `G_init` | None | Warm-start for \(\tilde{G}\), shape `(n_aug, n_control)` |

### Policy evaluation

| Parameter | Default | Description |
| --- | --- | --- |
| `policy_eval_window` | 200 | Sliding-window size of transitions used by batch LS |
| `policy_eval_every` | 50 | Stride between LS updates, in `learn()` ticks |
| `policy_eval_iterations` | 1 | Inner fixed-point sweeps per LS update |
| `policy_eval_regularization` | 1e-4 | Ridge term added to the normal equations |
| `policy_eval_warmup_updates` | 20 | Number of RLS updates to wait before the first LS |
| `enforce_psd` | True | Clip eigenvalues of \(\tilde{P}\) to stay positive-definite |
| `psd_floor` | 1e-6 | Lower bound used by the eigen-clip |
| `policy_eval_blend` | 1.0 | EMA coefficient for \(\tilde{P}\) updates. `1.0` replaces outright; smaller values (0.05–0.2) soft-update \(\tilde{P}\) like a target network and eliminate the control-trace sawtooth that otherwise appears at every LS tick. |
| `P_init` | `I` | Warm-start kernel matrix |

### Actuator bounds

| Parameter | Default | Description |
| --- | --- | --- |
| `dt` | 0.01 | Control step (s) |
| `u_magnitude_limit` | 25.0 | Hard magnitude clamp per channel |
| `u_rate_limit` | 60.0 | Max \(|\Delta\delta|\) per second per channel |
| `pinv_rcond` | 1e-8 | Cutoff for `np.linalg.pinv` on the policy-improvement matrix |

### Sequential-learning phase

| Parameter | Default | Description |
| --- | --- | --- |
| `model_learning_only_steps` | 0 | Open-loop identification window length |
| `excitation_signal` | None | Optional \((T, n_\text{control})\) schedule of absolute control values |

## Supported environments

- Any Gymnasium env whose observation vector contains the controlled states \(x_t\) and whose action space is a continuous control \(\delta_t\). The augmented state \(X_t = [x_t; x_t^r]\) is built internally, so the reference is passed alongside the observation in `predict` / `learn`.
- Typical targets: `NonlinearLongitudinalF16-v0` (pitch-rate or pitch-angle tracking via elevator), 6-DoF rate inner loops.

## Persistence

```python
run_dir = agent.save("./checkpoints")           # creates <date>_IADPAgent/
restored = IADPAgent.from_pretrained(run_dir)
agent.publish_to_hub("me/my-iadp", folder_path=run_dir, access_token="hf_...")
```

Saved artefacts:

- `config.json` — full `IADPConfig` + `n_state` / `n_control`, arrays stored as lists.
- `rls.npz` — RLS parameter matrix `theta`, covariance `Phi`, update counter, last residual.
- `value.npz` — current kernel matrix \(\tilde{P}\).
- `weights.npz` — active `Q` and `R` (including defaults filled in by the constructor).
- `loop_state.npz` — \(X_{t-1}\), last control, last increment, cached \(\Delta X\), step counter.
- `window.npz` — transition buffer used by the policy evaluator, stored as stacked arrays so reload is bit-identical.

## API reference

::: tensoraerospace.agent.iadp.model.IADPAgent

::: tensoraerospace.agent.iadp.model.IADPConfig

::: tensoraerospace.agent.iadp.rls.IncrementalRLS

## Sources

- Konatala, Milz, Weiser, Looye, van Kampen. *"Flight Testing Reinforcement Learning based Online Adaptive Flight Control Laws on CS-25 Class Aircraft"*, AIAA SCITECH 2024, [DOI 10.2514/6.2024-2402](https://doi.org/10.2514/6.2024-2402).
- Sieberling, Chu, Mulder. *"Robust Flight Control Using Incremental Nonlinear Dynamic Inversion and Angular Acceleration Prediction"*, J. Guid. Control Dyn., 2010.
- Lewis, Vrabie, Syrmos. *"Optimal Control"*, Wiley, 2012 — LQT theory underpinning the quadratic cost-to-go.
