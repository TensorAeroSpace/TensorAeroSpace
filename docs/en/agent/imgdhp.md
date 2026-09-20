# Incremental model-based GDHP (IM-GDHP)

The implementation follows **Bo Sun and Erik-Jan van Kampen (2021)**, [Intelligent adaptive optimal control using incremental model-based global dual heuristic programming subject to partial observability](https://doi.org/10.1016/j.asoc.2021.107153). The [author-hosted manuscript](https://pure.tudelft.nl/ws/portalfiles/portal/90401508/1_s2.0_S1568494621000764_main.pdf) is the equation reference. The paper calls the method **IGDHP**; the Python package retains the name `im_gdhp`.

## Architecture and equations

The actor and critic receive the tracking error. A window of **measured error and input increments** identifies hidden dynamics. Concatenating observation, reference and error does not replace this history or establish observability. Choose `history_length` for the actual plant/reference dynamics and provide sufficient excitation.

The critic has **one scalar output**. Autograd computes its exact input derivative and the mixed derivatives needed to train it; there is no independent costate head. Its layers have no bias, so `J(0)=0`. The actor has a constant input of 0.01 and a bounded tanh output. One hidden layer reproduces the paper architecture; additional layers are an optional extension.

Equations (31)–(38), (51)–(67), with column-vector Jacobians:

\[
e_t=y_t[\mathrm{tracking}]-r_t,\qquad
\hat e_{t+1}=e_t+\sum_{j=0}^{M-1}F_j\Delta e_{t-j}+\sum_{j=0}^{M-1}G_j\Delta u_{t-j}.
\]

\[
J(e)=w_{c2}^{T}\sigma(w_{c1}^{T}e),\qquad
\lambda(e)=\nabla_e J(e).
\]

\[
c=e^TQe+u^TRu,\quad
D=I+F_0+G_0\frac{\partial\pi}{\partial e},\quad
\lambda_{\rm target}=\nabla_e c+\gamma D^T\lambda(\hat e_{t+1}).
\]

\[
E_c=\frac{\beta}{2}(J(e_t)-c-\gamma J(\hat e_{t+1}))^2
+\frac{1-\beta}{2}\|\lambda(e_t)-\lambda_{\rm target}\|^2,
\qquad \beta=\frac{1}{1+\texttt{beta\_lambda}}.
\]

\[
E_a=\frac12 J(\hat e_{t+1})^2,\qquad
\nabla_w E_a=J(\hat e_{t+1})\frac{\partial\pi^T}{\partial w}G_0^T\lambda(\hat e_{t+1}).
\]

For nonzero `control_R`, the derivative of the immediate cost also includes the actor path, `(dπ/de).T @ (2 R u)`. The identity matrix in `D` comes from adding the current error to the predicted increment. The actor objective is the squared predicted cost-to-go, not an extra one-step tracking objective.

## Interaction with the environment

1. `predict(obs, reference, k)` computes the bounded command and records the current error.
2. `env.step(action)` advances the physical model.
3. `learn(next_obs, reference, k)` predicts the next error with the **previous identifier**, updates the networks using the previous policy/critic targets, then assimilates the measured transition into RLS (Algorithm 1).
4. Call `reset()` between episodes. It clears transition buffers but retains learned weights, RLS parameters and learning progress.

If the caller changes the command, pass `applied_action=` to `learn` in the same units as the policy output. Supply the entire reference array, including its next sample. In the F16 example observations are radians and commands are degrees.

## Known, discontinuous references

The paper identifies tracking-error dynamics when the reference dynamics are unknown (Section 3.3). Its reference representation assumes a continuous, piecewise differentiable signal. Fitting an abrupt commanded step directly as an error increment can contaminate the estimated aircraft dynamics.

When the command is available, select the explicit **known-reference extension**:

```python
cfg = IMGDHPConfig(identifier_mode="output", history_length=4)
```

RLS then fits measured output increments and control increments. Both actor and critic updates use `predicted_output - next_reference`. Network inputs and the tracking cost still contain only the tracking error. This keeps a commanded jump out of the plant identifier while preserving the published network losses and their derivative paths. No measurement of the next plant output is used in the predicted learning target.

The default `identifier_mode="tracking_error"` retains the paper's unknown-reference formulation. Select the mode before collecting transitions; changing it on an active agent would mix histories with different meanings. Checkpoints save the mode and the matching history.

## Configuration

| Parameter | Meaning |
| --- | --- |
| `identifier_mode` | Identifier input: `tracking_error` (paper default) or `output` (explicit known-reference extension). |
| `history_length` | Increment window M; default 1 is the full-state special case. The F16 lesson uses 4. |
| `track_Q`, `control_R` | Diagonal nonnegative tracking/input cost weights; R is zero when omitted. |
| `obs_scale` | Per-observation scaling. Network inputs and tracking cost use it; identification and lambda use physical error coordinates. |
| `beta_lambda` | Derivative/scalar loss ratio; the paper beta equals `1/(1+beta_lambda)`. Zero gives value-only critic learning. |
| `actor_lr`, `critic_lr` | Initial learning rates. |
| `actor_lr_decay`, `critic_lr_decay` | Per-update multipliers, default 1. |
| `actor_lr_min`, `critic_lr_min` | Lower bounds on learning rates, default 0. |
| `weight_limit` | Weight magnitude bound, default 20. |
| `forgetting`, `cov_init` | RLS forgetting factor and initial covariance: positive scalar or positive diagonal entries in regressor order. |
| `actor_bias_input` | Constant actor input ba in Eq. (64), default 0.01; its scale affects trim-command learning. |
| `optimizer` | `sgd` by default. `adam` is an optional extension. |
| `warmup_steps`, `critic_only_steps` | Initial identification period and optional extra critic-only updates. |
| `max_grad_norm` | Optional gradient clipping; disabled by default. |
| `target_update_tau` | Optional target smoothing; default 0 uses the current critic. |

The default initialization uses network weights in `[-0.1, 0.1]`, identity F blocks and zero G, following Section 5.2. Hyperparameters are configurable: defaults and the F16 lesson are **not** a reproduction of the paper's 1 kHz experiment, noise, actuator or Monte Carlo study. The lesson explicitly enables gradient clipping and critic warmup and scales the errors. Its learning rates are retuned for 100 Hz.

### Retuning an already trained actor

`agent.retune_actor_inputs(feedback_gain=..., bias_input=...)` is an SDK operation for a reproducible tuning stage between experiments. `feedback_gain` multiplies first-layer error weights; the bias change inversely rescales its weights so the zero-error action is preserved. With feedback gain one, the full current policy is preserved while its subsequent learning sensitivity changes. The method validates weight limits, rejects pending predict/learn transitions and clears optimizer moments for the transformed layer. Saving the agent includes the new bias scale and weights.

The F16 lesson derives a positive online SGD rate from the local Eq. (67) gradients and follows retuning with a zero-reference training episode. This is an explicit experimental procedure; the standard actor update does not add a PID or integral-control term.

## Complete examples

- [F16 lesson: linear tracking and nonlinear identification](../example/agent/imgdhp/example_imgdhp_nonlinear.md) — full construction, learning loop and plots.
- [Recipe 12](../cookbook/12_imgdhp.md) — executable SDK walkthrough and interpretation.

## Migration and checkpoints

The former implementation had independent J/lambda heads, no measured history, an incomplete costate target and an additional actor tracking/rate objective. Checkpoints from that architecture are rejected with a retraining message. Set `action_rate_penalty=0`; use `control_R` for the paper's input cost. Previous tuned learning rates are not interchangeable.

`save(path, save_gradients=True)` saves networks, optimizers, learning rates, RLS, history and exploration RNG. Load with `IMGDHPAgent.from_pretrained(path, load_gradients=True)`. Exact continuation also requires the same environment state, reference and time index. For a new episode, call `reset()`.

## Limits of the evidence

The numerical tests check the derivatives against independent equations and recover a hidden second-order system from measured history. A tracking experiment checks one chosen plant and set of signals. Neither establishes global closed-loop stability. The paper itself notes this limitation in Remark 2 and reports failures for some initial conditions. Accurate one-step identification of a nonlinear aircraft is not evidence of a successful nonlinear control policy.

## API

::: tensoraerospace.agent.im_gdhp.model.IMGDHPAgent

::: tensoraerospace.agent.im_gdhp.model.IMGDHPConfig

::: tensoraerospace.agent.im_gdhp.incremental_model.IncrementalModelRLS

::: tensoraerospace.agent.im_gdhp.networks.GDHPActor

::: tensoraerospace.agent.im_gdhp.networks.GDHPCritic

## Step-response validation

The complete F16 lesson commands **alpha only** (`reference_size=1`, `tracking_indices=[0]`). It uses known-reference identification and sensitivity-based tuning, followed by a zero-reference training episode. The primary step starts at 20 s and lasts 38 s in total; `ControlBenchmark` scores the response. **Recomputed result, seed 0:** zero-hold maximum error **0.000120°**, 38-second tail mean error **+0.000212°**, command overshoot **4.50%**, command-band settling **3.38 s**, **CPI 0.9237**. All 12 seed/amplitude assessments pass the final-five-second 0.1%-accuracy condition at 60 s. The command sequence retains larger finite-time residuals: its largest final-five-second mean magnitude is about 0.00192°. These are measured finite-horizon errors, not a claim of mathematically exact zero error for every signal.

### Scaling the RLS prior

`cov_init` also accepts `M * (n_tracked + n_action)` diagonal entries, ordered as all state-increment history blocks, then all action-increment history blocks. For example, with two radian measurements, degree-valued commands and M=4:

```python
scale = 180 / np.pi
covariance_diagonal = (1e4 * scale**2,) * 8 + (1e4,) * 4
cfg = IMGDHPConfig(track_Q=(1.0, 0.0), history_length=4,
                  cov_init=covariance_diagonal, actor_bias_input=0.01)
```

This prior is equivalent to starting a normalized-regressor RLS with covariance `1e4 * I`; the SDK still predicts in physical units. It does not guarantee convergence. Unit-conversion equivalence, covariance reset and checkpoint continuation are regression-tested. The original scalar covariance and bias-input defaults are preserved.

**Scope of the result:** the updated step, amplitude, seed, delayed-step and command-sequence checks use continuing adaptation. The 500-second rollout stays numerically bounded in alpha and control, but accumulates a large pitch angle; it is not nonlinear-flight validation. The 2° sine with a 4-second period is still tracked inaccurately (12-second RMSE 1.685°). Empirical tests do not establish stability for arbitrary signals or initial states.
