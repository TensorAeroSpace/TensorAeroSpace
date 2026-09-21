# Incremental Approximate Dynamic Programming (iADP)

iADP identifies an incremental plant model with fixed-forgetting RLS, fits a quadratic value function by batch least squares, and computes an analytic control increment. There is one implementation: unregularized critic fitting and the policy equation from [Konatala et al., AIAA 2024-2402](https://doi.org/10.2514/6.2024-2402).

## Start with a complete SDK example

The example below configures iADP, runs 80 s of continuous learning through an
unknown input-effectiveness change, measures tracking with `ControlBenchmark`
and plots the command, response, applied control and identified gain.
The scalar plant equation is explicit; only the plant receives its fault schedule.
Run the block with the installed `tensoraerospace` package.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig
from tensoraerospace.benchmark import ControlBenchmark

plt.rcParams.update(
    {
        "figure.dpi": 125,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
dt, duration = 0.001, 80.0
time = np.arange(round(duration / dt) + 1) * dt
identification_time = np.arange(round(20.0 / dt)) * dt
excitation = (
    0.15 * np.sin(2 * np.pi * 0.7 * identification_time)
    + 0.05 * np.sin(2 * np.pi * 1.7 * identification_time)
)[:, None]
period = np.arange(round(10.0 / dt)) * dt
ongoing_excitation = (0.015 * np.sin(2 * np.pi * 0.7 * period))[:, None]
config = IADPConfig.paper(
    excitation_signal=excitation,
    dt=dt,
    learning_mode="continuous",
    Q=np.array([[100.0]]),
    R=np.array([[0.0001]]),
    gamma=0.95,
    gamma_rls=0.999,
    phi_init=1e6,
    u_magnitude_limit=0.5,
    u_rate_limit=2.0,
    continuous_excitation_signal=ongoing_excitation,
)
agent = IADPAgent(1, 1, config)
x = np.zeros(1)
reference = np.array([0.05])
a = np.exp(-2 * dt)
b = (1 - a) / 2


states, actions, estimated_gain, true_gain = [x[0]], [], [], []
for k, t in enumerate(time[:-1]):
    command = agent.predict(x, reference, k)
    # The unknown effectiveness change belongs to the plant only.
    effectiveness = 1.0 if t < 60.0 else 0.7
    x = a * x + b * effectiveness * command
    agent.learn(x, reference, k, applied_action=command)
    if not np.isfinite(x).all() or not np.isfinite(agent.P).all():
        raise FloatingPointError(f"Nonfinite state or critic at {time[k+1]:g} s")
    states.append(x[0])
    actions.append(command[0])
    estimated_gain.append(agent.G[0, 0])
    true_gain.append(b * effectiveness)
states, actions, estimated_gain, true_gain = map(
    np.asarray, (states, actions, estimated_gain, true_gain)
)
assert len(actions) == len(time) - 1
assert agent.rls.num_updates == len(actions) - 1
print(f"Completed {time[-1]:g} s; RLS updates: {agent.rls.num_updates}")


benchmark = ControlBenchmark()
nominal = benchmark.tracking_metrics(0.05, states, dt, start=40.0, end=60.0)
faulty = benchmark.tracking_metrics(0.05, states, dt, start=65.0, end=80.0)
print("Nominal RMSE [rad/s]:", nominal["combined_rmse"])
print("Post-fault RMSE [rad/s]:", faulty["combined_rmse"])
print("Identified / true final G:", estimated_gain[-1], true_gain[-1])
print("Final minimum critic eigenvalue:", np.linalg.eigvalsh(agent.P).min())


fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
axes[0].plot(time, np.full_like(time, reference[0]), "k--", label="Reference")
axes[0].plot(time, states, color="#176b91", label="Plant state")
axes[0].set_ylabel("Rate [rad/s]")
axes[0].legend()
axes[1].plot(time[1:], actions, color="#40855b")
axes[1].set_ylabel("Applied control")
axes[2].plot(time[1:], true_gain, "k--", label="True discrete gain")
axes[2].plot(time[1:], estimated_gain, color="#176b91", label="Identified gain")
axes[2].set_ylabel("G estimate")
axes[2].legend()
for ax in axes:
    ax.axvspan(0, 20, alpha=0.08, color="#40855b")
    ax.axvline(60, color="#bb4040", linestyle=":")
axes[-1].set_xlabel("Time [s]")
fig.suptitle("iADP: identification, tracking and unknown effectiveness loss")
plt.show()
```

These gains and weights belong to this scalar plant. Initial excitation lasts
20 s; the later periodic excitation keeps providing identification data. The two
RMSE windows have different adaptation histories, so their values do not measure
the effect of damage alone.

### Continue with aircraft examples

- [B737: pitch step and elevator fault](../example/agent/iadp/example_iadp_nonlinear.md).
- [F-16: servo feedback and fault scenarios](../example/agent/iadp/example_iadp_small_fault_f16.md).
- [Executed notebook with this learning loop](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_iadp_paper.ipynb).

## State, output and cost

The augmented state is `X = [x; reference_state]`, with dimension `n_state + n_reference`. The plant and reference-generator states may have different sizes. Linear output maps define `y = C @ x` and `y_ref = Cr @ reference_state`. The stage cost is

\[
c_k = (Cx_k-C_r x_k^r)^T Q(Cx_k-C_r x_k^r) + \delta_k^T R\delta_k.
\]

Set `n_reference`, `output_matrix` and `reference_output_matrix` explicitly for a reference generator such as a sine oscillator. Identity output maps and equal state sizes are defaults. `Q` has shape `(n_output,n_output)`; `R` has shape `(n_control,n_control)`. Supplying one angular rate does not reconstruct a missing full aircraft state.

## Update equations

1. RLS fits `dX_next = F @ dX + G @ du`, using the **measured** next state and actual applied control. The first transition cannot update the incremental model because the previous state increment is unknown.
2. The critic stores the model prediction `X_next_hat = X + F @ dX + G @ du` and stage cost. Batch least squares fits `vec(P)` using quadratic state features and the current value of `P` on the right-hand side. SVD computes the Moore–Penrose solution; the resulting matrix is symmetrized.
3. Policy improvement solves

\[
(R+\gamma G^TPG)\Delta\delta =
-[R\delta_{k-1}+\gamma G^TPX_k+\gamma G^TPF\Delta X_k].
\]

Magnitude and per-second rate limits then constrain the requested command. There is no ridge penalty, PSD projection, critic blending or alternate pseudoinverse policy. A singular policy equation raises an error. The old options `policy_eval_regularization`, `enforce_psd`, `psd_floor`, `policy_eval_blend` and `pinv_rcond` have been removed.

## Continuous and sequential learning

`IADPConfig.paper(...)` supplies the reported experiment schedule. Direct `IADPConfig(...)` construction uses the same algorithm with configurable timings.

| Setting | Paper factory default |
| --- | --- |
| Control/model period | `dt=0.001` s |
| Critic update rate | 20 Hz |
| Initial open-loop identification | 20 s, caller-supplied varying excitation |
| Critic transition window | 20 s; wait for a full window |
| Learning approach | `continuous` (CLA) |
| SLA controller-training duration | 40 s after model identification |
| SLA critic fitting interval | Last 5 s of controller training: 55–60 s with the default schedule |

CLA keeps identifying the plant and training the critic online after initialization. SLA freezes the model after identification and the critic after training, as an explicit experimental mode from the paper. Use CLA for adaptation to unknown future faults. Neither mode receives a fault time. `continuous_excitation_signal` optionally adds a cyclic signal throughout controller training; amplitude/rate limits also apply to excitation.

## Actuator feedback and initialization

If the actuator clips, lags or changes the command, pass its actual input to `learn(..., applied_action=actual_input)`. The cost, RLS regressor and next increment use this feedback. Match units and trim offsets; B747/LAPAN environment feedback is in degrees. For a continuous servo, a transition-average position is an approximation to the effective input.

The default initial kernel is `[C,-Cr].T @ Q @ [C,-Cr] + 1e-6*I`, coupling plant and reference outputs. This is an implementation choice: the article does not publish its full initialization. Supply a physically consistent `P_init`, `F_init`, `G_init` when available. `reset(initial_action=trim_input)` initializes control history at a nonzero applied trim; reset preserves learned parameters.

No excitation with `G=0` still gives zero control. An uninformative window does not identify a full value function. Removing regularization does not prove bounded parameters or closed-loop stability. Check feature scales, excitation and actuator bandwidth on the target aircraft. Flight-test sensor preprocessing and complete tuning are not published or reproduced by this scalar check.

## Persistence and migration

`agent.save(path)` and `IADPAgent.from_pretrained(folder)` preserve output maps, phase, RLS, critic and transition history. Checkpoints/configurations containing the removed numerical options must be regenerated; their softened critic updates cannot be resumed as the same algorithm. Existing F-16 experiments now call the single published update law and need fresh evaluation; historical charts describe their earlier configurations.

## API reference

::: tensoraerospace.agent.iadp.model.IADPAgent

::: tensoraerospace.agent.iadp.model.IADPConfig

::: tensoraerospace.agent.iadp.rls.IncrementalRLS

## Nonlinear B737 notebook

[Run the B737 pitch-step example](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_iadp_nonlinear_b737.ipynb): the same cruise trim and +1° step as the IHDP notebook, with an explicit pitch-to-rate outer loop, continuous adaptation, saved plots and `ControlBenchmark` metrics. The example documents its nominal-model initialization and reduced-model assumptions.

## Fault examples

- [B737: 50% elevator-authority loss](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_iadp_fault_b737.ipynb), with continuous learning, actual surface feedback and post-fault error metrics.
