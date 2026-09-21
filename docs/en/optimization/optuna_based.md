# Controller tuning from environment settings

`ControllerTuner` is the main entry point: choose a native environment, its settings, one or more state steps, a controller and a search method. CPI is the default metric. The library constructs the agents, runs learning and simulation, and uses the native step benchmark.

```python
from tensoraerospace.optimization import ControllerTuner, Step

tuner = ControllerTuner(
    env="NonlinearB737-v0",
    env_kwargs={"trim_at": (10000.0, 600.0), "dt": 0.02},
    reference={"theta": Step(1.0, at=20.0, unit="deg")},
    controller="aa_indi",
    method="genetic",
    duration=40.0,
    method_options={"population_size": 8},
    seed=42,
)
result = tuner.optimize(n_trials=32)
print(result.best_params, result.best_value)
result.plot_response()
result.plot_history()
```

[Full executed tutorial: AA-INDI / nonlinear B737, several commanded states and validation](../example/optimization/example_optimization.md).

## Experiment settings

| Setting | Meaning |
|---|---|
| `env` | Registered Gymnasium ID or supported environment class; not a live instance |
| `env_kwargs` | Native environment settings; `dt`, initial state/trim, physical faults, etc. |
| `reference` | Mapping of physical state name to `Step`; exactly one step per channel |
| `Step(amplitude, at=20, initial=None, unit="native")` | Amplitude is the change; `at` is seconds; initial defaults to the initial physical state |
| `unit` | `native`, `rad`, `deg`; for angular-rate states the angular conversion also applies to rad/s ↔ deg/s |
| `duration=40` | Complete simulation horizon in seconds; duration and step times must align with `dt` |
| `controller` | Name/alias or a supported native agent class |
| `metric="cpi"` | Scalar metric to minimize: also `iae`, `ise`, `itae`, `relative_tail_error`, `theta.cpi`, etc. |
| `normalize=True` | Normalize each reference/response by its own step amplitude before benchmark computation |
| `constraints={}` | Inclusive metric upper bounds, checked on every seed; a missing settling time fails its bound |
| `state_limits={}` | Additional physical limits, e.g. `{"alpha": (-0.2, 0.2)}` in native state units |
| `seeds=(0,)` | Fixed simulation/training seeds for every candidate; their objective values are averaged |
| `seed=0` | Seed of the search algorithm |
| `training_episodes=0` | Additional unscored online episodes; weights persist into the final scored online episode |
| `torch_threads=1` | Torch CPU threads during each simulation; `None` preserves the current setting |
| `controller_options={}` | Fixed settings from the selected profile; see `ControllerTuner.profile(name)` |
| `search_space=None` | Built-in ranges by default; override a subset using `Float`, `Int`, `Categorical` |

Do not pass `reference_signal` or `number_time_steps` in `env_kwargs`: the tuner constructs both consistently. A successful trajectory includes time zero and the requested final time. Nonfinite states, model-envelope violations and early termination reject a candidate. Limits apply to all physical states, not just commanded ones. For Boeing the small-maneuver envelope is ±30° roll/pitch, 250–1000 ft/s speed, and altitude between 0 and 50,000 ft. F16 angle/rate limits are checked too.

The aggregate CPI is the mean of channel CPIs. Other aggregate metrics use the worst (maximum) channel value. Named keys such as `theta.cpi` and `phi.relative_tail_max_error` remain available. Pre-step hold error is separate from the native post-step CPI; add `pre_step_relative_error` to constraints when it matters.

## Search methods

| `method` | Algorithm | `method_options` examples |
|---|---|---|
| `"tpe"` (default) | Optuna TPE | `{"n_startup_trials": 5}` |
| `"annealing"` | Metropolis simulated annealing, geometric cooling, reflected bounded proposals | `{"temperature": 1.0, "cooling": 0.95, "step_size": 0.2}` |
| `"genetic"` | Optuna NSGA-II with one minimizing objective | `{"population_size": 8, "mutation_prob": 0.3}` |
| `"random"` | Seeded random sampling | `{}` |

Annealing explores logarithmic parameters in log coordinates and supports integer/categorical parameters too. Invalid trials never become its accepted state. Genetic evolution needs more evaluations than one population of completed candidates. Search is serial by default; TPE, genetic and random search support process batches with `n_jobs>1`. Every candidate has the same horizon/training budget. No partial-trajectory score can win.

`optimize(n_trials=30, timeout=None, target=None, patience=None, initial_params=None, show_progress_bar=True, n_jobs=1)` starts a new study. `initial_params` is a complete or partial dictionary of searched parameters evaluated first; the sampler fills missing fields. `target` and `patience` can stop a search early. Timeout is checked between trials, not inside a simulation. If all candidates fail, the tuner raises an error; inspect `tuner.diagnostics()` or the detailed `tuner.optimizer.study.trials`.

Algorithms: [Optuna NSGA-II](https://optuna.readthedocs.io/en/v3.6.2/reference/samplers/generated/optuna.samplers.NSGAIISampler.html), [Optuna custom sampler / annealing example](https://optuna.readthedocs.io/en/v3.6.2/tutorial/20_recipes/005_user_defined_sampler.html). The annealing implementation here is conventional simulated annealing, not SciPy dual annealing.

## Faster search on several CPU cores

Pass `n_jobs` to run independent controller candidates in separate processes. This works with all eight controller profiles and the `tpe`, `genetic` and `random` search methods:

```python
from tensoraerospace.optimization import ControllerTuner, Step

tuner = ControllerTuner(
    env="NonlinearB737-v0",
    env_kwargs={"trim_at": (10000.0, 600.0), "dt": 0.02},
    reference={"theta": Step(1.0, at=20.0, unit="deg")},
    controller="ihdp",
    method="tpe",
    duration=40.0,
    seed=42,
    torch_threads=1,  # Default: one Torch CPU thread per simulation
)
result = tuner.optimize(n_trials=120, n_jobs=4, target=3.7)
print(result.best_params, result.best_value)
result.plot_response()
```

In Jupyter, run the cell directly. In a Python script, put tuner construction and `optimize()` inside `if __name__ == "__main__":`. Each process imports the installed library; restart the notebook kernel after upgrading it. Settings must be picklable: use importable classes/functions instead of notebook-local callbacks or lambdas. Use `n_jobs=1` when process startup would outweigh a short evaluation.

Workers use fresh environments, adaptive weights and random states for each candidate. The step, dt, horizon, learning budget, physics checks and hard constraints are identical to serial execution. The parent alone updates the Optuna study. CPU process isolation prevents candidates from overwriting each other's Torch/NumPy seeds. `torch_threads=1` limits Torch CPU work for these small networks; `None` preserves each process's existing setting. Simulation restores the calling process's previous thread count, even on failure. This setting does not control NumPy/BLAS threads.

Candidates are proposed in batches of at most `n_jobs`, then results enter the study in trial-number order. This makes seeded searches reproducible for the same worker count and deterministic numerical backend. TPE uses `constant_liar=True` by default in parallel mode to avoid proposing similar running candidates; override through `method_options` if needed. A batch sees fewer completed results than an equally long serial search, so changing `n_jobs` can change the candidate sequence and final CPI. Faster execution does not guarantee fewer trials to a target. See [Optuna's ask-and-tell batch interface](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html).

A supplied `initial_params` candidate finishes before the first batch. `target`, `patience` and `timeout` prevent the next batch from starting; already dispatched candidates finish, so early stopping can evaluate up to `n_jobs - 1` extra candidates. The progress bar updates as results are recorded in trial-number order. `result.save(...)` includes worker/thread settings and per-trial worker timing. The best candidate is replayed automatically after the search.

Simulated annealing requires `n_jobs=1` because the next proposal depends on the previous acceptance decision. Use TPE or genetic search to parallelize. If you have a previously evaluated setting for the same experiment, pass it via `initial_params=previous_result.best_params`. `optimize()` starts a new study. Use `resume()` below to retain the complete history and the live sampler state.

Measured example: 24 identical random-search candidates, iHDP on nonlinear B737, a 1° pitch step at 20 s, dt=0.02 s and a 40 s horizon. Two serial runs averaged **44.7 s**, versus **14.9 s** with four workers, including startup and winner replay: approximately **3× faster** on the test machine. All candidate metrics and the winner's state/action trajectories matched exactly. This is a runtime comparison on fixed candidates, not a guarantee of speed or policy quality for other experiments.

## Find the best observed step response

`find_best_response()` provides budget-based minimization: it starts a new study on the first call and resumes the existing study on later calls. It minimizes the configured metric (`cpi` by default), with no stopping target.

```python
# Initial search; tuner has the environment, controller, metric and constraints.
result = tuner.find_best_response(n_trials=120, n_jobs=8)

# Another 120 candidates in the same study, retaining eight workers.
result = tuner.find_best_response(n_trials=120)
print(result.best_params, result.best_value)
result.plot_response()
result.plot_history()
```

The full signature is `find_best_response(n_trials=30, timeout=None, patience=None, initial_params=None, show_progress_bar=True, n_jobs=None)`. Timeout and optional patience can end the additional budget early. An earlier `optimize(target=...)` target does not carry over. `n_jobs=None` means one worker for a new search, or the previous worker count when continuing. This uses the same native controller, sampler and benchmark; it does not introduce a different CPI formula.

The result is the lowest **observed feasible** objective across the entire history. Hard `constraints` and all physical checks remain active. If no trial is feasible, the method raises the usual diagnostic error; it does not relax requirements. Changing the experiment requires a new study, just as with `resume()`.

For pure CPI minimization without accuracy ceilings, explicitly create a separate `ControllerTuner(..., metric="cpi", constraints={})`. Such a search may select a response that misses an accuracy goal; inspect command settling, overshoot, tail error and pre-step error before accepting it. The new study does not reuse old constrained-trial decisions. Physical envelope and actuator checks still apply.

With a finite budget, a global optimum or CPI=0 is not guaranteed. Increasing the budget can improve the best score or leave it unchanged. `find_best_response()` does not automatically compare controller types; it tunes the selected controller. The [executed AIDI/B737 tutorial](../example/optimization/example_optimization.md) demonstrates continuation, both constraint choices and a saved checkpoint.

## Continue a search and save checkpoints

`resume()` adds a new trial budget to the same study. It retains all completed/rejected trials, the best feasible result, the sampler's random generator and its search state. TPE keeps its history, genetic search keeps the current population/history, and annealing keeps the accepted point and current temperature.

```python
# Start the search once (tuner configured with method="tpe").
result = tuner.optimize(n_trials=120, n_jobs=8)

# Add up to 120 MORE candidates; keep the existing eight workers.
result = tuner.resume(n_trials=120)
# Equivalent entry point: result = result.resume(n_trials=120)

print("Total trials:", len(result.study.trials))
print("Best parameters:", result.best_params)
print("Best CPI:", result.best_value)
result.plot_history()  # The full history, including the first call
result.plot_response()
```

`resume(n_trials=30, timeout=None, target=None, patience=None, initial_params=None, show_progress_bar=True, n_jobs=None)` uses the existing study. `n_trials`, timeout and patience are new budgets for this call. `target` checks the best result across the entire history: if it is already reached, no additional candidates run. To continue exploring regardless of a prior target, omit `target` or specify a smaller value. The progress bar counts only the new budget and displays the previous best from the start.

`n_jobs=None` retains the previous worker count. You may explicitly change it, including switching between serial and process execution; sampler options/state remain intact. Changing the batch size can change subsequent adaptive proposals. Annealing still requires one worker. For deterministic numerical backends, splitting a run at a batch boundary with the same worker count preserves its candidate sequence. A split inside a batch can change TPE/genetic feedback timing.

The environment, reference channel order, duration, dt, controller, fixed settings, search space, metric, constraints, seeds, learning budget and method settings must stay unchanged. `resume()` detects changes before adding trials. Create a new tuner/search for another experiment. A new `optimize()` call replaces the active study; a result from an older study cannot resume that replacement accidentally. Reassign the returned result after continuation to use the newly selected best parameters and replay.

### Continue after restarting Jupyter

```python
# Before stopping the kernel:
tuner.save_checkpoint("ihdp-search.pkl")
# Equivalent: result.save_checkpoint("ihdp-search.pkl")
```

```python
# In the new kernel:
from tensoraerospace.optimization import ControllerTuner

tuner = ControllerTuner.load_checkpoint("ihdp-search.pkl")
result = tuner.resume(n_trials=200, target=3.7)
```

The checkpoint contains experiment settings, all trials, the live sampler and its RNG, and the worker count. Loading does not run simulations. Saving replaces the file atomically. Save after the search has stopped; checkpoints also work after `No feasible trial completed`, retaining the rejection history and the original constraints. `resume()` does not relax a failed constraint.

Checkpoints use pickle: load your own trusted files with a compatible Python/SDK/Optuna environment and importable settings. `result.save("report.json")` remains a portable metrics/parameter report; that JSON is not a resumable checkpoint. These methods continue **parameter optimization**. Each candidate still starts a fresh controller with the same online learning protocol; checkpoints do not contain the selected controller's trained neural weights.

## Live search progress

Progress is enabled by default in `ControllerTuner.optimize`, `ControlOptimizer.optimize` and `AgentClass.optimize`. Jupyter shows a widget when available; terminals use a text bar. It displays completed trials out of the budget, elapsed time, ETA, the best feasible objective and its trial number, and the rejected-trial count. If no candidate meets all constraints yet, the best field says `no feasible trial`.

```python
result = tuner.optimize(n_trials=100, target=0.6, show_progress_bar=True)
# For a quiet batch run:
# result = tuner.optimize(n_trials=100, show_progress_bar=False)
```

ETA becomes available after the first completed trial and is updated as trials finish. It estimates time to exhaust the trial budget; early stopping and different per-trial runtimes can shorten/change it. The automatic best-controller replay follows the search and is not included in its ETA. On an early stop, the bar preserves the actual evaluated count (e.g. 8/100), names the stopping reason and clears the remaining ETA. Invalid trials count as processed but cannot become the displayed best. Progress does not depend on Optuna log verbosity.

## When no candidate meets the constraints

`constraints={"cpi": 0.6}` requires CPI ≤ 0.6 in every evaluated seed/scenario. If all candidates exceed that value, the optimizer raises an error rather than selecting an infeasible controller. CPI is a weighted combination of ISE, ITAE and overshoot; `normalize=True` normalizes the step amplitude, **not CPI into [0, 1]**. A CPI limit is not a relative-error percentage.

The error now lists violated bounds, the number of evaluated cases and the lowest observed values. For the tutorial's B737 pitch step, a 12-trial search with the 0.6 ceiling observed a lowest CPI of 3.30518: all 12 trials violated the bound. This describes that search; it does not establish that 0.6 is unattainable with another controller configuration.

After a failed search, inspect the retained study without rerunning simulations:

```python
tuner.diagnostics()
```

The returned dictionary includes trial states, per-constraint `upper_bound`, `violating_cases`, `evaluated_cases`, `lowest_observed`, and rejection reasons. `lowest_fully_evaluated` includes only trials that evaluated every configured seed/scenario with a finite objective. It explicitly marks feasibility; rejected candidates remain rejected. Individual constraint minima may come from different candidates.

If 0.6 is a desired stopping score and you want the best feasible result when the budget ends, create the tuner without the hard CPI ceiling and run `tuner.optimize(n_trials=100, target=0.6)`. The remaining constraints still apply, and the returned CPI may exceed 0.6. If 0.6 is a mandatory requirement, retain the ceiling and revise the search/controller setup instead. Diagnostics never change the configured bounds.

## Controllers and physical adapters

| Profile | Environments and scope | Default search parameters |
|---|---|---|
| `aa_indi` | Nonlinear B737/B747; `phi`, `theta`, `psi`, `p`, `q`, `r` | `rate_gain`, `cutoff_hz`, `outer_gain`, `covariance_init` |
| `aidi` | Same aircraft/states, native rate-command boundary | `rate_gain`, `cutoff_hz`, `outer_gain`, `rls_cov_init`, `config.rls_sigma0` |
| `iadp` | Linear F16/B747, standard nonlinear F16/B737/B747 | `gamma`, `forgetting`, `control_weight`, `track_weight`, `phi_init`, `policy_eval_every` |
| `imgdhp` / `im_gdhp` | Same physical environments | `actor_lr`, `critic_lr`, `history_length`, `track_weight`, `control_weight`, `config.gamma`, `config.beta_lambda`, `config.forgetting` |
| `et_dhp` | Same physical environments | `actor_lr`, `critic_lr`, `rho`, `track_weight`, `control_weight`, `config.trigger_floor`, `config.gamma` |
| `ihdp` | Same physical environments; one online episode | `actor_lr`, `critic_lr`, `track_weight`, `gamma`, `hidden_size`, `excitation_amplitude`, `actor_settings.learning_rate_decay`, `critic_settings.learning_rate_decay` |
| `mpc` | Same physical environments; nominal linear prediction model | `horizon`, `control_weight`, `track_weight`, `terminal_weight`, `lr` |
| `hdp` | `ImprovedB747-v0`, `theta` only | `actor_lr`, `critic_lr`, `gamma`, `hidden_size`, `exploration_std` |

Unsupported combinations fail explicitly. Boeing state-space profiles observe body velocities/rates and attitude (nine channels), excluding drifting Earth-fixed positions. They do not implement position guidance. Angular F16 requires the standard 14-state, three-surface model with constant thrust. Nonlinear longitudinal F16 uses physical observations without damage metadata or a custom feedforward callback. AA-INDI/AIDI cannot simultaneously receive an angle and its body-rate command on the same axis.

State-space profiles subtract the initial physical state, scale angles/rates by one degree (tracked states by step amplitude), and normalize command increments around trim. Actual commands respect environment/actuator limits. Native environment observations and aerodynamic equations are unchanged. For the nonlinear F16, configure trim and `control_bias` yourself. Nominal-prior controllers require a nonzero one-step input Jacobian: include actuator states or use RK4. Reduced observed-state priors hold unobserved states at trim.

The iADP prior uses the healthy local model and a discounted Riccati value estimate; subsequent RLS and policy evaluation remain active. IM-GDHP uses its documented `identifier_mode="output"` extension for known discontinuous commands. ET-DHP pretrains its plant network on a seeded nominal transition dataset (`model_samples=512`, `model_epochs=200`); event-driven actor/critic learning continues online. `online_model_fit=True` enables its optional model-learning extension. MPC keeps a nominal linear model and does not preview future steps. AA-INDI/AIDI share a proportional outer attitude loop (0.35 s⁻¹ by default), body-rate command limit 3°/s and surface slew limit 20°/s; inner identification stays active.

These are explicit application profiles, not universal policy-quality guarantees or replacements for the agents' paper equations.

```python
print(ControllerTuner.available_controllers())
print(ControllerTuner.profile("iadp"))
```

## Search all active controller settings

The previous two-parameter IHDP profile was only a narrow preset, not the algorithm's parameter set. The automatic IHDP search now varies **eight parameters**: both initial learning rates, their decay factors, gamma, tracking cost, hidden width, and excitation amplitude. The other controllers also have broader automatic spaces (see the table above). `search_space` lets you select any active option or native path, not only keys in the default space. Searching every available parameter simultaneously is usually inefficient; first choose the parameter groups relevant to the observed error.

### IHDP: architecture, cost, learning and identification

```python
from tensoraerospace.optimization import ControllerTuner, Step, Float, Int, Categorical

profile = ControllerTuner.profile("ihdp")
print(profile["search_space"])          # automatic search ranges
print(profile["tunable_parameters"])   # shortcuts + native constructor paths

space = {
    "actor_lr": Float(1e-4, 0.3, log=True),
    "critic_lr": Float(1e-4, 0.1, log=True),
    "gamma": Float(0.8, 0.999),
    "track_weight": Float(0.1, 10.0, log=True),
    "actor_settings.layers.0": Int(4, 32, step=4),
    "critic_settings.layers.0": Int(4, 32, step=4),
    "actor_settings.activations.0": Categorical(["tanh", "relu"]),
    "actor_settings.learning_rate_decay": Float(0.995, 1.0),
    "critic_settings.learning_rate_decay": Float(0.995, 1.0),
    "actor_settings.learning_rate_min": Float(1e-7, 1e-5, log=True),
    "critic_settings.learning_rate_min": Float(1e-8, 1e-5, log=True),
    "warmup_steps": Int(5, 40),
    "excitation_amplitude": Float(0.001, 0.03, log=True),
    "actor_settings.pulse_length_3211": Int(5, 100),
    "incremental_settings.window_size": Int(20, 60, step=5),
    "actor_settings.WB_limits": Float(1.0, 10.0),
    "critic_settings.WB_limits": Float(1.0, 10.0),
}

tuner = ControllerTuner(
    env="NonlinearB737-v0",
    env_kwargs={"trim_at": (10000.0, 600.0), "dt": 0.02},
    reference={"theta": Step(1.0, at=20.0, unit="deg")},
    controller="ihdp",
    method="tpe",  # also: annealing, genetic, random
    metric="cpi",
    duration=40.0,
    search_space=space,
    seed=42,
)
result = tuner.optimize(n_trials=120, target=3.7)
print(result.best_params)
print(result.best_value, result.best_value <= 3.7)
result.plot_response()
```

This is a configurable search experiment, not a claim that 120 trials will reach CPI 3.7. A mandatory threshold belongs in `constraints={"cpi": 3.7}`; `target=3.7` permits returning the best result if that target is missed. The simulator, step and scoring horizon are fixed throughout the search.

IHDP's quadratic cost Q, discount gamma, network parameters and identification are part of the original architecture ([Zhou et al., IMAV 2016](https://www.imavs.org/papers/2016/25.pdf), Eqs. 2, 7, 12, 35). The exposed decay factors and floors are controls for this implementation's existing numerical schedule; they are not claimed to be paper-prescribed constants. Decay never raises a selected learning rate above its initial value. The LS window must contain at least `number_states + number_inputs` samples; sufficient length alone does not guarantee informative data.

### Native paths and active modes

| Controller | Additional tunable groups / example native paths |
|---|---|
| AA-INDI | Outer gain/rate cap; `config.sigma0`, `config.forgetting_min`, `config.covariance_init`, `config.rate_feedback.1`; observer noise, drift and `config.observer.hosm_gains.0` |
| AIDI | Outer gain/rate cap; `config.rls_lambda_min`, `config.rls_lambda_max`, `config.rls_memory_length`, `config.rls_sigma0`, consistency threshold, filter and allocator settings |
| iADP | `config.Q.0.0`, `config.R.0.0`, RLS covariance/forgetting, critic window, cadence, iterations and warmup. The nominal Riccati prior is recomputed using the selected Q, R and gamma |
| IM-GDHP | Actor/critic architectures, gamma, Q/R, `config.beta_lambda`, history, covariance/forgetting, learning schedules, optimizer, exploration, warmup, target smoothing, clipping and weight limits |
| ET-DHP | Actor/critic/model architectures and learning rates, Q/R, gamma, event threshold/floor, model training budget and fitting mode, updates per event, initialization scale |
| IHDP | Independent actor/critic layers and activations, Q, gamma, schedules, warmup, excitation, weight bounds and identification window |
| HDP | Learning rates, width, exploration, gamma, custom utility weights and actor/critic episode cycles. Custom weights require `controller_options={"dhp_use_env_cost": False}` |
| MPC | Prediction horizon, tracking/control/slew cost, terminal weight, solver learning rate/iterations, optimizer, warm start and best-iterate selection |

`ControllerTuner.profile(name)["native_parameters"]` lists supported constructor paths. Numeric path components address existing vector/matrix entries: `config.Q.0.0`, `weights.S_diag.0`, `actor_settings.layers.0`. Use `controller_options` for fixed vector/architecture templates, and `search_space` for selected entries. Example: `controller_options={"actor_hidden": (16, 8)}` with `search_space={"actor_settings.layers.0": Int(8, 32, step=8)}` changes the first hidden layer only. Actor/critic output dimensions must still match the environment.

Fixed settings are removed from the automatic search, including their aliases. An explicit custom search may override the same native fixed key or one of its elements. Duplicate search names for the same setting are rejected: do not search both `actor_lr` and `actor_settings.learning_rate`. `initial_params` may be partial; the sampler supplies omitted values for the first trial.

The profiles expose the settings used by their actual interaction loop. AIDI's `predict_rates` bypasses C*/roll/sideslip/speed guidance and PCH, so those knobs are not offered here. IHDP's cascaded/integral-augmented variants, iADP's offline/sequential protocols and HDP's optional expert baseline require an explicit research protocol through `AgentClass.optimize`. Protocol fields such as dt, seeds, state/output dimensions, aircraft geometry and physical actuator bounds are not search variables. Environment physics are unchanged.

## Results and replay

`tuner.simulate()` runs the profile defaults. `tuner.simulate(params, seed=17)` runs selected settings without searching. Both return `ControlRun`: physical `time`, `reference`, `output`, commands sent as `actions`, named `states`/`units`, and `metrics`. Arrays include all timestamps; actions contain one row per transition.

`result.best_params`, `best_value`, `study`, and `best_run` expose the selected trial and a replay on the first configured seed. With multiple seeds, `best_value` is their mean and may differ from that replay's CPI. `result.simulate(seed=17)` rebuilds and retrains the chosen profile. It does not load previously trained policy weights. `result.plot_response(baseline=baseline)` compares matching experiments. `result.save(path)` exports best parameters, protocol and trial history as JSON.

## Advanced custom protocols

`ControlOptimizer`, `StepResponseMetric` and `AgentClass.optimize(search_space, evaluate, agent_kwargs=...)` remain available when a custom environment, arbitrary reference or bespoke learning protocol needs its own evaluator. They are lower-level APIs; standard step tuning uses `ControllerTuner` above. Dotted constructor paths, multiple named cases, persistent Optuna studies and independent validation are supported by `ControlOptimizer`.

::: tensoraerospace.optimization.experiment.ControllerTuner

::: tensoraerospace.optimization.experiment.Step
