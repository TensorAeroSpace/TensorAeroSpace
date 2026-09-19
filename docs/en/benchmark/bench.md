# Benchmark

Tools for objectively comparing control systems using standard metrics.

![Benchmark report example](bench.png)

## What We Evaluate
- Overshoot
- Settling time
- Damping ratio
- Steady-state error

## API

::: tensoraerospace.benchmark.ControlBenchmark
    options:
      members: true

## Usage Example

```python
from tensoraerospace.benchmark import ControlBenchmark

bench = ControlBenchmark()
metrics = bench.benchmarking_one_step(control_signal, system_signal, 1.0, dt)

print("Steady-state error:", metrics['static_error'])
print("Settling time:", metrics['settling_time'])
print("Damping ratio:", metrics['damping_degree'])
print("Overshoot:", metrics['overshoot'])

# Visualize signal comparison and metrics
bench.plot(control_signal, system_signal, 1.0, dt, tps, figsize=(15, 5))
```

!!! note "Units and Input Data"
    - `control_signal`, `system_signal` — arrays of equal length
    - `1.0` — desired steady-state value (example)
    - `dt` — sampling step; `tps` — time axis

!!! info "Backward Compatibility"
    The old method name `becnchmarking_one_step` still works as an alias for `benchmarking_one_step` for backward compatibility.


## Tracking windows and commanded steps

`ControlBenchmark.tracking_metrics(reference, output, dt, start=..., end=...)`
accepts a scalar/channel-wise reference or a full schedule and an `(N, channels)`
response. It evaluates `(start, end]`: per-channel RMSE/MAE, combined RMSE, combined
IAE and final error (`reference - output`). With `tolerance`, recovery requires all
channels to stay within the command band until the end; `None` means no recovery.
Optional `(N-1, inputs)` applied actions add RMS, peak and total variation.
Use consistent units, for example degrees throughout the comparison.

`benchmarking_step_response(reference, output, signal_val, dt)` retains the
existing step metrics and adds `command_settling_time` and `command_overshoot`.
Here `signal_val` is the pre-step reference level. These additional metrics use
the requested step amplitude, including descending steps. Settling around the
observed final output alone does not establish that the command was reached.

## Aircraft protocols in the installed library

The protocols compose native environments, public agents and `ControlBenchmark`.
They require no imports from `example` and work from any working directory after
installing this revision of `tensoraerospace`.

```python
from tensoraerospace.benchmark import B737PitchStepBenchmark, B747EngineFailureBenchmark
from tensoraerospace.aerospacemodel.b737.nonlinear import ElevatorEffectiveness

pitch = B737PitchStepBenchmark(elevator_fault=ElevatorEffectiveness(time=30, effectiveness=0.5))
env, trim, trim_action = pitch.make_env()
state, _ = env.reset(seed=pitch.seed)
A, B = env.model.linearize(state, trim_action)  # Native-unit continuous Jacobians.
env.close()

comparison = B747EngineFailureBenchmark(duration=90, fault_time=30)
result = comparison.run("AA-INDI", fault=True)
print(result["after"])
```

`B737PitchStepBenchmark` supplies `reference`, `time`, `validate_transition`,
`evaluate`, `metric_table` and Matplotlib plots (`plot_reference`, `plot_response`,
`plot_step`). The notebook shows the actual `predict → step → learn` loop.

`B747EngineFailureBenchmark` supplies healthy `nominal_trim`/`nominal_model`,
`make_env`, `make_aaindi`, `make_lqr`, `tune_baselines`, `run`, `evaluate` and
`validate_additional_cases`. Tune baselines on healthy data, then pass the selected
settings to every comparison run. `run` returns states `(steps+1, 12)`, actual
inputs `(steps, 4)`, diagnostics, events and `before`/`after`/`whole` metrics;
`before` is `None` for a failure at time zero. Each run starts fresh.

These are documented local simulation protocols, with cruise-specific gains,
ideal sensors and explicit limits. See [the B737 walkthrough](../cookbook/14_aaindi.md)
and [the B747 comparison](../cookbook/09_fault_tolerance.md) for full setup and plots.
