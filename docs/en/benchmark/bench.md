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

