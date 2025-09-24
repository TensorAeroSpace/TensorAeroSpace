# Benchmark Metrics

A set of standard metrics for evaluating control-system quality: accuracy, transient speed, damping ratio, and overshoot.

## Quick Start

```python
from tensoraerospace.benchmark.function import (
    static_error, settling_time, damping_degree, overshoot
)

# control_signal and system_signal — arrays of equal length
e_ss = static_error(control_signal, system_signal)
ts   = settling_time(control_signal, system_signal, tolerance=0.02)
zeta = damping_degree(system_signal)
Ov   = overshoot(control_signal, system_signal)
```

---

## Steady-State Error

The steady-state error is the difference between the target value and the actual output in steady state:

\[ e_{ss}(t) = u(t) - y(t) \]

The closer to zero, the more precisely the system tracks the setpoint.

=== "API"

    ::: tensoraerospace.benchmark.function.static_error

---

## Damping Degree of the Transient

Measures the relative reduction in amplitude between successive peaks:

\[ \text{D} = 1 - \frac{A_n}{A_{n-1}} \]

where \(A_n\) is the current peak amplitude and \(A_{n-1}\) is the previous one. Higher values mean stronger damping.

=== "API"

    ::: tensoraerospace.benchmark.function.damping_degree

---

## Settling Time

The time required for the response to enter and remain within a tolerance band (typically ±2% or ±5%) around the steady-state value.

Algorithm:
- Estimate \(Y_{final}\) (e.g., the mean of the final samples).
- Find the first entry into the tolerance band and the moment the signal settles within the band.
- The difference between these times is the settling time.

=== "API"

    ::: tensoraerospace.benchmark.function.settling_time

---

## Overshoot

How much the response maximum exceeds the steady-state value, expressed in percent:

\[ \%OS = \frac{M - Y_{final}}{Y_{final}} \cdot 100\% \]

Lower values correspond to smoother transitions without excessive oscillations.

=== "API"

    ::: tensoraerospace.benchmark.function.overshoot

---

### Interpretation Tips

- **e_ss ≈ 0**: good steady-state accuracy; check for sensor drift/bias.
- **Low %OS and low T_s**: fast, non-oscillatory response.
- **High D**: strong damping; ensure it is not excessively slow.
