# Signals

Generators of standard test signals for modeling, identification, and verification of control systems in `TensorAeroSpace`.

## Quick Start

```python
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standart import unit_step, sinusoid
from tensoraerospace.signals.random import full_random_signal

# Time axis 0..20 s (default step)
tp = generate_time_period(tn=20)

# Step of 5° at t = 10 s
u_step = unit_step(degree=5, tp=tp, time_step=10, output_rad=False)

# Sine wave with amplitude 10 units, frequency 0.01 Hz
u_sin = sinusoid(tp=tp, amplitude=10, frequency=0.01)

# Random signal with random frequency and amplitude
u_rand = full_random_signal(0, 0.01, 20, (-0.5, 0.5), (-0.5, 0.5))
```

!!! tip "Units"
    For functions that support it, the `output_rad=False` parameter returns angles in degrees. Set it to `True` to get radians.

---

## Step Signal

A classic step input for exciting transient processes.

=== "API"

    ::: tensoraerospace.signals.standart.unit_step

=== "Example"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import unit_step

    tp = generate_time_period(tn=20)
    u = unit_step(degree=5, tp=tp, time_step=10, output_rad=False)
    ```

![Generated step signal](img/unit_step.png)

---

## Sinusoidal Signal

Used for frequency analysis and testing linear subsystems.

=== "API"

    ::: tensoraerospace.signals.standart.sinusoid

=== "Example"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import sinusoid

    tp = generate_time_period(tn=20)
    u = sinusoid(tp=tp, amplitude=10, frequency=0.01)
    ```

![Sinusoidal signal](img/sinusoid.png)

---

## Random Signal by Frequency and Amplitude

Generates a random test input with configurable frequency and amplitude ranges.

=== "API"

    ::: tensoraerospace.signals.random.full_random_signal

=== "Example"

    ```python
    from tensoraerospace.signals.random import full_random_signal

    # full_random_signal(t0, dt, tn, amplitude_range, frequency_range)
    u = full_random_signal(0, 0.01, 20, (-0.5, 0.5), (-0.5, 0.5))
    ```

![Random signal by frequency and amplitude](img/full_random.png)

---

### Notes

- Use `tensoraerospace.utils.generate_time_period` to build the time axis.
- All functions return an array of signal values that aligns with the `tp` time axis.
