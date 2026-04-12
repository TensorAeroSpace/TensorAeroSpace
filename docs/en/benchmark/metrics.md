# Benchmark Metrics

The `tensoraerospace.benchmark.function` module provides **17 functions** for evaluating control-system quality. These cover classical time-domain characteristics (overshoot, settling time, rise time, peak time), steady-state accuracy, damping and oscillation analysis, integral error criteria, and a composite performance index. Every metric operates on two NumPy arrays: a **control (reference) signal** and a **system (response) signal** of equal length.

All functions are importable from `tensoraerospace.benchmark.function`.

---

## Quick Start

The example below creates a step-response scenario using the `LinearLongitudinalF16` environment with a PID controller, then computes every available metric.

```python
import numpy as np
import gymnasium as gym

from tensoraerospace.agent.pid import PID
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.benchmark.function import (
    find_step_function,
    get_lower_upper_bound,
    find_longest_repeating_series,
    overshoot,
    settling_time,
    rise_time,
    peak_time,
    maximum_deviation,
    static_error,
    steady_state_value,
    damping_degree,
    oscillation_count,
    integral_absolute_error,
    integral_squared_error,
    integral_time_absolute_error,
    performance_index,
)

# --- Environment setup ---
dt = 0.01
tp = generate_time_period(tn=40, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)

reference_signals = np.reshape(
    unit_step(degree=5, tp=tp, time_step=10, output_rad=True),
    [1, -1],
)

env = gym.make(
    "LinearLongitudinalF16-v0",
    number_time_steps=number_time_steps,
    use_reward=False,
    initial_state=[[0], [0], [0]],
    reference_signal=reference_signals,
    state_space=["theta", "alpha", "q"],
    output_space=["theta", "alpha", "q"],
    tracking_states=["alpha"],
)
env.reset()

pid = PID(env, kp=-14.29, ki=-8.24, kd=-1.30, dt=dt)

# --- Run the simulation ---
xt, _ = env.reset()
for step in range(number_time_steps - 2):
    setpoint = reference_signals[0, step]
    ut = pid.select_action(setpoint, xt[1])
    xt, reward, terminated, truncated, info = env.step(np.array([ut]))

# --- Extract signals ---
system_signal = env.unwrapped.model.get_state("alpha", to_deg=True)
control_signal = np.rad2deg(reference_signals[0])[: len(system_signal)]

# --- Extract the step portion ---
ctrl_step, sys_step = find_step_function(control_signal, system_signal, signal_val=0)

# --- Compute ALL metrics ---
print(f"Overshoot:           {overshoot(ctrl_step, sys_step):.2f} %")
print(f"Settling time:       {settling_time(ctrl_step, sys_step)  * dt:.2f} s")
print(f"Rise time:           {rise_time(ctrl_step, sys_step) * dt:.3f} s")
print(f"Peak time:           {peak_time(sys_step) * dt:.3f} s")
print(f"Maximum deviation:   {maximum_deviation(ctrl_step, sys_step):.4f}")
print(f"Static error:        {static_error(ctrl_step, sys_step):.6f}")
print(f"Steady-state value:  {steady_state_value(ctrl_step):.4f}")
print(f"Damping degree:      {damping_degree(sys_step):.4f}")
print(f"Oscillation count:   {oscillation_count(sys_step)}")
print(f"IAE:                 {integral_absolute_error(ctrl_step, sys_step):.2f}")
print(f"ISE:                 {integral_squared_error(ctrl_step, sys_step):.2f}")
print(f"ITAE:                {integral_time_absolute_error(ctrl_step, sys_step, dt):.2f}")
print(f"Performance index:   {performance_index(ctrl_step, sys_step, dt):.3f}")
```

You can also use the high-level `ControlBenchmark` class to compute all metrics at once:

```python
from tensoraerospace.benchmark import ControlBenchmark

bench = ControlBenchmark()
metrics = bench.benchmarking_one_step(control_signal, system_signal, signal_val=0, dt=dt)
print(metrics)
```

---

## Time-Domain Metrics

### Overshoot

How much the response exceeds the steady-state value, expressed as a percentage.

$$
\sigma = \frac{M - y_{\text{final}}}{y_{\text{final}}} \cdot 100\%
$$

where \(M = \max(y(t))\) is the peak value of the system signal, and \(y_{\text{final}}\) is estimated as the mean of the last 10 % of the control (reference) signal.

Lower values indicate smoother transients with less oscillation.

**API**

```python
def overshoot(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
) -> float
```

| Parameter | Type | Description |
|---|---|---|
| `control_signal` | `np.ndarray` | Reference / command signal |
| `system_signal` | `np.ndarray` | System response signal |
| **Returns** | `float` | Overshoot in percent |

**Example**

```python
from tensoraerospace.benchmark.function import overshoot

ov = overshoot(control_signal, system_signal)
print(f"Overshoot: {ov:.2f}%")
```

---

### Settling Time

The number of samples (index) at which the response enters and permanently remains within a tolerance band around the steady-state value.

The tolerance band is defined as:

$$
y_{\text{final}} \cdot (1 - \delta) \;\leq\; y(t) \;\leq\; y_{\text{final}} \cdot (1 + \delta)
$$

where \(\delta\) is the `threshold` parameter (default 5 %). The function finds the longest consecutive run of indices inside this band and returns the start index. Multiply by `dt` to convert to seconds.

**API**

```python
def settling_time(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
    threshold: float = 0.05,
) -> Optional[int]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `control_signal` | `np.ndarray` | -- | Reference signal |
| `system_signal` | `np.ndarray` | -- | System response signal |
| `threshold` | `float` | `0.05` | Relative tolerance (fraction of steady-state value) |
| **Returns** | `Optional[int]` | -- | Index at which settling begins; `None` if never settled |

**Example**

```python
from tensoraerospace.benchmark.function import settling_time

ts = settling_time(control_signal, system_signal, threshold=0.05)
if ts is not None:
    print(f"Settling time: {ts * dt:.2f} s")
```

---

### Rise Time

The number of samples required for the response to travel from `low_threshold` to `high_threshold` of the final value.

$$
t_r = t_{high} - t_{low}
$$

where \(t_{low}\) is the first index at which \(y(t) \geq y_{\text{final}} \cdot \text{low\_threshold}\) and \(t_{high}\) is the first index at which \(y(t) \geq y_{\text{final}} \cdot \text{high\_threshold}\).

**API**

```python
def rise_time(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
    low_threshold: float = 0.1,
    high_threshold: float = 0.9,
) -> Optional[float]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `control_signal` | `np.ndarray` | -- | Reference signal |
| `system_signal` | `np.ndarray` | -- | System response signal |
| `low_threshold` | `float` | `0.1` | Lower fraction of steady-state value |
| `high_threshold` | `float` | `0.9` | Upper fraction of steady-state value |
| **Returns** | `Optional[float]` | -- | Rise time in samples; `None` if thresholds not reached |

**Example**

```python
from tensoraerospace.benchmark.function import rise_time

tr = rise_time(control_signal, system_signal)
if tr is not None:
    print(f"Rise time: {tr * dt:.3f} s")
```

---

### Peak Time

The sample index of the first peak in the response. If no peaks are detected by `scipy.signal.find_peaks`, the index of the global maximum is returned.

$$
t_p = \arg\max_{t} \; y(t)  \quad \text{(first peak)}
$$

**API**

```python
def peak_time(
    system_signal: np.ndarray,
) -> Optional[int]
```

| Parameter | Type | Description |
|---|---|---|
| `system_signal` | `np.ndarray` | System response signal |
| **Returns** | `Optional[int]` | Index of first peak (or global max) |

**Example**

```python
from tensoraerospace.benchmark.function import peak_time

tp = peak_time(system_signal)
print(f"Peak time: {tp * dt:.3f} s")
```

---

### Maximum Deviation

The largest absolute difference between the system response and the steady-state reference at any point in time.

$$
\Delta_{\max} = \max_t \; \lvert y(t) - y_{\text{final}} \rvert
$$

**API**

```python
def maximum_deviation(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
) -> float
```

| Parameter | Type | Description |
|---|---|---|
| `control_signal` | `np.ndarray` | Reference signal |
| `system_signal` | `np.ndarray` | System response signal |
| **Returns** | `float` | Maximum absolute deviation |

**Example**

```python
from tensoraerospace.benchmark.function import maximum_deviation

d_max = maximum_deviation(control_signal, system_signal)
print(f"Maximum deviation: {d_max:.4f}")
```

---

## Steady-State Metrics

### Static Error

The difference between the target value and the actual output in steady state.

$$
e_{ss} = r_{\text{final}} - y_{\text{final}}
$$

Both \(r_{\text{final}}\) and \(y_{\text{final}}\) are estimated as the mean of the last 10 % of their respective signals. A value close to zero means the controller tracks the setpoint with high accuracy.

**API**

```python
def static_error(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
) -> float
```

| Parameter | Type | Description |
|---|---|---|
| `control_signal` | `np.ndarray` | Reference signal |
| `system_signal` | `np.ndarray` | System response signal |
| **Returns** | `float` | Signed static error |

**Example**

```python
from tensoraerospace.benchmark.function import static_error

e_ss = static_error(control_signal, system_signal)
print(f"Static error: {e_ss:.6f}")
```

---

### Steady-State Value

Estimates the steady-state (final) value of a signal by averaging its tail.

$$
y_{ss} = \frac{1}{N_{\text{tail}}} \sum_{k=\lfloor (1 - p) \cdot N \rfloor}^{N-1} x[k]
$$

where \(p\) is the `percentage` parameter and \(N\) is the total number of samples.

**API**

```python
def steady_state_value(
    control_signal: np.ndarray,
    percentage: float = 0.1,
) -> float
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `control_signal` | `np.ndarray` | -- | Signal to summarize |
| `percentage` | `float` | `0.1` | Fraction of the tail used for averaging (last 10 % by default) |
| **Returns** | `float` | -- | Estimated steady-state value |

**Example**

```python
from tensoraerospace.benchmark.function import steady_state_value

y_ss = steady_state_value(system_signal)
print(f"Steady-state value: {y_ss:.4f}")
```

---

## Damping and Oscillation

### Damping Degree

Measures the relative reduction in amplitude between successive peaks of the response. Higher values indicate stronger damping.

$$
D = 1 - \frac{A_{n}}{A_{n-1}}
$$

The function finds all peaks using `scipy.signal.find_peaks`, computes the ratio for each consecutive pair, and returns the **mean** across all pairs. Returns `0.0` if fewer than two peaks exist.

**API**

```python
def damping_degree(
    system_signal: np.ndarray,
) -> float
```

| Parameter | Type | Description |
|---|---|---|
| `system_signal` | `np.ndarray` | System response signal |
| **Returns** | `float` | Average damping degree (0 to 1 for well-behaved systems) |

**Example**

```python
from tensoraerospace.benchmark.function import damping_degree

dd = damping_degree(system_signal)
print(f"Damping degree: {dd:.4f}")
```

---

### Oscillation Count

Estimates the number of full oscillation cycles in the transient response. The function counts peaks **and** valleys (via `scipy.signal.find_peaks` on the signal and its negation) whose amplitude exceeds `threshold`, then divides the total number of extrema by two.

$$
N_{\text{osc}} = \left\lfloor \frac{N_{\text{peaks}} + N_{\text{valleys}}}{2} \right\rfloor
$$

**API**

```python
def oscillation_count(
    system_signal: np.ndarray,
    threshold: float = 0.01,
) -> int
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `system_signal` | `np.ndarray` | -- | System response signal |
| `threshold` | `float` | `0.01` | Minimum peak magnitude to be counted |
| **Returns** | `int` | -- | Estimated number of full oscillation cycles |

**Example**

```python
from tensoraerospace.benchmark.function import oscillation_count

n_osc = oscillation_count(system_signal)
print(f"Oscillation count: {n_osc}")
```

---

## Integral Quality Criteria

Integral criteria accumulate the tracking error over the entire simulation. They are especially useful for comparing controllers: **lower values are always better**.

### IAE -- Integral Absolute Error

Penalizes all errors equally regardless of sign.

$$
\text{IAE} = \sum_{k=0}^{N-1} \lvert r[k] - y[k] \rvert
$$

**When to use:** General-purpose comparison where you want a single number representing total tracking accuracy. Equally weights early and late errors.

**API**

```python
def integral_absolute_error(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
) -> float
```

| Parameter | Type | Description |
|---|---|---|
| `control_signal` | `np.ndarray` | Reference signal |
| `system_signal` | `np.ndarray` | System response signal |
| **Returns** | `float` | IAE value |

**Example**

```python
from tensoraerospace.benchmark.function import integral_absolute_error

iae = integral_absolute_error(control_signal, system_signal)
print(f"IAE: {iae:.2f}")
```

---

### ISE -- Integral Squared Error

Penalizes large errors disproportionately more than small ones.

$$
\text{ISE} = \sum_{k=0}^{N-1} \bigl( r[k] - y[k] \bigr)^2
$$

**When to use:** When large deviations are unacceptable (e.g., safety-critical systems). ISE will strongly penalize spikes and overshoot.

**API**

```python
def integral_squared_error(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
) -> float
```

| Parameter | Type | Description |
|---|---|---|
| `control_signal` | `np.ndarray` | Reference signal |
| `system_signal` | `np.ndarray` | System response signal |
| **Returns** | `float` | ISE value |

**Example**

```python
from tensoraerospace.benchmark.function import integral_squared_error

ise = integral_squared_error(control_signal, system_signal)
print(f"ISE: {ise:.2f}")
```

---

### ITAE -- Integral Time-weighted Absolute Error

Weights the absolute error by time, so that **late errors are penalized more heavily** than early ones. This encourages fast settling.

$$
\text{ITAE} = \sum_{k=0}^{N-1} k \cdot \Delta t \cdot \lvert r[k] - y[k] \rvert
$$

**When to use:** When you need the controller to settle quickly and want persistent steady-state drift to dominate the score.

**API**

```python
def integral_time_absolute_error(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
    dt: float = 1.0,
) -> float
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `control_signal` | `np.ndarray` | -- | Reference signal |
| `system_signal` | `np.ndarray` | -- | System response signal |
| `dt` | `float` | `1.0` | Time step used for the time weight |
| **Returns** | `float` | -- | ITAE value |

**Example**

```python
from tensoraerospace.benchmark.function import integral_time_absolute_error

itae = integral_time_absolute_error(control_signal, system_signal, dt=0.01)
print(f"ITAE: {itae:.2f}")
```

---

### Performance Index

A composite quality index that combines ISE, ITAE, and overshoot with fixed weights. **Lower is better.**

$$
J = 0.4 \cdot \text{ISE} + 0.4 \cdot \text{ITAE} + 0.2 \cdot \lvert \sigma \rvert
$$

where \(\sigma\) is the overshoot in percent.

This single number provides a balanced assessment: ISE punishes large deviations, ITAE punishes slow settling, and overshoot punishes excessive peaks.

**API**

```python
def performance_index(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
    dt: float = 1.0,
) -> float
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `control_signal` | `np.ndarray` | -- | Reference signal |
| `system_signal` | `np.ndarray` | -- | System response signal |
| `dt` | `float` | `1.0` | Time step for the ITAE component |
| **Returns** | `float` | -- | Composite quality index (lower is better) |

**Example**

```python
from tensoraerospace.benchmark.function import performance_index

pi = performance_index(control_signal, system_signal, dt=0.01)
print(f"Performance index: {pi:.3f}")
```

---

## Utility Functions

### find_step_function

Extracts the step-response portion of a signal pair by trimming all leading samples where the control signal is at or below `signal_val`.

**API**

```python
def find_step_function(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
    signal_val: float = 0,
) -> Tuple[np.ndarray, np.ndarray]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `control_signal` | `np.ndarray` | -- | Reference signal |
| `system_signal` | `np.ndarray` | -- | System response signal |
| `signal_val` | `float` | `0` | Threshold below which samples are discarded |
| **Returns** | `Tuple[np.ndarray, np.ndarray]` | -- | Trimmed (control, system) pair |

**Example**

```python
from tensoraerospace.benchmark.function import find_step_function

ctrl_step, sys_step = find_step_function(control_signal, system_signal, signal_val=0)
print(f"Step portion length: {len(ctrl_step)} samples")
```

---

### get_lower_upper_bound

Computes the tolerance band (lower and upper bounds) around the final value of the control signal. Useful for visualizing the settling-time corridor.

$$
\text{upper}[k] = r_{\text{final}} \cdot (1 + \varepsilon) \qquad
\text{lower}[k] = r_{\text{final}} \cdot (1 - \varepsilon)
$$

where \(r_{\text{final}}\) is the last sample of `control_signal`.

**API**

```python
def get_lower_upper_bound(
    control_signal: np.ndarray,
    epsilon: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `control_signal` | `np.ndarray` | -- | Reference signal |
| `epsilon` | `float` | `0.05` | Half-width of the tolerance band as a fraction |
| **Returns** | `Tuple[np.ndarray, np.ndarray]` | -- | `(lower, upper)` arrays of the same shape |

**Example**

```python
from tensoraerospace.benchmark.function import get_lower_upper_bound

lower, upper = get_lower_upper_bound(control_signal, epsilon=0.05)
```

---

### find_longest_repeating_series

Internal helper that finds the longest run of consecutive integers in a list. Used by `settling_time` to locate the first sustained entry into the tolerance band.

**API**

```python
def find_longest_repeating_series(
    numbers: list,
) -> tuple
```

| Parameter | Type | Description |
|---|---|---|
| `numbers` | `list` | Sorted list of integers |
| **Returns** | `tuple` | `(start, end)` of the longest consecutive run; `(0, 0)` if empty |

**Example**

```python
from tensoraerospace.benchmark.function import find_longest_repeating_series

start, end = find_longest_repeating_series([0, 1, 2, 5, 6, 7, 8, 10])
print(f"Longest consecutive run: indices {start} to {end}")
# Output: Longest consecutive run: indices 5 to 8
```

---

## Visualization Example

The following self-contained script runs a PID-controlled step response and creates an annotated matplotlib plot showing key metrics. Copy and paste it directly.

```python
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tensoraerospace.agent.pid import PID
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.benchmark.function import (
    find_step_function,
    get_lower_upper_bound,
    overshoot,
    settling_time,
    rise_time,
    peak_time,
    maximum_deviation,
    static_error,
    steady_state_value,
    damping_degree,
    oscillation_count,
    integral_absolute_error,
    integral_squared_error,
    integral_time_absolute_error,
    performance_index,
)

# ── 1. Build environment and controller ──────────────────────
dt = 0.01
tp = generate_time_period(tn=40, dt=dt)
tps = np.array(convert_tp_to_sec_tp(tp, dt=dt))
n_steps = len(tp)

reference_signals = np.reshape(
    unit_step(degree=5, tp=tp, time_step=10, output_rad=True),
    [1, -1],
)

env = gym.make(
    "LinearLongitudinalF16-v0",
    number_time_steps=n_steps,
    use_reward=False,
    initial_state=[[0], [0], [0]],
    reference_signal=reference_signals,
    state_space=["theta", "alpha", "q"],
    output_space=["theta", "alpha", "q"],
    tracking_states=["alpha"],
)

pid = PID(env, kp=-14.29, ki=-8.24, kd=-1.30, dt=dt)

# ── 2. Run simulation ───────────────────────────────────────
xt, _ = env.reset()
for step in range(n_steps - 2):
    setpoint = reference_signals[0, step]
    ut = pid.select_action(setpoint, xt[1])
    xt, *_ = env.step(np.array([ut]))

system_signal = env.unwrapped.model.get_state("alpha", to_deg=True)
control_signal = np.rad2deg(reference_signals[0])[: len(system_signal)]

# ── 3. Extract step portion and compute metrics ─────────────
ctrl, sys = find_step_function(control_signal, system_signal, signal_val=0)
time = np.arange(len(ctrl)) * dt

ov    = overshoot(ctrl, sys)
ts    = settling_time(ctrl, sys)
tr    = rise_time(ctrl, sys)
tp_   = peak_time(sys)
md    = maximum_deviation(ctrl, sys)
e_ss  = static_error(ctrl, sys)
y_ss  = steady_state_value(ctrl)
lower, upper = get_lower_upper_bound(ctrl, epsilon=0.05)

# ── 4. Annotated plot ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

# Reference and response
ax.plot(time, ctrl, "k--", linewidth=2, label="Reference signal")
ax.plot(time, sys,  "b-",  linewidth=1.5, label="System response")

# Settling band (+-5%)
ax.fill_between(time, lower, upper, color="green", alpha=0.10, label="Settling band (+/-5%)")

# Steady-state value line
ax.axhline(y=y_ss, color="gray", linestyle=":", linewidth=1, label=f"Steady-state = {y_ss:.2f}")

# Overshoot point
if tp_ is not None:
    ax.plot(time[tp_], sys[tp_], "rv", markersize=12, zorder=5,
            label=f"Peak (overshoot = {ov:.1f}%)")
    ax.annotate(f"  {sys[tp_]:.2f}",
                xy=(time[tp_], sys[tp_]), fontsize=9, color="red")

# Settling time marker
if ts is not None and ts < len(time):
    ax.axvline(x=time[ts], color="green", linewidth=1.5, linestyle="--",
               label=f"Settling time = {time[ts]:.2f} s")

# Rise time markers
if tr is not None:
    y_final = np.mean(ctrl[int(0.9 * len(ctrl)):])
    low_val = y_final * 0.1
    high_val = y_final * 0.9
    low_idx = np.where(sys >= low_val)[0]
    high_idx = np.where(sys >= high_val)[0]
    if len(low_idx) > 0 and len(high_idx) > 0:
        t_low = time[low_idx[0]]
        t_high = time[high_idx[0]]
        ax.annotate("", xy=(t_high, y_final * 0.5), xytext=(t_low, y_final * 0.5),
                     arrowprops=dict(arrowstyle="<->", color="orange", lw=2))
        ax.text((t_low + t_high) / 2, y_final * 0.55,
                f"Rise time\n{tr * dt:.3f} s",
                ha="center", fontsize=9, color="orange", fontweight="bold")

# Peak time marker
if tp_ is not None:
    ax.axvline(x=time[tp_], color="red", linewidth=1, linestyle=":",
               label=f"Peak time = {time[tp_]:.2f} s")

# Static error annotation
ax.annotate(f"Static error = {e_ss:.4f}",
            xy=(time[-1], sys[-1]), xytext=(time[-1] - 5, sys[-1] - 0.8),
            fontsize=9, color="purple",
            arrowprops=dict(arrowstyle="->", color="purple"))

ax.set_xlabel("Time [s]", fontsize=12)
ax.set_ylabel("Amplitude [deg]", fontsize=12)
ax.set_title("Step Response Analysis with Benchmark Metrics", fontsize=14)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Interpretation Guide

The table below summarizes typical value ranges. Exact thresholds depend on the application; aerospace systems generally demand tighter bounds than general robotics.

| Metric | Excellent | Good | Acceptable | Poor |
|---|---|---|---|---|
| **Overshoot** (%) | < 5 | 5 -- 15 | 15 -- 25 | > 25 |
| **Settling time** (s) | < 2 | 2 -- 5 | 5 -- 15 | > 15 |
| **Rise time** (s) | < 1 | 1 -- 3 | 3 -- 10 | > 10 |
| **Peak time** (s) | < 1 | 1 -- 3 | 3 -- 10 | > 10 |
| **Maximum deviation** | < 0.5 | 0.5 -- 1.0 | 1.0 -- 2.0 | > 2.0 |
| **Static error** | < 0.01 | 0.01 -- 0.05 | 0.05 -- 0.2 | > 0.2 |
| **Damping degree** | > 0.8 | 0.5 -- 0.8 | 0.2 -- 0.5 | < 0.2 |
| **Oscillation count** | 0 -- 1 | 2 -- 3 | 4 -- 6 | > 6 |
| **IAE** | Low | -- | -- | High (context-dependent) |
| **ISE** | Low | -- | -- | High (context-dependent) |
| **ITAE** | Low | -- | -- | High (context-dependent) |
| **Performance index** | Low | -- | -- | High (lower is always better) |

**Key trade-offs to keep in mind:**

- Reducing **overshoot** often increases **rise time** and **settling time**.
- A very low **static error** may require high integral gain, which can increase **oscillation count**.
- **ISE** favors controllers that suppress large initial errors; **ITAE** favors controllers that settle quickly with minimal late-stage error.
- The **performance index** balances all three concerns (ISE, ITAE, overshoot) and is useful as a single-number ranking when comparing controllers.
