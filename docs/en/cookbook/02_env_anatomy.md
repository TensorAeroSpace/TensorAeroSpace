# Recipe 02 — Anatomy of a TensorAeroSpace env

Walk through every lever of `gym.make('<aircraft>-v0', ...)` with a small copy-and-run snippet at each step. After this recipe you'll know which argument does what on any env you meet (F-16, B747, rocket, UAV, …).

**Related.** [Recipe 01](01_hello.md) — first run, [Recipe 03](03_reference_signals.md) — the `reference_signal`, [Recipe 10](10_custom_plant.md) — writing your own env.

## The mental model

Every env maintains three parallel views of the state:

- **Internal state** — what the model integrates (e.g. `[theta, alpha, q, stab]`).
- **Observation** — the slice the agent sees, controlled by `state_space`.
- **Output** — the slice the reward function uses, controlled by `output_space`.

Separating observation from output lets you train on `(alpha, q)` but reward-shape on `theta`.

## Step 1 — Build a minimal env and inspect its spaces

```python
import gymnasium as gym
import numpy as np
import tensoraerospace  # noqa: F401

from tensoraerospace.signals.standart import unit_step
from tensoraerospace.utils import generate_time_period

dt = 0.01
tp = generate_time_period(tn=5, dt=dt)
ref = np.reshape(unit_step(tp=tp, degree=2, time_step=1.0, output_rad=True), (1, -1))

env = gym.make(
    'LinearLongitudinalF16-v0',
    number_time_steps=len(tp),
    use_reward=False,
    initial_state=[[0], [0], [0]],
    reference_signal=ref,
    state_space=['theta', 'alpha', 'q'],
    output_space=['theta', 'alpha', 'q'],
    tracking_states=['alpha'],
)
env.reset()

print('observation_space:', env.observation_space)
print('action_space:     ', env.action_space)
print('ref shape:        ', ref.shape)
```

**Expected output:**

```
observation_space: Box(-inf, inf, (3,), float64)
action_space:      Box(-0.436, 0.436, (1,), float64)
ref shape:         (1, 500)
```

Three-element observation matches `state_space=['theta', 'alpha', 'q']`. Single-element action matches the default `stab`. Reference shape is `(n_tracking, T)`.

## Step 2 — Every argument, in depth

### `number_time_steps: int`

Simulation horizon in **ticks**, not seconds. Use `generate_time_period(tn=T_seconds, dt=dt)` then `len(tp)`.

### `dt: float`

Control / integration step. Must match the agent's `dt`. `0.01` (100 Hz) is the repo default.

### `initial_state: list[list[float]]`

Column vector, shape `(n_full_state, 1)`. Each sub-list is a 1-element list. The order matches the env's internal state layout (documented per env).

### `reference_signal: np.ndarray`

Desired trajectory, shape `(len(tracking_states), number_time_steps)`. Always use `np.reshape(signal, (1, -1))` for single-channel tracking.

### `state_space: list[str]`

Names of states exposed as the agent's observation. Subset of the model's full state.

### `output_space: list[str]`

Names passed into the reward function. Often the same as `state_space`.

### `tracking_states: list[str]`

Names corresponding to `reference_signal`. `tracking_states = ['alpha']` means `reference_signal[0, t]` is the desired α at tick `t`.

### `control_space: list[str]` (nonlinear envs)

Channel names. `['aileron', 'stab', 'rudder']` for 6-DoF; `env.action_space` has `len(control_space)` entries.

### `use_reward: bool = True`

If `False`, `env.step()` returns `reward = 0`. Turn off for PID / MPC.

### `reward_func: Callable | None = None`

Custom `(obs, ref, u, info) -> float`. If `None`, defaults to a quadratic tracking reward.

### `integrator: str = 'rk4'` (nonlinear envs)

`'euler'` fast + fine for `dt ≤ 0.01`; `'rk4'` for aggressive dynamics.

### `control_bias: float | np.ndarray` (nonlinear envs)

Additive bias on each action before the plant. Operate around trim: agent outputs *deviation*, env adds the trim back.

## Step 3 — Tour across envs

| Env id | Model | Suggested `tracking_states` | Notes |
|---|---|---|---|
| `LinearLongitudinalF16-v0` | 4-state linear F-16 | `alpha`, `theta`, `wz` | Good for PID/MPC bootstrap. |
| `NonlinearLongitudinalF16-v0` | NumPy nonlinear F-16 (long) | `alpha`, `wz` | Needs `control_bias` for trim. |
| `NonlinearAngularF16-v0` | 6-DoF nonlinear F-16 | `p`, `q`, `r` | 3 control channels. |
| `LinearB747-v0` | Linear B747 | `theta`, `alpha` | Classical baseline. |
| `NonlinearB747-v0` | Nonlinear B747 | various | MPC-Transformer demo. |
| `UnityEnv-v0` | Unity-rendered | configurable | Requires Unity build. |

Run `gym.make(<id>).unwrapped.__doc__` or browse [Aerospace Models](../model/f16.md) for full per-env docs.

## Step 4 — The `.unwrapped` idiom

```python
env = gym.make('NonlinearLongitudinalF16-v0', ...).unwrapped
```

`.unwrapped` strips Gymnasium wrappers so you can reach env-specific helpers: `env.get_state()`, `env.ref_signal`, the internal integrator. All TensorAeroSpace examples use it.

## Where to go next

- **[Recipe 01 — Hello, TensorAeroSpace](01_hello.md)** — the minimal PID loop.
- **[Recipe 03 — Crafting reference signals](03_reference_signals.md)** — build `reference_signal` in every shape.
- **[Recipe 10 — Adding your own plant model](10_custom_plant.md)** — write your own env.
