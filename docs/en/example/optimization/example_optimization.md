# Hyperparameter search

Online hyperparameter search for `IHDPAgent` via the `HyperParamOptimizationOptuna`
wrapper on the `LinearLongitudinalF16-v0` environment. The idea: run a series of
trials, each of which lets Optuna sample actor/critic parameters, executes one
control episode, and returns the resulting reward; final values are extracted via
`get_best_param()`.

See also [Cookbook 07 — Hyperparameter search with Optuna](../../cookbook/07_optuna_search.md)
for a more general discussion of the approach.

## Imports and setup

```python
import numpy as np
import gymnasium as gym

from tensoraerospace.optimization import HyperParamOptimizationOptuna
from tensoraerospace.agent.ihdp.model import IHDPAgent
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standard import unit_step
```

!!! note
    `HyperParamOptimizationOptuna` is a thin wrapper around
    [Optuna](https://optuna.org/) that standardises logging and result
    serialisation. If the wrapper does not fit your use case, drop down to
    Optuna directly — `opt.study` is exposed on the wrapper.

```python
opt = HyperParamOptimizationOptuna(direction="minimize")
```

## Tracking signal and reference profile

```python
dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)

reference_signals = np.reshape(
    unit_step(degree=5, tp=tp, time_step=10.0, output_rad=True),
    [1, -1],
)
```

`time_step=10.0` is the moment the step fires, in **seconds** (the middle of a
20-second simulation).

## Optuna objective

```python
def objective(trial):
    env = gym.make(
        "LinearLongitudinalF16-v0",
        number_time_steps=number_time_steps,
        initial_state=[[0], [0]],
        reference_signal=reference_signals,
        tracking_states=["alpha"],
    )
    env.reset()

    actor_settings = {
        "start_training": trial.suggest_int("start_training", 5, 7, log=True),
        "layers": (trial.suggest_int("actor_layers", 20, 25, log=True), 1),
        "activations": ("tanh", "tanh"),
        "learning_rate": trial.suggest_int("learning_rate", 2, 5, log=True),
        "learning_rate_exponent_limit": 10,
        "type_PE": "combined",
        "amplitude_3211": 15,
        "pulse_length_3211": 5 / dt,
        "maximum_input": 25,
        "maximum_q_rate": 20,
        "WB_limits": 30,
        "NN_initial": 120,
        "cascade_actor": False,
        "learning_rate_cascaded": 1.2,
    }
    incremental_settings = {
        "number_time_steps": number_time_steps,
        "dt": dt,
        "input_magnitude_limits": 25,
        "input_rate_limits": 60,
    }
    critic_settings = {
        "Q_weights": [trial.suggest_int("Q_weights", 7, 9)],
        "start_training": -1,
        "gamma": 0.99,
        "learning_rate": 15,
        "learning_rate_exponent_limit": 10,
        "layers": (trial.suggest_int("critic_layers", 20, 25, log=True), 1),
        "activations": ("tanh", "linear"),
        "WB_limits": 30,
        "NN_initial": 120,
        "indices_tracking_states": env.unwrapped.indices_tracking_states,
    }

    model = IHDPAgent(
        actor_settings,
        critic_settings,
        incremental_settings,
        env.unwrapped.tracking_states,
        env.unwrapped.state_space,
        env.unwrapped.control_space,
        number_time_steps,
        env.unwrapped.indices_tracking_states,
    )

    xt = np.array([[np.deg2rad(3)], [0]])
    reward = 0.0
    for step in range(number_time_steps - 1):
        ut = model.predict(xt, reference_signals, step)
        xt, reward, terminated, truncated, info = env.step(np.array(ut))
        if terminated or truncated:
            break
    return reward
```

!!! note
    The objective minimises the absolute deviation of the current state from
    the reference: `reward = abs(state[0] - ref_signal[:, ts])`. The value is
    computed inside `tensoraerospace.envs.LinearLongitudinalF16.reward`.
    Alternative criteria (IAE/ISE/ITAE) are planned in future releases.

## Running the search

```python
opt.run_optimization(objective, n_trials=10)
```

Optuna will run 10 trials with different hyperparameter combinations and keep
the best result.

## Extracting the best parameters

```python
opt.get_best_param()
```

Example output:

```python
{
    "start_training": 5,
    "actor_layers": 25,
    "learning_rate": 5,
    "Q_weights": 8,
    "critic_layers": 25,
}
```

## Visualising the search history

```python
opt.plot_parms()
```

Returns a plot of the objective value vs. trial number plus parallel
coordinates — see details in
[Cookbook 07](../../cookbook/07_optuna_search.md).
