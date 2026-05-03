# Поиск гиперпараметров

Пример онлайн-поиска гиперпараметров `IHDPAgent` через обёртку
`HyperParamOptimizationOptuna` на среде `LinearLongitudinalF16-v0`. Идея:
запустить серию trial'ов, в каждом из которых Optuna сэмплирует параметры
актёра/критика, прогоняет один эпизод управления и возвращает значение
награды; финальные значения извлекаются через `get_best_param()`.

См. также [Cookbook 07 — Гиперпоиск через Optuna](../../cookbook/07_optuna_search.md)
с более общим обсуждением подхода.

## Импорты и настройки

```python
import numpy as np
import gymnasium as gym

from tensoraerospace.optimization import HyperParamOptimizationOptuna
from tensoraerospace.agent.ihdp.model import IHDPAgent
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standard import unit_step
```

!!! note
    `HyperParamOptimizationOptuna` — тонкая обёртка вокруг библиотеки
    [Optuna](https://optuna.org/) для удобства логирования и сериализации
    результатов. Если обёртка вам не подходит, можно работать с Optuna
    напрямую.

```python
opt = HyperParamOptimizationOptuna(direction="minimize")
```

## Сигнал слежения и опорный профиль

```python
dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)

reference_signals = np.reshape(
    unit_step(degree=5, tp=tp, time_step=10.0, output_rad=True),
    [1, -1],
)
```

`time_step=10.0` — момент срабатывания ступеньки в **секундах** (середина
20-секундной симуляции).

## Целевая функция Optuna

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
    Минимизируется модуль разности текущего состояния с заданным:
    `reward = abs(state[0] - ref_signal[:, ts])`. Расчёт происходит в
    `tensoraerospace.envs.LinearLongitudinalF16.reward`. В дальнейшем
    планируется добавить альтернативные критерии (например, IAE/ISE/ITAE).

## Запуск оптимизации

```python
opt.run_optimization(objective, n_trials=10)
```

Optuna прогонит 10 trial'ов с разными комбинациями гиперпараметров и
запомнит лучший результат.

## Извлечение лучших параметров

```python
opt.get_best_param()
```

Пример вывода:

```python
{
    "start_training": 5,
    "actor_layers": 25,
    "learning_rate": 5,
    "Q_weights": 8,
    "critic_layers": 25,
}
```

## Визуализация истории поиска

```python
opt.plot_parms()
```

Вернёт график зависимости значения целевой функции от номера trial и
параллельных координат — см. подробности в
[Cookbook 07](../../cookbook/07_optuna_search.md).
