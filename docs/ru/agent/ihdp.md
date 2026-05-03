# Incremental Heuristic Dynamic Programming (IHDP)

IHDP — инкрементальный вариант Heuristic Dynamic Programming из семейства Adaptive Critic Designs (ACD) для управления нелинейными объектами при неполном знании модели. В авиационных задачах используется для синтеза продольного управления. См. также модель F‑16: [LinearLongitudinalF16](../model/f16.md).

## Ключевые идеи

- Инкрементальная модель линеаризует динамику локально по данным онлайн
- Actor формирует управляющее воздействие по ошибке слежения
- Critic оценивает стоимостную функцию и даёт градиенты Actor

![Схема IHDP](../agent/img/ihdp/ihdp.png){ width=800 }

## Состав IHDP

| Компонент | Роль | Реализация |
| --- | --- | --- |
| Incremental model | Онлайн-идентификация и линеаризация динамики | `tensoraerospace.agent.ihdp.Incremental_model.IncrementalModel` |
| Actor | Генерация управляющего сигнала (NN) | `tensoraerospace.agent.ihdp.Actor` |
| Critic | Оценка J(x) и градиента dJ/dx (NN) | `tensoraerospace.agent.ihdp.Critic` |
| IHDPAgent | Оркестрация модулей, шаг predict и обучение | `tensoraerospace.agent.ihdp.model.IHDPAgent` |

## Быстрый старт

Пример инициализации агента и одного шага предсказания:

<!-- markdownlint-disable MD046 -->
```python
import numpy as np
from tensoraerospace.agent.ihdp.model import IHDPAgent

actor_settings = {
    "start_training": 100,
    "layers": (64, 32, 1),
    "activations": ("tanh", "tanh", "tanh"),
    "learning_rate": 0.01,
    "learning_rate_exponent_limit": 8,
    "type_PE": "3211",
    "amplitude_3211": 1,
    "pulse_length_3211": 15,
    "maximum_input": 25,
    "maximum_q_rate": 20,
    "WB_limits": 30,
    "NN_initial": None,
    "cascade_actor": False,
    "learning_rate_cascaded": 0.01,
}

critic_settings = {
    "Q_weights": np.eye(2),
    "start_training": 100,
    "gamma": 0.99,
    "learning_rate": 0.01,
    "learning_rate_exponent_limit": 8,
    "layers": (64, 32, 1),
    "activations": ("tanh", "tanh", "tanh"),
    "indices_tracking_states": [0, 1],
    "WB_limits": 30,
    "NN_initial": None,
}

incremental_settings = {
    "number_time_steps": 1000,
    "dt": 0.02,
    "input_magnitude_limits": 25,
    "input_rate_limits": 20,
}

tracking_states = ["alpha", "wz"]
selected_states = ["alpha", "wz"]
selected_input = ["elevator"]
number_time_steps = 1000
indices_tracking_states = [0, 1]

agent = IHDPAgent(
    actor_settings,
    critic_settings,
    incremental_settings,
    tracking_states,
    selected_states,
    selected_input,
    number_time_steps,
    indices_tracking_states,
)

# Один шаг предсказания
xt = np.zeros((len(selected_states), 1))
reference = np.zeros((len(selected_states), number_time_steps))
ut = agent.predict(xt, reference, time_step=0)
```
<!-- markdownlint-enable MD046 -->

!!! tip
    Убедитесь, что `indices_tracking_states` согласованы с порядком вектора состояний среды.

## Гиперпараметры

### Actor

| Параметр | Описание |
| --- | --- |
| layers, activations | Архитектура NN и активации |
| learning_rate, learning_rate_exponent_limit | Скорость обучения и предел масштабирования |
| type_PE, amplitude_3211, pulse_length_3211 | Персистентное возбуждение |
| maximum_input, maximum_q_rate, WB_limits | Ограничения и насыщения |
| cascade_actor, learning_rate_cascaded | Каскадный режим |

### Critic

| Параметр | Описание |
| --- | --- |
| Q_weights | Матрица весов функции стоимости |
| gamma | Дисконт |
| learning_rate, learning_rate_exponent_limit | Обучение |
| layers, activations | Архитектура |
| indices_tracking_states | Индексы отслеживаемых состояний |
| WB_limits, NN_initial | Ограничения/инициализация |

### Incremental model

| Параметр | Описание |
| --- | --- |
| number_time_steps, dt | Горизонт и шаг интегрирования |
| input_magnitude_limits | Ограничение по величине управления |
| input_rate_limits | Ограничение по скорости изменения |

## Поддерживаемые окружения

- `LinearLongitudinalF16-v0`

## Примеры

- Подробный пример для F‑16: [IHDP ↔ LinearLongitudinalF16](../example/agent/ihdp/example_ihdp.md)

## Документация API

::: tensoraerospace.agent.ihdp.model.IHDPAgent

::: tensoraerospace.agent.ihdp.Actor

::: tensoraerospace.agent.ihdp.Critic

::: tensoraerospace.agent.ihdp.IncrementalModel

## Источники

- [Incremental Model Based Heuristic Dynamic Programming for Nonlinear Adaptive Flight Control](https://www.researchgate.net/publication/313696777_Incremental_Model_Based_Heuristic_Dynamic_Programming_for_Nonlinear_Adaptive_Flight_Control)
- [IHDP (reference implementation)](https://github.com/joigalcar3/IHDP)
