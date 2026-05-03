# Incremental Model-based Global Dual Heuristic Programming (IMGDHP)

IMGDHP — инкрементальный модельно-ориентированный вариант Global Dual Heuristic Programming из семейства Adaptive Critic Designs (ACD). Предназначен для онлайн-адаптивного управления нелинейными объектами в условиях частичной наблюдаемости. Агент объединяет рекурсивный метод наименьших квадратов (RLS) для идентификации объекта с двуглавым критиком, оценивающим как функцию стоимости \(J\), так и вектор сопряжённых переменных \(\lambda = \partial J / \partial y\), что обеспечивает более информативный градиент для актора. См. также нелинейную модель F-16: [NonlinearLongitudinalF16](../model/f16_nonlinear_longitudinal.md).

## Ключевые идеи

- **Инкрементальная модель**: онлайн-идентификация локальной линеаризации \(\Delta y_{t+1} = A \Delta y_t + B \Delta u_t\) методом RLS — легковесно, интерпретируемо, не требует нейросети для идентификации
- **Двойной критик GDHP**: критик выдаёт как \(J(o)\) (скалярная стоимость), так и \(\lambda(o)\) (вектор сопряжённых переменных), обеспечивая более богатый градиентный сигнал для актора по сравнению со стандартным HDP/DHP
- **Модельно-предиктивное обновление актора**: градиент актора проходит через идентифицированные матрицы \(A\), \(B\), позволяя оптимизировать на один шаг вперёд
- **Частичная наблюдаемость**: расширенное наблюдение \(o = [y; r; e]\) позволяет агенту работать, когда наблюдение среды не совпадает с полным состоянием

## Отличия от близких методов

| Аспект | HDP | IHDP | **IMGDHP** |
| --- | --- | --- | --- |
| Идентификация | Известная модель | Онлайн NN | Онлайн RLS (инкрементальная линейная) |
| Выход критика | \(J(o)\) | \(J(o)\) | \(J(o)\) + \(\lambda(o)\) (двойной) |
| Обновление актора | Прямой градиент | По модели | Модельно-предиктивное через \(A\), \(B\) |
| Частичная наблюдаемость | Нет | Ограниченно | Основа архитектуры |
| Фреймворк | NumPy | NumPy | PyTorch |

## Состав IMGDHP

| Компонент | Роль | Реализация |
| --- | --- | --- |
| IncrementalModelRLS | Онлайн-идентификация матриц \(A\), \(B\) методом RLS | `tensoraerospace.agent.im_gdhp.IncrementalModelRLS` |
| GDHPActor | Детерминированная политика \(u = u_{\max} \tanh(\pi_\theta(o))\) | `tensoraerospace.agent.im_gdhp.GDHPActor` |
| GDHPCritic | Двуглавый критик: общий backbone + голова \(J\) + голова \(\lambda\) | `tensoraerospace.agent.im_gdhp.GDHPCritic` |
| IMGDHPAgent | Оркестрация всех компонент, цикл обучения, интерфейс predict/learn | `tensoraerospace.agent.im_gdhp.IMGDHPAgent` |

## Алгоритм

На каждом шаге \(t\), при наблюдении \(y_t\) и задании \(r_t\):

1. **Расширение наблюдения**: \(o_t = [y_t;\; r_t;\; e_t]\), где \(e_t = y_t[\text{tracking}] - r_t\)
2. **Актор формирует действие**: \(u_t = \pi_\theta(o_t)\)
3. **Исполнение** \(u_t\) в среде, получение \(y_{t+1}\)
4. **Одношаговая стоимость**: \(c_t = e_t^\top Q e_t + \rho \| u_t - u_{t-1} \|^2\)
5. **Обновление RLS** (при \(t \geq 2\)): по данным \((y_{t-2}, y_{t-1}, y_t, u_{t-2}, u_{t-1})\) получаем \(A_t\), \(B_t\)
6. **Обновление критика** (двойная функция потерь GDHP):

\[
L = \underbrace{\left( J(o_t) - (c_t + \gamma J(o_{t+1})) \right)^2}_{L_J} + \beta \underbrace{\left\| \lambda(o_t) - \left( \frac{\partial c_t}{\partial y} + \gamma A_t^\top \lambda(o_{t+1}) \right) \right\|^2}_{L_\lambda}
\]

7. **Обновление актора** (модельно-предиктивное):

\[
\min_\theta \; c_t + \gamma \, J\!\left(\hat{o}_{t+1}\right), \quad \text{градиент проходит через } B_t
\]

## Быстрый старт

```python
import numpy as np
import gymnasium as gym
from tensoraerospace.agent.im_gdhp import IMGDHPAgent, IMGDHPConfig
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standard import sinusoid

dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)
reference_signal = sinusoid(
    degree=3, tp=tp, frequency=0.1, output_rad=True
).reshape(1, -1)

env = gym.make(
    "NonlinearLongitudinalF16-v0",
    number_time_steps=number_time_steps,
    initial_state=np.array([0.0, 0.0]),
    reference_signal=reference_signal,
    dt=dt,
)

config = IMGDHPConfig(
    gamma=0.95,
    actor_hidden=(32, 32),
    critic_hidden=(64, 64),
    actor_lr=1e-3,
    critic_lr=5e-3,
    track_Q=(1.0,),
    warmup_steps=5,
    forgetting=0.9995,
    u_max=25.0,
)

agent = IMGDHPAgent(
    n_obs=2,
    n_action=1,
    reference_size=1,
    tracking_indices=[0],
    config=config,
)

obs, info = env.reset()
for t in range(number_time_steps - 1):
    action = agent.predict(obs, reference_signal, t)
    obs_next, reward, terminated, truncated, info = env.step(action)
    metrics = agent.learn(obs_next, reference_signal, t)
    obs = obs_next
    if terminated or truncated:
        break
```

!!! tip
    `tracking_indices` должны соответствовать индексам наблюдения, отслеживающим задающий сигнал. Например, если наблюдение — `[alpha, wz]` и вы отслеживаете `alpha`, используйте `tracking_indices=[0]`.

## Гиперпараметры

### Общие

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `gamma` | 0.95 | Коэффициент дисконтирования |
| `warmup_steps` | 5 | Шаги с замороженным актором/критиком (только исследование) |
| `critic_only_steps` | 0 | Дополнительные шаги с замороженным актором после прогрева |
| `seed` | None | Зерно ГСЧ для воспроизводимости |
| `device` | `"cpu"` | Устройство PyTorch |

### Актор

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `actor_hidden` | (32, 32) | Размеры скрытых слоёв |
| `actor_lr` | 1e-3 | Скорость обучения |
| `u_max` | 25.0 | Ограничение управляющего сигнала по каналу |
| `exploration_noise_std` | 0.0 | Гауссовский шум исследования при обучении |

### Критик

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `critic_hidden` | (64, 64) | Размеры скрытых слоёв backbone |
| `critic_lr` | 5e-3 | Скорость обучения |
| `beta_lambda` | 1.0 | Вес \(\lambda\)-потерь в двойной функции GDHP |
| `critic_updates_per_step` | 1 | Градиентных шагов на один переход в среде |
| `target_update_tau` | 0.0 | Коэффициент Поляка для целевого критика (0 = без целевой сети) |
| `critic_weight_decay` | 0.0 | L2-регуляризация |
| `max_grad_norm` | 5.0 | Порог отсечения градиента |

### Функция стоимости

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `track_Q` | (1.0,) | Диагональные веса стоимости слежения \(e^\top Q e\) |
| `action_rate_penalty` | 1e-3 | Коэффициент \(\rho\) штрафа \(\| \Delta u \|^2\) |

### Инкрементальная модель (RLS)

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `forgetting` | 0.9995 | Фактор забывания RLS \(\in (0, 1]\) |
| `cov_init` | 1e2 | Начальный масштаб ковариационной матрицы |

### Наблюдение

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `obs_scale` | None | Покомпонентное масштабирование наблюдений |

## Поддерживаемые окружения

- `NonlinearLongitudinalF16-v0`
- `LinearLongitudinalF16-v0`

## Документация API

::: tensoraerospace.agent.im_gdhp.model.IMGDHPAgent

::: tensoraerospace.agent.im_gdhp.model.IMGDHPConfig

::: tensoraerospace.agent.im_gdhp.incremental_model.IncrementalModelRLS

::: tensoraerospace.agent.im_gdhp.networks.GDHPActor

::: tensoraerospace.agent.im_gdhp.networks.GDHPCritic

## Источники

- Sun, Z. & van Kampen, E.-J. (2021). *Intelligent adaptive optimal control using incremental model-based global dual heuristic programming subject to partial observability*. Applied Soft Computing, 103, 107153.
- Zhou, Y., van Kampen, E.-J., & Chu, Q. P. (2020). *Incremental model based online dual heuristic programming for nonlinear adaptive control*. Control Engineering Practice, 95, 104242.
