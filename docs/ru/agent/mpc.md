# Model Predictive Control (MPC)

MPC использует модель динамики для прогнозирования поведения системы и выбора оптимальной последовательности управлений с учётом ограничений. На каждом шаге решается задача оптимизации, применяется первое управление из оптимальной последовательности, затем окно сдвигается и цикл повторяется.

![MPC схема](../agent/img/mpc/mpc.png){ width=800 }

## Теория (кратко)

- Дискретная динамика: \(x_{k+1} = f(x_k, u_k)\)
- Функция стоимости на горизонте \(N\):

$$
J = \sum_{i=0}^{N-1} (x_{k+i} - x^{\mathrm{ref}}_{k+i})^\top Q (x_{k+i} - x^{\mathrm{ref}}_{k+i})
    + u_{k+i}^\top R\, u_{k+i} + \Delta u_{k+i}^\top S\, \Delta u_{k+i}
    + \text{terminal\_weight} \cdot (x_{k+N}-x^{\mathrm{ref}}_{k+N})^\top Q (x_{k+N}-x^{\mathrm{ref}}_{k+N})
$$

- Приращение управления:

$$
\Delta u_{k+i} = u_{k+i} - u_{k+i-1}
$$

- Ограничения:

$$
\begin{aligned}
u_{\min} \le u_{k+i} \le u_{\max}, &\quad \Delta u_{\min} \le \Delta u_{k+i} \le \Delta u_{\max}, \\
\end{aligned}
$$

- Скользящий горизонт: решаем → применяем \(u_k\) → сдвигаем окно → повторяем
- Устойчивость: терминальный вес, достаточный \(N\), допустимость

## Архитектура

Модуль MPC состоит из:

| Компонент | Класс | Описание |
| --- | --- | --- |
| **Низкоуровневый солвер** | `MPC` | Проекционно-градиентный оптимизатор для дифференцируемой динамики |
| **Высокоуровневый агент** | `MPCAgent` | DSAC-подобная обёртка с обучаемой динамикой, буфером и обучением |
| **Конфигурация весов** | `MPCWeights` | Диагональные веса Q, R, S и терминальный вес |
| **Ограничения** | `MPCConstraints` | Box-ограничения для u и du |
| **Доп. штрафы** | `MPCTrackingExtraCostConfig`, `MPCStepResponseExtraCostConfig` | Штрафы за гладкость, перерегулирование, время установления |
| **Модели динамики** | `OneStepMLP`, `NARXDynamicsModel`, `TransformerDynamicsModel` | Нейросетевые модели для обучения динамики |
| **Нормализатор** | `MPCStandardScaler` | Нормализация признаков (mean/std) |

## Быстрый старт

### Базовый MPC с пользовательской функцией динамики

```python
import numpy as np
import torch
from tensoraerospace.agent.mpc import MPC, MPCWeights, MPCConstraints

state_dim = 4
action_dim = 1

# Определяем динамику: x_{t+1} = f(x_t, u_t)
def dynamics(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    # Простая линейная динамика для примера
    A = torch.eye(state_dim)
    B = torch.zeros(state_dim, action_dim)
    B[-1, 0] = 1.0  # управление влияет на последнее состояние
    return x @ A.T + u @ B.T

# Конфигурация весов
weights = MPCWeights(
    Q_diag=np.array([1.0, 1.0, 1.0, 10.0]),  # веса слежения за состоянием
    R_diag=np.array([0.01]),                   # штраф за управление
    S_diag=np.array([0.1]),                    # штраф за гладкость управления
    terminal_weight=2.0,
)

# Конфигурация ограничений
constraints = MPCConstraints(
    u_min=np.array([-1.0]),
    u_max=np.array([1.0]),
    du_min=np.array([-0.2]),
    du_max=np.array([0.2]),
)

# Создаём MPC солвер
mpc = MPC(
    dynamics=dynamics,
    state_dim=state_dim,
    action_dim=action_dim,
    horizon=20,
    weights=weights,
    constraints=constraints,
    iters=60,
    lr=0.05,
    optimizer="adam",
    warm_start=True,
)

# Решаем
x0 = np.zeros(state_dim)
x_ref = np.zeros((21, state_dim))  # опорная траектория (horizon+1)
x_ref[:, -1] = 0.1  # цель для последней компоненты состояния

result = mpc.solve(x0=x0, x_ref=x_ref, u_prev=None)
print("Первое управление:", result.u0)
print("Форма предсказанной траектории:", result.x_seq.shape)
```

### MPCAgent с обучаемой динамикой (рекомендуется)

`MPCAgent` предоставляет полный рабочий процесс: сбор данных, обучение динамики и MPC-управление.

```python
import gymnasium as gym
import numpy as np
from tensoraerospace.agent.mpc import (
    MPCAgent,
    MPCWeights,
    MPCConstraints,
    MPCStepResponseExtraCostConfig,
)

# Создаём среду
env = gym.make("LinearLongitudinalB747-v0", ...)

# Конфигурация весов
weights = MPCWeights(
    Q_diag=np.array([1.0, 1.0, 10.0, 100.0]),
    R_diag=np.array([0.01]),
    S_diag=np.array([0.5]),
    terminal_weight=1.0,
)

# Конфигурация ограничений
constraints = MPCConstraints(
    u_min=np.array([-0.3]),
    u_max=np.array([0.3]),
    du_min=np.array([-0.05]),
    du_max=np.array([0.05]),
)

# Доп. штрафы для качества переходного процесса
step_cfg = MPCStepResponseExtraCostConfig.from_degrees(
    tracked_idx=-1,        # последнее состояние = theta
    rate_idx=-2,           # предпоследнее = q (угловая скорость тангажа)
    dt=0.01,
    overshoot_limit_deg=0.05,
    settle_band_deg=0.1,
    settle_time_target_s=1.0,
)

# Создаём агента
agent = MPCAgent(
    env,
    horizon=30,
    weights=weights,
    constraints=constraints,
    tracking_type="step_response",
    step_response_config=step_cfg,
    hidden_layers=(256, 256),
    normalize=True,
    device="cuda",  # или "cpu"
)

# Собираем обучающие данные
agent.collect_data(num_episodes=50, exploration="signals")

# Обучаем модель динамики
agent.train_dynamics(epochs=10, batch_size=1024)

# Используем в цикле управления
obs, info = env.reset()
state = ...  # извлекаем внутреннее состояние из среды
x_ref = ...  # опорная траектория (horizon+1, state_dim)

action = agent.select_action(state, x_ref=x_ref)
obs, reward, done, truncated, info = env.step(action)

# Сохранение/загрузка чекпоинтов
path = agent.save("./runs")
agent.load(path)
```

### Использование различных моделей динамики

Вы можете подключить различные архитектуры нейронных сетей:

=== "MLP (по умолчанию)"

    ```python
    from tensoraerospace.agent.mpc import OneStepMLP

    model = OneStepMLP(
        input_dim=state_dim + action_dim,
        output_dim=state_dim,
        hidden_layers=(256, 256, 128),
        activation="relu",  # или "tanh", "gelu"
    )

    agent = MPCAgent(env, model=model, ...)
    ```

=== "NARX"

    ```python
    from tensoraerospace.agent.mpc import NARXDynamicsModel

    model = NARXDynamicsModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=256,
        num_layers=3,
        state_lags=1,
        control_lags=1,
    )

    agent = MPCAgent(env, model=model, ...)
    ```

=== "Transformer"

    ```python
    from tensoraerospace.agent.mpc import TransformerDynamicsModel

    model = TransformerDynamicsModel(
        input_dim=state_dim + action_dim,
        output_dim=state_dim,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    )

    agent = MPCAgent(env, model=model, ...)
    ```

## Дополнительные функции стоимости

### Режим слежения (`tracking`)

Добавляет штрафы за гладкость управления:

- `w_du`: вес для \(\sum (\Delta u)^2\)
- `w_jerk`: вес для \(\sum (\Delta^2 u)^2\)

```python
from tensoraerospace.agent.mpc import MPCTrackingExtraCostConfig

cfg = MPCTrackingExtraCostConfig(w_du=50.0, w_jerk=10.0)
agent = MPCAgent(env, tracking_type="tracking", tracking_config=cfg, ...)
```

### Режим переходного процесса (`step_response`)

Добавляет штрафы за перерегулирование, время установления, колебания:

```python
from tensoraerospace.agent.mpc import MPCStepResponseExtraCostConfig

cfg = MPCStepResponseExtraCostConfig.from_degrees(
    tracked_idx=-1,               # индекс отслеживаемого состояния (напр., theta)
    rate_idx=-2,                  # индекс скорости (напр., q)
    dt=0.01,                      # шаг по времени
    overshoot_limit_deg=0.05,     # макс. перерегулирование в градусах
    settle_band_deg=0.10,         # ширина полосы установления
    settle_time_target_s=1.0,     # целевое время установления
    w_overshoot=8000.0,           # вес штрафа за перерегулирование
    w_settle=8000.0,              # вес штрафа за установление
    w_osc=500.0,                  # вес штрафа за колебания
    w_jerk=50.0,                  # вес штрафа за рывки
)

agent = MPCAgent(env, tracking_type="step_response", step_response_config=cfg, ...)
```

Можно переключать режимы во время работы:

```python
agent.set_tracking_type("tracking", tracking_config=tracking_cfg)
# или
agent.set_tracking_type("step_response", step_response_config=step_cfg)
```

## Сбор данных

`MPCAgent.collect_data()` поддерживает две стратегии исследования:

| Стратегия | Описание |
| --- | --- |
| `"random"` | Случайные действия через `env.action_space.sample()` |
| `"signals"` | Богатая библиотека сигналов: ступеньки, рампы, синусоиды, chirp, дублеты и др. |

```python
agent.collect_data(
    num_episodes=50,
    max_steps=1000,
    exploration="signals",
    signal_kinds=["random_steps", "sinusoid", "chirp", "doublet"],
    action_amplitude_frac=0.8,
)
```

Доступные типы сигналов: `random_steps`, `unit_step`, `multi_step`, `ramp`, `sinusoid`, `multisine`, `chirp`, `square_wave`, `triangular_wave`, `sawtooth`, `doublet`, `pulse`, `gaussian_pulse`, `exponential`, `damped_sinusoid`.

## Гиперпараметры

### MPC Solver (`MPC`)

| Параметр | Описание | По умолчанию |
| --- | --- | --- |
| `horizon` | Горизонт предсказания | 20 |
| `iters` | Итерации оптимизации за один вызов solve | 60 |
| `lr` | Скорость обучения | 0.05 |
| `optimizer` | `"adam"` или `"sgd"` | `"adam"` |
| `warm_start` | Повторное использование предыдущего решения | `True` |
| `track_best` | Отслеживание лучшего решения при оптимизации | `True` |
| `compile_dynamics` | Использовать `torch.compile` (PyTorch 2.x) | `False` |

### MPCAgent

| Параметр | Описание | По умолчанию |
| --- | --- | --- |
| `hidden_layers` | Размеры скрытых слоёв MLP | `(256, 256)` |
| `normalize` | Нормализация входов/выходов | `True` |
| `dynamics_lr` | Скорость обучения модели динамики | `1e-3` |
| `grad_clip_norm` | Обрезка градиентов | `1.0` |
| `memory_capacity` | Размер replay buffer | `200_000` |
| `model_predict_delta` | Предсказывать \(\Delta x\) вместо \(x'\) | `True` |

!!! tip "Лучшие практики"
    - Используйте `exploration="signals"` для лучшего покрытия пространства состояний-действий
    - Начните с `horizon=20-30` и увеличивайте при необходимости
    - Включите `normalize=True` для нейросетевой динамики
    - Используйте `tracking_type="step_response"` для аэрокосмических задач управления
    - Для управления в реальном времени рассмотрите `compile_dynamics=True` на GPU

## Примеры

Полные end-to-end примеры, демонстрирующие MPC с различными моделями динамики на задаче продольного управления B747:

| Пример | Модель динамики | Описание |
| --- | --- | --- |
| [MPC + MLP](../example/agent/mpc/example-mpc-b747-torch-mpc-mlp.md) | `OneStepMLP` | Стандартное обучение динамики на MLP с отслеживанием переходного процесса |
| [MPC + NARX](../example/agent/mpc/example-mpc-b747-torch-mpc-narx.md) | `NARXDynamicsModel` | Нелинейная авторегрессия с экзогенными входами |
| [MPC + Transformer](../example/agent/mpc/example-mpc-b747-torch-mpc-transformer.md) | `TransformerDynamicsModel` | Transformer-энкодер для предсказания динамики |

Каждый пример демонстрирует полный пайплайн:

1. **Настройка среды** — Создание среды B747 со ступенчатым референсом по тангажу (θ)
2. **Сбор данных** — Сбор переходов с использованием богатых исследовательских сигналов
3. **Обучение динамики** — Обучение нейросети предсказывать переходы состояний
4. **MPC rollout** — Запуск управления с замкнутым контуром на обученной динамике
5. **Оценка** — Анализ качества переходного процесса (перерегулирование, время установления и т.д.)

### Ключевые результаты из примеров

| Модель | Перерегулирование | Время установления | Время нарастания | Статическая ошибка |
| --- | --- | --- | --- | --- |
| MLP | ~0.30% | ~1.7с | ~1.1с | ~0.001 |
| NARX | ~-1.9% | ~3.0с | ~1.0с | ~0.026 |
| Transformer | ~-0.10% | ~1.5с | ~0.8с | ~0.009 |

!!! note "Запуск примеров"
    Примеры — это Jupyter notebooks, расположенные в `example/mpc_controllers/`. Запустите их, чтобы увидеть полные логи обучения, графики и отчёты бенчмарков.

## Документация API

::: tensoraerospace.agent.mpc.MPC

::: tensoraerospace.agent.mpc.MPCAgent

::: tensoraerospace.agent.mpc.MPCWeights

::: tensoraerospace.agent.mpc.MPCConstraints

::: tensoraerospace.agent.mpc.MPCSolveResult

::: tensoraerospace.agent.mpc.MPCTrackingExtraCostConfig

::: tensoraerospace.agent.mpc.MPCStepResponseExtraCostConfig

::: tensoraerospace.agent.mpc.MPCStandardScaler

::: tensoraerospace.agent.mpc.OneStepMLP

::: tensoraerospace.agent.mpc.NARXDynamicsModel

::: tensoraerospace.agent.mpc.NARX

::: tensoraerospace.agent.mpc.TransformerDynamicsModel
