# Model Predictive Control (MPC)

MPC использует модель динамики, чтобы прогнозировать поведение системы и подбирать оптимальную последовательность управлений при учёте ограничений. На каждом шаге решается задача оптимизации, применяется первое управление из оптимальной последовательности, затем цикл повторяется.

![MPC схема](../agent/img/mpc/mpc.png){ width=800 }

## Теория (кратко)

- Динамика (дискретная): \(x_{k+1} = f(x_k, u_k)\)
- Стоимостная функция на горизонте \(N\):

$$
J = \sum_{i=0}^{N-1} (x_{k+i} - x^{\mathrm{ref}}_{k+i})^\top Q (x_{k+i} - x^{\mathrm{ref}}_{k+i})
    + u_{k+i}^\top R\, u_{k+i} + \Delta u_{k+i}^\top S\, \Delta u_{k+i}
    + (x_{k+N}-x^{\mathrm{ref}}_{k+N})^\top P (x_{k+N}-x^{\mathrm{ref}}_{k+N})
$$

- Приращение управления:

$$
\Delta u_{k+i} = u_{k+i} - u_{k+i-1}
$$

- Ограничения:

$$
\begin{aligned}
u_{\min} \le u_{k+i} \le u_{\max}, &\quad \|\Delta u_{k+i}\|_\infty \le \Delta u_{\max}, \\
x_{k+i} \in \mathcal{X}, &\quad i = 0,\dots,N-1
\end{aligned}
$$

- Принцип «скользящего горизонта»: решаем задачу → применяем \(u_k\) → сдвигаем окно → повторяем
- Стабильность: терминальный вес/набор \(P,\ \mathcal{X}_f\), достаточный \(N\), допустимость
- Для линейной динамики \(x_{k+1} = A x_k + B u_k\) и квадратичного \(J\) получаем выпуклую QP‑задачу

На практике в нашем `AircraftMPC` веса задаются полями `weights`, ограничения — `u_max`, `delta_u_max`, штраф за нарушения — `penalty_weight`.

### Как это соответствует реализации `AircraftMPC`

- Динамика: вызывается пользовательская функция `dynamics_model(xu)` итерированием в `predict_trajectory`:
  - вход: конкатенация состояния и управления \([x_t, u_t]\)
  - выход: предсказанное \(x_{t+1}\)
- Стоимость: в `cost_function` учитываются три слагаемых (см. `weights`):
  - `theta_tracking` — слежение за компонентой состояния (пример в коде использует индекс 3)
  - `control_effort` — энергия управления \(\sum u_t^2\)
  - `delta_control` — гладкость \(\sum (u_t - u_{t-1})^2\)
- Ограничения: применяются мягко через `penalty_function` и множитель `penalty_weight`:
  - насыщение \(|u_t| \le u_{\max}\)
  - ограничение приращений \(|u_t - u_{t-1}| \le \Delta u_{\max}\)
- Оптимизация: `optimize_control` использует численный градиент по вектору \(U\) и шаг `learning_rate` с проекцией на допустимый диапазон.

Таким образом, практическая постановка эквивалентна классической MPC с квадратичным функционалом и жесткими ограничениями, но реализована через штрафы + проекции для упрощения численной реализации.

## Компоненты

| Компонент | Роль | Реализация |
| --- | --- | --- |
| Динамика | Прогноз следующего состояния по текущему состоянию и управлению | Пользовательская модель или `DynamicsNN` |
| Стоимостная функция | Баланс слежения/энергии/гладкости | `AircraftMPC.cost_function` |
| Ограничения | Лимиты по управлению и его приращению | `u_max`, `delta_u_max` + штраф `penalty_weight` |
| Оптимизатор | Поиск последовательности управлений U | Численный градиент + `learning_rate`, `iterations` |

## Быстрый старт

=== "Линейная модель (A, B)"

    ```python
    import numpy as np
    import torch
    from tensoraerospace.agent.mpc.base import AircraftMPC

    # Простейшая линейная динамика x_{t+1} = A x_t + B u_t
    A = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=np.float32)
    B = np.array([[0], [0], [0], [1]], dtype=np.float32)

    def dyn_model(xu: torch.Tensor) -> torch.Tensor:
        x = xu[..., :4].numpy()
        u = xu[..., 4:].numpy()
        x_next = A @ x.T + B @ u.T
        return torch.tensor(x_next.T, dtype=torch.float32)

    mpc = AircraftMPC(dynamics_model=dyn_model, horizon=10, dt=0.05)

    x0 = np.zeros(4, dtype=np.float32)
    # Опорная траектория по theta (последний компонент в примере cost_function)
    theta_ref = np.zeros(mpc.horizon + 1, dtype=np.float32)

    u0, X_pred = mpc.optimize_control(x0, theta_ref)
    ```

=== "NN‑модель динамики (из примера)"

    ```python
    import numpy as np
    import torch
    import torch.nn as nn
    import gymnasium as gym
    from tensoraerospace.agent.mpc.base import AircraftMPC
    from tensoraerospace.agent.mpc.dynamics import DynamicsNN
    from tensoraerospace.signals.standart import unit_step
    from tensoraerospace.utils import generate_time_period

    # 1) Среда и матрицы A, B (пример для F16)
    dt = 0.1
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)

    env = gym.make(
        'LinearLongitudinalF16-v0',
        number_time_steps=number_time_steps,
        initial_state=[[0],[0]],
        reference_signal=unit_step(degree=2, tp=tp, time_step=int(5/dt), output_rad=True).reshape(1, -1),
    )
    state, info = env.reset()
    A = np.array(env.unwrapped.model.A, dtype=np.float32)
    B = np.array(env.unwrapped.model.B, dtype=np.float32)

    # 2) NN модель динамики f([x,u]) -> x_{t+1}
    model = nn.Sequential(
        nn.Linear(4 + 1, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 4)
    )
    dyn = DynamicsNN(model)

    # 3) Генерация обучающих данных (как в примерах)
    state_ranges = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    states, controls, next_states = dyn.generate_training_data(
        num_samples=30000,
        state_dim=4,
        control_dim=1,
        state_ranges=state_ranges,
        control_signals=["sine", "step", "sine_09", "sine_07", "gaussian_noise", "linear_up", "linear_down"],
        A=A, B=B,
    )

    dyn.train_and_validate(states, controls, next_states, epochs=20, batch_size=512, verbose_epoch=5)

    # 4) MPC поверх NN динамики
    def dyn_model(xu: torch.Tensor) -> torch.Tensor:
        return model(xu)

    mpc = AircraftMPC(dynamics_model=dyn_model, horizon=2, dt=dt)

    x0 = np.zeros(4, dtype=np.float32)
    # Опорная траектория по theta
    theta_ref = unit_step(degree=2, tp=np.arange(mpc.horizon+1)*dt, time_step=0, output_rad=True).astype(np.float32)

    u0, X_pred = mpc.optimize_control(x0, theta_ref)
    ```

Полный разбор с визуализацией: [пример MPC](../example/agent/mpc/example_mpc.md)

## Вариации MPC

- **Градиентный MPC агент** (`MPCOptimizationAgent`): оптимизирует последовательности действий градиентными методами поверх обучаемой динамики. Подходит для детерминированных систем и задач слежения.
- **Стохастический MPC агент** (`MPCAgent`): учитывает распределения действий/неопределённость, использует стохастические выборки и регуляризацию; полезен при шуме и ограничениях.
- **Модели динамики**:
  - `DynamicsNN`: базовая NN‑аппроксимация f([x,u]) → x′
  - `NARX`: нелинейная авторегрессия с экзогенными входами
  - `TransformerDynamicsModel`: трансформер‑архитектура для последовательностей состояний/действий

!!! note
    Выбор вариации зависит от свойств вашей системы: детерминированность, шумы, размерность, доступность данных.

## Гиперпараметры и советы

- `horizon`: больше — лучше качество (дороже вычислительно)
- `weights.theta_tracking`: приоритет слежения по `theta`
- `weights.control_effort`, `weights.delta_control`: ограничивают энергию и «дёрганье» управления
- `u_max`, `delta_u_max`: физические лимиты; увеличивайте `penalty_weight` при частых нарушениях
- `learning_rate`, `iterations`: скорость и точность оптимизации

!!! tip
    Для реальных систем используйте нормализацию признаков и регуляризацию NN‑модели динамики.

## Документация API

::: tensoraerospace.agent.mpc.base.AircraftMPC

::: tensoraerospace.agent.mpc.dynamics.DynamicsNN

::: tensoraerospace.agent.mpc.gradient.MPCOptimizationAgent

::: tensoraerospace.agent.mpc.stochastic.MPCAgent

::: tensoraerospace.agent.mpc.narx.NARX

::: tensoraerospace.agent.mpc.transformers.TransformerDynamicsModel
