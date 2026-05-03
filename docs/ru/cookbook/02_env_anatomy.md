# Рецепт 02 — Устройство среды TensorAeroSpace

Пройти по каждому рычагу `gym.make('<aircraft>-v0', ...)` с копируемым сниппетом на каждом шаге. После этого рецепта вы знаете, что делает каждый аргумент в любой среде (F-16, B747, ракета, БПЛА, …).

**Связано.** [Рецепт 01](01_hello.md) — первый прогон; [Рецепт 03](03_reference_signals.md) — `reference_signal`; [Рецепт 10](10_custom_plant.md) — своя среда.

## Ментальная модель

Каждая среда поддерживает три параллельных взгляда на состояние:

- **Внутреннее состояние** — то, что интегрирует модель (например `[theta, alpha, q, stab]`).
- **Наблюдение** — срез, видимый агенту, задаётся `state_space`.
- **Выход** — срез для функции награды, задаётся `output_space`.

Разделение наблюдения и выхода позволяет тренировать на `(alpha, q)`, а награду формировать по `theta`.

## Шаг 1 — Построить минимальную среду и посмотреть пространства

```python
import gymnasium as gym
import numpy as np
import tensoraerospace  # noqa: F401

from tensoraerospace.signals.standard import unit_step
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

**Ожидаемый вывод:**

```
observation_space: Box(-inf, inf, (3,), float64)
action_space:      Box(-0.436, 0.436, (1,), float64)
ref shape:         (1, 500)
```

Трёхэлементное наблюдение соответствует `state_space=['theta', 'alpha', 'q']`. Однокомпонентное действие — дефолтному `stab`. Форма `ref` — `(n_tracking, T)`.

## Шаг 2 — Каждый аргумент в деталях

### `number_time_steps: int`

Горизонт симуляции в **тиках**, не в секундах. Используйте `generate_time_period(tn=T_секунды, dt=dt)` и затем `len(tp)`.

### `dt: float`

Шаг управления / интегрирования. Должен совпадать с `dt` агента. `0.01` (100 Гц) — default в репозитории.

### `initial_state: list[list[float]]`

Столбец, форма `(n_full_state, 1)`. Каждый подсписок — один элемент. Порядок соответствует внутренней раскладке состояния (описана в документации среды).

### `reference_signal: np.ndarray`

Эталонная траектория, форма `(len(tracking_states), number_time_steps)`. Для одного канала всегда `np.reshape(signal, (1, -1))`.

### `state_space: list[str]`

Имена состояний, образующих наблюдение. Подмножество полного состояния.

### `output_space: list[str]`

Имена, передаваемые в функцию награды. Часто совпадает со `state_space`.

### `tracking_states: list[str]`

Имена, соответствующие `reference_signal`. `tracking_states = ['alpha']` означает, что `reference_signal[0, t]` — желаемый α в тике `t`.

### `control_space: list[str]` (нелинейные)

Имена каналов. `['aileron', 'stab', 'rudder']` для 6-DoF; `env.action_space` содержит `len(control_space)` элементов.

### `use_reward: bool = True`

Если `False`, `env.step()` возвращает `reward = 0`. Отключайте для PID / MPC.

### `reward_func: Callable | None = None`

Кастомная `(obs, ref, u, info) -> float`. При `None` используется квадратичная награда слежения.

### `integrator: str = 'rk4'` (нелинейные)

`'euler'` быстрее и достаточно для `dt ≤ 0.01`; `'rk4'` — для агрессивной динамики.

### `control_bias: float | np.ndarray` (нелинейные)

Аддитивное смещение действия перед передачей в объект. Работа вокруг трима: агент выдаёт *отклонение*, среда добавляет трим.

## Шаг 3 — Обзор сред

| Env id | Модель | `tracking_states` (по умолчанию) | Примечания |
|---|---|---|---|
| `LinearLongitudinalF16-v0` | 4-стат. линейная F-16 | `alpha`, `q` | Bootstrap для PID/MPC. Допустимо переопределить, например `["alpha"]` или `["alpha","theta","wz"]`. |
| `NonlinearLongitudinalF16-v0` | NumPy нелинейная F-16 (прод.) | `alpha` | Нужен `control_bias` для трима. |
| `NonlinearAngularF16-v0` | 6-DoF нелинейная F-16 | настраивается | 3–4 канала управления (см. `split_stab`). |
| `LinearLongitudinalB747-v0` | Линейный B747 | `theta` | Классический baseline. |
| `ImprovedB747-v0` | Нормализованная нелинейная B747 | настраивается | Используется в большинстве RL-экспериментов и MPC-демо. |

Среды Unity не регистрируются как `gym.make`-id и инициализируются через
фабрики `get_plane_env()` / `unity_discrete_env()` (см.
[Unity-окружение](../guide/unity_env.md)).

Полный список регистраций — в `tensoraerospace/__init__.py` (есть также
`LinearLongitudinalX15-v0`, `LinearLongitudinalLAPAN-v0`,
`LinearLongitudinalUAV-v0`, `LinearLongitudinalUltrastick-v0`,
`LinearLongitudinalELVRocket-v0`, `LinearLongitudinalMissileModel-v0`,
`LinearLongitudinalF4C-v0`, `GeoSat-v0`, `ComSat-v0` и их Improved-варианты).

Запустите `gym.make(<id>).unwrapped.__doc__` или смотрите [Объекты управления](../model/f16.md) для полной документации.

## Шаг 4 — Идиома `.unwrapped`

```python
env = gym.make('NonlinearLongitudinalF16-v0', ...).unwrapped
```

`.unwrapped` снимает обёртки Gymnasium — доступны env-специфичные помощники: `env.get_state()`, `env.ref_signal`, внутренний интегратор. Все примеры TensorAeroSpace её используют.

## Куда дальше

- **[Рецепт 01 — Hello, TensorAeroSpace](01_hello.md)** — минимальный PID-цикл.
- **[Рецепт 03 — Задающие сигналы](03_reference_signals.md)** — все формы `reference_signal`.
- **[Рецепт 10 — Добавить свой объект управления](10_custom_plant.md)** — написать свою среду.
