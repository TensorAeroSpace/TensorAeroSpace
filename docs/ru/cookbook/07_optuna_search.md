# Рецепт 07 — Гиперпоиск через Optuna

Настройка PID-коэффициентов от и до: задать objective (RMSE на позднем окне), запустить 25 Optuna-trial'ов, получить лучшие коэффициенты, построить историю поиска.

Исходный ноутбук: [`example/cookbook/recipe_07_optuna.ipynb`](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/cookbook/recipe_07_optuna.ipynb).

## Шаг 1 — Импорты

```python
import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import tensoraerospace  # noqa: F401
from tensoraerospace.agent.pid import PID
from tensoraerospace.optimization import HyperParamOptimizationOptuna
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.utils import generate_time_period

dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
N = len(tp)
reference_signal = np.reshape(
    unit_step(tp=tp, degree=5, time_step=2.0, output_rad=True), (1, -1)
)

def make_env():
    env = gym.make(
        'LinearLongitudinalF16-v0',
        number_time_steps=N,
        use_reward=False,
        initial_state=[[0], [0], [0]],
        reference_signal=reference_signal,
        state_space=['theta', 'alpha', 'q'],
        output_space=['theta', 'alpha', 'q'],
        tracking_states=['alpha'],
    ).unwrapped
    env.reset()
    return env
```

## Шаг 2 — Objective-функция

```python
def objective(trial):
    kp = trial.suggest_float('kp', -25.0, -3.0)
    ki = trial.suggest_float('ki', -15.0, -1.0)
    kd = trial.suggest_float('kd', -3.0, -0.1)

    env = make_env()
    pid = PID(env, kp=kp, ki=ki, kd=kd, dt=dt)

    alpha_meas = []
    xt = np.zeros(3)
    for step in range(N - 2):
        setpoint = float(reference_signal[0, step])
        ut = pid.select_action(setpoint, float(xt[1]))
        xt, *_ = env.step(np.array([ut]))
        alpha_meas.append(float(xt[1]))

    alpha_meas = np.asarray(alpha_meas)
    ref = reference_signal[0, : len(alpha_meas)]
    late = slice(-500, None)
    return float(np.sqrt(np.mean((alpha_meas[late] - ref[late]) ** 2)))
```

Три ручки, разумные диапазоны, возврат RMSE на позднем окне (меньше — лучше). Меняйте знак возврата, если ваша objective — накопленная награда.

## Шаг 3 — Запустить study

```python
opt = HyperParamOptimizationOptuna(direction='minimize')
opt.run_optimization(objective, n_trials=25)

best = opt.get_best_param()
print('Best gains:')
for k, v in best.items():
    print(f'  {k} = {v:+.3f}')
print(f'Best RMSE:  {opt.study.best_value:.5f} рад '
      f'({np.degrees(opt.study.best_value):.4f} град)')
```

**Ожидаемый вывод (ваши числа в пределах ±10 %):**

```
Best gains:
  kp = -24.659
  ki = -8.745
  kd = -1.675
Best RMSE:  0.00004 рад (0.0020 град)
```

25 trial'ов достаточно для 3 примерно-выпуклых PID-параметров. Дефолтный `TPESampler` Optuna сходится в нужный бассейн за ~15 trial'ов.

## Шаг 4 — График истории поиска

```python
fig = opt.plot_parms(figsize=(14, 5))
plt.show()
```

**Эталонный график:**

![История поиска Optuna](img/07_search_history.png)

Каждая метка по X — один trial с аннотацией `(kp, ki, kd)`. Objective (ось Y) резко падает за первые ~5 trial'ов и выходит на плато в лучшем бассейне.

## Шаг 5 — Прогнать лучшие коэффициенты, построить слежение

```python
env = make_env()
pid = PID(env, **best, dt=dt)
alpha_meas, u_trace = [], []
xt = np.zeros(3)
for step in range(N - 2):
    setpoint = float(reference_signal[0, step])
    ut = pid.select_action(setpoint, float(xt[1]))
    xt, *_ = env.step(np.array([ut]))
    alpha_meas.append(float(xt[1])); u_trace.append(float(ut))

alpha_meas = np.asarray(alpha_meas); u_trace = np.asarray(u_trace)
ref = reference_signal[0, : len(alpha_meas)]
t = tp[: len(alpha_meas)]

fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
axes[0].plot(t, np.degrees(ref), 'k--', label='команда alpha')
axes[0].plot(t, np.degrees(alpha_meas), label='alpha (тюнед PID)')
axes[0].set_ylabel('alpha [град]'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(t, u_trace); axes[1].set_ylabel('stab команда [рад]')
axes[1].set_xlabel('время [с]'); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

**Эталонный график:**

![Тюнед PID — слежение](img/07_tuned_tracking.png)

α достигает команды 5° без заметного перерегулирования и устанавливается за ~0.5 с. RMSE на позднем окне в диапазоне `< 0.01°` — на порядок плотнее, чем руками настроенные коэффициенты из [Рецепта 01](01_hello.md).

## Типичные отклонения

| Симптом | Причина |
|---|---|
| Все trial'ы дают одинаковый RMSE | Диапазоны не накрывают стабильный регион — расширьте (kp до -30, ki до -20). |
| Лучшее значение не улучшается после trial 5 | Мало trial'ов ИЛИ плоская objective — попробуйте `n_trials=50` или добавьте параметры. |
| Взрывается runtime | Слишком большой `number_time_steps` на trial; используйте короче горизонт (10 с вместо 20 с). |

## Варианты, стоящие изучения

- **Добавить прунер.** `optuna.pruners.MedianPruner` рано обрывает плохие trial'ы. Подключите через `opt.study = optuna.create_study(..., pruner=MedianPruner())`.
- **Параллельные воркеры.** Общее `sqlite://`-хранилище для нескольких параллельных trial'ов.
- **Multi-objective.** Возвращайте кортеж `(rmse, расход_управления)` и выбирайте с Парето-фронта.

## Куда дальше

- **[Ноутбук-пример Optuna](../example/optimization/example_optimization.md)** — настройка PID с богаче графиком.
- **[Рецепт 08 — Save/load/публикация в HuggingFace](08_huggingface.md)** — опубликовать настроенного агента.
