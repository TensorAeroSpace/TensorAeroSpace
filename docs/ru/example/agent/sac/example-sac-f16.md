# Пример: SAC на линейной F-16 — слежение за ступенькой по α

В этом примере **Soft Actor-Critic (SAC)** обучается отслеживать ступеньку по углу атаки 5° на среде `LinearLongitudinalF16-v0`. SAC — это off-policy стохастический actor-critic с максимизацией энтропии, хорошо подходящий для непрерывных задач управления, где важно исследование, а градиентно-свободные альтернативы (например, IHDP) с трудом покрывают пространство состояний.

Исходный ноутбук: `example/reinforcement_learning/deep_rl/example-sac-f16.ipynb`.

## Когда выбирать SAC, а когда — адаптивный критик

| Аспект | SAC | IHDP / DHP |
|---|---|---|
| Стиль обучения | Off-policy, replay-buffer | Онлайн, по одному переходу |
| Модель ОУ | Не нужна | Онлайн инкрементальная линеаризация |
| Исследование | Стохастическая политика + энтропия | Persistent excitation |
| Сложность тюнинга | Низкая–средняя | Низкая (при удачной инициализации/FF) |
| Эффективность по данным | Низкая (10⁴–10⁶ шагов среды) | Высокая (1–10 эпизодов) |
| Интерпретируемость | Низкая | Высокая (явные \(F, G\)) |

SAC уместен, когда есть бюджет на длительное обучение и нужен робастный «чёрный ящик». IHDP (см. [пример с линейным IHDP](../ihdp/example_ihdp.md)) выбирают для быстрой онлайн-адаптации.

## 1. Импорты

```python
import itertools

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from tensoraerospace.agent.sac import SAC
from tensoraerospace.signals.standard import unit_step
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period
```

## 2. Сетка времени и опорный сигнал

Эпизод 20 секунд при \(dt = 0{,}01\) с (2 000 шагов). Задание — ступенька 5° в момент \(t = 10\) с — та же задача, что в примере с линейным IHDP, для прямого сравнения.

```python
dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)

reference_signals = np.reshape(
    unit_step(degree=5, tp=tp, time_step=10, output_rad=True),
    [1, -1],
)
```

## 3. Сборка среды

```python
env = gym.make(
    'LinearLongitudinalF16-v0',
    number_time_steps=number_time_steps,
    initial_state=[[0], [0]],
    reference_signal=reference_signals,
)
state, info = env.reset()
```

Среда отдаёт двумерное наблюдение `[alpha, wz]` (радианы, рад/с) и одномерное действие — команда руля высоты (радианы). Встроенная функция награды штрафует ошибку слежения по \(\alpha\) плюс малый член по угловой скорости тангажа; см. `LinearLongitudinalF16.default_reward`.

## 4. Создание SAC-агента

Класс `SAC` сам владеет replay-буфером, парой критиков, target-сетями и стохастической гауссовой политикой. Гиперпараметры по умолчанию следуют референсной статье; здесь переопределены только `hidden_size` и `device` для лёгкого запуска.

```python
seed = 42
batch_size = 256
updates_per_step = 1
num_steps = 100_000   # суммарно шагов среды по всем эпизодам обучения

torch.manual_seed(seed)
np.random.seed(seed)

agent = SAC(
    env=env,
    hidden_size=32,
    batch_size=batch_size,
    updates_per_step=updates_per_step,
    memory_capacity=1_000_000,
    device="cpu",
    seed=seed,
)
```

Ключевые гиперпараметры (все взяты по умолчанию, если не указано иное):

| Параметр | Значение | Смысл |
|---|---|---|
| `gamma` | 0.99 | Дисконт |
| `tau` | 0.005 | Поляк-усреднение target-сетей |
| `alpha` | 0.2 | Температура энтропии (вес энтропийного бонуса) |
| `policy_type` | "Gaussian" | Squashed-Gaussian стохастическая политика |
| `automatic_entropy_tuning` | False | Если True, `alpha` обучается онлайн |
| `lr` / `policy_lr` | 3e-4 | Learning rate критика и политики |

## 5. Цикл обучения

Классический off-policy роллаут. На каждом шаге: сэмплирование действия → переход → запись в replay → `updates_per_step` градиентных шагов на критике и политике, как только в буфере накопится достаточно сэмплов.

```python
total_numsteps = 0
updates = 0

for _ in itertools.count(1):
    episode_reward = 0.0
    episode_steps = 0
    state, info = env.reset()
    state = np.array(state, dtype=np.float32).reshape(-1)
    reward_per_step = []

    for _ in tqdm(range(number_time_steps - 1)):
        action = agent.select_action(state)

        # Делаем градиентные шаги, как только накоплено достаточно опыта.
        if len(agent.memory) > batch_size:
            for _ in range(updates_per_step):
                (c1_loss, c2_loss, policy_loss,
                 ent_loss, alpha_val) = agent.update_parameters(
                    agent.memory, batch_size, updates
                )
                updates += 1

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_state, dtype=np.float32).reshape(-1)

        episode_steps += 1
        total_numsteps += 1
        episode_reward += float(reward)
        reward_per_step.append(float(reward))

        # mask=1 при truncated (бутстрап Q), 0 при terminated.
        mask = 1.0 if (episode_steps == number_time_steps - 1) else float(
            not (terminated or truncated)
        )
        agent.memory.push(state, action, float(reward), next_state, mask)
        state = next_state

        if terminated or truncated:
            break

    print(f"episode reward={episode_reward:+.3f}  avg/step={np.mean(reward_per_step):+.4f}")

    if total_numsteps > num_steps:
        break
```

!!! note "Стоимость обучения"
    100 000 шагов среды × батчи по 256 займут ~10–20 минут на CPU. Для быстрого smoke-теста снизьте `num_steps` до 10 000 — агент не сойдётся полностью, но кривая награды выйдет из шума за несколько эпизодов. Для серьёзных запусков ставьте `device="cuda"` и `hidden_size=128` или `256`.

## 6. Детерминированная оценка

После обучения переходим в режим `evaluate=True` — политика возвращает математическое ожидание squashed-Gaussian вместо сэмпла.

```python
state, info = env.reset()
state = np.array(state, dtype=np.float32).reshape(-1)

alpha_log, action_log = [], []
total_rew = 0.0
for step in range(number_time_steps - 1):
    action = agent.select_action(state, evaluate=True)
    state, reward, terminated, truncated, info = env.step(action)
    state = np.array(state, dtype=np.float32).reshape(-1)
    alpha_log.append(float(state[0]))
    action_log.append(float(action[0]))
    total_rew += float(reward)
    if terminated or truncated:
        break

print(f"eval total reward = {total_rew:+.3f}")
```

## 7. Визуализация слежения

```python
env.unwrapped.model.plot_transient_process(
    'alpha', tps, reference_signals[0], to_deg=True, figsize=(15, 4)
)
```

Встроенные plot-хелперы среды обращаются к полной траектории состояний через `env.unwrapped.model` — это форма после Gymnasium-обёртки, поэтому `unwrapped` обязателен для выхода на нижележащий объект `ModelBase`.

## 8. Количественная проверка

```python
alpha_hist = env.unwrapped.model.get_state('alpha', to_deg=True)
ref_deg = np.rad2deg(reference_signals[0, :len(alpha_hist)])

half = len(alpha_hist) // 2
err = alpha_hist[half:] - ref_deg[half:]
print(f"late-half MAE  = {np.mean(np.abs(err)):.4f}°")
print(f"late-half RMSE = {np.sqrt(np.mean(err ** 2)):.4f}°")
print(f"late-half max  = {np.max(np.abs(err)):.4f}°")
```

## Заметки и подводные камни

- **Маленькая сеть.** `hidden_size=32` — агрессивный выбор: на этой низкоразмерной задаче такая сеть сходится быстрее 256-юнитной, но чувствительнее к настройке `alpha` (энтропии). Если политика коллапсирует, включите `automatic_entropy_tuning=True`.
- **Прогрев replay-буфера.** Первые ~256 шагов среды градиентных обновлений нет — `agent.memory` только заполняется. Ранние эпизоды будут выглядеть как шум.
- **Масштаб награды.** `LinearLongitudinalF16.default_reward` возвращает значения порядка `-0.1` за шаг. Если кривая награды плоская, переключитесь на `use_reward=False` и подставьте свою функцию награды при сборке среды.
- **Масштабирование действия.** Гауссова политика SAC выдаёт действия, уже отображённые в `action_space.low`/`action_space.high` среды. Внешний `Box` или клипирование на стороне агента не нужны.

## См. также

- **Документация API SAC:** [`docs/ru/agent/sac.md`](../../../agent/sac.md) — теория и полный API.
- **Альтернативный регулятор:** [`Пример: IHDP на линейной F-16`](../ihdp/example_ihdp.md) — онлайн-адаптация на порядки меньшим числом шагов среды.
- **Более тяжёлый ОУ:** [`Пример: SAC на B-747`](example-sac-b747.md) — тот же рецепт SAC на продольной модели B-747.
