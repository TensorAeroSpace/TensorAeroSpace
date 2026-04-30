# Рецепт 05 — Обучение deep-RL от и до

Пошаговый скелет прогона PPO / SAC / DSAC на одной из сред библиотеки. Указываем на три полных рабочих ноутбука в репозитории и показываем короткую версию, которую можно скопировать в свой скрипт.

**Связано.** [Рецепт 04](04_choosing_agent.md) — выбор семейства; [Рецепт 07](07_optuna_search.md) — тюнинг; [Рецепт 08](08_huggingface.md) — публикация.

## Пять шагов

1. Подготовить обучающую среду (`use_reward=True`).
2. Построить агента с гиперпараметрами.
3. Обучить на `n_episodes` роллаутах.
4. Оценить детерминированно на отложенной эталонной траектории.
5. Сохранить / опубликовать.

## Шаг 1 — Обучающая среда

```python
import gymnasium as gym
import numpy as np

import tensoraerospace  # регистрирует среды
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standart import unit_step

dt = 0.01
tp = generate_time_period(tn=20, dt=dt)

def make_train_env():
    ref = np.reshape(unit_step(tp=tp, degree=5, time_step=2.0, output_rad=True), (1, -1))
    return gym.make(
        'LinearLongitudinalF16-v0',
        number_time_steps=len(tp),
        use_reward=True,                  # ← обязательно для RL
        initial_state=[[0], [0], [0]],
        reference_signal=ref,
        state_space=['theta', 'alpha', 'q'],
        output_space=['theta', 'alpha', 'q'],
        tracking_states=['alpha'],
    )
```

**Совет.** Рандомизируйте `initial_state` и время ступеньки между эпизодами внутри `make_train_env`; агенты, обученные на одной траектории, быстро переобучаются.

## Шаг 2 — Конструируем агента

```python
from tensoraerospace.agent.ppo import PPO

agent = PPO(
    env=make_train_env(),
    gamma=0.99,
    max_episodes=200,
    rollout_len=2048,
    clip_pram=0.2,
    actor_lr=1e-3,
    critic_lr=5e-3,
    seed=0,
)
```

SAC и DSAC принимают аналогичный `env=` и похожие гиперпараметры. Полные списки ручек — на страницах агентов в [Agents](../agent/ppo.md).

## Шаг 3 — Обучение

Самый простой путь:

```python
agent.learn()          # запускает тренировочный цикл PPO на max_episodes
```

Обучение на F-16 с нуля — задача неприветливая: пространство вознаграждения почти плоское, и энтропия часто коллапсирует до того, как политика успела изучить среду. Самый надёжный deep-RL-демо в репо — это **DSAC на нормированной B747**, где награда плотнее и пространство действий предварительно нормировано:

![DSAC на нормированной B747 — слежение за ступенькой](img/05_dsac_b747_tracking.png)

Этот график — eval-вывод ноутбука [`eval_dsac_b747_step_response.ipynb`](../example/agent/dsac/example-dsac-b747.md), который оценивает агента, обученного ~1000 эпизодов скриптом `train-dsac-b747-step-response.py`. Воспроизведение ~30 минут на CPU; CUDA-GPU ужимает до ~5 минут.

Если SAC / PPO у вас не сходятся, обычные причины:

- **Слишком разреженная награда.** Нужна плотная `-||y - y_r||²` вместо пороговых сигналов.
- **Не нормированное действие.** Среды `_improved` (`LinearB747Improved-v0`, …) уже нормируют действие к `[-1, 1]`. Предпочитайте их, когда можно.
- **Слишком короткий горизонт обучения.** `max_episodes = 30` в дефолте PPO — это разогрев, не политика. Поднимайте до 200+ для чего-то нетривиального.

## Шаг 4 — Оценка

Держите эталонный сигнал, который агент не видел:

```python
eval_env = make_eval_env()               # другое время ступеньки или амплитуда
obs, _ = eval_env.reset()
returns = 0.0
for _ in range(len(tp) - 2):
    action = agent.select_action(obs, deterministic=True)
    obs, r, terminated, truncated, _ = eval_env.step(action)
    returns += r
print(f'eval return: {returns:.2f}')
```

Используйте `tensoraerospace.benchmark.function` для `overshoot`, `settling_time`, `static_error` — чтобы агенты сравнивались по-честному.

## Шаг 5 — Save / reload

```python
agent.save('./checkpoints/ppo_f16')
# позже
from tensoraerospace.agent.ppo import PPO
agent2 = PPO.load('./checkpoints/ppo_f16', env=make_eval_env())
```

У пяти онлайн-адаптивных агентов (IHDP, IM-GDHP, ET-DHP, AA-INDI, iADP) API богаче (полный HuggingFace Hub round-trip) — см. [Рецепт 08](08_huggingface.md).

## Готовые ноутбуки для копирования

| Задача | Ноутбук |
|---|---|
| **DSAC на B747 со ступенчатым воздействием (источник графика выше)** | [`example/agent/dsac/example-dsac-b747.md`](../example/agent/dsac/example-dsac-b747.md) |
| DSAC на B747, синусоидальное слежение | [`example/agent/dsac/train-dsac-b747-tracking.md`](../example/agent/dsac/train-dsac-b747-tracking.md) |
| PPO на B747 (нормированная среда) | `example/reinforcement_learning/example_a2c_b747_improved.ipynb` и соседние PPO-ноутбуки |
| SAC на линейной F-16 | [`example/agent/sac/example-sac-f16.md`](../example/agent/sac/example-sac-f16.md) — медленнее сходится, полезно как референс архитектуры |
| SAC на B747 | [`example/agent/sac/example-sac-b747.md`](../example/agent/sac/example-sac-b747.md) |

## Подводные камни

- **Слишком узкий обучающий сигнал.** Всегда рандомизируйте reference_signal между эпизодами.
- **Длина эпизода = `number_time_steps - 2`.** Среда «закладывает» два тика в конце.
- **Несоответствие амплитуды действий.** PPO/SAC выдают нормированные [-1, 1]; среда масштабирует в физические единицы автоматически. При написании своего агента учитывайте `env.action_space.low / high`.

## Куда дальше

- **[Рецепт 06 — Онлайн-адаптивные агенты](06_online_adaptive.md)** — когда полное обучение не по силам.
- **[Рецепт 07 — Гиперпоиск через Optuna](07_optuna_search.md)** — автоматизация тюнинга.
- **[Рецепт 08 — Save/load/публикация в HuggingFace](08_huggingface.md)** — отправка обученной модели.
