# Generative Adversarial Imitation Learning (GAIL)

Generative Adversarial Imitation Learning (GAIL) — алгоритм имитационного обучения, который восстанавливает стратегию эксперта по наблюдаемым траекториям, используя состязательное обучение актор–дискриминатор.

## Обзор

Цель: обучить политику, воспроизводящую поведение эксперта, без явной функции вознаграждения.

Архитектура:
- Policy (Actor): генерирует действие по состоянию.
- Discriminator: отличает пары (состояние, действие) эксперта от сгенерированных агентом.
- Обучение: актор максимизирует «обман» дискриминатора; дискриминатор минимизирует ошибку классификации. Для стабильности актор обычно обновляется PPO/TRPO.

## Как работает GAIL

GAIL на каждом шаге порождает батч траекторий агента и сравнивает их с экспертными данными:

1. Дискриминатор обучается классифицировать `(s, a)` как «эксперт/агент».
2. Актор получает псевдо‑награду от дискриминатора и обновляется (в TensorAeroSpace — PPO).
3. Повторяем до сходимости: действия агента становятся неотличимыми от экспертных.

## Зачем применять в авиации

- Когда функция вознаграждения неоднозначна (комфорт/стиль пилотирования, мягкость касания и т.п.).
- Для переноса человеческой экспертизы (демонстрации пилотов/автопилотов) в обучаемую политику.
- Для инициализации RL‑агента перед последующим дообучением с подкреплением.

## Формат данных эксперта

Ожидается массив `expert_data` формы `[N, obs_dim + act_dim]`: для каждой строки конкатенация текущего состояния и выбранного действия эксперта.

## Пример (LinearLongitudinalF16‑v0)

Полный пример — в ноутбуке `example/reinforcement_learning/example_gail.ipynb`. Ниже краткая схема:

```python
import gymnasium as gym
import numpy as np
from tensoraerospace.agent.gail.model import GAIL
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standart import unit_step

dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)
reference_signals = np.reshape(
    unit_step(degree=5, tp=tp, time_step=1000, output_rad=True), [1, -1]
)

env = gym.make(
    'LinearLongitudinalF16-v0',
    number_time_steps=number_time_steps,
    initial_state=[[0], [0], [0]],
    reference_signal=reference_signals,
    use_reward=False,
    state_space=["theta", "alpha", "q"],
    output_space=["theta", "alpha", "q"],
    control_space=["ele"],
    tracking_states=["alpha"],
)

expert_data = np.load('expert_f16.npy')
agent = GAIL(env, learning_rate=3e-3, max_steps=20, mini_batch_size=16, epochs=4, data=expert_data)
agent.learn(max_frames=5000, max_reward=-1)
```

## Практические замечания

- Нормируйте состояния/действия; проверяйте размеры входов дискриминатора.
- Качество и разнообразие демонстраций критично для обобщающей способности политики.
- Для стабильности применяйте более крупные мини‑батчи и меньшие скорости обучения.

## Документация API

::: tensoraerospace.agent.gail.model.GAIL

## Источники

- [Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476)

## На каких средах протестили:

- Unity среда
- LinearLongitudinalF16‑v0 (пример в репозитории)