# Generative Adversarial Imitation Learning (GAIL)

GAIL — имитационное обучение через состязание актор–дискриминатор: политика учится генерировать траектории, неотличимые от экспертных, без явной функции вознаграждения.

## Компоненты

- Actor‑Critic: `ActorCritic` генерирует действие и оценивает \(V(s)\)
- Discriminator: `Discriminator` отличает пары \((s,a)\) эксперта от агента
- Оптимизатор политики: обновление актёра по PPO (клиппированный суррогат)

## Теория

- Минмакс‑задача (формально):

$$
\min_{\pi} \max_{D} \; \mathbb{E}_{(s,a)\sim \pi}[\log D(s,a)] + \mathbb{E}_{(s,a)\sim \pi_E}[\log (1 - D(s,a))]
$$

- Псевдо‑награда от дискриминатора (для актёра):

$$
 r_D(s,a) = -\log D(s,a)
$$

- PPO‑обновление актёра (в нашей реализации):

$$
\mathcal{L}_\text{actor} = -\,\mathbb{E}\Big[ \min\big(r_t A_t,\ \mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon) A_t\big) \Big],\quad
r_t = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})
$$

- Advantage через GAE:

$$
\delta_t = r_D(s_t,a_t) + \gamma V(s_{t+1}) - V(s_t),\quad
\hat{A}_t = \sum_{l\ge 0} (\gamma\lambda)^l\, \delta_{t+l}
$$

## Данные эксперта

Ожидается массив `expert_data` формы `[N, obs_dim + act_dim]` — конкатенация состояния и действия.

## Обучение (контур по реализации)

1. Собираем траектории: исходные действия политики, исполненные команды, логарифмы вероятностей и оценки ценности.
2. Обучаем дискриминатор на этих командах и данных эксперта: `D(fake)=1`, `D(real)=0`.
3. Пересчитываем награды обновлённым дискриминатором, затем вычисляем GAE и целевые возвраты.
4. Обновляем актёр и критик мини-батчами PPO.
5. Между фрагментами оцениваем политику и применяем early stopping по `max_reward`.

`Discriminator.logits()` возвращает неограниченную оценку `z`. Для обучения
используется `BCEWithLogitsLoss`, для награды — `softplus(-z)`, эквивалент
`-log(sigmoid(z))` без ограничения вероятности снизу. Градиент сохраняется
даже при уверенно неверном ответе дискриминатора. `forward()` по-прежнему
возвращает вероятность; имена и размеры сохраняемых параметров не изменены.

Обновление дискриминатора перед вычислением наград соответствует
[алгоритму 1 исходной статьи GAIL](https://arxiv.org/html/1606.03476v1).
Для шага политики здесь используется PPO. Устойчивые потери и правильный
порядок обновлений не гарантируют сходимость: необходимо проверять точность
управления и нарушения ограничений на разных seed и бюджетах обучения.

## Пример (LinearLongitudinalF16‑v0)

```python
import gymnasium as gym
import numpy as np
from tensoraerospace.agent.gail.model import GAIL
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standard import unit_step

dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)
reference_signals = unit_step(degree=5, tp=tp, time_step=1000, output_rad=True).reshape(1, -1)

env = gym.make('LinearLongitudinalF16-v0',
               number_time_steps=number_time_steps,
               initial_state=[[0],[0],[0]],
               reference_signal=reference_signals,
               use_reward=False,
               state_space=["theta","alpha","q"],
               output_space=["theta","alpha","q"],
               control_space=["ele"],
               tracking_states=["alpha"],)

expert_data = np.load('expert_f16.npy')
agent = GAIL(env, learning_rate=3e-3, max_steps=20, mini_batch_size=16, epochs=4, data=expert_data)
agent.learn(max_frames=5000, max_reward=-1)

# Унифицированный API (обёртка над learn)
agent.train(num_episodes=250, max_steps=20, max_reward=-1)
```

## Унифицированный интерфейс обучения

GAIL предоставляет общий унифицированный API `train()` из `BaseRLModel`.
Внутри он делегирует вызов устаревшему методу `learn()`, пересчитывая
`num_episodes * max_steps` в бюджет `max_frames`. GAIL‑специфичные
параметры, принимаемые через `**kwargs`:

- `max_frames` (`int`): переопределяет вычисленный бюджет шагов.
- `max_reward` (`float`): порог раннего останова по средней награде на
  тесте.

!!! tip
    Качество `expert_data` критично: добавьте демо с разными начальными условиями и манёврами.

!!! warning "Gymnasium 5-tuple API"
    Реализация использует современный 5‑элементный API `step` из Gymnasium:
    ```python
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    ```
    Если вы переходите со старого кода с 4‑элементным API (`next_state, reward, done, info = env.step(action)`), убедитесь, что среда совместима с Gymnasium и возвращает 5‑элементный кортеж.

## Документация API

::: tensoraerospace.agent.gail.model.GAIL

## Источники

- [Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476)

## Где тестировалось

- Unity‑среда
- LinearLongitudinalF16‑v0 (пример в репозитории)

## Границы траекторий

При сборе опыта наблюдения копируются: среда может повторно использовать
буфер numpy. При лимите времени сохраняется оценка последнего наблюдения,
при настоящем завершении — нет. Бюджет шагов обучения соблюдается точно.
Каждые 1 000 шагов собранный фрагмент обрабатывается до запуска оценки
в той же среде. Затем обучение начинает новый эпизод; прерванный эпизод
логируется как усечённый. Награда имитации остаётся конечной при нулевом
выходе дискриминатора.

Политика обновляется по каждому собранному фрагменту с заданным числом `epochs`.
Минибатчи перемешиваются без повторений и включают неполный остаток. PPO
использует совместную вероятность многомерного действия, нормализацию
преимуществ и ограничение нормы градиента. Среда и дискриминатор получают
ограниченные команды; для вычисления вероятности PPO сохраняет исходную
выборку гауссовской политики.

Bootstrap при лимите времени использует `final_observation` или
`terminal_observation`; при отсутствии обоих используется возвращённое наблюдение.
