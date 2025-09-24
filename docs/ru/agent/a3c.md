# A3C (Asynchronous Advantage Actor‑Critic)

A3C сочетает преимущества policy‑based и value‑based подходов: несколько асинхронных рабочих агентов параллельно исследуют среду и обновляют общую (глобальную) сеть, используя функцию преимущества.

![A3C схема](../agent/img/a3c/a3c.png){ width=800 }

## Компоненты

- Глобальная сеть: общие параметры Actor (политики) и Critic (оценки V)
- Воркеры: независимые копии среды и локальные копии сети для сбора опыта
- Advantage‑обновление: градиент политики взвешивается \(A_t\)

## Теория (на базе реализации)

### Политика (Actor) — гауссовская, параметризация через \(\mu\) и \(\sigma\)

Сеть актёра выводит среднее \(\mu(s)\) (через `tanh` и масштабирование до `action_bound`) и \(\sigma(s)\) (через `softplus`, затем клиппинг в `[std_min, std_max]`). Действие семплируется как:

$$
 a \sim \mathcal{N}\big(\mu(s),\ \sigma^2(s)\big)
$$

Лог‑плотность гаусса (для многомерного действия суммируется по осям):

$$
\log \pi_\theta(a|s) = -\tfrac{1}{2}\,\frac{(a-\mu)^2}{\sigma^2} - \tfrac{1}{2}\,\log(2\pi\sigma^2)
$$

Потеря актёра (с отрицательным знаком, чтобы минимизировать):

$$
\mathcal{L}_\text{actor}(\theta) = -\,\mathbb{E}\big[\log \pi_\theta(a_t|s_t)\, A_t\big]
$$

(энтропия политики может добавляться как регуляризатор: \(+\beta\,\mathbb{E}[\mathcal{H}[\pi]]\)).

### Оценщик (Critic) — скалярная V‑сеть

Критик оценивает \(V_\phi(s)\). Потеря критика — MSE между n‑шаговым таргетом и предсказанием:

$$
\mathcal{L}_\text{critic}(\phi) = \mathbb{E}\big[\big(R_t^{(n)} - V_\phi(s_t)\big)^2\big]
$$

n‑шаговый таргет (в коде реализован упрощённый вариант, без \(\gamma\) внутри цикла и без «bootstrap» от \(V(s_{t+n})\); это допускается как эвристика):

$$
R_t^{(n)} \approx \sum_{k=0}^{n-1} r_{t+k} \quad (\text{эпизодический финал обнуляет «хвост»})
$$

Классическая форма, к которой можно вернуться при доработке:

$$
R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n})
$$

### Advantage

$$
A_t = R_t^{(n)} - V_\phi(s_t)
$$

Он масштабирует лог‑градиент политики.

### Асинхронность и синхронизация

- Каждый воркер периодически (каждые `update_interval` шагов или при завершении эпизода) формирует батчи и обновляет глобальные сети (в реализации вектор обновлений актёра/критика закомментирован, оставлена синхронизация локальных весов с глобальными).
- Синхронизация локальных копий с глобальными: `sync_with_global()` копирует веса глобальных сетей актёра и критика в локальные.

### Расписания и гиперпараметры

- Скорости обучения: `actor_lr`, `critic_lr`
- Скидка: `gamma`
- Размер скрытых слоёв: `hidden_size`
- Интервал обновлений: `update_interval`
- Лимит эпизодов: `max_episodes`
- Пределы действий и стандартного отклонения: `action_bound`, `std_bound`

## Асинхронное обучение (псевдокод)

```text
parallel for worker in 1..W:
  sync local nets from global
  s = env.reset()
  trajectory = []
  for t in range(T_max):
    a ~ N(mu_theta(s), sigma_theta(s)) ; a = clip(a, action_bound)
    s', r, done = env.step(a)
    push (s,a,r,s') into trajectory
    if done or len(trajectory) == update_interval:
      # в текущей версии R^n — накопленная сумма наград
      compute R^n from rewards (and optionally bootstrap V(s'))
      A = R^n - V_phi(s)
      # (обновления глобальной сети могут быть добавлены здесь)
      sync local nets from global
      clear trajectory
    s = s'
    if done: s = env.reset()
```

## Быстрый старт

```python
import gymnasium as gym
import numpy as np

from tensoraerospace.agent.a3c.model import Agent, setup_global_params

# Глобальные гиперпараметры
actor_lr = 0.0005
critic_lr = 0.001
gamma = 0.99
hidden_size = 128
update_interval = 5
max_episodes = 100

setup_global_params(actor_lr, critic_lr, gamma, hidden_size, update_interval, max_episodes)

# Обёртка для окружения (пример)
def env_function(worker_id: int):
    env = gym.make('LinearLongitudinalF16-v0', number_time_steps=2000)
    return env

agent = Agent(env_function, gamma)
agent.train()
```

!!! tip
    Для непрерывных действий не забывайте клиппировать действие по `action_bound`; полезно ограничивать `std` снизу (например, `1e-2`) для численной стабильности.

## Документация API

::: tensoraerospace.agent.a3c.model.Agent

::: tensoraerospace.agent.a3c.model.Worker

::: tensoraerospace.agent.a3c.model.Actor

::: tensoraerospace.agent.a3c.model.Critic

## Источники

- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

## Где тестировалось

- Unity‑среда
