# DQN (Deep Q-Network)

DQN — классический метод обучения с подкреплением, аппроксимирующий Q‑функцию нейросетью. В реализации используется целевая сеть для стабилизации обучения и приоритезированный опыт (PER) для более информативных обновлений.

![DQN схема](../agent/img/dqn/DQN.png){ width=800 }

## Компоненты

- Основная Q‑сеть: оценивает \(Q_\theta(s,a)\) и выбирает действия
- Целевая сеть: \(Q_{\theta^-}\) для вычисления целевых значений, реже обновляется
- Буфер повторов (PER): хранит переходы и отдаёт мини‑батчи с приоритетами
- Выбор действия: \(\epsilon\)-жадная стратегия

## Теория

### 1) Уравнение оптимальности Беллмана

Оптимальная Q‑функция удовлетворяет:

$$
Q^*(s,a) = \mathbb{E}\Big[r + \gamma \max_{a'} Q^*(s', a')\,\Big|\, s,a\Big]
$$

Стохастический градиентный спуск по MSE‑приближению решает:

$$
\min_\theta \;\mathbb{E}_{(s,a,r,s')}\big[\big(y - Q_\theta(s,a)\big)^2\big],\quad
\text{где } y = r + \gamma\, \max_{a'} Q_{\theta^-}(s', a')
$$

### 2) Double DQN против классического DQN

- Классический DQN завышает оценки из‑за \(\max\) по одной и той же сети. Double DQN разносит выбор и оценку:

$$
 y = r + \gamma\, Q_{\theta^-}\Big(s', \operatorname*{argmax}\limits_{a'} Q_{\theta}(s', a')\Big)
$$

Это снижает переоценку и стабилизирует обучение.

### 3) Целевая сеть и её обновление

- Целевая сеть \(Q_{\theta^-}\) копируется с онлайн‑сети каждые `target_update_iter` шагов:

$$
\theta^- \leftarrow \theta \quad \text{(периодически)}
$$

Фиксированный таргет на коротком интервале уменьшает разрушение целевой функции.

### 4) Приоритезированный опыт (PER) и SumTree

- Приоритет перехода i:

$$
 p_i = |\delta_i| + \varepsilon_{\text{margin}} \quad \text{(далее может быть отсечён сверху: } p_i \le \text{abs\_error\_upper)}
$$

- Вероятность выборки:

$$
 P(i) = \frac{p_i^{\alpha}}{\sum_j p_j^{\alpha}}, \quad \alpha \in [0,1]
$$

- Веса важности (importance sampling) для коррекции смещения:

$$
 w_i = \Big( \frac{1}{N\, P(i)} \Big)^{\beta}, \quad \tilde{w}_i = \frac{w_i}{\max_j w_j}, \quad \beta \nearrow 1
$$

- Обновление приоритета после шага обучения: \(p_i \leftarrow |\delta_i| + \varepsilon_{\text{margin}}\)

- Структура SumTree обеспечивает \(\mathcal{O}(\log N)\) обновления/выборку по приоритетам.

### 5) \(\epsilon\)-жадная стратегия и её расписание

- С вероятностью \(\epsilon\) выбирается случайное действие, иначе \(\arg\max_a Q_\theta(s,a)\).
- Эксплорейшн уменьшается: \(\epsilon \leftarrow \max(\text{min\_epsilon}, \epsilon \cdot \text{epsilon\_decay})\).

### 6) Общий цикл обучения

1. Сбор опыта в буфер (первыe K шагов — без обучения).
2. Каждые `replay_period` шагов: выборка мини‑батча из PER, вычисление \(y\), TD‑ошибок \(\delta\), IS‑весов и обновление \(\theta\) по взвешенной MSE.
3. Обновление приоритетов \(p_i\) и параметра \(\beta\).
4. Каждые `target_update_iter` шагов: \(\theta^- \leftarrow \theta\).

Псевдокод:

```text
predict_q = Q_theta(s_batch)
best_action = argmax_a predict_q
target_q = Q_theta_minus(s_next_batch)
y = r_batch + gamma * target_q[range, best_action]

# TD-ошибка и приоритеты
delta = y - predict_q[range, a_batch]
priority = clip(|delta| + margin, 0, abs_error_upper) ** alpha

# веса важности и взвешенная MSE
w = ((buffer_size * P(i)) ** -beta) / max_w
loss = mean(w * (y - Q_theta(s_batch, a_batch))^2)
update theta by SGD

# обновить приоритеты, увеличить beta, периодически обновить target
```

### 7) Стабилизационные приёмы

- Нормализация/клиппинг градиентов
- Ограничение TD‑ошибки сверху (как в коде — `abs_error_upper`)
- Выбор Huber‑потерь вместо MSE (в нашей версии — взвешенная MSE)
- Регулярное обновление target‑сети, достаточный размер буфера

### 8) Соответствие параметрам реализации

- `alpha` — степень приоритезации (0 → равновероятно, 1 → строго по ошибке)
- `beta`, `beta_increment_per_sample` — сила IS‑весов и её рост
- `target_update_iter` — период синхронизации целевой сети
- `replay_period` — как часто тренируемся
- `epsilon`, `epsilon_decay`, `min_epsilon` — расписание \(\epsilon\)
- `margin` (`\varepsilon` в формулах), `abs_error_upper` — формирование приоритетов

## Быстрый старт

```python
import gymnasium as gym
import numpy as np

from tensoraerospace.agent.dqn.model import Model, DQNAgent

env = gym.make('LinearLongitudinalF16-v0', number_time_steps=2000)
num_actions = env.action_space.n

model = Model(num_actions)
target_model = Model(num_actions)

agent = DQNAgent(
    model=model,
    target_model=target_model,
    env=env,
    train_nums=10000,
    epsilon=1.0,
    epsilon_dacay=0.995,
    min_epsilon=0.05,
)

agent.train()
```

!!! tip
    Для непрерывного действия используйте дискретизацию или политику на базе DDPG/SAC.

## Документация API

::: tensoraerospace.agent.dqn.model.Model

::: tensoraerospace.agent.dqn.model.SumTree

::: tensoraerospace.agent.dqn.model.DQNAgent
