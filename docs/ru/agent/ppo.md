# Proximal Policy Optimization (PPO)

PPO — надёжный policy‑gradient метод, сочетающий простоту реализации и стабильность обучения. В нашей реализации актор и критик обучаются на батчах собранных роллаутов, используется клиппированный суррогат, энтропия политики и оценка преимуществ с обобщённой ошибкой (GAE‑подобная).

![PPO схема](../agent/img/ppo.png){ width=800 }

## Компоненты

- Актор (гауссовская политика): параметры \(\mu, \sigma\) → распределение \(\mathcal{N}(\mu, \sigma^2)\)
- Критик: скалярная оценка \(V(s)\)
- Сбор опыта: роллаут длины `rollout_len` с записью \(s,a,\log\pi(a|s), r, d, V(s)\)
- Обучение: мини‑батчи по несколько эпох `num_epochs` с клиппингом вероятностных отношений

## Теория

- Отношение вероятностей:

$$
 r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} = \exp\big(\log \pi_\theta - \log \pi_{\theta_{\text{old}}}\big)
$$

- Клиппированный суррогат (Actor):

$$
\mathcal{L}_\text{actor} = -\,\mathbb{E}\Big[\min\big( r_t\,A_t,\ \mathrm{clip}(r_t,\ 1-\varepsilon,\ 1+\varepsilon)\,A_t \big) \Big]
$$

- Потеря критика (Value):

$$
\mathcal{L}_\text{critic} = \mathbb{E}\big[ (R_t - V_\phi(s_t))^2 \big]
$$

- Энтропийная регуляризация (стохастичность политики):

$$
\mathcal{L}_\text{entropy} = -\beta\,\mathbb{E}\big[\mathcal{H}[\pi_\theta(\cdot|s_t)]\big]
$$

- Полная цель: \(\mathcal{L} = \mathcal{L}_\text{actor} + \mathcal{L}_\text{critic} + \mathcal{L}_\text{entropy}\)

- Преимущество (GAE‑подобное): в `preprocess1` возвращается \(\text{return} = V + \sum\gamma\lambda\,\delta\), а \(A = \text{return} - V\)

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),\quad
\hat{A}_t \approx \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l}
$$

### Детали реализации

- Политика: `Actor.forward(..., continous_actions=True)` выводит `mu = tanh(Wx)` и `log_std = tanh(Wx)` с последующим линеарным растяжением в диапазон `[log_std_min, log_std_max]`; \(\sigma = e^{\log \sigma}\). Действие семплируется из `Normal(mu, sigma)`.
- Отношения вероятностей: берутся через разность лог‑плотностей `new_probs - old_probs`, затем экспонента (`torch.exp`) — это численно устойчивее, чем делить плотности напрямую.
- Энтропия: в коде в `actor_loss` подаётся отрицательная энтропия `-new_distr.entropy().mean()`, а затем добавляется как `+ entropy_coef * entropy`. Эффект равнозначен вычитанию энтропии с коэффициентом (стимулируется стохастичность политики).
- GAE и бустрап: в `preprocess1` добавляется `next_value` в `values`, затем по реверсу считается \(\delta\) и аккумулируется \(g\) с \(\lambda=0.8\); в итоге `returns = V + g`, `advantages = returns - V`.
- Мини‑батчи: итератор `ppo_iter` случайно выбирает индексы размера `mini_batch_size` многократно в течение `epoch`.
- Доп. голова r: актор возвращает ещё и предсказание наград (линия `self.r`), для вспомогательной задачи `auxillary_task` (MSE по наградам); по умолчанию в loss не добавляется.

### Псевдокод обучения

```text
for episode in range(max_episodes):
  rollout = collect(rollout_len)
  next_value = V(s_T)
  returns, advantages = GAE(rollout.rewards, rollout.values, dones, gamma, lambda)
  for epoch in range(num_epochs):
    for batch in mini_batches(rollout, returns, advantages):
      ratios = exp(new_logp - old_logp)
      a_loss = -mean(min(ratios*A, clip(ratios)*A)) + entropy_coef * (-entropy)
      c_loss = mse(returns - V(s))
      update(actor, critic)
  log TensorBoard metrics
```

### Гиперпараметры и соответствие коду

- `clip_pram = ε` — порог клиппинга вероятностных отношений
- `num_epochs`, `batch_size` — количество проходов и размер мини‑батча для обновлений
- `rollout_len` — длина роллаута перед обновлениями
- `entropy_coef` — вес энтропийного члена (учитывая знак в реализации)
- `actor_lr`, `critic_lr` — скорости обучения оптимизаторов Adam
- `gamma`, `lambda(=0.8)` — скидка и параметр GAE внутри `preprocess1`

## Быстрый старт

```python
import gymnasium as gym
from tensoraerospace.agent.ppo.model import PPO

# Создаём среду (пример — F16)
env = gym.make('LinearLongitudinalF16-v0', number_time_steps=2000)

# Инициализация PPO
agent = PPO(
    env=env,
    gamma=0.99,
    max_episodes=50,
    rollout_len=2048,
    clip_pram=0.2,
    num_epochs=64,
    batch_size=64,
    entropy_coef=0.005,
    actor_lr=1e-3,
    critic_lr=5e-3,
)

# Обучение
agent.train()

# Сохранение
agent.save('./runs')
```

!!! tip
    Для непрерывных действий используем гауссовскую политику; полезно ограничивать `log_std` (как в коде) и нормировать признаки.

## Практические советы

- Увеличивайте `rollout_len` для более стабильной оценки преимуществ
- Балансируйте `clip_pram` (обычно 0.1–0.3) и `entropy_coef` для исследовательности
- Несколько эпох (`num_epochs`) и мелкие `batch_size` улучшают сходимость, но следите за переобучением

## Вспомогательные задачи (Auxiliary Tasks) {#auxiliary-tasks}

Реализация PPO включает опциональный механизм **вспомогательных задач** (auxiliary tasks) для предсказания награды. Этот дополнительный выход помогает агенту формировать более качественные представления состояний, предсказывая ожидаемые награды параллельно с основной оптимизацией политики.

### Как это работает

- Сеть `Actor` включает дополнительный выходной слой `self.r`, который предсказывает награду
- Вспомогательная ошибка вычисляется как MSE между предсказанной и фактической наградой
- Эта ошибка может быть добавлена к основной ошибке PPO через метод `auxillary_task` в классе `Agent`

### Использование

```python
# Вспомогательная задача вычисляется отдельно от основного обучения
aux_loss = agent.auxillary_task(states, rewards)
```

Вспомогательная задача стимулирует сеть кодировать признаки, релевантные для награды, в скрытых представлениях, что потенциально улучшает эффективность использования выборки и обобщающую способность.

## Унифицированный интерфейс обучения

PPO следует общему унифицированному API `train()` из `BaseRLModel`:

```python
stats = agent.train(
    num_episodes=200,   # необязательно: переопределяет self.max_episodes
    max_steps=1024,     # необязательно: переопределяет self.rollout_len
)
```

Вызов `agent.train()` без аргументов также поддерживается — в этом случае
используются гиперпараметры, заданные при создании. Обратите внимание,
что метод PPO `learn(states, actions, adv, old_probs, returns, rewards, old_values)`
является внутренним помощником, выполняющим один шаг градиентного
обновления по батчу, и не затрагивается унифицированным интерфейсом.

## Документация API

::: tensoraerospace.agent.ppo.model.PPO

::: tensoraerospace.agent.ppo.model.Actor

::: tensoraerospace.agent.ppo.model.Critic

::: tensoraerospace.agent.ppo.model.ppo_iter

## Источники

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

## Где тестировалось

- Unity‑среда
