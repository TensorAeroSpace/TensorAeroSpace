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

- Политика: `Actor.forward(...)` выводит `mu = tanh(Wx)` и `log_std = tanh(Wx)` с последующим линеарным растяжением в диапазон `[log_std_min, log_std_max]`; \(\sigma = e^{\log \sigma}\). Действие семплируется из `Normal(mu, sigma)`.
- Отношения вероятностей: берутся через разность лог‑плотностей `new_probs - old_probs`, затем экспонента (`torch.exp`) — это численно устойчивее, чем делить плотности напрямую.
- Энтропия: в коде в `actor_loss` подаётся отрицательная энтропия `-new_distr.entropy().mean()`, а затем добавляется как `+ entropy_coef * entropy`. Эффект равнозначен вычитанию энтропии с коэффициентом (стимулируется стохастичность политики).
- GAE и бустрап: в `preprocess1` добавляется `next_value` в `values`, затем по реверсу считается \(\delta\) и аккумулируется \(g\) с \(\lambda=0.8\); в итоге `returns = V + g`, `advantages = returns - V`.
- Мини‑батчи: итератор `ppo_iter` случайно выбирает индексы размера `mini_batch_size` многократно в течение `epoch`.
- Предсказание награды: при `auxiliary_coef > 0` метод `actor.predict_reward(states)` использует `self.r`; его MSE автоматически добавляется к ошибке актора.

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

# Сохраняем модель и освобождаем ресурсы исходного агента
checkpoint_dir = agent.save('./runs')
agent.close()
env.close()

# Загружаем каталог, возвращённый save()
agent = PPO.from_pretrained(str(checkpoint_dir))
```

!!! tip
    Для непрерывных действий используем гауссовскую политику; полезно ограничивать `log_std` (как в коде) и нормировать признаки.

## Практические советы

- Увеличивайте `rollout_len` для более стабильной оценки преимуществ
- Балансируйте `clip_pram` (обычно 0.1–0.3) и `entropy_coef` для исследовательности
- Несколько эпох (`num_epochs`) и мелкие `batch_size` улучшают сходимость, но следите за переобучением

## Воспроизводимость и завершение работы

`seed` определяет начальные веса актора и критика, включая включённый выход
предсказания награды. Для сравнения запусков обучения отдельно задавайте seed среды.

Синхронные (`save_best_async=False`) и фоновые сохранения лучшей модели записывают
веса сетей, состояния оптимизаторов Adam, лучшую награду и включённую статистику
нормализации. Загружайте каталог через `PPO.from_pretrained(...)`, чтобы продолжить
обучение с этими состояниями.

После работы вызывайте `agent.close()`: метод завершает запись checkpoint и
закрывает TensorBoard/W&B. Повторный вызов допустим. Среду закрывайте отдельно
через `agent.env.close()`.

## Вспомогательные задачи (Auxiliary Tasks) {#auxiliary-tasks}

Задайте `auxiliary_coef > 0` при создании `PPO`, чтобы включить предсказание
непосредственной награды. Дополнительный линейный выход `actor.r` использует
два общих скрытых слоя актора и предсказывает одну награду на наблюдение.
Вспомогательная ошибка обучает как этот выход, так и общие слои:

```text
auxiliary_loss = mean((predicted_reward - immediate_reward) ** 2)
actor_objective = actor_loss + auxiliary_coef * auxiliary_loss
```

`learn()` автоматически добавляет этот член, в том числе при вызове из `train()`
для обычной или векторной среды. Цели — непосредственные награды из собранных
переходов, а не дисконтированные возвраты; `normalize_reward` нормирует только
возвраты. По умолчанию `auxiliary_coef=0.0`: параметры дополнительного выхода
не создаются, старые checkpoint продолжают загружаться. Коэффициент и веса
выхода сохраняются и восстанавливаются через `save()`, лучшие checkpoint
и `from_pretrained()`. `save()` и асинхронные лучшие checkpoint также сохраняют
состояние оптимизатора.

### Использование

```python
agent = PPO(env=env, auxiliary_coef=0.1)
agent.train(num_episodes=2, max_steps=128)
```

`learn()` возвращает MSE без коэффициента в `metrics["auxiliary_loss"]`.
В TensorBoard/WandB она записывается под тегом `loss/auxiliary`.
Метрика `actor_loss` по-прежнему показывает ошибку политики без вспомогательного
слагаемого.

Для собственного цикла обучения `agent.auxiliary_task(states, rewards)`
возвращает дифференцируемую скалярную ошибку без шага оптимизатора.
`states` — непустой тензор `(batch_size, obs_dim)` с той же предобработкой,
что у входов политики; `rewards` — тензор `(batch_size,)` или `(batch_size, 1)`.
Награды отделяются от графа градиентов; оба тензора переносятся на устройство
агента. Ранее описанное написание `auxillary_task` сохранено как alias.
Если агент создан с `auxiliary_coef=0`, оба метода выдают понятную ошибку.

`Actor.forward(states)` возвращает прежнюю пару `(action, distribution)`.
Для предсказания наград используйте `actor.predict_reward(states)` с тензором
на устройстве актора; этот метод не семплирует действия.

### Пример на F-16

Из корня репозитория:

```bash
poetry run python example/reinforcement_learning/deep_rl/example_ppo_auxiliary_tasks.py \
  --episodes 2 --steps 128 --log-dir runs/ppo_auxiliary_f16
```

Скрипт обучает PPO на линейной F-16, выводит изменение весов предсказателя
награды и сохраняет checkpoint. Это демонстрация работы задачи, а не
утверждение о сходимости регулятора за два коротких запуска. Коэффициент
зависит от масштаба наград; предсказание награды не гарантирует улучшения
качества управления.

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

## Границы эпизодов и начальное исследование

При ограничении времени PPO учитывает стоимость конечного наблюдения, но обрывает рекурсию GAE на границе эпизода. Векторная среда с автосбросом должна передавать `final_observation` и необязательную маску `_final_observation`; без них нельзя использовать наблюдение нового эпизода для этой оценки. Непосредственная награда для метрик и auxiliary prediction сохраняется. Нормализация наблюдений согласована при векторном сборе, обновлении и применении политики.

Новые политики начинают с логарифмом стандартного отклонения около `-0.5` внутри заданных границ. Это устраняет почти детерминированный старт в середине диапазона `[-20, 0]`. Имена параметров и преобразование выходов сохранены для старых checkpoint. Выборка из одного перехода использует дисперсию с делением на N, исключая NaN выборочного стандартного отклонения.
