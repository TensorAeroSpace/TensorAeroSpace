# A2C с NARX‑Critic

A2C (Advantage Actor‑Critic) использует актёра для выбора действий и критика для оценки состояний. В нашей реализации критик — NARX (Nonlinear AutoRegressive with eXogenous inputs), что позволяет лучше моделировать динамику и историю за счёт явного учёта прошедших состояний.

![A2C-NARX схема](../agent/img/a2c_narx.png){ width=800 }

## Компоненты

- Актор: гауссовская политика \(\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2)\); параметры — `Actor` (PyTorch)
- Критик (NARX): оценка \(V(s)\) на основе расширенного входа (текущее состояние + предыдущие сигналы); классы `Critic` (A2C) и `NARX` (модульная NARX‑сеть)
- Сбор опыта: `Runner` собирает траектории, клиппирует действия под `action_space`
- Обучение: `A2CLearner.learn` — обновления актёра/критика со стабилизацией (клиппинг градиента, энтропия)

## Целевые значения и история

Критик получает `z_t = [s_t, s_(t-1)]`, а оценка следующего состояния
использует `[s_(t+1), s_t]`. История сохраняется между вызовами `Runner.run`
и обнуляется при новом эпизоде. Наблюдения копируются при сборе опыта.

При `discount_rewards=False` используется одношаговая цель:

$$
y_t = r_t + \gamma (1-\mathrm{terminated}_t) V_\phi(z_{t+1}),
\qquad A_t = y_t - V_\phi(z_t).
$$

При `discount_rewards=True` возвраты накапливаются до конца эпизода или
собранного фрагмента. При настоящем завершении продолжение равно нулю;
при лимите времени или незавершённом фрагменте добавляется оценка критика
для последнего наблюдения. Награды следующего эпизода не попадают в возврат.
Цели вычисляются без градиента. Потери критика — MSE, актора —
`-mean(log_prob * advantage) - entropy_beta * entropy`.
Для многомерного действия логарифмы вероятностей суммируются по компонентам.

`Runner` возвращает `NARXTransition`: распаковка остаётся
`(action, reward, state, next_state, done)`, дополнительно сохраняются
`terminated` и `previous_state`. Старые кортежи из пяти элементов принимаются,
но их `done=True` трактуется как настоящее завершение, а история первого
перехода заполняется нулями. Для корректной обработки лимита времени
используйте `Runner` или явно создавайте `NARXTransition`.

## Быстрый старт

```python
import gymnasium as gym
import torch
from tensoraerospace.agent.a2c.narx import Actor, Critic, A2CLearner, Runner

env = gym.make('LinearLongitudinalF16-v0', number_time_steps=2000)
actor = Actor(state_dim=env.observation_space.shape[0], n_actions=env.action_space.shape[0])
critic = Critic(state_dim=env.observation_space.shape[0])
learner = A2CLearner(actor, critic, gamma=0.99, entropy_beta=0.01)
runner = Runner(env, actor, learner.writer)

memory = runner.run(max_steps=2048)
learner.learn(memory, steps=2048, discount_rewards=True)
```

!!! tip
    Для систем с сильной инерцией используйте `discount_rewards=False`, чтобы критик обучался по TD‑таргету с \(V(s')\).

## Документация API

::: tensoraerospace.agent.a2c.narx.A2CLearner

::: tensoraerospace.agent.a2c.narx.Runner

<!-- ::: tensoraerospace.agent.narx.model.NARX -->
## Основной агент A2C и история команд

`agent.a2c.model.A2C` и `A2CWithNARXCritic` используют `run_episode()` и `learn()`.
Для них действуют описанные выше правила целей при завершении и лимите времени.
`RolloutTransition` сохраняет распаковку пяти значений; действие в кортеже —
исходная выборка Gaussian-политики. Отдельное поле `executed_action` содержит
ограниченную команду, переданную объекту. Вероятность политики вычисляется
для выборки, а NARX-критик использует исполненные команды.

При длине истории `h` вход `A2CWithNARXCritic` имеет вид
`[s_t, ..., s_(t-h+1), u_(t-1), ..., u_(t-h)]`.
Следующий вход сдвигает обе истории, добавляя текущее наблюдение и исполненную
команду. История сохраняется между фрагментами, сбрасывается между эпизодами
и копируется в каждый переход. В старых кортежах этих метаданных нет, поэтому
первая строка использует нулевую историю. Обновление по одному переходу
сохраняет преимущество: оно не обнуляется центрированием и не создаёт NaN
при вычислении стандартного отклонения.
