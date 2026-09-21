# Deep Deterministic Policy Gradient (DDPG)

DDPG — off‑policy актор‑критик для непрерывных действий: обучает детерминированную стратегию и Q‑функцию, используя буфер повторов и целевые сети со «мягким» обновлением.

## Компоненты

- Политика (Actor): `PolicyNetwork(s) -> a`, детерминированное действие через `tanh`
- Критик (Q‑сеть): `ValueNetwork(s,a) -> Q(s,a)`
- Целевые сети: `target_policy_net`, `target_value_net` для стабильности
- Реплей‑буфер: `ReplayBuffer` для выборки мини‑батчей
- Эксплорейшн: орнштейн–уленбековский шум `OUNoise`

## Теория (на базе реализации)

- Градиент политики (DPG):

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s\sim \mathcal{D}}\Big[\nabla_a Q(s,a)\big|_{a=\pi_\theta(s)}\, \nabla_\theta \pi_\theta(s)\Big]
$$

В коде минимизируется \(-Q(s,\pi(s))\), что эквивалентно градиентному подъёму по \(J\).

- Обновление критика (таргет Беллмана с целевыми сетями):

$$
\hat{Q}(s,a) = r + \gamma\,(1-\text{done})\, Q_{\text{target}}(s', \pi_{\text{target}}(s'))
$$

Лосс критика — MSE: \(\mathcal{L}_Q = (Q(s,a) - \hat{Q})^2\).

- Мягкое обновление целевых сетей:

$$
\theta^- \leftarrow (1-\tau)\,\theta^- + \tau\,\theta
$$

## Быстрый старт

```python
import gymnasium as gym
import numpy as np
from tensoraerospace.agent.ddpg.model import DDPG
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standard import unit_step

# Временная сетка и референс
dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)
reference_signals = unit_step(degree=5, tp=tp, time_step=1000, output_rad=True).reshape(1, -1)

# Среда F‑16
env = gym.make('LinearLongitudinalF16-v0',
               number_time_steps=number_time_steps,
               initial_state=[[0],[0],[0]],
               reference_signal=reference_signals,
               use_reward=True,
               state_space=["theta","alpha","q"],
               output_space=["theta","alpha","q"],
               control_space=["ele"],
               tracking_states=["alpha"],)

agent = DDPG(env, value_lr=1e-3, policy_lr=1e-4, replay_buffer_size=1_000_000)
agent.learn(max_frames=12000, max_steps=500, batch_size=128)
```

!!! tip
    Эксплорейшн обеспечивается OU‑шумом: контролируйте `sigma` и `decay_period`, чтобы плавно снижать силу шума.

## Унифицированный интерфейс обучения

DDPG поддерживает общий унифицированный API `train()` из `BaseRLModel`:

```python
agent.train(
    num_episodes=24,
    max_steps=500,
    batch_size=128,
    warmup_frames=2_000,
)
```

Под капотом `train()` пересчитывает `num_episodes * max_steps` в бюджет
`max_frames` и вызывает устаревший метод `learn()`. Поддерживаемые
DDPG‑специфичные именованные аргументы (передаются через `**kwargs`):

- `max_frames`, `batch_size`, `gamma`, `soft_tau`, `warmup_frames`,
  `updates_per_step`, `target_value_clip`.

Старый вызов `agent.learn(max_frames=..., max_steps=..., batch_size=...)`
продолжает работать без изменений.

## Наблюдения и совместимость checkpoint

DDPG сохраняет в replay независимые копии **исходных наблюдений**. При выборке
обе стороны перехода нормализуются текущими средним и дисперсией. Статистика
обновляется по снимкам до `env.step()`: общий массив среды не изменяет прошлые
состояния. При ручном заполнении replay передавайте ненормализованные наблюдения.

Ограничение времени сохраняет bootstrap; настоящее терминальное состояние
отключает его. При автоматическом сбросе конечное состояние берётся из
`final_observation` или `terminal_observation`.

Файловые checkpoint помечают исходные наблюдения полем
`replay_observation_format="raw_v1"`. При загрузке восстанавливаются режим и
статистика нормализации. Старый нормализованный replay нельзя достоверно
восстановить: переходы могли использовать разные статистики. Он пропускается
с предупреждением, обучение начинается с пустого replay; веса сетей загружаются.
Старый replay без нормализации совместим. `load_replay=False` явно отключает
загрузку replay.

Исправление переходов не гарантирует сходимость политики или сохранение кривой
обучения при прежних гиперпараметрах.

При `min_sigma < max_sigma` шум OU уменьшается по общему счётчику шагов
внутри `learn()`, без перезапуска расписания на границах эпизодов. Масштаб
текущего шага применяется до генерации случайного возмущения.

## Применение обученной политики

Передавайте исходные наблюдения в `agent.predict(observation)`. Метод применяет
сохранённую нормализацию и возвращает детерминированные действия для одного
наблюдения или пакета. Статистика не обновляется, шум не добавляется. Прямой
вызов `policy_net.get_action()` обходит нормализацию; передача в него исходных
наблюдений может изменить поведение обученной политики.

## Проверенный эксперимент слежения B747

В ранее выполненном регрессионном эксперименте слежения линейного B747 снижение LR критика до
`1e-4` и уменьшение sigma шума OU с `0.3` до `0.05` за 15 000 шагов снизили
среднюю финальную RMSE после 60 000 шагов с 5,13° до 1,14° на seed 11, 29 и 47.
LR актора остался `1e-4`, нормализация наблюдений включена. В финальных оценках
нарушений границ тангажа нет. Контрольный PD дал 0,954°: превосходство над
классическим контроллером и сходимость на других задачах этим не подтверждены.
Короткие запуски и промежуточные оценки остаются неоднородными. Одно отключение
нормализации проблему не устранило.

Полный процесс обучения через SDK приведён в [ноутбуке DDPG/B747](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/deep_rl/example_ddpg_b747_improved.ipynb). Исторические метрики выше относятся к регрессионному эксперименту и не гарантируют такой же результат каждого запуска ноутбука.

## Документация API

::: tensoraerospace.agent.ddpg.model.DDPG

## Источники

- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

## Где тестировалось

- Unity‑среда
- LinearLongitudinalF16‑v0 (пример в репозитории)
