---
hide:
  - navigation
  - toc
---

<div class="hero">
  <h1>TensorAeroSpace</h1>
  <p class="tagline">Реалистичные аэрокосмические среды и алгоритмы RL для обучения систем управления</p>
  <p>
    <a href="guide/installation/" class="md-button md-button--primary">Установка</a>
    <a href="lesson/0intro/" class="md-button">Учебные уроки</a>
    <a href="agent/sac/" class="md-button">Алгоритмы</a>
    <a href="model/f16/" class="md-button">Модели</a>
  </p>
  <p>
    <a href="https://github.com/TensorAeroSpace/TensorAeroSpace"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-TensorAeroSpace-000?logo=github"></a>
    <a href="https://pypi.org/project/tensoraerospace/"><img alt="PyPI" src="https://img.shields.io/pypi/v/tensoraerospace?color=3775A9&logo=pypi&label=PyPI"></a>
    <a href="https://huggingface.co/TensorAeroSpace"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TensorAeroSpace-FFD21E"></a>
    <a href="https://pypi.org/project/tensoraerospace/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/tensoraerospace?logo=python&label=Python"></a>
    <a href="https://pypi.org/project/tensoraerospace/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/tensoraerospace?label=Downloads"></a>
    <a href="https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
    <a href="https://deepwiki.com/TensorAeroSpace/TensorAeroSpace"><img alt="Ask DeepWiki" src="https://deepwiki.com/badge.svg"></a>
  </p>
</div>

<style>
.hero {
  text-align: center;
  margin: 2rem 0 2.5rem 0;
  padding: 2.2rem 1rem;
  background: linear-gradient(120deg, rgba(59,130,246,.18), rgba(59,130,246,0) 50%),
              radial-gradient(60rem 60rem at 10% -20%, rgba(59,130,246,.25), transparent 40%),
              radial-gradient(50rem 50rem at 90% 120%, rgba(59,130,246,.18), transparent 40%),
              linear-gradient(135deg, rgba(59,130,246,.08), rgba(59,130,246,0));
  background-size: 200% 200%, auto, auto, auto;
  animation: gradientShift 12s ease-in-out infinite alternate;
  border-radius: 16px;
}
@keyframes gradientShift {
  0% { background-position: 0% 50%, 0 0, 0 0, 0 0; }
  100% { background-position: 100% 50%, 0 0, 0 0, 0 0; }
}
.hero .tagline {
  font-size: 1.08rem;
  color: var(--md-default-fg-color--light);
  margin-top: .3rem;
}
.hero .md-button { margin: .25rem .25rem; }
.hero a img { vertical-align: middle; margin: 0 .22rem; }
.cards .card-icon { font-size: 1.6rem; }
.stats { text-align: center; margin: 1.5rem 0 0; color: var(--md-default-fg-color--light); }
.logos { display: flex; gap: 1.2rem; align-items: center; justify-content: center; flex-wrap: wrap; margin: 1rem 0; }
.logos img { height: 42px; opacity: .9; filter: saturate(0) contrast(1.1); }
</style>

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Быстрый старт**

    Установите библиотеку, выберите модель и запустите первого агента.

    [:octicons-arrow-right-24: Установка](guide/installation/)

-   :material-robot-outline: **Алгоритмы RL**

    Современные алгоритмы: DQN, A3C/A2C‑NARX, PPO, SAC, DDPG, GAIL.

    [:octicons-arrow-right-24: Смотреть](agent/sac/)

-   :material-airplane-takeoff: **Модели объектов**

    F‑16, Boeing‑747, X‑15, спутники и ракеты с готовыми средами.

    [:octicons-arrow-right-24: Перейти](model/f16/)

-   :material-cog-outline: **Интеграция с Gym**

    Совместимые environments и простой API для обучения и оценки.

    [:octicons-arrow-right-24: Подробнее](example/enviroment/gymnasium/)

-   :material-school-outline: **Учебные уроки**

    Практика по XFLR5, Simulink, SimInTech и теории управления.

    [:octicons-arrow-right-24: К урокам](lesson/0intro/)

-   :material-chart-line: **Бенчмаркинг**

    Метрики, сравнение агентов и примеры экспериментов.

    [:octicons-arrow-right-24: Метрики](benchmark/metrics/)

</div>

---

## Основные преимущества

<div class="grid cards" markdown>

-   :material-speedometer: **Производительность**

    Лёгкие среды и быстрые эксперименты — меньше кода, больше результатов.

-   :material-brain: **Современный RL стек**

    DDPG, SAC, PPO, GAIL и др. с удобным API и примерами.

-   :material-cube-outline: **Физически корректные модели**

    Линейные модели продольной динамики, ракеты, самолёты, спутники.

-   :material-puzzle-outline: **Интеграции**

    Gymnasium, Simulink/Matlab, SimInTech — готовые связки.

-   :material-book-open-variant: **Понятная документация**

    Пошаговые уроки, рецепты, best‑practices и разборы типичных задач.

-   :material-chart-areaspline: **Бенчмаркинг**

    Метрики, сравнения и воспроизводимые эксперименты.

</div>

## Обзор функционала

=== "Агенты"

    - IHDP, DQN, A3C/A2C‑NARX, PPO, SAC, DDPG, GAIL
    - Буферы опыта, шум OUNoise, политики стохастические/детерминированные
    - GAE, PPO‑update, дискриминатор GAIL

=== "Модели"

    - F‑16, B747, X‑15, типичная ракета, спутники
    - Матрицы состояния, линейные/линеаризованные модели
    - Примеры с обучением контроллеров

=== "Документация"

    - Пошаговые уроки по XFLR5/Simulink/SimInTech
    - Руководства и примеры интеграции
    - Ссылки на примеры и бенчмарки

---

## Установка

=== "pip"

    ```bash
    pip install tensoraerospace
    ```

=== "poetry"

    ```bash
    poetry add tensoraerospace
    ```

=== "conda"

    ```bash
    conda create -n tas python=3.10
    conda activate tas
    pip install tensoraerospace
    ```

## Пример запуска

```python
import gymnasium as gym
import numpy as np

from tensoraerospace.agent.pid import PID
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standart import unit_step

# Параметры симуляции
dt = 0.01
tp = generate_time_period(tn=10, dt=dt)  # 10 секунд
N = len(tp)

# Опорный сигнал для слежения по alpha (ступенька 5° в радианах)
reference = unit_step(degree=5, tp=tp, time_step=100, output_rad=True).reshape(1, -1)

# Создание среды F-16 (порядок состояний: [alpha, q])
env = gym.make(
    'LinearLongitudinalF16-v0',
    number_time_steps=N,
    initial_state=[[0], [0]],
    reference_signal=reference,
    use_reward=False,
)

# ПИД-контроллер (коэффициенты из примера PID)
pid = PID(env, kp=-14.290139135229715, ki=-8.240470780203491, kd=-1.2991634935096958, dt=dt)

obs, info = env.reset()
for t in range(N - 1):
    setpoint = reference[0, t]
    alpha = float(obs[0])  # env возвращает [alpha, q]
    u = pid.select_action(setpoint, alpha)
    action = np.array([[float(u)]], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Почему TensorAeroSpace?

- Реалистичные аэродинамические модели и матрицы состояния
- Интеграция с MATLAB/Simulink и SimInTech
- Готовые environments и шаблоны обучения контроллеров

## Полезные ссылки

- Руководство: [guide/installation](guide/installation/)
- Уроки: [lesson/0intro](lesson/0intro/)
- Модели: [model/f16](model/f16/), [model/b747](model/b747/)
- Алгоритмы: [agent/sac](agent/sac/), [agent/ppo](agent/ppo/), [agent/ddpg](agent/ddpg/)
- Примеры: [example/enviroment/gymnasium](example/enviroment/gymnasium/)

---

Нужна помощь? Откройте issue в GitHub или загляните в раздел уроков.

---

## Немного цифр

<div class="grid cards" markdown>

-   :material-brain: **RL‑алгоритмы**

    8+ реализованных методов: IHDP, DQN, A3C/A2C‑NARX, PPO, SAC, DDPG, GAIL

-   :material-airplane: **Аэрокосмические модели**

    10+ моделей: F‑16, B747, X‑15, ракеты и спутники

-   :material-python: **Поддержка Python**

    3.9 — 3.12, совместимость с ecosystem Gymnasium

-   :material-license: **Лицензия**

    MIT — свободно для науки и индустрии

</div>

## Где используют

<div class="logos">
  <img src="logo.png" alt="TensorAeroSpace">
  <span>… и ещё исследовательские группы и энтузиасты</span>
  
</div>

<div style="text-align:center; margin: 1.2rem 0 0.2rem;">
  <a href="guide/installation/" class="md-button md-button--primary">Начать сейчас</a>
  <a href="example/enviroment/gymnasium/" class="md-button">Посмотреть примеры</a>
</div>
