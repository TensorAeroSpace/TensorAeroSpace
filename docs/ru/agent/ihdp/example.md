# Пример: управление F‑16 с IHDP (LinearLongitudinalF16‑v0)

Ниже — понятный пошаговый пример на базе `example_ihdp_beautiful.ipynb`: от создания опорного сигнала до запуска IHDP и интерпретации графиков. В конце приведены ответы на частые вопросы.

— [Шаг 1. Время и опорный сигнал](#шаг-1-время-симуляции-и-опорный-сигнал)

— [Шаг 2. Среда F‑16](#шаг-2-инициализация-среды-f16)

— [Шаг 3. Параметры IHDP](#шаг-3-параметры-actor--critic--incremental-model)

— [Шаг 4. Создание агента](#шаг-4-создание-агента-ihdp)

— [Шаг 5. Симуляция](#шаг-5-основной-цикл-симуляции)

— [Шаг 6. Графики и ожидания](#шаг-6-визуализация-результатов)

— [FAQ и советы](#частые-вопросы-и-советы)

## Шаг 1. Время симуляции и опорный сигнал {#шаг-1-время-симуляции-и-опорный-сигнал}

Задаём дискретизацию и горизонт моделирования. Формируем ступенчатый опорный сигнал по углу атаки `alpha`: это цель, к которой агент будет стремиться, минимизируя ошибку слежения.

<!-- markdownlint-disable MD046 -->
```python
import numpy as np
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.signals.standart import unit_step

dt = 0.01  # дискретизация, с
tp = generate_time_period(tn=20, dt=dt)  # массив шагов времени
tps = convert_tp_to_sec_tp(tp, dt=dt)    # время в секундах для графиков
number_time_steps = len(tp)

# Заданная ступенька по углу атаки (5°) на 10‑м шаге
reference_signals = unit_step(degree=5, tp=tp, time_step=10, output_rad=True).reshape(1, -1)
```
<!-- markdownlint-enable MD046 -->

## Шаг 2. Инициализация среды F‑16 {#шаг-2-инициализация-среды-f16}

Создаём среду `LinearLongitudinalF16-v0`. Указываем начальное состояние, состав вектора состояний/выхода и канал управления (руль высоты — `ele`). Включаем `tracking_states=["alpha"]`, чтобы ориентировать обучение на слежение за углом атаки.

<!-- markdownlint-disable MD046 -->
```python
import gymnasium as gym

env = gym.make(
    'LinearLongitudinalF16-v0',
    number_time_steps=number_time_steps,
    initial_state=[[0], [0], [0]],   # порядок: [theta, alpha, q]
    reference_signal=reference_signals,
    use_reward=False,
    state_space=["theta", "alpha", "q"],
    output_space=["theta", "alpha", "q"],
    control_space=["ele"],
    tracking_states=["alpha"],
)

obs, info = env.reset()
```
<!-- markdownlint-enable MD046 -->

## Шаг 3. Параметры Actor / Critic / Incremental model {#шаг-3-параметры-actor--critic--incremental-model}

IHDP использует три модуля:

- `Actor` — нейросеть, выдающая управляющее воздействие по ошибке слежения.
- `Critic` — нейросеть, оценивающая функционал качества J(x) и его градиент.
- `Incremental model` — онлайн‑идентификация и локальная линеаризация динамики.

Параметры ниже подобраны под демонстрационный кейс: сильная персистентная составляющая (`type_PE="combined"`) ускоряет идентификацию; высокие `learning_rate` уместны при коротком горизонте и насыщениях (`WB_limits`).

<!-- markdownlint-disable MD046 -->
```python
actor_settings = {
    "start_training": 5,
    "layers": (25, 1),
    "activations": ("tanh", "tanh"),
    "learning_rate": 2.0,
    "learning_rate_exponent_limit": 10,
    "type_PE": "combined",
    "amplitude_3211": 15,
    "pulse_length_3211": 5 / dt,
    "maximum_input": 25,
    "maximum_q_rate": 20,
    "WB_limits": 30,
    "NN_initial": 120,
    "cascade_actor": False,
    "learning_rate_cascaded": 1.2,
}

critic_settings = {
    "Q_weights": [8],
    "start_training": -1,
    "gamma": 0.99,
    "learning_rate": 15.0,
    "learning_rate_exponent_limit": 10,
    "layers": (25, 1),
    "activations": ("tanh", "linear"),
    "WB_limits": 30,
    "NN_initial": 120,
    "indices_tracking_states": env.unwrapped.indices_tracking_states,
}

incremental_settings = {
    "number_time_steps": number_time_steps,
    "dt": dt,
    "input_magnitude_limits": 25,
    "input_rate_limits": 60,
}
```
<!-- markdownlint-enable MD046 -->

## Шаг 4. Создание агента IHDP {#шаг-4-создание-агента-ihdp}

Передаём агенту конфигурацию модулей и метаданные среды. Следите за согласованностью `indices_tracking_states` с порядком состояний среды.

<!-- markdownlint-disable MD046 -->
```python
from tensoraerospace.agent.ihdp.model import IHDPAgent

agent = IHDPAgent(
    actor_settings,
    critic_settings,
    incremental_settings,
    env.unwrapped.tracking_states,
    env.unwrapped.state_space,
    env.unwrapped.control_space,
    number_time_steps,
    env.unwrapped.indices_tracking_states,
)
```
<!-- markdownlint-enable MD046 -->

## Шаг 5. Основной цикл симуляции {#шаг-5-основной-цикл-симуляции}

На каждом шаге:

- `agent.predict` возвращает управляющее воздействие с учётом текущего состояния и опорного сигнала,
- среда интегрирует динамику и возвращает новое состояние,
- при необходимости можно учитывать `reward` (здесь отключён для чистоты контроля по слежению).

<!-- markdownlint-disable MD046 -->
```python
from tqdm import tqdm

xt = np.array([[0], [0], [0]])  # начальное состояние [theta, alpha, q]
for step in tqdm(range(number_time_steps - 3)):
    ut = agent.predict(xt, reference_signals, step)
    xt, reward, terminated, truncated, info = env.step(np.array(ut))
    if terminated or truncated:
        break
```
<!-- markdownlint-enable MD046 -->

## Шаг 6. Визуализация результатов {#шаг-6-визуализация-результатов}

Сравниваем траекторию `alpha` с опорным сигналом и анализируем динамику `wz`. Сглаженность и отсутствие перерегулирования зависят от весов критика, ограничений и скоростей обучения.

<!-- markdownlint-disable MD046 -->
```python
env.model.plot_transient_process('alpha', tps, reference_signals[0], to_deg=True, figsize=(15, 4))
```
<!-- markdownlint-enable MD046 -->

![Переходный процесс по углу атаки](../../example/agent/ihdp/img/output_9_0.png){ width=960 }

<!-- markdownlint-disable MD046 -->
```python
env.model.plot_state('wz', tps, to_deg=True, figsize=(15, 4))
```
<!-- markdownlint-enable MD046 -->

![Динамика угловой скорости wz](../../example/agent/ihdp/img/output_10_1.png){ width=960 }

Ожидаемые признаки корректной работы:

- Ошибка `alpha` быстро стремится к нулю без устойчивых колебаний
- `wz` не выходит за физически разумные пределы и затухает
- Управляющий сигнал (если вывести) не насыщается длительно

!!! note
    Параметры обучения подобраны для демонстрации и могут отличаться от оптимальных для вашей задачи.

## Sanity‑checks (быстрые проверки)

- Осреднённая абсолютная ошибка по `alpha` за финальные 20% горизонта меньше 5–10% от амплитуды ступеньки
- Нет длительного насыщения управляющего сигнала
- Дискретный шаг `dt` согласован с динамикой (нет «пилообразного» шума)

## Единицы измерения и нормализация

- Углы `theta`, `alpha`, `q` в моделях и графиках — в радианах, для визуализации переводим в градусы (`to_deg=True`)
- Управление `ele` — в радианах; ограничения задаются в тех же единицах
- Если используете собственные признаки — нормализуйте входы для стабильного обучения

## Частые вопросы и советы {#частые-вопросы-и-советы}

- Почему добавляется персистентное возбуждение (PE)?
  Для качественной идентификации инкрементальной модели требуется богатая динамика входа; иначе актор и критик быстро «залипают».

- Как выбрать `Q_weights` у критика?
  Увеличение веса по ошибке слежения (например, по `alpha`) повышает приоритет точности слежения относительно энергозатрат.

- Что делать при дрожании управления?
  Снизьте `learning_rate`, увеличьте `WB_limits` осторожно, ограничьте `maximum_input`/`maximum_q_rate`, проверьте масштабирование признаков.

- Как ускорить выход на режим?
  Поднимите `learning_rate` актёра/критика, но следите за устойчивостью; увеличьте амплитуду PE, если идентификация идёт медленно.

См. также обзор алгоритма и гиперпараметры: [IHDP](../../agent/ihdp.md).
