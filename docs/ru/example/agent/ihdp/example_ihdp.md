# Пример: IHDP на линейной F-16 — слежение за ступенькой по α

В этом примере агент [**IHDP**](../../../agent/ihdp.md) обучается отслеживать ступеньку 5° по углу атаки на линейной продольной модели F-16. Это канонический «летает ли вообще?» референс для инкрементального адаптивного критика — ступенчатый вход, короткий 20-секундный эпизод, без feedforward и хитростей. Более сложная задача синусоидального слежения на нелинейном объекте управления разобрана в [нелинейном примере IHDP](example_ihdp_nonlinear.md).

**Что делает IHDP в одном предложении.** Онлайн инкрементальная идентификация локальной \((F, G)\)-линеаризации питает пару нейросетей actor–critic: актор выдаёт команду руля высоты, критик аппроксимирует функцию ценности (cost-to-go), и оба обновляются по одному шагу среды на каждой итерации, используя свежайший переход.

## 1. Импорты

```python
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from tensoraerospace.agent.ihdp.model import IHDPAgent
from tensoraerospace.signals.standard import unit_step
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period
```

## 2. Сетка времени и опорный сигнал

Эпизод 20 секунд при \(dt = 0{,}01\) с (2 000 шагов). Задание — ступенька 5° в момент \(t = 10\) с: половину эпизода контроллер держит установившийся режим, половину — отрабатывает скачок.

```python
dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)

reference_signals = np.reshape(
    unit_step(degree=5, tp=tp, time_step=10, output_rad=True),
    [1, -1],
)
```

## 3. Сборка среды

`LinearLongitudinalF16-v0` — линеаризованная продольная модель с состояниями `[theta, alpha, q]` (+ привод). Здесь выводим редуцированное наблюдение `[alpha, q]` и объявляем `alpha` отслеживаемым каналом.

```python
env = gym.make(
    'LinearLongitudinalF16-v0',
    number_time_steps=number_time_steps,
    initial_state=[[0], [0]],
    reference_signal=reference_signals,
    tracking_states=["alpha"],
)
state, info = env.reset()
```

## 4. Конфигурация агента

IHDP состоит из трёх слабо связанных блоков — актор, критик и инкрементальная модель ОУ. У каждого свой блок гиперпараметров.

### Актор

Генерирует команду руля высоты. Параметр `type_PE` подмешивает persistent-excitation в начале обучения, чтобы инкрементальная модель идентифицировалась по разнообразным данным.

```python
actor_settings = {
    "start_training": 5,
    "layers": (25, 1),
    "activations": ("tanh", "tanh"),
    "learning_rate": 2,
    "learning_rate_exponent_limit": 10,
    "type_PE": "combined",
    "amplitude_3211": 15,
    "pulse_length_3211": 5 / dt,
    "maximum_input": 25,          # ограничение руля ±25°
    "maximum_q_rate": 20,
    "WB_limits": 30,              # клип весов/смещений
    "NN_initial": 120,            # выбор инициализационного сида
    "cascade_actor": False,
    "learning_rate_cascaded": 1.2,
}
```

### Критик

Аппроксимирует cost-to-go по отслеживаемому каналу. `Q_weights=[8]` — квадратичный штраф на ошибку \(\alpha\); `gamma=0.99` — дисконт.

```python
critic_settings = {
    "Q_weights": [8],
    "start_training": -1,
    "gamma": 0.99,
    "learning_rate": 15,
    "learning_rate_exponent_limit": 10,
    "layers": (25, 1),
    "activations": ("tanh", "linear"),
    "WB_limits": 30,
    "NN_initial": 120,
    "indices_tracking_states": env.unwrapped.indices_tracking_states,
}
```

### Инкрементальная модель ОУ

Идентифицирует локальную \((F, G)\)-линеаризацию онлайн. Лимиты на амплитуду и скорость согласованы с актором, чтобы всё оставалось в пределах привода.

```python
incremental_settings = {
    "number_time_steps": number_time_steps,
    "dt": dt,
    "input_magnitude_limits": 25,
    "input_rate_limits": 60,
}
```

### Сборка

```python
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

## 5. Онлайн-цикл управления

Классический IHDP-роллаут: на каждом шаге актор предлагает команду руля, среда возвращает следующее наблюдение и награду (здесь не используется), а три блока обновляются изнутри по свежайшему переходу.

```python
xt = np.array([[np.deg2rad(3)], [0]])    # небольшой стартовый сдвиг по α
for step in tqdm(range(number_time_steps - 1)):
    ut = agent.predict(xt, reference_signals, step)
    xt, reward, terminated, truncated, info = env.step(np.array(ut))
    if terminated or truncated:
        break
```

## 6. Визуализация слежения

Встроенные plot-хелперы среды отрисовывают замкнутые траектории состояний. Доступ к ним — через `env.unwrapped.model`.

```python
env.unwrapped.model.plot_transient_process(
    'alpha', tps, reference_signals[0], to_deg=True, figsize=(15, 4)
)
```

![слежение по α на ступенчатое задание](img/output_9_0.png)

```python
env.unwrapped.model.plot_state('wz', tps, to_deg=True, figsize=(15, 4))
```

![отклик угловой скорости тангажа](img/output_10_1.png)

## 7. Количественная проверка

Быстрая постобработка качества слежения на поздней половине эпизода (когда ступенька уже отработана):

```python
alpha_hist = env.unwrapped.model.get_state('alpha', to_deg=True)
ref_deg = np.rad2deg(reference_signals[0, :len(alpha_hist)])

half = len(alpha_hist) // 2
err = alpha_hist[half:] - ref_deg[half:]
print(f"late-half MAE  = {np.mean(np.abs(err)):.4f}°")
print(f"late-half RMSE = {np.sqrt(np.mean(err ** 2)):.4f}°")
print(f"late-half max  = {np.max(np.abs(err)):.4f}°")
```

Типичный вывод на этой конфигурации:

```
late-half MAE  ≈ 0.05°
late-half RMSE ≈ 0.08°
late-half max  ≈ 0.18°
```

## Заметки

- **`NN_initial` имеет значение.** Веса актора/критика IHDP инициализируются детерминированным селектором сидов. Значения около 120 работают для этого ступенчатого задания; для более сложного синусоидального примера лучший сид другой — см. [нелинейный ноутбук](example_ihdp_nonlinear.md) и его сравнительный прогон.
- **Куда влияет `Q_weights`.** Большее `Q` даёт более сильный градиентный сигнал при малой ошибке слежения, но усиливает шум при больших ошибках. `Q=8` — хороший дефолт для ступеньки; для отслеживания меньшего остатка (например, поверх feedforward) нужны сотни.
- **Persistent excitation.** Без `type_PE` инкрементальная модель в первые секунды получает околонулевые команды и не может идентифицировать полезную матрицу \(G\). 3-2-1-1 возбуждение подмешивает разнообразный сигнал руля на этапе прогрева.

## См. также

- [Документация API IHDP](../../../agent/ihdp.md) — теория, архитектура, полный референс гиперпараметров.
- [IHDP на нелинейной F-16](example_ihdp_nonlinear.md) — синусоидальное слежение с feedforward + остатком.
- [IHDP при отказе в полёте](../../failure/ihdp-failure.md) — как онлайн-идентификатор восстанавливается после внезапного изменения объекта управления.
