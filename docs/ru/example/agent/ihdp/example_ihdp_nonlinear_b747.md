# Пример: IHDP на нелинейном B-747 — слежение за ступенькой по тангажу

Пример обучает агент **Incremental Heuristic Dynamic Programming (IHDP)** удерживать угол тангажа \(\theta\) [нелинейной модели Boeing 747-100 6-DoF](../../../model/b747_nonlinear.md) на ступенчатом задании на крейсерском режиме (FL200, \(V \approx 674\) ft/s). Исходный ноутбук: `example/reinforcement_learning/incremental_adp/example_ihdp_nonlinear_b747.ipynb`.

## Чем это отличается от примера IHDP на F-16

[Нелинейный пример IHDP для F-16](example_ihdp_nonlinear.md) отслеживает синусоиду по \(\alpha\) и использует обратно-модельный feedforward для компенсации фазового лага. B-747 — другой объект: на крейсе он **тяжёлый** (\(W \approx 636\,600\) lb, \(I_y \approx 33{,}1 \times 10^6\) slug·ft²), и короткопериодическая мода значительно медленнее F-16. Поэтому:

1. **Ступенчатое задание здесь — правильная стартовая задача.** Время нарастания короткопериодической моды на FL200 — около 3–4 с; ступенька даёт агенту чистый, устойчивый сигнал ошибки, что и есть стихия IHDP.
2. **Feedforward не нужен.** При амплитуде ступеньки 1° мы остаёмся в линейной окрестности тейлоровского разложения аэродинамики вокруг трим-точки — реактивной онлайн-политики достаточно.
3. **Обучение в пространстве отклонений от трима.** Один раз вычисляем трим \((\alpha_0, \delta_{e,0}, \delta_{T,0})\) методом Ньютона-Рафсона; IHDP видит и выдаёт только *отклонения* от рабочей точки. Постоянные трим-значения добавляются обратно внутри цикла.

## Архитектура

```
δ_e(t) = δ_{e,trim}  +  agent_residual(t)
δ_a    = 0,  δ_r = 0,  δ_T = δ_{T,trim}
```

**Состояние, видимое агентом** (извлекается из 12-мерного состояния среды):

* `d_theta` = \(\theta - \theta_{\text{trim}}\) — отклонение тангажа [рад]
* `q`       = угловая скорость тангажа [рад/с]

**Отслеживаемая величина:** `d_theta` следует за ступенькой 1°, срабатывающей в \(t = 5\) с.

## 1. Импорты и трим

```python
import math

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from tensoraerospace.aerospacemodel.b747.nonlinear import B747Configuration, trim
from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env
from tensoraerospace.agent.ihdp.model import IHDPAgent
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period
```

Решаем \(\dot u = \dot w = \dot q = 0\) при \(h = 20\,000\) ft, \(V = 674\) ft/s (\(M \approx 0{,}65\)) — триммер возвращает \(\alpha\), \(\delta_e\), throttle и готовый 12-мерный вектор начального состояния:

```python
trim_result = trim(altitude_ft=20_000.0, V_ft_s=674.0,
                   config=B747Configuration.NOMINAL)

alpha_trim_rad   = float(trim_result.alpha_rad)
theta_trim_rad   = alpha_trim_rad         # горизонтальный полёт: theta = alpha
delta_e_trim_deg = math.degrees(trim_result.elevator_rad)
throttle_trim    = float(trim_result.throttle)
# trim: alpha = +3.603°, delta_e = -0.722°, throttle = 0.555
```

## 2. Сетка времени и ступенчатое задание

60-секундный эпизод при \(dt = 0{,}02\) с. Первые 5 с задание удерживается на триме (это даёт инкрементальной модели IHDP время идентифицировать пригодную локальную линеаризацию), затем — ступенька на \(\theta_{\text{trim}} + 1^{\circ}\) до конца эпизода:

```python
dt = 0.02
tp = generate_time_period(tn=60.0, dt=dt)        # 3001 шаг
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)

ref_dev_deg = np.zeros(number_time_steps)
ref_dev_deg[int(5.0/dt):] = 1.0                 # +1° ступенька в t=5с

reference_signal = np.deg2rad(ref_dev_deg).reshape(1, -1)
```

Задание выражено как **отклонение от трима**: отслеживаемая величина агента идёт от 0 до 1° в момент ступеньки. При сравнении с фактическим тангажом мы прибавляем \(\theta_{\text{trim}}\) обратно в коде визуализации.

## 3. Среда

```python
env = NonlinearB747Env(
    trim_at=(20_000.0, 674.0),
    number_time_steps=number_time_steps,
    dt=dt, integrator="rk4",
    action_space="virtual",
    config=B747Configuration.NOMINAL,
)
obs, _ = env.reset()
```

Action space — `"virtual"`: среда принимает «сырое» 4-канальное управление \([\delta_e, \delta_a, \delta_r, \delta_T]\) в физических единицах (рад / [0, 1] для тяги). Слежение, индексация задания и сборка трим-команд выполняются в цикле rollout — это самый чистый способ подключить IHDP к произвольной Gymnasium-среде объекта управления, у которой нет встроенной trackin g-машинерии.

## 4. Агент IHDP

```python
state_space = ["d_theta", "q"]
tracking_states = ["d_theta"]
control_space = ["d_elev"]
indices_tracking_states = [0]

actor_settings = {
    "start_training": 5, "layers": (25, 1), "activations": ("tanh", "tanh"),
    "learning_rate": 2.0, "learning_rate_exponent_limit": 10,
    "type_PE": "combined", "amplitude_3211": 3, "pulse_length_3211": 5/dt,
    "maximum_input": 5, "maximum_q_rate": 20,
    "WB_limits": 30, "NN_initial": 47,
    "cascade_actor": False, "learning_rate_cascaded": 1.2,
}
critic_settings = {
    "Q_weights": [200], "start_training": -1, "gamma": 0.99,
    "learning_rate": 15, "learning_rate_exponent_limit": 10,
    "layers": (25, 1), "activations": ("tanh", "linear"),
    "WB_limits": 30, "NN_initial": 47,
    "indices_tracking_states": indices_tracking_states,
}
incremental_settings = {
    "number_time_steps": number_time_steps, "dt": dt,
    "input_magnitude_limits": 5, "input_rate_limits": 20,
}

agent = IHDPAgent(actor_settings, critic_settings, incremental_settings,
                  tracking_states, state_space, control_space,
                  number_time_steps, indices_tracking_states)
```

Три параметра отличаются от дефолтов линейного F-16 IHDP:

* **`Q_weights = [200]`** взвешивает ошибку слежения в квадратичной стоимости критика. Амплитуда ступеньки B-747 мала (\(\Delta\theta = 0{,}0175\) рад); без усиления ошибки градиент, обучающий актор, слишком слаб. \(Q = 200\) приводит стоимость в рабочий диапазон и совпадает с примером F-16 нелинейный.
* **`maximum_input = 5`** ограничивает ел евторный residual агента до ±5°. Полный ход руля высоты B-747 — ±25°, но для 1° по тангажу residual, требующийся агенту, существенно меньше 3°. Узкий клип не даёт PE-возбуждению расшатывать самолёт в первые секунды.
* **`NN_initial = 47`** — нейросети актора и критика чувствительны к инициализации; seed 47 выбран по короткому свипу на этом конкретном задании. Для других заданий лучшие сиды могут отличаться.

## 5. Онлайн-цикл обучения

```python
def deviation_state(obs):
    return np.array([[obs[7] - theta_trim_rad], [obs[4]]])  # [d_theta, q]

obs, _ = env.reset()
xt = deviation_state(obs)

for step in tqdm(range(number_time_steps - 3)):
    ut = agent.predict(xt, reference_signal, step)
    delta_e_residual_deg = float(np.asarray(ut).flatten()[0])
    delta_e_total_deg = delta_e_trim_deg + delta_e_residual_deg

    action = np.array([
        math.radians(delta_e_total_deg),  # руль высоты [рад]
        0.0, 0.0,                          # элероны, руль направления
        throttle_trim,                     # тяга [0, 1]
    ])
    obs, _, _, trunc, _ = env.step(action)
    xt = deviation_state(obs)
```

На каждом шаге:

1. Извлекаем `[d_theta, q]` из 12-мерного состояния (индексы 7 и 4 соответственно).
2. Запрашиваем у агента residual руля высоты (град).
3. Собираем полное 4-канальное virtual-управление: трим-руль + residual на канале 0, трим-тяга на канале 3, нули по элеронам и рулю направления.
4. Делаем шаг среды. Актор, критик и инкрементальная модель обновляются один раз за шаг по полученному переходу.

## 6. Метрики слежения (типичный прогон)

| Метрика                | Значение      |
|------------------------|--------------:|
| Длина эпизода          | 60 с          |
| MAE на второй половине | **0,043°**    |
| RMSE на второй половине| 0,049°        |
| Пиковая ошибка пере́хода | 3,12°         |
| Установившийся offset  | 0,04°         |

MAE на второй половине эпизода 0,043° — это 4,3% амплитуды ступеньки: агент успевает выучить динамический residual за первые ~30 с и устанавливается практически точно на цели trim+1°. Пик 3,12° около \(t = 5\) с — это короткопериодический перерегуляционный отклик, присущий планеру B-747 на этой трим-точке; именно его агенту сложнее всего подавить одним каналом руля высоты при ограниченной мощности привода.

## 7. Замечания и следующие шаги

* **Допущение по рабочей точке.** Агент обучен на одной трим-точке (FL200, \(V=674\) ft/s). Амплитуды ступеньки, выводящие систему за линейную окрестность (≥ 5°), потребуют новой настройки гиперпараметров и более длинных эпизодов — IHDP справится, но инкрементальной модели придётся агрессивнее отслеживать движущуюся локальную линеаризацию.
* **Тангаж против \(\alpha\).** Здесь отслеживается \(\theta\), а не \(\alpha\). На горизонтальном крейсе они совпадают на триме, но во время переходного процесса \(\theta\) опережает \(\alpha\) (самолёт «закидывает» нос раньше, чем поток подстраивается). Переход на отслеживание \(\alpha\) требует извлечения \(\alpha = \arctan(w/u)\) из состояния среды.
* **Синусоидальные задания.** Чистая 0,1 Гц синусоида по \(\theta\) сильно нагрузит короткопериодическую моду B-747. Как и в примере F-16 IHDP, для синусоид правильное решение — feedforward-ветвь (lookahead + amplitude gain): замените ступеньку из раздела 2 и добавьте `ff_fn`, возвращающую \(\delta_{e,\text{trim}}\) как функцию опережённого задания.
* **Сценарии повреждений.** Запуск этого же примера с `damage_profile=ELEVATOR_50PCT_LOSS` (см. `tensoraerospace.aerospacemodel.b747.nonlinear.damage`) демонстрирует онлайн-адаптацию IHDP: агент восстанавливает слежение после падения эффективности руля, с переходным выбросом ошибки в момент срабатывания.
* **Другие трим-точки.** Замените `trim_at` на `flight_condition_id ∈ {1, …, 10}` — стартанёте на одной из 10 опубликованных CR-2144 анкер-точек вместо численного трима.

## См. также

* [Boeing 747-100 (нелинейный 6-DoF)](../../../model/b747_nonlinear.md) — обзор модели, геометрия, EOM, двигатель JT9D, подсистема повреждений.
* [Примеры использования B-747](../../../model/b747_examples.md) — практические рецепты для трима, крейса, повреждений.
* [IHDP на нелинейной F-16](example_ihdp_nonlinear.md) — синусоидальное слежение с feedforward + residual; ближайший родственник этого примера.
