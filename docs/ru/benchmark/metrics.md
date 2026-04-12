# Метрики качества управления

Модуль `tensoraerospace.benchmark.function` содержит **17 функций** для расчёта показателей качества переходных процессов в системах автоматического управления. Метрики охватывают все классические аспекты анализа: временные характеристики, установившийся режим, затухание колебаний и интегральные критерии качества.

```python
from tensoraerospace.benchmark.function import (
    overshoot,
    settling_time,
    rise_time,
    peak_time,
    maximum_deviation,
    static_error,
    steady_state_value,
    damping_degree,
    oscillation_count,
    integral_absolute_error,
    integral_squared_error,
    integral_time_absolute_error,
    performance_index,
    find_step_function,
    get_lower_upper_bound,
    find_longest_repeating_series,
)
```

---

## Быстрый старт

Полный рабочий пример: создание ступенчатого отклика с PID-регулятором на модели Boeing 747, вычисление **всех** метрик.

```python
import gymnasium as gym
import numpy as np

from tensoraerospace.agent.pid import PID
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.benchmark.function import (
    find_step_function,
    overshoot,
    settling_time,
    rise_time,
    peak_time,
    maximum_deviation,
    static_error,
    steady_state_value,
    damping_degree,
    oscillation_count,
    integral_absolute_error,
    integral_squared_error,
    integral_time_absolute_error,
    performance_index,
    get_lower_upper_bound,
)

# --- Параметры моделирования ---
dt = 0.01
tn = 40  # секунд
tp = generate_time_period(tn=tn, dt=dt)
N = len(tp)

# Ступенчатый сигнал задания: 5 градусов, начало на 1 секунде
reference = unit_step(degree=5, tp=tp, time_step=100, output_rad=True).reshape(1, -1)

# Создание среды Boeing 747
env = gym.make(
    "LinearLongitudinalB747-v0",
    number_time_steps=N,
    initial_state=[[0], [0], [0], [0]],
    reference_signal=reference,
    tracking_states=["theta"],
    state_space=["u", "w", "q", "theta"],
    control_space=["delta_e"],
    output_space=["u", "w", "q", "theta"],
    use_reward=False,
    dt=dt,
)

# PID-регулятор
pid = PID(env, kp=-10.0, ki=-5.0, kd=-1.5, dt=dt)

# --- Симуляция ---
obs, info = env.reset()
control_log = []
output_log = []

for t in range(N - 1):
    setpoint = reference[0, t]
    measurement = float(obs[0])
    u = pid.select_action(setpoint, measurement)
    action = np.array([[float(u)]], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    control_log.append(setpoint)
    output_log.append(measurement)
    if terminated or truncated:
        break

control_signal = np.array(control_log)
system_signal = np.array(output_log)

# Выделение участка ступенчатого отклика
control_signal, system_signal = find_step_function(control_signal, system_signal)

# --- Расчёт всех метрик ---
print(f"Перерегулирование:                {overshoot(control_signal, system_signal):.2f} %")
print(f"Время регулирования (5%):         {settling_time(control_signal, system_signal, threshold=0.05)}")
print(f"Время нарастания (10%-90%):       {rise_time(control_signal, system_signal)}")
print(f"Время достижения пика:            {peak_time(system_signal)}")
print(f"Максимальное отклонение:          {maximum_deviation(control_signal, system_signal):.6f}")
print(f"Статическая ошибка:               {static_error(control_signal, system_signal):.6f}")
print(f"Установившееся значение (выход):  {steady_state_value(system_signal):.6f}")
print(f"Степень затухания:                {damping_degree(system_signal):.4f}")
print(f"Число колебаний:                  {oscillation_count(system_signal)}")
print(f"IAE:                              {integral_absolute_error(control_signal, system_signal):.4f}")
print(f"ISE:                              {integral_squared_error(control_signal, system_signal):.4f}")
print(f"ITAE:                             {integral_time_absolute_error(control_signal, system_signal, dt=dt):.4f}")
print(f"Комплексный индекс качества:      {performance_index(control_signal, system_signal, dt=dt):.4f}")
```

---

## Временные метрики

Характеризуют скорость и характер переходного процесса до достижения установившегося режима.

---

### Перерегулирование (Overshoot)

Показывает, насколько максимум отклика системы превышает установившееся значение (в процентах).

**Формула:**

$$
\sigma = \frac{M - y_{\text{уст}}}{y_{\text{уст}}} \cdot 100\%
$$

где:

- $M = \max(y(t))$ --- максимальное значение выходного сигнала,
- $y_{\text{уст}}$ --- установившееся значение (среднее последних 10% управляющего сигнала).

**Интерпретация:** чем меньше перерегулирование, тем плавнее переходный процесс. Значение $\sigma = 0$ означает апериодический (без колебаний) отклик.

**API:**

```python
def overshoot(control_signal: np.ndarray, system_signal: np.ndarray) -> float
```

| Параметр | Тип | Описание |
|---|---|---|
| `control_signal` | `np.ndarray` | Управляющий (задающий) сигнал |
| `system_signal` | `np.ndarray` | Выходной сигнал системы |
| **Возврат** | `float` | Перерегулирование в процентах |

**Пример:**

```python
from tensoraerospace.benchmark.function import overshoot

sigma = overshoot(control_signal, system_signal)
print(f"Перерегулирование: {sigma:.2f} %")
```

---

### Время регулирования (Settling Time)

Время (в отсчётах), за которое выходной сигнал входит в коридор допустимого отклонения от установившегося значения и остаётся в нём.

**Алгоритм:**

1. Вычисляется установившееся значение: $y_{\text{уст}} = \text{mean}(r[\,0.9N\!:\!N\,])$.
2. Определяются границы коридора: $y_{\text{уст}} \cdot (1 \pm \varepsilon)$, где $\varepsilon$ --- порог (по умолчанию 5%).
3. Находятся все индексы, в которых сигнал лежит внутри коридора.
4. Среди них выбирается начало самой длинной непрерывной серии --- это и есть время регулирования.

**API:**

```python
def settling_time(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
    threshold: float = 0.05
) -> Optional[int]
```

| Параметр | Тип | Описание |
|---|---|---|
| `control_signal` | `np.ndarray` | Управляющий сигнал |
| `system_signal` | `np.ndarray` | Выходной сигнал системы |
| `threshold` | `float` | Допустимое относительное отклонение (по умолчанию `0.05` = 5%) |
| **Возврат** | `Optional[int]` | Индекс начала установившегося режима. Если система не вошла в коридор --- возвращается длина массива |

**Пример:**

```python
from tensoraerospace.benchmark.function import settling_time

# Время регулирования с допуском ±5%
ts_5 = settling_time(control_signal, system_signal, threshold=0.05)

# Время регулирования с допуском ±2%
ts_2 = settling_time(control_signal, system_signal, threshold=0.02)

print(f"Время регулирования (5%): {ts_5} отсчётов")
print(f"Время регулирования (2%): {ts_2} отсчётов")
```

---

### Время нарастания (Rise Time)

Время (в отсчётах), за которое выходной сигнал нарастает от нижнего порога до верхнего порога установившегося значения.

**Формула:**

$$
t_r = t_{0.9} - t_{0.1}
$$

где $t_{0.1}$ --- момент первого достижения 10% от $y_{\text{уст}}$, $t_{0.9}$ --- момент первого достижения 90% от $y_{\text{уст}}$.

**API:**

```python
def rise_time(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
    low_threshold: float = 0.1,
    high_threshold: float = 0.9,
) -> Optional[float]
```

| Параметр | Тип | Описание |
|---|---|---|
| `control_signal` | `np.ndarray` | Управляющий сигнал |
| `system_signal` | `np.ndarray` | Выходной сигнал системы |
| `low_threshold` | `float` | Нижний порог (доля от $y_{\text{уст}}$, по умолчанию `0.1`) |
| `high_threshold` | `float` | Верхний порог (доля от $y_{\text{уст}}$, по умолчанию `0.9`) |
| **Возврат** | `Optional[float]` | Время нарастания в отсчётах. `None` --- если пороги не были достигнуты |

**Пример:**

```python
from tensoraerospace.benchmark.function import rise_time

tr = rise_time(control_signal, system_signal)
print(f"Время нарастания (10%->90%): {tr} отсчётов")

# Пользовательские пороги: 20% -> 80%
tr_custom = rise_time(control_signal, system_signal, low_threshold=0.2, high_threshold=0.8)
print(f"Время нарастания (20%->80%): {tr_custom} отсчётов")
```

---

### Время достижения пика (Peak Time)

Индекс (в отсчётах), при котором выходной сигнал достигает первого пика.

**API:**

```python
def peak_time(system_signal: np.ndarray) -> Optional[int]
```

| Параметр | Тип | Описание |
|---|---|---|
| `system_signal` | `np.ndarray` | Выходной сигнал системы |
| **Возврат** | `Optional[int]` | Индекс первого пика. Если пиков не обнаружено --- индекс максимального значения |

**Алгоритм:** использует `scipy.signal.find_peaks` для обнаружения пиков. При отсутствии пиков возвращается `np.argmax(system_signal)`.

**Пример:**

```python
from tensoraerospace.benchmark.function import peak_time

tp_idx = peak_time(system_signal)
print(f"Время достижения пика: индекс {tp_idx}, значение {system_signal[tp_idx]:.6f}")
```

---

### Максимальное отклонение (Maximum Deviation)

Максимальное абсолютное отклонение выходного сигнала от установившегося значения задания.

**Формула:**

$$
\Delta_{\max} = \max_{t} \left| y(t) - y_{\text{уст}} \right|
$$

**API:**

```python
def maximum_deviation(control_signal: np.ndarray, system_signal: np.ndarray) -> float
```

| Параметр | Тип | Описание |
|---|---|---|
| `control_signal` | `np.ndarray` | Управляющий сигнал |
| `system_signal` | `np.ndarray` | Выходной сигнал системы |
| **Возврат** | `float` | Максимальное абсолютное отклонение |

**Пример:**

```python
from tensoraerospace.benchmark.function import maximum_deviation

delta_max = maximum_deviation(control_signal, system_signal)
print(f"Максимальное отклонение: {delta_max:.6f}")
```

---

## Установившиеся метрики

Характеризуют точность системы в установившемся режиме --- после завершения переходного процесса.

---

### Статическая ошибка (Static Error)

Разница между целевым значением задания и фактическим выходом системы в установившемся состоянии.

**Формула:**

$$
e_{\text{уст}} = r_{\text{уст}} - y_{\text{уст}}
$$

где:

- $r_{\text{уст}} = \text{mean}(r[\,0.9N\!:\!N\,])$ --- среднее последних 10% управляющего сигнала,
- $y_{\text{уст}} = \text{mean}(y[\,0.9N\!:\!N\,])$ --- среднее последних 10% выходного сигнала.

**Интерпретация:** значение, близкое к нулю, указывает на высокую статическую точность системы.

**API:**

```python
def static_error(control_signal: np.ndarray, system_signal: np.ndarray) -> float
```

| Параметр | Тип | Описание |
|---|---|---|
| `control_signal` | `np.ndarray` | Управляющий сигнал |
| `system_signal` | `np.ndarray` | Выходной сигнал системы |
| **Возврат** | `float` | Статическая ошибка |

**Пример:**

```python
from tensoraerospace.benchmark.function import static_error

e_ss = static_error(control_signal, system_signal)
print(f"Статическая ошибка: {e_ss:.6f}")
```

---

### Установившееся значение (Steady-State Value)

Оценка установившегося значения сигнала по среднему последних $p\%$ отсчётов.

**Формула:**

$$
y_{\text{уст}} = \frac{1}{|T_{\text{tail}}|} \sum_{t \in T_{\text{tail}}} x(t)
$$

где $T_{\text{tail}}$ --- множество индексов, соответствующих последним `percentage * 100 %` отсчётов (по умолчанию 10%).

**API:**

```python
def steady_state_value(control_signal: np.ndarray, percentage: float = 0.1) -> float
```

| Параметр | Тип | Описание |
|---|---|---|
| `control_signal` | `np.ndarray` | Сигнал, для которого оценивается установившееся значение |
| `percentage` | `float` | Доля хвоста сигнала, используемая для усреднения (по умолчанию `0.1` = 10%) |
| **Возврат** | `float` | Оценка установившегося значения |

**Пример:**

```python
from tensoraerospace.benchmark.function import steady_state_value

# Установившееся значение выходного сигнала (по последним 10%)
y_ss = steady_state_value(system_signal)

# По последним 20% для более стабильной оценки
y_ss_20 = steady_state_value(system_signal, percentage=0.2)

print(f"Установившееся значение: {y_ss:.6f}")
```

---

## Затухание и колебания

Метрики, характеризующие колебательный процесс и степень его затухания.

---

### Степень затухания (Damping Degree)

Оценивает относительное уменьшение амплитуды между последовательными пиками колебаний.

**Формула:**

$$
D = 1 - \frac{A_{n}}{A_{n-1}}
$$

где $A_n$ --- амплитуда $n$-го пика, $A_{n-1}$ --- амплитуда предыдущего. Итоговое значение --- **среднее** по всем парам соседних пиков.

**Интерпретация:**

- $D \to 1$ --- быстрое затухание, система хорошо демпфирована.
- $D \to 0$ --- слабое затухание, колебания медленно убывают.
- $D < 0$ --- колебания нарастают (неустойчивая система!).

**API:**

```python
def damping_degree(system_signal: np.ndarray) -> float
```

| Параметр | Тип | Описание |
|---|---|---|
| `system_signal` | `np.ndarray` | Выходной сигнал системы |
| **Возврат** | `float` | Средняя степень затухания. Возвращает `0.0`, если менее двух пиков |

**Пример:**

```python
from tensoraerospace.benchmark.function import damping_degree

d = damping_degree(system_signal)
print(f"Степень затухания: {d:.4f}")
```

---

### Число колебаний (Oscillation Count)

Оценивает количество полных колебательных циклов (пар «пик -- впадина») в переходном процессе.

**Алгоритм:**

1. Обнаруживает пики сигнала (`find_peaks(y, height=threshold)`).
2. Обнаруживает впадины (`find_peaks(-y, height=threshold)`).
3. Суммарное число экстремумов делится на 2 --- получается число полных колебаний.

**API:**

```python
def oscillation_count(system_signal: np.ndarray, threshold: float = 0.01) -> int
```

| Параметр | Тип | Описание |
|---|---|---|
| `system_signal` | `np.ndarray` | Выходной сигнал системы |
| `threshold` | `float` | Минимальная амплитуда пика для учёта (по умолчанию `0.01`) |
| **Возврат** | `int` | Число полных колебаний |

**Пример:**

```python
from tensoraerospace.benchmark.function import oscillation_count

n_osc = oscillation_count(system_signal)
print(f"Число колебаний: {n_osc}")

# Учитывать только крупные колебания
n_osc_large = oscillation_count(system_signal, threshold=0.1)
print(f"Число крупных колебаний: {n_osc_large}")
```

---

## Интегральные критерии качества

Интегральные критерии оценивают накопленную ошибку за весь переходный процесс. Они чувствительны к форме и продолжительности ошибки и широко применяются при оптимизации регуляторов.

---

### IAE --- интегральная абсолютная ошибка (Integral Absolute Error)

Суммарная абсолютная ошибка за всё время моделирования.

**Формула:**

$$
\text{IAE} = \sum_{t=0}^{N} \left| r(t) - y(t) \right|
$$

**Свойства:**

- Одинаково штрафует любые ошибки вне зависимости от их знака.
- Менее чувствителен к большим кратковременным выбросам, чем ISE.

**API:**

```python
def integral_absolute_error(
    control_signal: np.ndarray,
    system_signal: np.ndarray
) -> float
```

| Параметр | Тип | Описание |
|---|---|---|
| `control_signal` | `np.ndarray` | Управляющий (задающий) сигнал |
| `system_signal` | `np.ndarray` | Выходной сигнал системы |
| **Возврат** | `float` | Значение IAE |

**Пример:**

```python
from tensoraerospace.benchmark.function import integral_absolute_error

iae = integral_absolute_error(control_signal, system_signal)
print(f"IAE: {iae:.4f}")
```

---

### ISE --- интегральная квадратичная ошибка (Integral Squared Error)

Суммарная квадратичная ошибка за всё время моделирования.

**Формула:**

$$
\text{ISE} = \sum_{t=0}^{N} \left( r(t) - y(t) \right)^2
$$

**Свойства:**

- Сильно штрафует большие ошибки (квадратичная зависимость).
- Предпочитает системы с малым перерегулированием, даже если затухание медленное.
- Широко используется как целевая функция при оптимизации параметров регуляторов.

**API:**

```python
def integral_squared_error(
    control_signal: np.ndarray,
    system_signal: np.ndarray
) -> float
```

| Параметр | Тип | Описание |
|---|---|---|
| `control_signal` | `np.ndarray` | Управляющий сигнал |
| `system_signal` | `np.ndarray` | Выходной сигнал системы |
| **Возврат** | `float` | Значение ISE |

**Пример:**

```python
from tensoraerospace.benchmark.function import integral_squared_error

ise = integral_squared_error(control_signal, system_signal)
print(f"ISE: {ise:.4f}")
```

---

### ITAE --- интегральная времявзвешенная абсолютная ошибка (Integral Time Absolute Error)

Абсолютная ошибка, взвешенная по времени: ошибки на поздних этапах переходного процесса наказываются сильнее.

**Формула:**

$$
\text{ITAE} = \sum_{t=0}^{N} t \cdot \left| r(t) - y(t) \right|
$$

**Свойства:**

- Штрафует затянувшиеся переходные процессы.
- Нечувствителен к начальным большим ошибкам (когда $t$ мало).
- Считается лучшим критерием для настройки регуляторов, обеспечивающих быстрое затухание.

**API:**

```python
def integral_time_absolute_error(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
    dt: float = 1.0
) -> float
```

| Параметр | Тип | Описание |
|---|---|---|
| `control_signal` | `np.ndarray` | Управляющий сигнал |
| `system_signal` | `np.ndarray` | Выходной сигнал системы |
| `dt` | `float` | Шаг дискретизации по времени (по умолчанию `1.0`) |
| **Возврат** | `float` | Значение ITAE |

**Пример:**

```python
from tensoraerospace.benchmark.function import integral_time_absolute_error

itae = integral_time_absolute_error(control_signal, system_signal, dt=0.01)
print(f"ITAE: {itae:.4f}")
```

---

### Комплексный индекс качества (Performance Index)

Составной показатель качества переходного процесса, объединяющий несколько критериев с фиксированными весами. **Чем меньше --- тем лучше.**

**Формула:**

$$
J = 0.4 \cdot \text{ISE} + 0.4 \cdot \text{ITAE} + 0.2 \cdot |\sigma|
$$

где $\sigma$ --- перерегулирование (в процентах).

**Свойства:**

- Позволяет одним числом сравнить качество различных регуляторов.
- Веса можно адаптировать в зависимости от задачи (в текущей реализации фиксированы).

**API:**

```python
def performance_index(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
    dt: float = 1.0
) -> float
```

| Параметр | Тип | Описание |
|---|---|---|
| `control_signal` | `np.ndarray` | Управляющий сигнал |
| `system_signal` | `np.ndarray` | Выходной сигнал системы |
| `dt` | `float` | Шаг дискретизации по времени (по умолчанию `1.0`) |
| **Возврат** | `float` | Значение комплексного индекса (меньше = лучше) |

**Пример:**

```python
from tensoraerospace.benchmark.function import performance_index

J = performance_index(control_signal, system_signal, dt=0.01)
print(f"Комплексный индекс качества: {J:.4f}")
```

---

## Вспомогательные функции

Служебные функции, используемые другими метриками или полезные при подготовке данных.

---

### find_step_function

Выделяет участок ступенчатого отклика из записанных сигналов. Обрезает начальную часть, в которой управляющий сигнал равен нулю (или заданному значению `signal_val`), и возвращает только участок от момента появления ступеньки.

**API:**

```python
def find_step_function(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
    signal_val: float = 0
) -> Tuple[np.ndarray, np.ndarray]
```

| Параметр | Тип | Описание |
|---|---|---|
| `control_signal` | `np.ndarray` | Управляющий сигнал |
| `system_signal` | `np.ndarray` | Выходной сигнал системы |
| `signal_val` | `float` | Пороговое значение, от которого начинается ступенька (по умолчанию `0`) |
| **Возврат** | `Tuple[np.ndarray, np.ndarray]` | Обрезанные массивы `(control_signal, system_signal)` |

**Raises:** `ValueError` --- если длины входных массивов не совпадают.

**Пример:**

```python
from tensoraerospace.benchmark.function import find_step_function

# Обрезать начальный участок до ступенчатого воздействия
ctrl, sys = find_step_function(control_signal, system_signal)

# С пользовательским порогом
ctrl, sys = find_step_function(control_signal, system_signal, signal_val=0.01)
```

---

### get_lower_upper_bound

Возвращает массивы нижней и верхней границ коридора допуска вокруг конечного значения управляющего сигнала. Полезно для визуализации зоны допуска при анализе времени регулирования.

**Формула:**

$$
\text{upper}(t) = r_{\text{final}} \cdot (1 + \varepsilon), \quad \text{lower}(t) = r_{\text{final}} \cdot (1 - \varepsilon)
$$

где $r_{\text{final}}$ --- последнее значение управляющего сигнала.

**API:**

```python
def get_lower_upper_bound(
    control_signal: np.ndarray,
    epsilon: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]
```

| Параметр | Тип | Описание |
|---|---|---|
| `control_signal` | `np.ndarray` | Управляющий сигнал |
| `epsilon` | `float` | Относительная величина допуска (по умолчанию `0.05` = 5%) |
| **Возврат** | `Tuple[np.ndarray, np.ndarray]` | `(lower, upper)` --- массивы той же формы, что и `control_signal` |

**Пример:**

```python
from tensoraerospace.benchmark.function import get_lower_upper_bound

lower, upper = get_lower_upper_bound(control_signal, epsilon=0.05)

# Использование для визуализации
import matplotlib.pyplot as plt

t = np.arange(len(control_signal))
plt.fill_between(t, lower, upper, alpha=0.2, color="green", label="Коридор ±5%")
plt.plot(t, system_signal, label="Выход системы")
plt.plot(t, control_signal, "--", label="Задание")
plt.legend()
plt.show()
```

---

### find_longest_repeating_series

Находит самую длинную серию последовательных целых чисел в массиве. Используется внутри `settling_time` для определения начала устойчивого нахождения сигнала в коридоре допуска.

**API:**

```python
def find_longest_repeating_series(numbers: list) -> tuple
```

| Параметр | Тип | Описание |
|---|---|---|
| `numbers` | `list` | Список целых чисел |
| **Возврат** | `tuple` | Кортеж `(start, end)` --- начало и конец самой длинной непрерывной серии. Для пустого списка `(0, 0)` |

**Пример:**

```python
from tensoraerospace.benchmark.function import find_longest_repeating_series

# Индексы, в которых сигнал был в коридоре
indices = [2, 3, 4, 10, 11, 12, 13, 14, 20]
start, end = find_longest_repeating_series(indices)
print(f"Самая длинная серия: от {start} до {end}, длина = {end - start + 1}")
# Вывод: Самая длинная серия: от 10 до 14, длина = 5
```

---

## Пример визуализации

Полный пример: моделирование ступенчатого отклика, расчёт метрик и построение аннотированного графика.

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from tensoraerospace.agent.pid import PID
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.benchmark.function import (
    find_step_function,
    overshoot,
    settling_time,
    rise_time,
    peak_time,
    maximum_deviation,
    static_error,
    steady_state_value,
    get_lower_upper_bound,
)

# --- 1. Моделирование ---
dt = 0.01
tn = 40
tp = generate_time_period(tn=tn, dt=dt)
N = len(tp)

reference = unit_step(degree=5, tp=tp, time_step=100, output_rad=True).reshape(1, -1)

env = gym.make(
    "LinearLongitudinalB747-v0",
    number_time_steps=N,
    initial_state=[[0], [0], [0], [0]],
    reference_signal=reference,
    tracking_states=["theta"],
    state_space=["u", "w", "q", "theta"],
    control_space=["delta_e"],
    output_space=["u", "w", "q", "theta"],
    use_reward=False,
    dt=dt,
)

pid = PID(env, kp=-10.0, ki=-5.0, kd=-1.5, dt=dt)

obs, info = env.reset()
control_log, output_log = [], []

for t in range(N - 1):
    setpoint = reference[0, t]
    measurement = float(obs[0])
    u = pid.select_action(setpoint, measurement)
    obs, reward, terminated, truncated, info = env.step(
        np.array([[float(u)]], dtype=np.float32)
    )
    control_log.append(setpoint)
    output_log.append(measurement)
    if terminated or truncated:
        break

ctrl = np.array(control_log)
sys_out = np.array(output_log)

# Выделение ступенчатого отклика
ctrl, sys_out = find_step_function(ctrl, sys_out)

# --- 2. Расчёт метрик ---
sigma = overshoot(ctrl, sys_out)
ts = settling_time(ctrl, sys_out, threshold=0.05)
tr = rise_time(ctrl, sys_out)
tp_idx = peak_time(sys_out)
e_ss = static_error(ctrl, sys_out)
y_ss = steady_state_value(sys_out)
lower, upper = get_lower_upper_bound(ctrl, epsilon=0.05)

# --- 3. Построение графика ---
time_axis = np.arange(len(ctrl)) * dt

fig, ax = plt.subplots(figsize=(14, 7))

# Сигналы
ax.plot(time_axis, ctrl, "k--", linewidth=1.5, label="Задание $r(t)$")
ax.plot(time_axis, sys_out, "b-", linewidth=1.5, label="Выход $y(t)$")

# Коридор допуска ±5%
ax.fill_between(time_axis, lower, upper, alpha=0.15, color="green", label="Коридор ±5%")

# Установившееся значение
ax.axhline(y=y_ss, color="gray", linestyle=":", linewidth=1, label=f"$y_{{уст}}$ = {y_ss:.4f}")

# Статическая ошибка
r_final = np.mean(ctrl[int(0.9 * len(ctrl)):])
ax.annotate(
    f"$e_{{уст}}$ = {e_ss:.4f}",
    xy=(time_axis[-1], y_ss),
    xytext=(time_axis[-1] * 0.85, (y_ss + r_final) / 2),
    fontsize=10, color="red",
    arrowprops=dict(arrowstyle="->", color="red"),
)

# Пик и перерегулирование
if tp_idx is not None and tp_idx < len(time_axis):
    ax.plot(time_axis[tp_idx], sys_out[tp_idx], "rv", markersize=10)
    ax.annotate(
        f"Пик: {sys_out[tp_idx]:.4f}\n$\\sigma$ = {sigma:.1f}%",
        xy=(time_axis[tp_idx], sys_out[tp_idx]),
        xytext=(time_axis[tp_idx] + 0.5, sys_out[tp_idx] + 0.002),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="red"),
    )

# Время регулирования
if ts is not None and ts < len(time_axis):
    ax.axvline(x=time_axis[ts], color="orange", linestyle="--", linewidth=1,
               label=f"Время регулирования = {time_axis[ts]:.2f} с")

# Время нарастания
if tr is not None:
    y_final_est = np.mean(ctrl[int(0.9 * len(ctrl)):])
    low_val = y_final_est * 0.1
    high_val = y_final_est * 0.9
    low_indices = np.where(sys_out >= low_val)[0]
    high_indices = np.where(sys_out >= high_val)[0]
    if len(low_indices) > 0 and len(high_indices) > 0:
        t_low = time_axis[low_indices[0]]
        t_high = time_axis[high_indices[0]]
        ax.axvspan(t_low, t_high, alpha=0.1, color="purple", label=f"Время нарастания = {tr} отсч.")

ax.set_xlabel("Время, с", fontsize=12)
ax.set_ylabel("Амплитуда", fontsize=12)
ax.set_title("Ступенчатый отклик с аннотациями метрик качества", fontsize=14)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Таблица интерпретации метрик

Ориентировочные значения для оценки качества переходного процесса. Конкретные нормативы зависят от задачи и требований к системе управления.

| Метрика | Обозначение | Хорошо | Удовлетворительно | Плохо | Единицы |
|---|---|---|---|---|---|
| Перерегулирование | $\sigma$ | < 5% | 5--20% | > 20% | % |
| Время регулирования (5%) | $t_s$ | Малое (быстрый отклик) | Среднее | Большое (медленная стабилизация) | отсчёты / с |
| Время нарастания | $t_r$ | Малое | Среднее | Большое | отсчёты / с |
| Время пика | $t_p$ | Малое | --- | Большое | отсчёты / с |
| Максимальное отклонение | $\Delta_{\max}$ | Близко к 0 | Умеренное | Большое | ед. сигнала |
| Статическая ошибка | $e_{\text{уст}}$ | $\approx 0$ | < 2% от $r$ | > 5% от $r$ | ед. сигнала |
| Установившееся значение | $y_{\text{уст}}$ | $\approx r_{\text{уст}}$ | --- | Значительно отличается | ед. сигнала |
| Степень затухания | $D$ | $\to 1$ | 0.3--0.7 | < 0.3 или < 0 | безразм. |
| Число колебаний | $N_{\text{osc}}$ | 0--2 | 3--5 | > 5 | шт. |
| IAE | IAE | Малое | Среднее | Большое | ед. сигнала |
| ISE | ISE | Малое | Среднее | Большое | (ед. сигнала)$^2$ |
| ITAE | ITAE | Малое | Среднее | Большое | ед. сигнала $\cdot$ с |
| Комплексный индекс | $J$ | Минимальный | --- | Большой | безразм. |

### Рекомендации по выбору критерия

- **IAE** --- универсальный критерий, подходит для большинства задач. Даёт сбалансированную настройку.
- **ISE** --- если критичны большие отклонения. Обеспечивает малое перерегулирование, но может давать медленный отклик.
- **ITAE** --- если важно быстрое затухание колебаний. Наказывает «хвосты» переходного процесса.
- **Performance Index** ($J$) --- комплексная оценка для сравнения нескольких алгоритмов на одной задаче.

### Типичные компромиссы

| Требование | Что улучшается | Что ухудшается |
|---|---|---|
| Уменьшить перерегулирование | Плавность отклика, $\sigma \downarrow$ | Время регулирования $t_s \uparrow$ |
| Уменьшить время регулирования | Быстрота $t_s \downarrow$ | Перерегулирование $\sigma \uparrow$, число колебаний $\uparrow$ |
| Уменьшить статическую ошибку | Точность $e_{\text{уст}} \to 0$ | Увеличение интегральной составляющей может привести к колебаниям |
| Увеличить степень затухания | Устойчивость, $D \uparrow$ | Медленный отклик, большое $t_s$ |
