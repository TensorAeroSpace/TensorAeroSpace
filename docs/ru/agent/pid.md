# PID-регулятор

PID (Пропорционально-Интегрально-Дифференциальный) регулятор — классический алгоритм управления с обратной связью, широко применяемый в аэрокосмической отрасли, робототехнике и промышленной автоматизации. Наша реализация следует соглашениям MATLAB/Simulink и включает автоматический подбор коэффициентов в стиле MATLAB.

![Блок-схема PID](../agent/img/pid/pid_diagram.svg){ width=700 }

## Теория

PID-регулятор вычисляет управляющий сигнал \(u(t)\) на основе ошибки \(e(t) = r(t) - y(t)\) между уставкой \(r(t)\) и измеренным выходом \(y(t)\):

$$
u(t) = K_p e(t) + K_i \int_0^t e(\tau)\,d\tau + K_d \frac{de(t)}{dt}
$$

### Компоненты

| Составляющая | Роль | Эффект |
|--------------|------|--------|
| **Пропорциональная (P)** | Реагирует на текущую ошибку | Быстрый отклик, может давать статическую ошибку |
| **Интегральная (I)** | Накапливает прошлую ошибку | Устраняет статическую ошибку, может вызывать перерегулирование |
| **Дифференциальная (D)** | Предсказывает будущую ошибку | Демпфирует колебания, чувствительна к шуму |

### Дискретная реализация

В дискретном времени с шагом \(\Delta t\):

$$
u_k = K_p e_k + K_i \sum_{j=0}^{k} e_j \Delta t + K_d \frac{y_{k-1} - y_k}{\Delta t}
$$

!!! note "Производная по измерению"
    Наша реализация использует **производную по измерению** (а не по ошибке), как это принято по умолчанию в Simulink. Это позволяет избежать "derivative kick" при резком изменении уставки.

### Anti-Windup

Когда управляющий выход насыщается (достигает пределов исполнительного механизма), интегральная составляющая может "накручиваться", вызывая большое перерегулирование. Наша реализация включает **условное интегрирование (anti-windup)**: интегратор замораживается при насыщении выхода.

## Быстрый старт

```python
import numpy as np
import gymnasium as gym

from tensoraerospace.agent.pid import PID
from tensoraerospace.signals.standard import unit_step
from tensoraerospace.utils import generate_time_period

# Опорный сигнал — ступенька 5° по тангажу
dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)
reference_signal = np.reshape(
    unit_step(degree=5, tp=tp, time_step=2.0, output_rad=True),
    [1, -1],
)

# Создаём окружение
env = gym.make(
    'LinearLongitudinalB747-v0',
    number_time_steps=number_time_steps,
    reference_signal=reference_signal,
)

# Создаём PID-регулятор
pid = PID(env=env, kp=-0.1, ki=-0.01, kd=-0.05, dt=dt)

# Цикл управления
obs, info = env.reset()
for k in range(number_time_steps):
    reference = float(reference_signal[0, k])
    measurement = obs[3]  # theta (угол тангажа)
    action = pid.select_action(reference, measurement)
    obs, reward, terminated, truncated, info = env.step([action])
    if terminated or truncated:
        break
```

## Автоматический подбор в стиле MATLAB

Метод `tune_matlab_style()` автоматически находит оптимальные коэффициенты PID с помощью глобальной оптимизации, аналогично PID Tuner в MATLAB Simulink.

### Как это работает

1. **Извлекает модель пространства состояний** (матрицы A, B, C, D) из окружения
2. **Автоматически определяет знак контура** используя анализ статического коэффициента усиления (DC gain)
3. **Запускает дифференциальную эволюцию** для минимизации функции стоимости
4. **Оптимизирует на робастность**: учитывает и переходную характеристику, И качество слежения

### Режимы настройки

=== "Режим Step Response"

    Оптимизирует чистую переходную характеристику с быстрым временем установления и минимальным перерегулированием.

    ```python
    pid = PID(env=env)
    result = pid.tune_matlab_style(
        track_state_idx=3,      # Индекс состояния theta
        mode="step_response",
        target_settling_time=5.0,
        target_overshoot=10.0,
        n_iterations=100
    )
    print(result)
    # MATLABTuneResult(Kp=-0.1234, Ki=-0.0456, Kd=-0.0789, ...)
    ```

    **Функция стоимости минимизирует:**
    - Время установления (время достижения ±2% от конечного значения)
    - Перерегулирование выше целевого порога
    - Статическую ошибку
    - Интеграл квадрата ошибки (ISE)
    - Затраты на управление и насыщение

    **Также учитывает** качество слежения как вторичную цель (вес 25%), чтобы настроенный PID не отказывал на синусоидальных сигналах.

=== "Режим Tracking"

    Оптимизирует точное слежение за изменяющимися во времени сигналами (синусоиды, рампы).

    ```python
    pid = PID(env=env)
    result = pid.tune_matlab_style(
        track_state_idx=3,
        mode="tracking",
        n_iterations=100
    )
    ```

    **Функция стоимости минимизирует:**
    - Среднеквадратичную ошибку (RMSE)
    - Интеграл абсолютной ошибки (IAE)
    - Фазовое запаздывание
    - Затраты на управление и насыщение

    **Также учитывает** переходную характеристику как вторичную цель (вес 25%), чтобы обеспечить устойчивость при резких изменениях уставки.

### Пример использования с B747

```python
import gymnasium as gym
import numpy as np
from tensoraerospace.agent.pid import PID
from tensoraerospace.signals.standard import unit_step
from tensoraerospace.utils import generate_time_period

# Настройка
dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
n_steps = len(tp)

# Создаём ступенчатый сигнал задания (5 градусов)
reference = unit_step(degree=5, tp=tp, time_step=100, output_rad=False)

env = gym.make(
    'LinearLongitudinalB747-v0',
    number_time_steps=n_steps,
    initial_state=np.array([[0], [0], [0], [0]]),
    reference_signal=reference.reshape(1, -1),
    track_state='theta'
)

# Создаём и настраиваем PID
pid = PID(env=env, dt=dt)
result = pid.tune_matlab_style(
    track_state_idx=3,        # индекс theta
    mode="step_response",
    target_settling_time=5.0,
    target_overshoot=10.0,
    n_iterations=150,
    verbose=True
)

print(f"Настроенный PID: Kp={pid.kp:.4f}, Ki={pid.ki:.4f}, Kd={pid.kd:.4f}")
print(f"Время установления: {result.settling_time:.2f}с")
print(f"Перерегулирование: {result.overshoot:.1f}%")
```

### Пример вывода

```
📊 MATLAB-Style PID Optimization (Step Response)
------------------------------------------------------------
   System dimension: 4 states
   Matrices: A=(4, 4), B=(4, 1), C=(4, 4), D=(4, 1)
   Simulation steps: 2000, dt: 0.01s
   Mode: Step Response
   Target settling time: 5.0s
   Target overshoot: 10.0%
   DC Gain: -0.0421

   🔄 Running optimization (150 iterations)...
   Optimization: 100%|██████████| 150/150 [00:45<00:00]

   ✅ Optimization completed!
   Kp=-0.1523, Ki=-0.0234, Kd=-0.0891
   [Primary step] Settling time: 4.32s
   [Primary step] Overshoot: 8.45%
   [Primary step] Static error: 0.0012
   [Secondary sine] RMSE: 0.3421
```

## Основные параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `kp` | Пропорциональный коэффициент | 1.0 |
| `ki` | Интегральный коэффициент | 1.0 |
| `kd` | Дифференциальный коэффициент | 0.5 |
| `dt` | Шаг времени (секунды) | 0.01 |
| `env` | Gymnasium окружение | None |

### Параметры `tune_matlab_style()`

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `track_state_idx` | Индекс управляемого состояния | Обязательный |
| `mode` | `"step_response"` или `"tracking"` | `"step_response"` |
| `target_settling_time` | Желаемое время установления (с) | Авто |
| `target_overshoot` | Максимальное допустимое перерегулирование (%) | 10.0 |
| `n_iterations` | Итерации оптимизации | 100 |
| `verbose` | Выводить прогресс | True |

## Сравнение с другими методами

| Метод | Плюсы | Минусы | Лучше всего для |
|-------|-------|--------|-----------------|
| **PID** | Простой, быстрый, понятный | Ограниченная производительность на сложной динамике | Линейные системы, быстрое прототипирование |
| **MPC** | Учитывает ограничения, оптимальный | Вычислительно затратный | Системы с ограничениями, траектории |
| **RL (SAC/PPO)** | Адаптируется к нелинейной динамике | Требует обучения, менее интерпретируемый | Сложные нелинейные системы |

## Практические советы

!!! tip "Когда использовать PID vs другие методы"
    - **Используйте PID** когда система приблизительно линейна и нужен простой, интерпретируемый регулятор
    - **Используйте MPC** когда есть явные ограничения на состояния или управления
    - **Используйте RL** когда динамика сильно нелинейна или неизвестна

!!! warning "Согласованность единиц"
    Убедитесь, что сигнал задания и наблюдения используют одинаковые единицы. Наш тюнер автоматически обрабатывает преобразование градусы/радианы для окружений B747.

!!! tip "Начальная точка"
    Для большинства аэрокосмических систем начните с `mode="step_response"` и `target_overshoot=10.0`. Это даёт хороший баланс между скоростью и устойчивостью.

## Документация API

::: tensoraerospace.agent.pid.PID

::: tensoraerospace.agent.pid.MATLABTuneResult

::: tensoraerospace.agent.pid.StateSpaceNotAvailable

