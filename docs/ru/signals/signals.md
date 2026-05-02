# Сигналы

Генераторы типовых тестовых сигналов для моделирования, идентификации и проверки систем управления в `TensorAeroSpace`.

TensorAeroSpace предоставляет **17 типов сигналов** для комплексного тестирования и анализа систем:

- **Базовые сигналы**: Ступенчатый, Линейный, Импульсный, Константный
- **Периодические сигналы**: Синусоидальный, Прямоугольный, Треугольный, Пилообразный
- **Сложные сигналы**: Чирп, Дублет, Мульти-шаг, Экспоненциальный, Гауссов импульс, Мульти-синус, Затухающая синусоида
- **Случайные сигналы**: Полностью случайный сигнал

## Быстрый старт

```python
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standard import unit_step, sinusoid, chirp, doublet
from tensoraerospace.signals.random import full_random_signal
import numpy as np

# Временная ось 0..20 с (шаг по умолчанию)
tp = generate_time_period(tn=20)

# Ступень 5° в момент t=10 с
u_step = unit_step(degree=5, tp=tp, time_step=10, output_rad=False)

# Синус 10 ед. амплитуда, частота 0.01 Гц
u_sin = sinusoid(tp=tp, amplitude=10, frequency=0.01)

# Чирп-сигнал для частотного анализа
u_chirp = chirp(tp, f0=0.1, f1=2.0, amplitude=2.0, method='linear')

# Дублет для аэрокосмических маневров
u_doublet = doublet(tp, amplitude=np.deg2rad(5), time_start=5.0, width=1.0)

# Случайный сигнал со случайными частотой и амплитудой
u_rand = full_random_signal(0, 0.01, 20, (-0.5, 0.5), (-0.5, 0.5))
```

!!! tip "Единицы измерения"
    В функциях, где доступно, параметр `output_rad=False` возвращает углы в градусах. Установите `True` для радиан.

---

## Базовые сигналы

### Ступенчатый сигнал

Классический ступенчатый вход для возбуждения переходных процессов и анализа реакции системы.

=== "API"

    ::: tensoraerospace.signals.standard.unit_step

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import unit_step

    tp = generate_time_period(tn=20)
    u = unit_step(degree=5, tp=tp, time_step=10, output_rad=False)
    ```

![Сгенерированный ступенчатый сигнал](img/unit_step.png)

---

### Линейный сигнал (Ramp)

Линейно возрастающий сигнал для тестирования способности системы отслеживать линейно изменяющиеся траектории.

=== "API"

    ::: tensoraerospace.signals.standard.ramp

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import ramp

    tp = generate_time_period(tn=20)
    u = ramp(tp, slope=0.5, time_start=2.0)
    ```

![Линейный сигнал](img/ramp.png)

---

### Импульсный сигнал

Прямоугольный импульс для анализа импульсной характеристики и переходных процессов.

=== "API"

    ::: tensoraerospace.signals.standard.pulse

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import pulse

    tp = generate_time_period(tn=20)
    u = pulse(tp, amplitude=5.0, time_start=5.0, width=3.0)
    ```

![Импульсный сигнал](img/pulse.png)

---

### Константный сигнал

Постоянный опорный сигнал для отслеживания уставки и анализа установившегося режима.

=== "API"

    ::: tensoraerospace.signals.standard.constant_line

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import constant_line

    tp = generate_time_period(tn=20)
    u = constant_line(tp, value_state=3.0)
    ```

![Константный сигнал](img/constant_line.png)

---

## Периодические сигналы

### Синусоидный сигнал

Используется для частотного анализа и тестирования линейных подсистем.

=== "API"

    ::: tensoraerospace.signals.standard.sinusoid

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import sinusoid

    tp = generate_time_period(tn=20)
    u = sinusoid(tp=tp, amplitude=10, frequency=0.01)
    ```

![Синусоидный сигнал](img/sinusoid.png)

---

### Синусоида со смещением

Синусоидальный сигнал с постоянной составляющей для тестирования систем с ненулевой рабочей точкой.

=== "API"

    ::: tensoraerospace.signals.standard.sinusoid_vertical_shift

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import sinusoid_vertical_shift

    tp = generate_time_period(tn=20)
    u = sinusoid_vertical_shift(tp, frequency=0.5, amplitude=2.0, vertical_shift=5.0)
    ```

![Синусоида со смещением](img/sinusoid_vertical_shift.png)

---

### Прямоугольный сигнал

Периодический прямоугольный сигнал для переключательного управления и систем с релейными элементами.

=== "API"

    ::: tensoraerospace.signals.standard.square_wave

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import square_wave

    tp = generate_time_period(tn=20)
    u = square_wave(tp, frequency=0.5, amplitude=3.0, duty_cycle=0.5)
    ```

![Прямоугольный сигнал](img/square_wave.png)

---

### Треугольный сигнал

Плавный периодический сигнал с симметричными временами нарастания и спада.

=== "API"

    ::: tensoraerospace.signals.standard.triangular_wave

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import triangular_wave

    tp = generate_time_period(tn=20)
    u = triangular_wave(tp, frequency=0.3, amplitude=4.0)
    ```

![Треугольный сигнал](img/triangular_wave.png)

---

### Пилообразный сигнал

Периодический пилообразный сигнал с линейным нарастанием от отрицательной до положительной амплитуды.

=== "API"

    ::: tensoraerospace.signals.standard.sawtooth

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import sawtooth

    tp = generate_time_period(tn=20)
    u = sawtooth(tp, frequency=0.4, amplitude=3.0)
    ```

![Пилообразный сигнал](img/sawtooth.png)

---

## Сложные сигналы

### Чирп-сигнал

Сигнал с изменяющейся частотой для идентификации систем и анализа частотных характеристик.

=== "API"

    ::: tensoraerospace.signals.standard.chirp

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import chirp

    tp = generate_time_period(tn=20)
    u = chirp(tp, f0=0.1, f1=2.0, amplitude=2.0, method='linear')
    ```

![Чирп-сигнал](img/chirp.png)

---

### Дублет

Аэрокосмический маневр, состоящий из положительного и отрицательного импульсов для анализа устойчивости.

=== "API"

    ::: tensoraerospace.signals.standard.doublet

=== "Пример"

    ```python
    import numpy as np
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import doublet

    tp = generate_time_period(tn=20)
    u = doublet(tp, amplitude=np.deg2rad(10), time_start=5.0, width=1.0)
    ```

![Дублет](img/doublet.png)

---

### Мульти-шаговый сигнал

Последовательность ступенчатых изменений для тестирования отслеживания множественных уставок.

=== "API"

    ::: tensoraerospace.signals.standard.multi_step

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import multi_step

    tp = generate_time_period(tn=20)
    u = multi_step(tp, step_times=[2, 5, 8, 12, 16], step_values=[1, 2, -1, 3, -2])
    ```

![Мульти-шаговый сигнал](img/multi_step.png)

---

### Экспоненциальный сигнал

Плавный экспоненциальный подход к конечному значению, моделирующий реакцию системы первого порядка.

=== "API"

    ::: tensoraerospace.signals.standard.exponential

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import exponential

    tp = generate_time_period(tn=20)
    u = exponential(tp, amplitude=10.0, time_constant=2.0, time_start=3.0)
    ```

![Экспоненциальный сигнал](img/exponential.png)

---

### Гауссов импульс

Колоколообразный импульс для плавных возмущений и полосно-ограниченных возбуждений.

=== "API"

    ::: tensoraerospace.signals.standard.gaussian_pulse

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import gaussian_pulse

    tp = generate_time_period(tn=20)
    u = gaussian_pulse(tp, amplitude=8.0, center=10.0, width=1.5)
    ```

![Гауссов импульс](img/gaussian_pulse.png)

---

### Мульти-синусоидальный сигнал

Сумма нескольких синусоид для многочастотного возбуждения систем и анализа MIMO систем.

=== "API"

    ::: tensoraerospace.signals.standard.multisine

=== "Пример"

    ```python
    import numpy as np
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import multisine

    tp = generate_time_period(tn=20)
    u = multisine(tp, frequencies=[0.2, 0.5, 1.0, 1.5], 
                  amplitudes=[2.0, 1.5, 1.0, 0.5],
                  phases=[0, np.pi/4, np.pi/2, np.pi])
    ```

![Мульти-синусоидальный сигнал](img/multisine.png)

---

### Затухающая синусоида

Экспоненциально затухающие колебания, характерные для недодемпфированных систем.

=== "API"

    ::: tensoraerospace.signals.standard.damped_sinusoid

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import damped_sinusoid

    tp = generate_time_period(tn=20)
    u = damped_sinusoid(tp, frequency=1.0, amplitude=5.0, damping=0.3, time_start=2.0)
    ```

![Затухающая синусоида](img/damped_sinusoid.png)

---

## Случайные сигналы

### Случайный сигнал по частоте и амплитуде

Генерирует случайный тестовый вход с варьируемыми диапазонами частоты и амплитуды для моделирования возмущений.

=== "API"

    ::: tensoraerospace.signals.random.full_random_signal

=== "Пример"

    ```python
    from tensoraerospace.signals.random import full_random_signal

    # full_random_signal(t0, dt, tn, amplitude_range, frequency_range)
    u = full_random_signal(0, 0.01, 20, (-0.5, 0.5), (-0.5, 0.5))
    ```

![Случайный сигнал по частоте и амплитуде](img/full_random.png)

---

## Примечания

- Для построения временной оси используйте `tensoraerospace.utils.generate_time_period`.
- Все функции возвращают массив значений сигнала, совместимый с временной осью `tp`.
- Для аэрокосмических приложений дублет особенно полезен для тестирования систем управления полетом.
- Чирп-сигналы идеально подходят для идентификации систем и анализа частотных характеристик.
- Комбинируйте различные сигналы для создания сложных тестовых сценариев.
