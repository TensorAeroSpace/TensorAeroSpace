# Сигналы

Генераторы типовых тестовых сигналов для моделирования, идентификации и проверки систем управления в `TensorAeroSpace`.

## Быстрый старт

```python
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standart import unit_step, sinusoid
from tensoraerospace.signals.random import full_random_signal

# Временная ось 0..20 с (шаг по умолчанию)
tp = generate_time_period(tn=20)

# Ступень 5° в момент t=10 с
u_step = unit_step(degree=5, tp=tp, time_step=10, output_rad=False)

# Синус 10 ед. амплитуда, частота 0.01 Гц
u_sin = sinusoid(tp=tp, amplitude=10, frequency=0.01)

# Случайный сигнал со случайными частотой и амплитудой
u_rand = full_random_signal(0, 0.01, 20, (-0.5, 0.5), (-0.5, 0.5))
```

!!! tip "Единицы измерения"
    В функциях, где доступно, параметр `output_rad=False` возвращает углы в градусах. Установите `True` для радиан.

---

## Ступенчатый сигнал

Классический ступенчатый вход для возбуждения переходных процессов.

=== "API"

    ::: tensoraerospace.signals.standart.unit_step

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import unit_step

    tp = generate_time_period(tn=20)
    u = unit_step(degree=5, tp=tp, time_step=10, output_rad=False)
    ```

![Сгенерированный ступенчатый сигнал](img/unit_step.png)

---

## Синусоидный сигнал

Используется для частотного анализа и тестирования линейных участков.

=== "API"

    ::: tensoraerospace.signals.standart.sinusoid

=== "Пример"

    ```python
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import sinusoid

    tp = generate_time_period(tn=20)
    u = sinusoid(tp=tp, amplitude=10, frequency=0.01)
    ```

![Синусоидный сигнал](img/sinusoid.png)

---

## Случайный сигнал по частоте и амплитуде

Генерирует случайный тестовый вход с варьируемыми диапазонами частоты и амплитуды.

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

### Примечания

- Для построения временной оси используйте `tensoraerospace.utils.generate_time_period`.
- Все функции возвращают массив значений сигнала, совместимый с временной осью `tp`.
