# Benchmark

Инструменты для объективного сравнения систем управления по стандартным метрикам.

![Пример отчёта бенчмарка](bench.png)

## Что оцениваем
- Перерегулирование
- Время переходного процесса
- Степень затухания
- Статическую ошибку

## API

::: tensoraerospace.benchmark.ControlBenchmark
    options:
      members: true

## Пример использования

```python
from tensoraerospace.benchmark import ControlBenchmark

bench = ControlBenchmark()
metrics = bench.benchmarking_one_step(control_signal, system_signal, 1.0, dt)

print("Статическая ошибка:", metrics['static_error'])
print("Время переходного процесса:", metrics['settling_time'])
print("Степень затухания:", metrics['damping_degree'])
print("Перерегулирование:", metrics['overshoot'])

# Визуализация сравнения сигналов и метрик
bench.plot(control_signal, system_signal, 1.0, dt, tps, figsize=(15, 5))
```

!!! note "Единицы и входные данные"
    - `control_signal`, `system_signal` — массивы одинаковой длины
    - `1.0` — желаемое установившееся значение (пример)
    - `dt` — шаг дискретизации; `tps` — временная ось

!!! info "Обратная совместимость"
    Старое имя метода `becnchmarking_one_step` по-прежнему работает как псевдоним для `benchmarking_one_step` для обеспечения обратной совместимости.

