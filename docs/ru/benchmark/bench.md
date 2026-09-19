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


## Окна оценки и метрики относительно команды

`ControlBenchmark.tracking_metrics(reference, output, dt, start=..., end=...)`
принимает постоянное задание либо весь задающий сигнал и отклик `(N, channels)`.
Метрики вычисляются на `(start, end]`: RMSE/MAE по каналам, суммарные RMSE и IAE,
конечная ошибка `reference - output`. При заданной `tolerance` восстановление
означает, что все каналы остаются в полосе до конца окна; `None` означает, что
восстановление не достигнуто. Фактически приложенные управления `(N-1, inputs)`
в аргументе `actions` добавляют RMS, максимум и суммарную вариацию. Единицы
измерения задания и отклика должны совпадать.

`benchmarking_step_response(reference, output, signal_val, dt)` сохраняет
существующие метрики ступеньки и добавляет `command_settling_time` и
`command_overshoot`. Здесь `signal_val` — уровень задания до ступеньки.
Дополнительные метрики используют заданную амплитуду, включая отрицательную
ступеньку: установление около конечного выхода ещё не означает выход на задание.

## Самолётные протоколы в установленной библиотеке

Протоколы используют штатные среды, публичные агенты и `ControlBenchmark`.
Импорт каталога `example` не нужен: после установки этой версии
`tensoraerospace` код работает из любого рабочего каталога.

```python
from tensoraerospace.benchmark import B737PitchStepBenchmark, B747EngineFailureBenchmark
from tensoraerospace.aerospacemodel.b737.nonlinear import ElevatorEffectiveness

pitch = B737PitchStepBenchmark(elevator_fault=ElevatorEffectiveness(time=30, effectiveness=0.5))
env, trim, trim_action = pitch.make_env()
state, _ = env.reset(seed=pitch.seed)
A, B = env.model.linearize(state, trim_action)  # Непрерывные якобианы в единицах модели.
env.close()

comparison = B747EngineFailureBenchmark(duration=90, fault_time=30)
result = comparison.run("AA-INDI", fault=True)
print(result["after"])
```

`B737PitchStepBenchmark` предоставляет `reference`, `time`, `validate_transition`,
`evaluate`, `metric_table` и графики Matplotlib: `plot_reference`, `plot_response`,
`plot_step`. Цикл `predict → step → learn` остаётся виден в ноутбуке.

`B747EngineFailureBenchmark` предоставляет здоровую балансировку и модель
(`nominal_trim`, `nominal_model`), `make_env`, `make_aaindi`, `make_lqr`,
`tune_baselines`, `run`, `evaluate`, `validate_additional_cases`.
Подберите базовые регуляторы на исправном объекте и передавайте выбранные
настройки во все сравниваемые прогоны. `run` возвращает состояния `(steps+1, 12)`,
управления `(steps, 4)`, диагностику, события и метрики `before`/`after`/`whole`.
При отказе с нулевого момента `before` равно `None`. Каждый прогон начинает новый
эпизод и создаёт новый регулятор.

Это локальные протоколы моделирования с настройками для указанного крейсерского
режима, идеальными датчиками и явными ограничениями. Полные инструкции и графики:
[пример B737](../cookbook/14_aaindi.md), [сравнение B747](../cookbook/09_fault_tolerance.md).
