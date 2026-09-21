# AA-INDI: управление нелинейным B737 и отказ руля

Полный пример ниже использует установленную библиотеку `tensoraerospace`:
создаёт среду, настраивает агент, выполняет `predict → step → learn`, показывает
задание и отклик и вычисляет метрики переходного процесса.

Сценарий: B737-800, 20 000 ft и 650 ft/s, 60 с с шагом 0.02 с. На 15 с тангаж
увеличивается на 1°, на 30 с руль теряет 50% аэродинамической эффективности.
Газ фиксирован в балансировочном положении; скорость и высота отслеживаются
на графиках, но не удерживаются отдельными регуляторами. Агент не получает
расписание отказа и обучается весь полёт.

[Ноутбук без отказа](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_nonlinear_b737.ipynb) ·
[ноутбук с отказом](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_fault_b737.ipynb) ·
[пошаговый cookbook](../../../cookbook/14_aaindi.md).

## Настройка, цикл управления и графики

```python
import numpy as np
import matplotlib.pyplot as plt
from tensoraerospace.agent.aa_indi import (
    AAINDIAgent,
    AAINDIConfig,
    AircraftGeometry,
    FlightMeasurement,
    ObserverConfig,
)
from tensoraerospace.aerospacemodel.b737.nonlinear import ElevatorEffectiveness
from tensoraerospace.benchmark import B737PitchStepBenchmark

experiment = B737PitchStepBenchmark(
    duration=60.0,
    dt=0.02,
    step_time=15.0,
    step_deg=1.0,
    elevator_fault=ElevatorEffectiveness(time=30.0, effectiveness=0.5),
)
env, trim, trim_action = experiment.make_env()
state, _ = env.reset(seed=experiment.seed)
theta_trim = state[7]
geometry = AircraftGeometry.from_parameters(env.model.param)
measurement = FlightMeasurement.from_model(
    env.model,
    applied_action=trim_action,
    surface_indices=(0,),
)
_, B = env.model.linearize(state, trim_action)
nominal_derivatives = geometry.coefficients(
    np.zeros(3),
    B[3:6, 0],
    measurement.density,
    measurement.airspeed,
)[:, None]
agent = AAINDIAgent(
    AAINDIConfig(
        geometry=geometry,
        nominal_derivatives=nominal_derivatives,
        observer=ObserverConfig(
            dt=experiment.dt, gravity=env.model.param.g_ft_s2 * 0.3048
        ),
        covariance_init=1.0,
        rate_feedback=np.full(3, 3.0),
        acceleration_cutoff_hz=5.0,
        magnitude_limit=env.model.param.elevator_max_rad,
        rate_limit=np.deg2rad(20.0),
        enable_sensor_correction=True,
    )
)
states, actions, rate_commands = [state.copy()], [], []
try:
    for k in range(experiment.steps):
        q_ref = np.clip(
            0.8 * (theta_trim + experiment.reference[k] - state[7]),
            -np.deg2rad(3.0),
            np.deg2rad(3.0),
        )
        command = agent.predict(measurement, np.array([0.0, q_ref, 0.0]))
        action = trim_action.copy()
        action[0] = command[0]
        state, _, terminated, truncated, _ = env.step(action)
        experiment.validate_transition(state, terminated, truncated, k)
        applied = env.model.applied_action
        measurement = FlightMeasurement.from_model(env.model, surface_indices=(0,))
        agent.learn(measurement, applied_action=applied[:1])
        states.append(state.copy())
        actions.append(applied)
        rate_commands.append(q_ref)
finally:
    env.close()
states, actions, rate_commands = map(np.asarray, (states, actions, rate_commands))
experiment.plot_response(states, actions, rate_commands, "AA-INDI: B737 elevator fault")
plt.show()
windows, physical_metrics = experiment.evaluate(states, actions)
print(experiment.metric_table(windows).to_string())
print(physical_metrics)
```

## Как читать результаты

Состояния модели используют ft/s, ft и радианы. `AircraftGeometry.from_parameters`
и `FlightMeasurement.from_model` выполняют преобразование в СИ. Для первого
измерения явно передаётся балансировочное управление; после `step` берётся
фактический вход объекта. `predict` возвращает абсолютный угол руля в радианах.

Сначала сравните заданный и фактический тангаж, затем угловую скорость, руль,
накопленную ошибку, высоту и скорость. Таблица различает установление около
конечного выхода и около команды. Для исправного сравнения задайте
`elevator_fault=None` и повторите весь запуск с новым агентом.

## Пример с отказом гироскопа

[Отдельный ноутбук](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_sensor_actuator_faults.ipynb) показывает
создание `AAINDIAgent` и `FlightMeasurement` напрямую, синтетический объект,
введение отказов и цикл обучения. На 20 с теряется 30% эффективности привода
тангажа и появляется смещение гироскопа 0.02 рад/с. Сравниваются новые агенты
с `enable_sensor_correction=True` и `False` при одинаковом шуме.

Независимая скорость измеряется на 10 Гц, IMU и ориентация — на 100 Гц.
Ноутбук строит графики задания, истинных скоростей, оценки смещения и рулей,
вычисляет RMSE через `ControlBenchmark`. Это аналитическое твёрдое тело;
параметры и результаты относятся к этому объекту, а не к полной аэродинамике B737.

[Измерения и API](../../../agent/aa_indi.md) ·
[AA-INDI против PID, LQR и LQI на B747](../../../comparison/aaindi_vs_pid_lqr_lqi_b747.md).
