# Active-Adaptive Incremental Nonlinear Dynamic Inversion (AA-INDI)

Используйте `AAINDIAgent(AAINDIConfig(...))` для идентификации физических моментов и оценки отказов OTSEKF–HOSM с независимой навигацией. Это единственная реализация AA-INDI. Конфигурация требует геометрии самолёта и начальных производных по рулям, управление принимает пакеты `FlightMeasurement`. Старый агент по одним угловым скоростям и эвристика смещения удалены.

Источники: [Atmaca et al., AA-INDI, 2026](https://doi.org/10.2514/6.2026-1743) и подробная [статья авторов об OTSEKF–HOSM, 2025](https://doi.org/10.2514/1.G009147).

## Начните с полного примера SDK

Пример создаёт штатную среду B737, инициализирует AA-INDI по исправной балансировке,
задаёт ступеньку тангажа +1° на 15 с и потерю 50% эффективности руля на 30 с.
Выполните блок с установленным пакетом `tensoraerospace`: он проходит все 3000
переходов, показывает задание и отклик, выводит метрики библиотеки.

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

`predict` возвращает абсолютный угол руля в радианах; повторно прибавлять балансировку
не нужно. `FlightMeasurement.from_model` формирует измерения в СИ и обратную связь
по фактическому рулю. Обучение продолжается каждый такт. Для исправного объекта
повторите запуск с `elevator_fault=None`. Высота и скорость меняются в рамках
задачи управления тангажом при фиксированном газе.

### Выберите следующий пример

- [Подробный пример B737](../example/agent/aa_indi/example_aaindi_nonlinear.md).
- [Отказы привода и гироскопа: с коррекцией и без неё](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_sensor_actuator_faults.ipynb).
- [AA-INDI против PID, LQR и LQI на B747](../comparison/aaindi_vs_pid_lqr_lqi_b747.md).

## Архитектура по статье

| Блок | Реализация | Источник |
| --- | --- | --- |
| Оценка кинематического состояния с независимой навигацией | `OTSEKFHOSMObserver`, `OptimalTwoStageEKF` | 2026, III.A; 2025, (20)–(42) |
| Четырёхзвенный нерекурсивный дифференциатор | `HOSMDifferentiator` | 2025, (47)–(50) |
| Восстановление моментов твёрдого тела | `AircraftGeometry.coefficients` | 2026, (7)–(16) |
| Производные по рулям: отдельный скалярный VFF-RLS на каждую ось | `MomentIdentifier` | 2026, (50)–(57) |
| Инкрементальная инверсия по заданному угловому ускорению | `AAINDIAgent.predict_acceleration` | 2026, (9) |

Измеренный момент восстанавливается по формуле
\[
M = J\dot\omega + \omega\times J\omega,
\quad C_M = \frac{M}{\bar q S [b,c,b]^T},
\quad G=J^{-1}\bar q S\operatorname{diag}(b,c,b) C_\delta.
\]
Деление при вычислении коэффициентов покомпонентное. Идентификатор использует **абсолютные измеренные положения рулей** и коэффициенты моментов; забывание вычисляется независимо по трём осям. По умолчанию заданы значения из статьи: `sigma0=15`, `forgetting_min=0.25`, максимум фактора забывания 1. Фильтры положений рулей и моментов имеют одну постоянную времени и инициализируются данными одного интервала.

Основной интерфейс получает виртуальное угловое ускорение от внешнего регулятора. Метод `predict(measurement, rate_reference)` добавляет настраиваемую пропорциональную обратную связь по угловой скорости. Этот адаптер не воспроизводит внешние контуры C*/крена/скольжения Flying-V.

## Измерения и единицы

Один пакет `FlightMeasurement` содержит:

| Поле | Смысл / единицы |
| --- | --- |
| `angular_rate` | Связанные скорости `[p,q,r]`, рад/с; могут содержать отказ |
| `specific_force` | Удельная сила акселерометра `[Ax,Ay,Az]`, м/с²; не инерциальное ускорение |
| `ground_velocity` | Независимая путевая скорость в NED, м/с |
| `attitude` | Независимые `[крен,тангаж,рыскание]`, радианы, углы Эйлера 3-2-1 |
| `surface_position` | Фактические углы рулей за предыдущий интервал, рад; постоянное значение либо измеренное среднее |
| `airspeed`, `density` | Воздушная скорость, м/с, и плотность воздуха, кг/м³ |

Связанные оси: вперёд/вправо/вниз; навигационные: север/восток/вниз. При горизонтальном полёте без ускорения акселерометр выдаёт `[0,0,-g]`. Полный тензор инерции задаётся в кг·м², площадь — в м², размах и хорда — в метрах. Управление среды в градусах нужно явно преобразовывать.

Первый пакет требует скорости и ориентации. В дальнейшем `None` обозначает отсутствие нового измерения навигационного канала: например, GPS 10 Гц при IMU 100 Гц. Временные метки должны отличаться на `observer.dt`; следующий `predict` использует тот же пакет, что уже получил `learn`. Ориентация, полученная интегрированием того же неисправного гироскопа, не является независимым измерением. При ветре нельзя подменять воздушную скорость модулем скорости GPS.

## Интерпретация и оставшиеся отличия

Двухэтапное разложение ковариации проверено сравнением с независимым расширенным фильтром Калмана, включая случайный дрейф смещения и коррелированные процессные шумы. HOSM использует одновременное обновление состояний и опубликованные степени `3/4`, `2/3`, `1/2`, `sign`.

**Интерпретация координаты дрейфа:** статья называет вторую координату фильтра дрейфом состояния, но в её распространении использует интегрированную матрицу входного шума. Здесь эта координата имеет единицы входного смещения. Поэтому HOSM дифференцирует накопленный *физический дрейф состояния*: интеграл неточной кинематической модели минус оценка, скорректированная независимой навигацией. Производная пересчитывается в отказы IMU через коэффициенты кинематических каналов. Это явная размерностная интерпретация архитектуры, а не подтверждённое воспроизведение неопубликованного кода авторов. Дифференцирование уже оценённого смещения скорости дало бы его изменение и потеряло постоянный отказ.

Настройки процессного шума, коэффициенты/масштабы HOSM, частота фильтра и начальная ковариация параметров заданы явно. Статьи не содержат полного исполняемого набора настроек. Проверки относятся к этой реализации и синтетическим моделям, а не к полётным результатам авторов. При малых безразмерных невязках `sigma0=15` может давать медленную идентификацию привода: хорошее слежение ещё не доказывает сходимость производных. Кинематика Эйлера ограничивает работу вдали от тангажа ±90° и плохо обусловленных положений для восстановления отказов.

`agent.save(path)` и `AAINDIAgent.from_pretrained(folder)` сохраняют наблюдатель, HOSM, RLS, фильтры и ожидающий переход. Проверена идентичность продолжения после загрузки. Старые checkpoint не содержат необходимой геометрии и независимой навигации: создайте новую конфигурацию и checkpoint.

## API новой архитектуры

::: tensoraerospace.agent.aa_indi.model.AAINDIAgent

::: tensoraerospace.agent.aa_indi.model.AAINDIConfig

::: tensoraerospace.agent.aa_indi.observer.ObserverConfig

::: tensoraerospace.agent.aa_indi.kinematics.FlightMeasurement

## Ноутбук с нелинейным B737

[Пример отработки ступеньки тангажа на B737](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_nonlinear_b737.ipynb): те же крейсерский режим и ступенька +1°, что в ноутбуке IHDP. Внешний контур тангажа задан явно; адаптация продолжается во время моделирования. Сохранены графики и метрики `ControlBenchmark`, описаны инициализация по номинальной модели и допущения упрощённых моделей. Ноутбук оформлен на английском.

## Примеры с отказами

- [B737: потеря 50% эффективности руля высоты](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_fault_b737.ipynb): непрерывное обучение, обратная связь по фактическому углу руля и метрики после отказа.
- [B747: сравнение AA-INDI с PID, LQR и LQI при отказе двигателя](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_vs_pid_lqr_b747.ipynb): настройка базовых регуляторов на исправном объекте и отдельная проверка до 500 с.

[Сравнение с графиками и метриками](../comparison/aaindi_vs_pid_lqr_lqi_b747.md).

## Измерения симулятора через публичный API

Для штатных нелинейных моделей B737/B747 используйте
`AircraftGeometry.from_parameters(model.param)` для геометрии и инерции в СИ и
`FlightMeasurement.from_model(model, surface_indices=(0,))` для руля высоты,
либо `(1, 2)` для элеронов и руля направления. До первого перехода передайте
`applied_action=trim_action` явно; далее пакет использует `model.applied_action`
и `model.current_time`. Индексы относятся к физическому входу
`[elevator, aileron, rudder, throttle]`.

Адаптер моделирует идеальную IMU и независимую навигацию через `model.dynamics`.
Из ускорения исключаются гравитация и переносные слагаемые вращающейся системы
координат; величины переводятся из американских единиц в СИ. Ветра, шума и отказов
датчиков адаптер не добавляет. Для аппаратных или шумных датчиков создавайте
`FlightMeasurement` из реальных потоков измерений. Вызов адаптера не продвигает
время модели.

`model.linearize(state, trim_action)` возвращает непрерывные якобианы A/B
в единицах модели для номинальной инициализации. Рассчитывайте их на исправной
модели до полёта; не подменяйте их истинными производными после отказа.
