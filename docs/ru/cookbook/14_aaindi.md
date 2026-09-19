# Рецепт 14 — AA-INDI на нелинейном B737: от датчиков до ступенчатого отклика

**Цель:** собрать физически согласованный регулятор AA-INDI, отработать ступеньку
тангажа +1° на B737, проследить непрерывную идентификацию и оценить переходный
процесс через `ControlBenchmark`. Затем тот же полёт можно повторить с потерей
50% эффективности руля высоты. Выполняйте блоки Python по порядку с установленной текущей версией `tensoraerospace`.

**Полные ноутбуки:** [исправный B737](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_nonlinear_b737.ipynb)
· [B737 с отказом руля высоты](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_fault_b737.ipynb).
В [теории и API агента](../agent/aa_indi.md) разобраны используемые здесь
наблюдатель OTSEKF–HOSM и идентификация физических моментов.

## 1. Разберитесь с двумя контурами управления

Внешний контур превращает ошибку тангажа в заданную угловую скорость:

\[
q_{\mathrm{ref}} = \operatorname{clip}\left(
0.8(\theta_{\mathrm{trim}}+\Delta\theta_{\mathrm{ref}}-\theta),
-3^\circ/\mathrm{s},\;3^\circ/\mathrm{s}\right).
\]

Далее `predict(measurement, rate_reference)` в AA-INDI превращает ошибку скорости
в виртуальное угловое ускорение и выполняет инкрементальное обращение динамики.
Если приложение уже вычисляет виртуальное ускорение, используйте
`predict_acceleration`; не применяйте оба внешних закона скорости к одной команде.

Регулятор использует восстановленное угловое ускорение и фактические рули
для непрерывной оценки производных моментов. Независимая навигация позволяет
оценивать отказы IMU. Это разные задачи оценивания; малая ошибка слежения сама
по себе не подтверждает правильность каждой оценки параметров.

## 2. Создайте самолёт, балансировку и задание

Используется нелинейная конфигурация B737-800 на высоте 20,000 ft и скорости
650 ft/s, интегратор RK4 и период управления 0.02 с. До 15 с задан балансировочный
тангаж, затем балансировочный тангаж +1°. Адаптируется только руль высоты; элероны,
руль направления и газ остаются в балансировочных положениях. Скорость и высота
наблюдаются, но отдельного контура их удержания в этом рецепте нет.


```python
import numpy as np
import matplotlib.pyplot as plt
from tensoraerospace.benchmark import B737PitchStepBenchmark, ControlBenchmark
from tensoraerospace.agent.aa_indi import AircraftGeometry, FlightMeasurement
from tensoraerospace.aerospacemodel.b737.nonlinear import ElevatorEffectiveness
from tensoraerospace.agent.aa_indi import AAINDIAgent, AAINDIConfig, ObserverConfig

USE_ELEVATOR_FAULT = False
fault = ElevatorEffectiveness(time=30.0, effectiveness=0.5) if USE_ELEVATOR_FAULT else None

experiment = B737PitchStepBenchmark(elevator_fault=fault, duration=60.0, dt=0.02, step_time=15.0, step_deg=1.0)
env, trim_result, trim_action = experiment.make_env()
state, _ = env.reset(seed=experiment.seed)
params = env.unwrapped.model.param
theta_trim = trim_result.alpha_rad
reference = experiment.reference
OUTER_PITCH_GAIN = 0.8
print("Trim residual:", trim_result.residual)
print("Trim pitch/elevator [deg]:", np.rad2deg([theta_trim, trim_action[0]]))
experiment.plot_reference(theta_trim)
plt.show()
```


![Задание тангажа B737 и момент подачи ступеньки](../../assets/images/cookbook_14_aaindi_reference.png)

В `B737PitchStepBenchmark.reference` содержится **3,001 отсчёт** на **3,000 переходов**, включая
последнее измерение. `B737PitchStepBenchmark.make_env()` проверяет сходимость балансировки
до создания окружения. Порядок действий — `[elevator, aileron, rudder, throttle]`:
первые три величины в радианах, газ нормирован.

Порядок состояний — `[u, v, w, p, q, r, phi, theta, psi, x_N, y_E, z_D]`.
Линейные скорости и координаты этой модели заданы в ft/s и ft; угловые состояния —
в радианах и рад/с. Адаптер датчиков переводит измерения для AA-INDI в СИ.

## 3. Передайте физически корректный пакет измерений

[`FlightMeasurement.from_model` из SDK](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/tensoraerospace/agent/aa_indi/kinematics.py)
собирает `FlightMeasurement` из моделируемых датчиков:

| Поле | Единицы и смысл |
|---|---|
| `time` | Секунды; при каждом переходе увеличивается на `ObserverConfig.dt` |
| `angular_rate` | Связанные угловые скорости `[p, q, r]`, рад/с |
| `specific_force` | Показания акселерометра, м/с²; без гравитации |
| `ground_velocity` | Независимая навигационная скорость NED, м/с |
| `attitude` | Независимые крен, тангаж и курс, радианы |
| `surface_position` | Фактический угол руля высоты за предыдущий интервал, радианы |
| `airspeed`, `density` | Скорость относительно воздуха, м/с; плотность, кг/м³ |

При связанных скоростях \(v_b\), угловых скоростях \(\omega\) и матрице поворота
из связанных осей в NED \(R\), синтетический акселерометр вычисляется как
\(f_b=\dot v_b+\omega\times v_b-R^T[0,0,g]^T\) с переводом ft/s² в м/с².
Так гравитация и вращение системы координат не учитываются дважды. Правая часть
модели используется для симуляции акселерометра; истинные моменты и угловое
ускорение напрямую идентификатору не передаются.

В реальной системе независимая навигация должна быть независима от отказавшего IMU.
При инициализации нужны и скорость, и ориентация. Позже можно передавать `None`,
если более медленный датчик ещё не выдал нового отсчёта; устаревшие наблюдения
не следует повторно выдавать за новые независимые измерения. Здесь идеальная
навигация доступна каждый такт; ветра нет, поэтому воздушная скорость равна норме
связанной линейной скорости.

## 4. Инициализируйте модель моментов и регулятор

Исправная балансировка даёт однократную локальную производную
`b = ∂q_dot/∂elevator`. Геометрический адаптер переводит полный тензор инерции,
включая центробежный момент инерции, в СИ. `geometry.coefficients` переводит
угловое ускорение в безразмерную производную момента с учётом инерции и скоростного
напора. Полученный массив имеет размер **(3, 1)**: три оси момента, один вход руля высоты.


```python
from tensoraerospace.agent.aa_indi import AAINDIAgent, AAINDIConfig, ObserverConfig

geometry = AircraftGeometry.from_parameters(params)
measurement = FlightMeasurement.from_model(env.model, applied_action=trim_action, surface_indices=(0,))
A_cont, B_cont = env.model.linearize(state, trim_action)
b = B_cont[4, 0]
nominal_derivatives = geometry.coefficients(
    np.zeros(3), np.array([0.0, b, 0.0]), measurement.density, measurement.airspeed,
)[:, None]
agent = AAINDIAgent(AAINDIConfig(
    geometry=geometry, nominal_derivatives=nominal_derivatives,
    observer=ObserverConfig(dt=experiment.dt, gravity=params.g_ft_s2 * 0.3048),
    sigma0=15.0, forgetting_min=0.25, covariance_init=1.0,
    rate_feedback=np.full(3, 3.0), acceleration_cutoff_hz=5.0,
    magnitude_limit=params.elevator_max_rad, rate_limit=np.deg2rad(20.0),
    enable_sensor_correction=True,
))
print(f"Nominal Cm_delta_e: {nominal_derivatives[1, 0]:.6f} 1/rad")
print(f"Trim airspeed: {measurement.airspeed:.3f} m/s; density: {measurement.density:.6f} kg/m³")
print("Trim specific force [m/s²]:", measurement.specific_force)
```


| Настройка | Назначение |
|---|---|
| `rate_feedback = [3, 3, 3]` | Пропорциональный внешний переход от ошибки скорости к ускорению |
| `acceleration_cutoff_hz = 5` | Общая фильтрация восстановленного момента и регрессоров рулей |
| `covariance_init = 1` | Начальная неопределённость идентификации производных моментов |
| `sigma0 = 15`, `forgetting_min = 0.25` | Настройки идентификации с переменным забыванием |
| `rate_limit = 20°/s` | Ограничение скорости перекладки; в конфигурации переведено в рад/с |
| `enable_sensor_correction = True` | Использование скорректированных наблюдателем скоростей весь прогон |

Номинальная инициализация — явно используемое знание модели. Один руль высоты
не позволяет независимо задавать три угловых ускорения; в симметричном опыте
по тангажу задания скоростей крена и рыскания нулевые. Внешние коэффициенты
и настройки датчиков относятся к приложению и не воспроизводят лётную настройку
самолёта из статьи.

## 5. Выполните полный цикл predict → step → learn


```python
states, actions, rate_commands, learning = [state.copy()], [], [], []
try:
    for k in range(experiment.steps):
        q_ref = float(np.clip(OUTER_PITCH_GAIN * (theta_trim + reference[k] - state[7]),
                              -np.deg2rad(3.0), np.deg2rad(3.0)))
        command = agent.predict(measurement, np.array([0.0, q_ref, 0.0]))
        action = trim_action.copy()
        action[0] = command[0]  # AA-INDI returns the absolute surface angle.
        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_state, _, terminated, truncated, _ = env.step(action)
        applied = env.model.applied_action
        experiment.validate_transition(next_state, terminated, truncated, k)
        measurement = FlightMeasurement.from_model(env.model, surface_indices=(0,))
        diagnostics = agent.learn(measurement, applied_action=applied[:1])
        learning.append([
            agent.identifier.derivatives[1, 0], agent.G[1, 0],
            agent.observer.faults[4], diagnostics["moment_residual_norm"],
        ])
        states.append(next_state.copy())
        actions.append(applied)
        rate_commands.append(q_ref)
        state = next_state
finally:
    env.close()
states, actions, rate_commands, learning = map(np.asarray, (states, actions, rate_commands, learning))
assert np.isfinite(learning).all()
update_counts = [estimator.num_updates for estimator in agent.identifier.estimators]
assert update_counts == [experiment.steps] * 3
print(f"Completed {experiment.duration:.1f} s; VFF-RLS updates per moment axis: {update_counts}")
print(f"Final fitted Cm_delta_e: {learning[-1, 0]:.6f} 1/rad")
print(f"Final reconstructed q-gyro bias: {np.rad2deg(learning[-1, 2]):.6f} deg/s (true bias: zero)")
```


В этом цикле важны следующие детали:

1. `predict` возвращает **абсолютный угол руля высоты в радианах**. Заменяйте
   `action[0]`; повторное добавление балансировочного угла учтёт его дважды.
2. После ограничения и интегрирования считайте фактическое действие из истории
   модели. В пакет и `applied_action` передаётся один и тот же физический угол.
3. Следующий пакет описывает новое состояние на `time[k + 1]`. `learn` обрабатывает
   его один раз; следующий `predict` повторно использует тот же пакет на той же метке.
4. Обучение активно на каждом переходе. Ожидаемые счётчики идентификаторов —
   `[3000, 3000, 3000]`: отдельная оценка для каждой оси момента.
5. Проверяйте конечность всей траектории и достижение заданного горизонта.
   Досрочно завершившемуся полёту нельзя приписывать метрики полного эпизода.

Цикл закрывает окружение через `finally`. При интерактивной настройке закрывайте
неиспользуемое окружение перед повторным созданием после ошибки.

## 6. Постройте отклики и диагностику оценивания


```python
experiment.plot_response(states, actions, rate_commands, "AA-INDI · nonlinear B737 pitch step")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, constrained_layout=True)
axes[0, 0].plot(experiment.time[1:], learning[:, 0], color="#176b87", label="Control-only regression")
axes[0, 0].axhline(nominal_derivatives[1, 0], color="#bd6230", linestyle="--", label="Nominal trim derivative")
axes[0, 0].set(title="Fitted elevator moment coefficient", ylabel="Cm_delta_e [1/rad]")
axes[0, 0].legend(fontsize=9)
axes[0, 1].plot(experiment.time[1:], learning[:, 1], color="#176b87")
axes[0, 1].set(title="Angular-acceleration input gain", ylabel="G_q [1/s²]")
axes[1, 0].plot(experiment.time[1:], np.rad2deg(learning[:, 2]), color="#8064a2", label="OTSEKF–HOSM reconstruction")
axes[1, 0].axhline(0, color="#bd6230", linestyle="--", label="No injected bias")
axes[1, 0].set(title="Pitch-gyro bias estimate", ylabel="Bias [deg/s]")
axes[1, 0].legend(fontsize=9)
axes[1, 1].plot(experiment.time[1:], learning[:, 3], color="#176b87")
axes[1, 1].set(title="Moment regression residual", ylabel="Coefficient residual norm")
for axis in axes.ravel():
    axis.axvline(experiment.step_time, color="#64748b", linestyle=":")
    axis.set_xlabel("Time [s]")
    axis.grid(alpha=0.22)
fig.suptitle("Continuous adaptation on an aircraft with unmodeled moment terms", fontsize=16)
plt.show()
```


![Ступенчатый отклик тангажа B737 и связанные физические каналы](../../assets/images/cookbook_14_aaindi_response.png)

Сопоставьте задание и тангаж, затем проверьте заданную угловую скорость,
фактический руль, накопленную ошибку и высоту. Изменение воздушной скорости
приведено в физической сводке ниже. Отсутствие контура удержания скорости/высоты
существенно для интерпретации результата по тангажу.
Графики из исполняемого примера подписаны на английском; pitch — тангаж,
elevator — руль высоты, airspeed — воздушная скорость, altitude — высота.

![Производная момента AA-INDI, эффективность по ускорению, оценка смещения гироскопа и невязка](../../assets/images/cookbook_14_aaindi_diagnostics.png)

Регрессия учитывает только моменты от управляющей поверхности. Полные моменты B737
зависят также от угла атаки, демпфирования и других факторов. Дрейф оценённого
коэффициента руля может поглощать эти неучтённые вклады; это не прямое измерение
исправности руля. Истинное смещение гироскопа в опыте с идеальными датчиками равно
нулю, что даёт ориентир для графика наблюдателя.

## 7. Оцените ступеньку библиотечным бенчмарком


```python
windows, physical_metrics = experiment.evaluate(states, actions)
print(experiment.metric_table(windows).to_string(na_rep="Not reached"))
for name, value in physical_metrics.items():
    print(f"{name}: {value:.6f}")
experiment.plot_step(states, windows)
plt.show()
```


`evaluate_step` вызывает `ControlBenchmark.benchmarking_one_step` проекта
и возвращает два окна: первые 15 с после команды и весь интервал после ступеньки.
Углы передаются в бенчмарк в радианах; физическая сводка выводит угловые ошибки
в градусах.

Бенчмарк использует конечный выход как опорный уровень для расчёта времени
установления и перерегулирования. Поэтому `ControlBenchmark.benchmarking_step_response` дополнительно сообщает
**время установления относительно задания** в полосе ±5% от ступеньки 1°
и **перерегулирование относительно команды**. Регулятор с остаточным смещением
может установиться около собственного выхода и всё ещё не достичь задания.
Сопоставляйте конечную ошибку и метрики по команде со штатными метриками библиотеки.

![Оценка ступенчатого отклика B737 относительно заданного тангажа](../../assets/images/cookbook_14_aaindi_step.png)

Для исправного прогона длительностью 60 с выполненный пример даёт:

| Величина | Результат |
|---|---:|
| RMSE тангажа после ступеньки | 0.135474° |
| Конечная ошибка «задание минус тангаж» | −0.005090° |
| Время установления по команде, ±5% | 2.88 с |
| Конечное изменение высоты | +397.753 ft |
| Конечное изменение воздушной скорости | −18.840 ft/s |

RMSE всего интервала после ступеньки включает начальную ошибку от команды.
Хорошее регулирование тангажа здесь сопровождается набором высоты и потерей
скорости, поскольку газ остаётся балансировочным. Это пример управления тангажом,
а не одновременного удержания ориентации, высоты и скорости.

## 8. Повторите полёт с отказом руля высоты

Установите `USE_ELEVATOR_FAULT = True` в первом блоке и выполните **все блоки заново
с новым агентом**. На 30 с штатная модель отказа B737 передаёт половину физического
угла руля в исходную аэродинамическую таблицу, согласованно пересчитывая подъёмную
силу, сопротивление и момент тангажа. Датчик положения продолжает сообщать
физический угол. Расписание отказа получает только объект и генератор
синтетических датчиков.


```python
if USE_ELEVATOR_FAULT:
    fault_metrics = ControlBenchmark().tracking_metrics(
        np.rad2deg(reference), np.rad2deg(states[:, 7] - theta_trim),
        experiment.dt, start=fault.time, tolerance=0.05,
    )
    print("Post-fault RMSE [deg]:", fault_metrics["combined_rmse"])
    print("Post-fault IAE [deg·s]:", fault_metrics["iae"])
    print("Final error [deg]:", fault_metrics["final_error"][0])
    print("Recovery to ±0.05° [s]:", fault_metrics["recovery_time"])
```


В сохранённом примере с отказом RMSE тангажа после события составляет
**0.003756°**, пиковая ошибка — **0.012846°**, конечная ошибка «задание минус тангаж» —
**−0.002334°**. Эти числа относятся к `(30, 60]` с и данной конфигурации.
При обсуждении восстановления разделяйте переход от команды на 15 с и переход
от возмущения на 30 с.

Для одновременного отказа датчика и привода откройте
[полный ноутбук с твёрдым телом](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_sensor_actuator_faults.ipynb).
В ячейках видны создание `AAINDIAgent`, пакеты `FlightMeasurement`, переход
объекта и `learn`, затем метрики библиотеки и графики. Сравните
`enable_sensor_correction=True` и `False` с новыми регуляторами и одинаковыми
seed. На 20 с вносятся потеря 30% эффективности привода тангажа и смещение
гироскопа +0.02 рад/с. Объект и быстродействие отличаются от B737: сравнивайте
два режима этого опыта между собой, а не с ошибками данного управления тангажом.

## Типичные проблемы и ограничения модели

| Симптом | Что проверить |
|---|---|
| Сразу запрошен большой угол руля | Абсолютный угол или отклонение от балансировки, радианы/градусы, знак производной и единицы скоростного напора. |
| Несовпадение меток времени или пакетов | Ровно один новый пакет на физический переход; повторное использование пакета после `learn`. |
| Дрейф при нулевом внесённом смещении гироскопа | Независимость навигации, учёт гравитации акселерометром и настройки наблюдателя. |
| Тангаж отслеживается, но коэффициенты меняются | Неучтённые аэродинамические члены и возбуждение параметров; исследуйте невязку, прежде чем заявлять сходимость. |
| Потеря скорости или высоты | В этом опыте по тангажу газ остаётся балансировочным; для другой задачи добавьте и проверьте продольное удержание. |

Конфигурация B737-800 использует производные B737-100 с изменённой геометрией
и инерцией. Датчики идеальны, динамика приводов отсутствует. Локальный отказ руля —
параметрическая потеря эффективности, а не проверенная модель разрушения.
Результаты подтверждают поведение этого симулятора в проверяемом диапазоне.

**Далее:** [Рецепт 09 — Отказ двигателя и сравнение с PID/LQR/LQI](09_fault_tolerance.md)
· [Рецепт 08 — Сохранение полного состояния регулятора](08_huggingface.md).
