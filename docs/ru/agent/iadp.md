# Incremental Approximate Dynamic Programming (iADP)

iADP идентифицирует инкрементальную модель через RLS с постоянным забыванием, обучает квадратичную функцию ценности пакетным МНК и вычисляет аналитическое приращение управления. Оставлена одна реализация: критик без регуляризации и закон управления из [Konatala et al., AIAA 2024-2402](https://doi.org/10.2514/6.2024-2402).

## Начните с полного примера SDK

Код ниже настраивает iADP, выполняет 80 с непрерывного обучения при неизвестном
изменении эффективности входа, вычисляет метрики через `ControlBenchmark` и
показывает задание, отклик, управление и оценку входного коэффициента. Уравнение
скалярного объекта задано явно; расписание отказа получает только объект.
Выполните блок с установленным пакетом `tensoraerospace`.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig
from tensoraerospace.benchmark import ControlBenchmark

plt.rcParams.update(
    {
        "figure.dpi": 125,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
dt, duration = 0.001, 80.0
time = np.arange(round(duration / dt) + 1) * dt
identification_time = np.arange(round(20.0 / dt)) * dt
excitation = (
    0.15 * np.sin(2 * np.pi * 0.7 * identification_time)
    + 0.05 * np.sin(2 * np.pi * 1.7 * identification_time)
)[:, None]
period = np.arange(round(10.0 / dt)) * dt
ongoing_excitation = (0.015 * np.sin(2 * np.pi * 0.7 * period))[:, None]
config = IADPConfig.paper(
    excitation_signal=excitation,
    dt=dt,
    learning_mode="continuous",
    Q=np.array([[100.0]]),
    R=np.array([[0.0001]]),
    gamma=0.95,
    gamma_rls=0.999,
    phi_init=1e6,
    u_magnitude_limit=0.5,
    u_rate_limit=2.0,
    continuous_excitation_signal=ongoing_excitation,
)
agent = IADPAgent(1, 1, config)
x = np.zeros(1)
reference = np.array([0.05])
a = np.exp(-2 * dt)
b = (1 - a) / 2


states, actions, estimated_gain, true_gain = [x[0]], [], [], []
for k, t in enumerate(time[:-1]):
    command = agent.predict(x, reference, k)
    # The unknown effectiveness change belongs to the plant only.
    effectiveness = 1.0 if t < 60.0 else 0.7
    x = a * x + b * effectiveness * command
    agent.learn(x, reference, k, applied_action=command)
    if not np.isfinite(x).all() or not np.isfinite(agent.P).all():
        raise FloatingPointError(f"Nonfinite state or critic at {time[k+1]:g} s")
    states.append(x[0])
    actions.append(command[0])
    estimated_gain.append(agent.G[0, 0])
    true_gain.append(b * effectiveness)
states, actions, estimated_gain, true_gain = map(
    np.asarray, (states, actions, estimated_gain, true_gain)
)
assert len(actions) == len(time) - 1
assert agent.rls.num_updates == len(actions) - 1
print(f"Completed {time[-1]:g} s; RLS updates: {agent.rls.num_updates}")


benchmark = ControlBenchmark()
nominal = benchmark.tracking_metrics(0.05, states, dt, start=40.0, end=60.0)
faulty = benchmark.tracking_metrics(0.05, states, dt, start=65.0, end=80.0)
print("Nominal RMSE [rad/s]:", nominal["combined_rmse"])
print("Post-fault RMSE [rad/s]:", faulty["combined_rmse"])
print("Identified / true final G:", estimated_gain[-1], true_gain[-1])
print("Final minimum critic eigenvalue:", np.linalg.eigvalsh(agent.P).min())


fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
axes[0].plot(time, np.full_like(time, reference[0]), "k--", label="Reference")
axes[0].plot(time, states, color="#176b91", label="Plant state")
axes[0].set_ylabel("Rate [rad/s]")
axes[0].legend()
axes[1].plot(time[1:], actions, color="#40855b")
axes[1].set_ylabel("Applied control")
axes[2].plot(time[1:], true_gain, "k--", label="True discrete gain")
axes[2].plot(time[1:], estimated_gain, color="#176b91", label="Identified gain")
axes[2].set_ylabel("G estimate")
axes[2].legend()
for ax in axes:
    ax.axvspan(0, 20, alpha=0.08, color="#40855b")
    ax.axvline(60, color="#bb4040", linestyle=":")
axes[-1].set_xlabel("Time [s]")
fig.suptitle("iADP: identification, tracking and unknown effectiveness loss")
plt.show()
```

Эти коэффициенты и веса относятся к скалярному объекту. Начальное возбуждение
длится 20 с, затем небольшое периодическое воздействие продолжает давать данные
идентификатору. У окон RMSE разная предыстория адаптации: их разность не измеряет
только влияние отказа.

### Перейдите к самолётным примерам

- [B737: ступенька тангажа и отказ руля](../example/agent/iadp/example_iadp_nonlinear.md).
- [F-16: фактическое положение привода и сценарии отказа](../example/agent/iadp/example_iadp_small_fault_f16.md).
- [Выполненный ноутбук с этим циклом обучения](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_iadp_paper.ipynb).

## Состояние, выход и стоимость

Расширенное состояние `X = [x; reference_state]` имеет размерность `n_state + n_reference`. Размерности объекта и генератора задания могут различаться. Линейные отображения задают `y = C @ x` и `y_ref = Cr @ reference_state`. Стоимость шага:

\[
c_k = (Cx_k-C_r x_k^r)^T Q(Cx_k-C_r x_k^r) + \delta_k^T R\delta_k.
\]

Для генератора задания, например синусоидального осциллятора, задайте `n_reference`, `output_matrix`, `reference_output_matrix`. По умолчанию используются единичные отображения и равные размерности. Размер `Q` — `(n_output,n_output)`, `R` — `(n_control,n_control)`. Единственное измерение угловой скорости не восстанавливает недостающие состояния самолёта.

## Уравнения обновления

1. RLS оценивает `dX_next = F @ dX + G @ du` по **измеренному** следующему состоянию и фактическому управлению. Первый переход не обновляет инкрементальную модель: предыдущее приращение неизвестно.
2. Критик сохраняет прогноз `X_next_hat = X + F @ dX + G @ du` и стоимость. Пакетный МНК оценивает `vec(P)` по квадратичным признакам; правая часть использует текущую `P`. Решение Мура–Пенроуза вычисляется через SVD, затем матрица симметризуется.
3. Улучшение политики решает систему

\[
(R+\gamma G^TPG)\Delta\delta =
-[R\delta_{k-1}+\gamma G^TPX_k+\gamma G^TPF\Delta X_k].
\]

Затем команда ограничивается по амплитуде и скорости. Ридж, PSD-проекция, сглаживание критика и альтернативная псевдообратная политика удалены. При вырожденной системе выдаётся ошибка. Старые параметры `policy_eval_regularization`, `enforce_psd`, `psd_floor`, `policy_eval_blend`, `pinv_rcond` больше не принимаются.

## Непрерывное и последовательное обучение

`IADPConfig.paper(...)` задаёт расписание опубликованного эксперимента. Прямой конструктор `IADPConfig(...)` использует тот же алгоритм с настраиваемым временем фаз.

| Настройка | Значение фабрики по умолчанию |
| --- | --- |
| Период управления/модели | `dt=0.001` с |
| Обновление критика | 20 Гц |
| Начальная разомкнутая идентификация | 20 с, переменное возбуждение задаёт пользователь |
| Окно критика | 20 с; ожидание полного окна |
| Подход | `continuous` (CLA) |
| Обучение регулятора SLA | 40 с после идентификации |
| Оценка критика SLA | Последние 5 с обучения: 55–60 с при стандартном расписании |

В CLA модель и критик продолжают адаптироваться после начальной фазы. SLA замораживает модель после идентификации, критик — после обучения: это явно выбираемый экспериментальный режим из статьи. Для неизвестных будущих отказов используйте CLA. Время отказа алгоритму не передаётся. `continuous_excitation_signal` добавляет циклическое возбуждение во время обучения регулятора; оно также проходит ограничения привода.

## Фактическое управление и инициализация

При ограничении, задержке или изменении команды приводом передавайте его фактический вход: `learn(..., applied_action=actual_input)`. Он используется в стоимости, регрессоре RLS и следующем приращении. Единицы и триммерные смещения должны совпадать; обратная связь сред B747/LAPAN возвращается в градусах. Среднее положение непрерывного сервопривода за переход — приближение эффективного входа.

Начальная матрица по умолчанию `[C,-Cr].T @ Q @ [C,-Cr] + 1e-6*I` связывает выходы объекта и задания. Это выбор реализации: полная инициализация в статье не опубликована. При наличии задайте согласованные `P_init`, `F_init`, `G_init`. Вызов `reset(initial_action=trim_input)` начинает историю управления с ненулевого трима; обученные параметры сохраняются.

При `G=0` без возбуждения управление по-прежнему остаётся нулевым. Неинформативное окно не идентифицирует всю функцию ценности. Удаление регуляризации само по себе не доказывает ограниченность параметров или устойчивость: нужны проверки масштаба признаков, возбуждения и полосы привода на конкретном самолёте. Подготовка полётных измерений и все настройки авторов не опубликованы; скалярный пример не воспроизводит их полётные испытания.

## Сохранение и переход на новый вариант

`agent.save(path)` и `IADPAgent.from_pretrained(folder)` сохраняют отображения выходов, фазу, RLS, критик и историю переходов. Конфигурации/checkpoint с удалёнными численными опциями нужно создать заново: прежнее сглаженное обновление критика нельзя продолжить как тот же алгоритм. Примеры F-16 теперь используют единый закон и требуют новой оценки; исторические графики относятся к прежним настройкам.

## API

::: tensoraerospace.agent.iadp.model.IADPAgent

::: tensoraerospace.agent.iadp.model.IADPConfig

::: tensoraerospace.agent.iadp.rls.IncrementalRLS

## Ноутбук с нелинейным B737

[Пример отработки ступеньки тангажа на B737](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_iadp_nonlinear_b737.ipynb): те же крейсерский режим и ступенька +1°, что в ноутбуке IHDP. Внешний контур тангажа задан явно; адаптация продолжается во время моделирования. Сохранены графики и метрики `ControlBenchmark`, описаны инициализация по номинальной модели и допущения упрощённых моделей. Ноутбук оформлен на английском.

## Примеры с отказами

- [B737: потеря 50% эффективности руля высоты](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_iadp_fault_b737.ipynb): непрерывное обучение, обратная связь по фактическому углу руля и метрики после отказа.
