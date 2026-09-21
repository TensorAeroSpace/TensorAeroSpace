# iADP: непрерывное обучение через SDK

Пример полностью показывает настройку `IADPAgent`, переходы объекта, вызовы
`predict`/`learn`, расчёт метрик `ControlBenchmark` и графики. Все импорты относятся
к установленному `tensoraerospace` и обычным библиотекам NumPy/Matplotlib.

Объект задан уравнением `x_dot = -2*x + effectiveness*u`. На 60 с эффективность
входа уменьшается с 1 до 0.7. Задание — 0.05 рад/с. Первые 20 с отведены
идентификации с возбуждением; далее идентификатор и критик продолжают обновляться.
Регулятор не получает время отказа и истинный коэффициент объекта.

[Выполненный ноутбук](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_iadp_paper.ipynb) ·
[подробный cookbook](../../../cookbook/06_online_adaptive.md).

## Полный цикл и графики

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

## Интерпретация

Метрики считаются по наблюдениям на (40, 60] с и (65, 80] с. Они исключают
начальное возбуждение, а второе окно — ещё и первые 5 с после изменения объекта.
Разная история адаптации объясняет, почему эти RMSE нельзя трактовать как
изолированное влияние отказа. Истинный входной коэффициент на графике доступен
только для оценки: агент идентифицирует его по переходам.

## Перейдите к самолётным примерам

- [B737: ступенька тангажа](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_iadp_nonlinear_b737.ipynb) — штатная среда, инициализация по балансировке, внешний контур тангажа и полный цикл SDK.
- [B737: потеря эффективности руля](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_iadp_fault_b737.ipynb) — непрерывное обучение и отдельные метрики после отказа.
- [F-16: измеренное положение привода и отказы](example_iadp_small_fault_f16.md) — учитывайте фактическое время завершения длинных прогонов.

[Уравнения, конфигурация и API](../../../agent/iadp.md).
