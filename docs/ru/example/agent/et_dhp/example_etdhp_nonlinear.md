# Пример: ET-DHP на нелинейной F-16 — слежение за синусоидальным α

Пример обучает агент **Event-Triggered Dual Heuristic Programming (ET-DHP)** на том же задании, что и пример IHDP: синусоида по углу атаки амплитудой 3° и частотой 0.1 Гц на [нелинейной модели F-16](../../../model/f16_nonlinear_longitudinal.md). Оба реактивных агента можно сравнивать на равных, потому что у них одинаковый глобальный трим, одинаковый inverse-model feedforward и одинаковый 80-секундный задающий сигнал. Исходный ноутбук: `example/reinforcement_learning/incremental_adp/example_etdhp_nonlinear_f16.ipynb`.

**Источник:** Bo Sun, Cheng Liu, Killian Dally, Erik-Jan van Kampen. *"Intelligent Aircraft Stabilization Control with Event-Triggered Scheme"*, CEAS EuroGNC 2022.

## Стратегия слежения

ET-DHP — регулятор (\(u(0)=0\) по построению), поэтому для слежения за движущимся заданием мы:

- передаём среде **feedforward trim-кривую** \(\delta_e^{\text{trim}}(\alpha_{\text{ref}})\), удерживающую самолёт на текущем задающем α,
- предоставляем агенту выучить **residual** поверх этого FF,
- определяем регуляционное состояние через **ошибку слежения**: \(\tilde x = [\alpha - \alpha_{\text{ref}}(t),\, w_z,\, \delta_{\text{stab}} - \delta_e^{\text{trim}}(\alpha_{\text{ref}}(t)),\, \dot\delta_{\text{stab}}]\) в градусах.

Когда слежение идеально, \(\tilde x = 0\) и агент отдаёт нулевой residual — FF удерживает задание один.

## 1. Импорты, трим, FF-кривая

```python
import math
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from scipy.optimize import fsolve

import tensoraerospace.envs
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import f16_ode_long
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params import default_parameters
from tensoraerospace.agent.et_dhp import ETDHPAgent, ETDHPConfig
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period

params = default_parameters()

def trim_residual(z):
    alpha, stab = z
    x = np.array([alpha, 0.0, stab, 0.0])
    return list(f16_ode_long(x, np.array([stab]), 0.0, params)[:2])

sol, *_ = fsolve(trim_residual, x0=[math.radians(2.0), math.radians(-2.0)], full_output=True)
alpha_trim_rad, stab_trim_rad = float(sol[0]), float(sol[1])
alpha_trim_deg = math.degrees(alpha_trim_rad)
stab_trim_deg = math.degrees(stab_trim_rad)

def stab_for_alpha(alpha_rad):
    def res(s):
        x = np.array([alpha_rad, 0.0, s[0], 0.0])
        return f16_ode_long(x, np.array([s[0]]), 0.0, params)[1]
    return float(fsolve(res, x0=[math.radians(-2.0)])[0])

alphas_grid_deg = np.linspace(-12.0, 17.0, 61)
stabs_grid_deg = np.array([math.degrees(stab_for_alpha(a)) for a in np.deg2rad(alphas_grid_deg)])
```

![Feedforward trim-кривая](img/etdhp_ff_curve.png)

## 2. Задающий сигнал

80-секундный эпизод при \(dt = 0.01\) с. Первые 2 с — удержание на триме, затем синусоида 3° / 0.1 Гц центрированная на триме — то же задание, что и в примере IHDP.

```python
dt = 0.01
tp = generate_time_period(tn=80, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)

amplitude_deg = 3.0; freq_hz = 0.1
warmup_steps = int(2.0 / dt)
ref_alpha_rad = np.full(number_time_steps, alpha_trim_rad)
active_t = np.arange(number_time_steps - warmup_steps) * dt
ref_alpha_rad[warmup_steps:] = (
    alpha_trim_rad + math.radians(amplitude_deg) * np.sin(2 * np.pi * freq_hz * active_t)
)
reference_signals = ref_alpha_rad.reshape(1, -1)
```

![Задающий сигнал](img/etdhp_reference.png)

## 3. Feedforward, преобразование состояния, среда

```python
lookahead_sec = 0.85
lookahead_steps = int(lookahead_sec / dt)
ff_gain = 1.55

def feedforward_fn(time_step, ref_signal):
    k = min(time_step + lookahead_steps, ref_signal.shape[1] - 1)
    alpha_ref_deg = math.degrees(ref_signal[0, k])
    alpha_eff_deg = alpha_trim_deg + ff_gain * (alpha_ref_deg - alpha_trim_deg)
    return float(np.interp(alpha_eff_deg, alphas_grid_deg, stabs_grid_deg))


def state_transform(obs, ref, ts):
    """Преобразование сырого [alpha, wz, stab, dstab] (рад) в регуляционное состояние (град).

    Компоненты α и δₑ перецентрируются на текущий задающий сигнал, так что
    x̃ = 0 означает идеальное слежение.
    """
    obs_deg = np.degrees(np.asarray(obs, dtype=np.float64).reshape(-1))
    if ref is not None and int(ts) < ref.shape[1]:
        alpha_ref_deg_now = math.degrees(float(ref[0, int(ts)]))
    else:
        alpha_ref_deg_now = alpha_trim_deg
    stab_ref_deg_now = float(np.interp(alpha_ref_deg_now, alphas_grid_deg, stabs_grid_deg))
    return np.array([
        obs_deg[0] - alpha_ref_deg_now,
        obs_deg[1],
        obs_deg[2] - stab_ref_deg_now,
        obs_deg[3],
    ])


def make_env():
    return gym.make(
        'NonlinearLongitudinalF16-v0',
        number_time_steps=number_time_steps,
        initial_state=[alpha_trim_rad, 0.0, stab_trim_rad, 0.0],
        reference_signal=reference_signals,
        state_space=['alpha', 'wz', 'stab', 'dstab'],   # полное 4-мерное наблюдение
        control_space=['stab'],
        tracking_states=['alpha'],
        use_reward=False,
        dt=dt,
        integrator='euler',
        feedforward_fn=feedforward_fn,
    ).unwrapped
```

4-мерное наблюдение важно при \(dt = 0.01\) с: при только `[alpha, wz]` эффективность управления на одном шаге доминируется запаздыванием привода, и нейросетевой модели объекта не удаётся выделить канал команды руля из шума. Добавление актуаторных состояний `[stab, dstab]` восстанавливает чистый якобиан управления.

## 4. PE-данные и предобучение модели объекта

Модель объекта предобучается на 30-секундном многочастотном PE-возбуждении около трима (FF отключён, постоянный bias трима):

```python
N_ID = 3000
env_id = gym.make(
    'NonlinearLongitudinalF16-v0', number_time_steps=N_ID + 2,
    initial_state=[alpha_trim_rad, 0.0, stab_trim_rad, 0.0],
    reference_signal=np.full((1, N_ID + 2), alpha_trim_rad),
    state_space=['alpha', 'wz', 'stab', 'dstab'], control_space=['stab'],
    tracking_states=['alpha'], use_reward=False, dt=dt, integrator='euler',
    control_bias=stab_trim_deg,
).unwrapped

rng = np.random.default_rng(0)
states_buf, actions_buf, next_states_buf = [], [], []
ref_at_trim = np.full((1, 1), alpha_trim_rad)
obs, _ = env_id.reset()
for t in range(N_ID):
    u_t = 2.0 * (
        0.6 * np.sin(2*np.pi*0.3*t*dt)
        + 0.3 * np.sin(2*np.pi*0.9*t*dt)
        + 0.1 * rng.normal()
    )
    x_curr = state_transform(obs, ref_at_trim, 0)
    obs_next, _, done, _, _ = env_id.step(np.array([u_t]))
    x_next = state_transform(obs_next, ref_at_trim, 0)
    states_buf.append(x_curr); actions_buf.append([u_t]); next_states_buf.append(x_next)
    obs = obs_next
    if done: break

states_arr = np.asarray(states_buf, dtype=np.float32)
actions_arr = np.asarray(actions_buf, dtype=np.float32)
next_states_arr = np.asarray(next_states_buf, dtype=np.float32)

cfg = ETDHPConfig(
    actor_hidden=(24, 24), critic_hidden=(24, 24), model_hidden=(24, 24),
    actor_lr=1e-3, critic_lr=1e-3, model_lr=5e-3,
    model_epochs=300,
    Q=[10.0, 0.1, 0.0, 0.0],
    R=[1.0],
    gamma=0.95,
    num_epochs_per_trigger=3,
    u_bound=2.0,
    rho=0.2,
    trigger_floor=0.1,
    weight_init_scale=0.2,
    seed=0,
)
agent = ETDHPAgent(n_state=4, n_control=1, state_transform=state_transform, config=cfg)
model_losses = agent.fit_plant_model(states_arr, actions_arr, next_states_arr,
                                     batch_size=128, verbose=True)
```

![Потери предобучения модели](img/etdhp_plant_model.png)

## 5. Обучение в замкнутом контуре с событийным триггером

**Без force-trigger.** Липшицев триггер сам решает, когда выполнять обновления, поэтому веса меняются только когда ошибка слежения растёт выше порога. Это естественным образом разрывает положительную обратную связь между шумным критиком и необученным актором, которая иначе разваливает обычный временной цикл на почти-решённой задаче.

```python
NUM_EPISODES = 12
train_log = {'ep': [], 'mae_late': [], 'rmse_late': [], 'triggers': []}

for ep in range(NUM_EPISODES):
    env = make_env()
    obs, _ = env.reset()
    agent.reset()
    n_trig = 0
    for k in range(number_time_steps - 2):
        agent.predict(obs, reference_signals, k)
        u_cmd = agent.last_action()
        obs_next, _, done, _, _ = env.step(u_cmd)
        m = agent.learn(obs_next, reference_signals, k, dt=dt)
        n_trig += int(m['triggered'])
        obs = obs_next
        if done: break
    # ... детерминированная оценка (без learn) для метрик на эпизод ...
```

Ошибки слежения и число срабатываний по эпизодам:

![Прогресс обучения: ошибка и число срабатываний](img/etdhp_training.png)

Число срабатываний падает с нескольких сотен в эпизоде в начале до ~100 после стабилизации — чем лучше контур замкнут, тем меньше событий нужно.

## 6. Финальная оценка слежения

Детерминированный прогон на том же 80-секундном задании:

![Итоговое слежение ET-DHP + FF](img/etdhp_tracking.png)

| Конфигурация | Late-half MAE | RMSE | Макс. |
|---|---|---|---|
| Только FF (случайный актор) | 0.27° | 0.33° | 0.59° |
| **ET-DHP + FF** | **0.13°** | **0.16°** | **0.30°** |

Обученный residual улучшает FF-базу примерно на 50 %.

## Итоги

- **Та же постановка, что в IHDP.** Глобальный трим, inverse-model feedforward с lookahead + gain, 3° sin / 0.1 Гц, 80-секундный эпизод.
- **FF + residual-регулятор.** Среда ведёт самолёт по FF-рулю под текущий задающий α, а ET-DHP обучает небольшой residual. Поскольку \(u(0) = 0\) зашито в актора, идеальное слежение соответствует нулевому residual.
- **Естественный event-триггер.** Липшицево правило само решает, когда срабатывать — первые эпизоды содержат по несколько сотен срабатываний, поздние стабилизируются на ~100 за 80 с.
- **Качество слежения.** Late-half MAE ≈ 0.13° (около 4 % от амплитуды 3°), сопоставимо с IHDP-примером (~0.05°) на том же задании, при этом event-триггер экономит заметную часть обновлений actor/critic.

### Замечания по гиперпараметрам

- `u_bound = 2.0°` — малое, потому что работа агента — снять доли градуса residual с почти идеального FF.
- `Q = [10, 0.1, 0, 0]` — большой вес на ошибку α, умеренный на \(w_z\), ноль на актуаторные состояния, чтобы регулятор не боролся с физикой привода.
- `rho = 0.2`, `trigger_floor = 0.1°` — не выполнять обновления, пока ошибка слежения в пределах 0.1°-полосы шума; за ней — срабатывать активнее.
