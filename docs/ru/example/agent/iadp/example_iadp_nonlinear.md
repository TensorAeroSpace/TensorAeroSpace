# Пример: iADP на нелинейной F-16 — слежение за угловой скоростью тангажа с инъекцией отказа

Пример запускает агента [**iADP**](../../../agent/iadp.md) на синусоидальной команде по угловой скорости тангажа на [нелинейной продольной модели F-16](../../../model/f16_nonlinear_longitudinal.md) и демонстрирует восстановление регулятора после отказа привода (потеря 50 % эффективности руля в середине эпизода). Исходный ноутбук: `example/reinforcement_learning/example_iadp_nonlinear_f16.ipynb`.

## Ключевая идея

iADP объединяет **онлайн-RLS-идентификатор** инкрементной модели \(\Delta X_{t+1} \approx \tilde{F} \Delta X_t + \tilde{G} \Delta\delta_t\) с **пакетным МНК для политики оценки** квадратичной функции ценности \(V_\pi(X_t) = X_t^T \tilde{P} X_t\). Политика — аналитический минимизатор правой части уравнения Беллмана (eq. (11)):

\[
\Delta\delta_t = -(R + \gamma\,\tilde{G}^T \tilde{P} \tilde{G})^{-1}\bigl[R\,\delta_{t-1} + \gamma\,\tilde{G}^T \tilde{P} X_t + \gamma\,\tilde{G}^T \tilde{P} \tilde{F} \Delta X_t\bigr].
\]

Поскольку политика оценивается по *идентифицированной* модели, внезапное уменьшение эффективности привода в два раза отрабатывается автоматически — RLS обновляет \(\tilde{G}\), МНК переподбирает \(\tilde{P}\), политика следует. Никакой явной логики обнаружения отказов, никакой перетюнинг.

## 1. Импорты и трим

```python
import math
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import fsolve

import tensoraerospace
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import f16_ode_long
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params import default_parameters
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig

dt = 0.01
params = default_parameters()

def trim_residual(z):
    alpha, stab = z
    x = np.array([alpha, 0.0, stab, 0.0])
    return list(f16_ode_long(x, np.array([stab]), 0.0, params)[:2])

sol, *_ = fsolve(trim_residual, x0=[math.radians(2.0), math.radians(-2.0)], full_output=True)
alpha_trim_rad, stab_trim_rad = float(sol[0]), float(sol[1])
# глобальный трим: α = +4.918°, руль = -4.447°
```

## 2. Конструктор среды

```python
def make_env(n_steps):
    env = gym.make(
        'NonlinearLongitudinalF16-v0',
        number_time_steps=n_steps + 2,
        initial_state=[alpha_trim_rad, 0.0, stab_trim_rad, 0.0],
        reference_signal=np.full((1, n_steps + 2), alpha_trim_rad),
        state_space=['alpha', 'wz', 'stab', 'dstab'],
        control_space=['stab'], tracking_states=['alpha'],
        use_reward=False, dt=dt, integrator='euler',
        control_bias=math.degrees(stab_trim_rad),
    ).unwrapped
    env.reset()
    return env
```

## 3. Warm-start `F_init`, `G_init` по короткому PE-возбуждению

Политика iADP (eq. (11)) вырождена, пока \(\tilde{G}\) не имеет нетривиальной оценки — при \(\tilde{G} \approx 0\) закон управления сводится к \(\Delta\delta_t \approx -\delta_{t-1}\), что фиксирует привод и лишает RLS сигнала для идентификации. Поэтому запускаем короткое мультисинусоидное возбуждение вокруг трима и офлайн подбираем дискретную скалярную форму \(\Delta\omega_{z,t+1} \approx F \Delta\omega_{z,t} + G \Delta\delta_t\).

```python
N_PE = 300
env_pe = make_env(N_PE); obs, _ = env_pe.reset()
wz_hist, u_hist = [float(obs[1])], [0.0]
for t in range(N_PE):
    u = 2.0*math.sin(2*math.pi*0.7*t*dt) + 1.0*math.sin(2*math.pi*1.5*t*dt)
    obs, *_ = env_pe.step(np.array([u]))
    wz_hist.append(float(obs[1])); u_hist.append(float(u))

wz = np.asarray(wz_hist); us = np.asarray(u_hist)
dwz, du = np.diff(wz), np.diff(us)
A_pe = np.column_stack([dwz[:-1], du[:-1]])
F_wz, G_wz = np.linalg.lstsq(A_pe, dwz[1:], rcond=None)[0]

F_init = np.array([[F_wz, 0.0], [0.0, 1.0]])     # строка референса стационарна
G_init = np.array([[G_wz], [0.0]])               # руль влияет только на строку ω_z
# PE: F_wz ≈ 1.00, G_wz ≈ -0.0014 (per °, дискретное время)
```

## 4. Warm-start `P_init` по дискретному LQT-DARE

Пакетный МНК policy-evaluator подгоняет \(\tilde{P}\) инкрементно на окнах переходов. На чистых синтетических данных он сходится к LQT-решению на бесконечном горизонте, но на реальной траектории МНК смещён конечным окном и обеднён кросс-информацией между \(x\) и \(x_r\), когда референс держится около одного уровня — политика работает, но стабильно на несколько процентов ниже оптимального коэффициента.

Обходим это, вычисляя офлайн аналитическое решение **дискретного алгебраического уравнения Риккати (DARE)** по warm-start модели и используя его как `P_init`. Дисконт \(\gamma\) заходит через стандартную подстановку \(\bar{F} = \sqrt{\gamma}F,\, \bar{G} = \sqrt{\gamma}G\) — так что `scipy.linalg.solve_discrete_are` применяется напрямую:

```python
Q_val = 30_000.0       # вес ошибки слежения
R_val = 0.1            # вес управления
gamma = 0.9            # дисконт Беллмана

# Стоимость (wz - wz_ref)^2 · Q  =  X' · Q_aug · X на дополненном состоянии.
Q_aug = Q_val * np.array([[1.0, -1.0], [-1.0, 1.0]])
R_aug = np.array([[R_val]])

P_dare = solve_discrete_are(
    np.sqrt(gamma) * F_init, np.sqrt(gamma) * G_init,
    Q_aug, R_aug,
)
```

С хорошо засеянной \(\tilde{P}\) онлайн-МНК только *поддерживает* её в актуальном состоянии, не ищет с нуля — базовый RMSE на этом объекте падает с ≈ 0.13 °/с (ad-hoc блочная инициализация) до ≈ 0.09 °/с.

## 5. Плавное обновление \(\tilde{P}\): `policy_eval_blend`

По умолчанию пакетный МНК *заменяет* \(\tilde{P}\) каждые `policy_eval_every` тиков. На реальном объекте это маленький **ступенчатый скачок**, и поскольку политика линейна по \(\tilde{P}\), командный руль высоты показывает заметный **пилообразный эффект** на каждом такте обновления. Без сглаживания пиковый скачок на этом демо — ≈ 0.14° на тик, прямо видно на графике.

`policy_eval_blend` подмешивает свежее МНК-решение в действующую \(\tilde{P}\) через экспоненциальное скользящее среднее,

\[
\tilde{P} \leftarrow \beta\,\tilde{P}_\text{solved} + (1-\beta)\,\tilde{P}_\text{prev}.
\]

При малом \(\beta\) и более частом `policy_eval_every` матрица \(\tilde{P}\) становится сглаженным фильтром МНК-решения вместо серии ступенчатых изменений — это аналог *soft-target* трюка из RL. Два немедленных выигрыша на этом объекте:

- **Гладкость управления.** Пиковый скачок за тик \(|\Delta\delta|\) падает в ~14× (с 0.14° до 0.01°) — пилообразный эффект на трассе руля исчезает.
- **Выше допустимый коэффициент.** Тот же `Q`, который раньше вызывал нестабильность «заменили-и-насытили», теперь безопасен. Можно поднять `Q` с 10 000 до 30 000 — это ещё на ~25 % уменьшает ошибку слежения.

## 6. Harness для замкнутого контура

```python
def run_iadp(wz_cmd, n_steps, *, fault_gain=1.0, fault_at_step=None):
    cfg = IADPConfig(
        dt=dt,
        Q=np.array([[Q_val]]), R=np.array([[R_val]]),
        gamma=gamma, gamma_rls=0.9999, phi_init=1.0,
        policy_eval_window=300, policy_eval_every=5,
        policy_eval_warmup_updates=20,
        policy_eval_regularization=1e-10,
        policy_eval_blend=0.10,        # EMA soft-update для P̃
        F_init=F_init, G_init=G_init,
        P_init=P_dare,
        u_magnitude_limit=8.0, u_rate_limit=200.0,
        seed=0,
    )
    agent = IADPAgent(n_state=1, n_control=1, config=cfg)
    env = make_env(n_steps); obs, _ = env.reset()
    logs = {k: [] for k in ('wz', 'u_agent', 'u_applied', 'G_est', 'residual')}
    current_gain = 1.0
    for k in range(n_steps):
        if fault_at_step is not None and k >= fault_at_step:
            current_gain = fault_gain
        u_agent = agent.predict(np.array([float(obs[1])]),
                                np.array([wz_cmd[k]]), k)
        u_applied = u_agent * current_gain
        obs, *_ = env.step(u_applied)
        m = agent.learn(np.array([float(obs[1])]),
                        np.array([wz_cmd[k]]), k)
        logs['wz'].append(float(obs[1]))
        logs['u_agent'].append(float(u_agent[0]))
        logs['u_applied'].append(float(u_applied[0]))
        logs['G_est'].append(float(agent.G[0, 0]))
        logs['residual'].append(m['rls_pred_error_norm'])
    return {k: np.asarray(v) for k, v in logs.items()}
```

!!! warning "Масштабируйте `policy_eval_regularization` под величину состояния"
    Для угловых скоростей в рад/с (≈ 0.01) элементы внешнего произведения регрессора ≈ \(10^{-4}\). Ридж по умолчанию `1e-4` подавит нормальные уравнения и обнулит \(\tilde{P}\) — политика перестанет выдавать управление. Здесь задаём `1e-10`; PSD-проекция сохраняет численную устойчивость.

## 7. Базовое слежение — синусоидальная команда по угловой скорости

```python
N = 1800
t_arr = np.arange(N) * dt
wz_cmd = math.radians(0.8) * np.sin(2*math.pi*0.12*t_arr)  # амплитуда 0.8 °/с, T ≈ 8.3 с
baseline = run_iadp(wz_cmd, N)
# RMSE на окне t ≥ 5 с:  ≈ 0.072 °/с  (9.0 % от амплитуды)
# Пиковый |Δδₑ| за тик:  ≈ 0.010 °
# Финальный G̃:           ≈ -0.00013
```

Политика отрабатывает синусоиду \(\pm 0.8\) °/с в точке трима F-16 с RMSE около **0.07 °/с** (≈ 9 % от амплитуды). Команда на руль не выходит за \(\pm 0.5\)° и меняется максимум на **0.01° за тик** — пилообразный эффект версии без смешивания пропал.

## 8. Инъекция отказа — 50 % потери эффективности руля в t = 10 с

```python
fault_step = int(10.0 / dt)
fault = run_iadp(wz_cmd, N, fault_gain=0.5, fault_at_step=fault_step)
# RMSE до отказа (5 с ≤ t < 10 с):   ≈ 0.090 °/с
# RMSE после (11 с ≤ t ≤ 18 с):      ≈ 0.098 °/с
```

В момент \(t = 10\,\text{с}\) среда умножает команду агента на `0.5` перед передачей в объект. Политика iADP опирается на *идентифицированную* \(\tilde{G}\); по мере роста невязки RLS и смещения \(\tilde{G}\) аналитическое выражение eq. (11) увеличивает \(\Delta\delta\) для компенсации. Слежение слабо деградирует в переходном процессе и восстанавливается после сходимости RLS.

| Метрика | Базовый | С отказом |
|---|---|---|
| RMSE на позднем окне (ω_z) | ≈ 0.07 °/с | ≈ 0.10 °/с (после) |
| Пиковый \|Δδ\| (агент) | ≈ 0.5 ° | ≈ 0.5 ° |
| Пиковый \|Δδ\| за тик | ≈ 0.01 ° | ≈ 0.01 ° |
| Поведение при отказе | n/a | Всплеск невязки RLS, сдвиг G̃, слежение восстановлено |

## Заметки

- **Warm-start критичен.** Случайная \(\tilde{G}\) приведёт либо к коллапсу \(\Delta\delta \to -\delta_{t-1}\), либо к загону привода в лимитный цикл. Короткое (2–3 с) офлайн-PE-возбуждение (как здесь) или бортовая линеаризация в реальном деплое достаточно.
- **Warm-start `P_init` по DARE** — самый большой выигрыш по точности после базовой настройки (Q, R, γ). Ставит политику почти в оптимум с шага 0, вместо ожидания, пока онлайн-МНК сам соберёт \(\tilde{P}\).
- **Soft-update (`policy_eval_blend`)** убирает пилообразный эффект на командном руле, который иначе возникает на каждом такте пакетного МНК, и позволяет поднять вес ошибки `Q` в 3 раза без дестабилизации контура. И точность, и гладкость улучшаются одновременно.
- **Ридж policy-evaluation масштабируется с состоянием.** При скоростях или углах порядка O(1) дефолтное `1e-4` нормально; при угловых скоростях в рад/с нужно уменьшить на несколько порядков.
- **Нет внешнего PI-контура.** В отличие от AA-INDI (где вокруг INDI-инверсии закрывается PI), квадратичная функция ценности iADP *и есть* внешний контур — кнопки `(Q, R, γ)` задают компромисс «точность/расход управления» напрямую.
- **Остаточные ~9 %** от амплитуды — это внутренний предел фазового отставания политики iADP (пропорциональная + \(\delta^2\)-штраф) на 0.12 Гц-синусоиде. Опускаться ниже — задача расширений за пределами статьи (feedforward, дополнение производной референса, интегральные члены).

## См. также

- [Документация iADP](../../../agent/iadp.md) — теория, полный API, референс гиперпараметров.
- [AA-INDI на нелинейной F-16](../aa_indi/example_aaindi_nonlinear.md) — INDI-альтернатива на том же объекте и том же профиле отказа.
- [IM-GDHP на нелинейной F-16](../imgdhp/example_imgdhp_nonlinear.md) — инкрементный подход на основе GDHP.
- [ET-DHP на нелинейной F-16](../et_dhp/example_etdhp_nonlinear.md) — событийный DHP.
