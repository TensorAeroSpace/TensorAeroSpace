# ImprovedUltrastickEnv -- БПЛА Ultrastick-25e, отслеживание тангажа (нормализованная, RL-совместимая)

На этой странице описана среда `ImprovedUltrastickEnv`, предназначенная для обучения и оценки контроллеров на основе обучения с подкреплением в продольном канале БПЛА Ultrastick-25e. В отличие от большинства других улучшенных сред в этой библиотеке, данная среда имеет **2D пространство действий** (руль высоты + тяга), 5D вектор наблюдений, бонус формирования прогресса, а также ограничение скорости и фильтрацию нижних частот для руля высоты.

Эталонная реализация находится в `tensoraerospace/envs/ultrastick.py` (класс `ImprovedUltrastickEnv`).

## Краткое описание

- Пространство наблюдений: 5D, нормализовано в [-1, 1]
- Пространство действий: 2D, нормализовано в [-1, 1] (отклонение руля высоты + команда тяги)
- Цель: отслеживание изменяющегося во времени задания по тангажу при сохранении плавности управления
- Награда: квадратичные штрафы за ошибку отслеживания тангажа, рассогласование угловой скорости, величины руля высоты и тяги, их скоростей и рывков, плюс бонус за прогресс
- Завершение: выход за безопасный диапазон тангажа или достижение горизонта по времени

## Наблюдение и действие

Пусть \(\theta\) -- текущий угол тангажа (рад), \(q\) -- угловая скорость тангажа (рад/с), а \(\theta_{ref}(t)\) -- целевой тангаж.

Наблюдение на шаге \(t\) представляет собой 5-мерный вектор:

\[
\mathbf{o}_t = \big[\underbrace{\mathrm{clip}\!\Big(\frac{\theta_{ref}(t) - \theta(t)}{\theta_{max}},\,-1,\,1\Big)}_{\text{[0] norm\_pitch\_error}},\; \underbrace{\mathrm{clip}\!\Big(\frac{q(t)}{q_{max}},\,-1,\,1\Big)}_{\text{[1] norm\_q}},\; \underbrace{\mathrm{clip}\!\Big(\frac{\theta(t)}{\theta_{max}},\,-1,\,1\Big)}_{\text{[2] norm\_\(\theta\)}},\; \underbrace{u_{e,t-1}}_{\text{[3] prev elev}},\; \underbrace{u_{\delta_t,t-1}}_{\text{[4] prev throttle}}\big]
\]

где

- \(\theta_{max} = 30°\) (внутри преобразуется в радианы),
- \(q_{max} = 30°/\text{s}\) (внутри преобразуется в рад/с),
- \(u_{e,t-1} \in [-1, 1]\) -- предыдущая нормализованная команда руля высоты,
- \(u_{\delta_t,t-1} \in [-1, 1]\) -- предыдущая нормализованная команда тяги.

**Примечание**: Порядок выходных переменных модели Ultrastick: `[Va, alpha, theta, q, h]`. Среда извлекает \(\theta\) по индексу 2 и \(q\) по индексу 3.

Действие -- 2D нормализованная команда \(\mathbf{a}_t = [a_e, a_{\delta_t}] \in [-1, 1]^2\):

- **Руль высоты** (\(a_e\)): отображается в градусы с фильтрацией нижних частот и ограничением скорости:
  \[
  \delta_e^{pre} = \alpha \cdot a_e \cdot \delta_{e,max}^° + (1-\alpha) \cdot \delta_{e,t-1}^°, \quad \alpha = 0.5, \quad \delta_{e,max}^° = 15°,
  \]
  затем ограничивается скоростью \(\pm 300 \cdot \Delta t\) град за шаг и преобразуется в радианы для модели.
- **Тяга** (\(a_{\delta_t}\)): отображается из \([-1, 1]\) в физическую тягу \([0, 1]\):
  \[
  \delta_t(t) = 0.5 \cdot (a_{\delta_t} + 1).
  \]

## Функция награды

Среда использует формированную награду с восемью квадратичными штрафными членами (отдельно для руля высоты и тяги) и бонусом формирования прогресса. Награда за шаг:

\[
r_t = -s \cdot C_t + P_t,
\]

где стоимость:

\[
C_t = w_\theta\,e_\theta^2 + w_q\,e_q^2 + w_{u_e}\,u_e^2 + w_{u_{\delta_t}}\,u_{\delta_t}^2 + w_{\Delta_e}\,(\Delta u_e)^2 + w_{\Delta_{\delta_t}}\,(\Delta u_{\delta_t})^2 + w_{\Delta^2_e}\,(\Delta^2 u_e)^2 + w_{\Delta^2_{\delta_t}}\,(\Delta^2 u_{\delta_t})^2,
\]

а бонус за прогресс:

\[
P_t = k_p \cdot \Big[\big(e_{\theta,t-1}^2 + e_{q,t-1}^2\big) - \big(e_\theta^2 + e_q^2\big)\Big].
\]

Веса и масштаб по умолчанию:

\[
\begin{aligned}
&w_\theta=8.0,\quad w_q=0.5, \\
&w_{u_e}=0.003,\quad w_{u_{\delta_t}}=0.001, \\
&w_{\Delta_e}=0.003,\quad w_{\Delta_{\delta_t}}=0.002, \\
&w_{\Delta^2_e}=0.0005,\quad w_{\Delta^2_{\delta_t}}=0.0003, \\
&s=0.08,\quad k_p=0.05.
\end{aligned}
\]

Члены ошибки определяются как

\[
e_\theta = \frac{\theta(t)-\theta_{ref}(t)}{\theta_{max}},\qquad
e_q = \frac{q(t)-\dot\theta_{ref}(t)}{q_{max}},
\]

где \(\dot\theta_{ref}(t)\) -- конечно-разностная производная эталонного тангажа, вычисленная с шагом \(\Delta t\). Члены гладкости воздействия используют

\[
\Delta u_t = u_t - u_{t-1},\qquad \Delta^2 u_t = u_t - 2 u_{t-1} + u_{t-2},
\]

вычисляемые отдельно для каналов руля высоты и тяги.

Замечания:

- Бонус за прогресс \(P_t\) даёт положительную награду при уменьшении ошибок отслеживания и отрицательную при их увеличении, ускоряя сходимость.
- Фильтр нижних частот руля высоты (\(\alpha=0.5\)) и ограничение скорости (\(300°/\text{s}\)) сглаживают резкие команды, предотвращая высокочастотный дребезг.
- Раздельные веса для руля высоты и тяги позволяют тонко настраивать штрафование каждого канала привода.

## Завершение и усечение

- **Аварийное завершение**: если \(|\theta| > \theta_{max}\), эпизод завершается немедленно (явный штраф не добавляется, помимо потерянной будущей награды; флаг terminated сигнализирует о неудаче).
- **Усечение**: эпизод усекается при достижении заданного горизонта (число временных шагов).

## Динамика эпизода

На каждом шаге:

1. Агент выдаёт \(\mathbf{a}_t = [a_e, a_{\delta_t}] \in [-1, 1]^2\).
2. Среда применяет фильтрацию нижних частот и ограничение скорости к команде руля высоты, отображает тягу в \([0, 1]\) и продвигает внутреннюю модель Ultrastick с \([\delta_e\text{ (рад)}, \delta_t]\).
3. Наблюдение \(\mathbf{o}_{t+1}\) строится с использованием нормализованных сигналов.
4. Награда \(r_t\) вычисляется по формуле выше.
5. Проверяются условия завершения/усечения.
6. Словарь `info` содержит диагностические значения: `elevator_deg`, `throttle`, `reward`, `cost`, `progress` и квадратичные ошибки по каждому слагаемому.

## Пример использования

```python
import numpy as np
from tensoraerospace.envs.ultrastick import ImprovedUltrastickEnv
from tensoraerospace.signals.standard import sinusoid_vertical_shift
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp

dt = 0.01
tn = 200
tp = generate_time_period(tn=tn, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)

# Эталон: плавный синусоидальный тангаж в радианах (амплитуда 1 град)
reference_signal = np.reshape(
    sinusoid_vertical_shift(
        tp=np.asarray(tps), frequency=0.05, amplitude=np.deg2rad(1.0), vertical_shift=0.0
    ),
    (1, -1),
)

# Начальное состояние: [u, w, q, theta, h] -- полное состояние Ultrastick
initial_state = np.array([0, 0, 0, 0, 0], dtype=np.float32)

env = ImprovedUltrastickEnv(
    initial_state=initial_state,
    reference_signal=reference_signal,
    number_time_steps=number_time_steps,
    initial_elevator_deg=0.0,
    initial_throttle=0.0,
    use_initial_action_on_first_step=True,
    dt=dt,
)

obs, info = env.reset()
done = False
while not done:
    # простое пропорциональное управление: руль высоты по ошибке тангажа, тяга нейтральная
    elev_cmd = float(np.clip(2.0 * float(obs[0]), -1.0, 1.0))
    thr_cmd = 0.0  # нейтральная тяга
    action = np.array([elev_cmd, thr_cmd], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
```

## Замечания по реализации

- Среда надёжно приводит входящие действия к форме `(2,)` через внутренний помощник `_to_norm_action`. Передача скаляра или 1D действия будет дополнена предыдущим значением тяги.
- Границы нормализации наблюдений (\(\theta_{max} = 30°\), \(q_{max} = 30°/\text{s}\)) шире, чем в других средах, для учёта диапазона динамики БПЛА.
- Веса награды доступны как атрибуты экземпляра (`w_theta`, `w_q`, `w_action_elev`, `w_action_thr`, `w_smooth_elev`, `w_smooth_thr`, `w_jerk_elev`, `w_jerk_thr`, `reward_scale`, `k_progress`) и могут быть изменены для соответствия конкретным целям управления.
- Модель Ultrastick принимает 2D вход `[elevator (рад), throttle]` на каждом шаге; среда выполняет все преобразования автоматически.

## Литература

- `tensoraerospace/envs/ultrastick.py` -- полная реализация среды
- Модель динамики Ultrastick-25e: `tensoraerospace/aerospacemodel/ultrastick.py`
