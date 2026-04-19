# Active-Adaptive Incremental Nonlinear Dynamic Inversion (AA-INDI)

AA-INDI — **отказоустойчивый регулятор** для управления полётом, построенный на основе Incremental Nonlinear Dynamic Inversion (INDI). Сочетает классический INDI-закон с онлайн-идентификацией матрицы эффективности управления методом **Variable-Forgetting-Factor RLS** (что даёт быструю адаптацию к отказам приводов), и лёгким сенсорным фильтром, имитирующим блок OTSEKF-HOSM из исходной статьи. См. также нелинейную модель F-16: [NonlinearLongitudinalF16](../model/f16_nonlinear_longitudinal.md).

**Источник**: Sun et al., *"Active Incremental Nonlinear Dynamic Inversion for Sensor and Actuator Fault Diagnosis and Fault-Tolerant Flight Control"*, TU Delft Aerospace, [research.tudelft.nl](https://research.tudelft.nl/en/publications/active-incremental-nonlinear-dynamic-inversion-for-sensor-and-act/).

## Ключевые идеи

- **INDI-закон**: приращение управления \(\Delta u = G^+ \cdot (\nu_{\text{des}} - \dot{\omega}_{\text{meas}})\) требует только матрицы эффективности управления \(G\), а не полной нелинейной динамики \(f\). Это устраняет чувствительность к неопределённости модели.
- **Эталонная модель**: фильтр второго порядка формирует из задающего угловой скорости плавную целевую скорость и её производную \(\nu_{\text{des}} = \dot{\omega}_{\text{ref}}\).
- **VFF-RLS**: фактор забывания \(\lambda_k\) уменьшается к нижнему пределу при росте невязки (быстрая адаптация при отказах/манёврах) и релаксирует к верхнему в спокойном режиме (подавление шума).
- **Сенсорный фильтр**: низкочастотный дифференциатор даёт \(\dot{\omega}\) из сырого \(\omega\), а экспоненциальный оценщик смещения даёт грубую оценку смещения ИИУ, которую агент вычитает из измерений — минимальная замена полного стека OTSEKF-HOSM из статьи.

## Отличия от близких методов

| Аспект | INDI | Adaptive INDI | **AA-INDI** |
| --- | --- | --- | --- |
| Эффективность управления \(G\) | Оффлайн / фикс. | Онлайн (базовый RLS) | Онлайн VFF-RLS |
| Отказы датчиков | Не обрабатываются | Не обрабатываются | Оценщик смещения (сурогат OTSEKF-HOSM) |
| Реакция на резкие отказы | Слабая | Средняя | Быстрая (λ сжимается при больших невязках) |
| Подавление шума на крейсере | Хорошее | Среднее | Хорошее (λ релаксирует к макс.) |

## Состав AA-INDI

| Компонент | Роль | Реализация |
| --- | --- | --- |
| VFFRLSEstimator | Онлайн-идентификация \(G = \partial \dot{\omega}/\partial u\) с переменным забыванием | `tensoraerospace.agent.aa_indi.VFFRLSEstimator` |
| LowPassDerivative | Причинный дифференциатор (замена HOSM) | `tensoraerospace.agent.aa_indi.LowPassDerivative` |
| BiasEstimator | Экспоненциальный оценщик смещения ИИУ | `tensoraerospace.agent.aa_indi.BiasEstimator` |
| Эталонная модель | Фильтр 2-го порядка для \(\nu_{\text{des}}\) | Встроен в `AAINDIAgent` |
| AAINDIAgent | Оркестрирует INDI, оценщики, фильтр | `tensoraerospace.agent.aa_indi.AAINDIAgent` |

## Алгоритм

На каждом шаге управления \(k\), при измерении \(\omega_k\) и команде \(r_k\):

1. **Подготовка измерений.** Вычесть текущую оценку смещения (если включено): \(\omega_k^c = \omega_k - \hat{b}\). Низкочастотный дифференциатор даёт \(\dot{\omega}_k^{\text{meas}}\) (продвигается в `learn()`, чтобы не подавать одно измерение дважды).
2. **Эталонная модель.** Фильтр 2-го порядка:

\[
\ddot{r} = -2\zeta\omega_n \dot{r} + \omega_n^2 (r_{\text{cmd}} - r), \qquad \nu_{\text{des}} = \dot{r}.
\]

3. **INDI-закон.**

\[
\Delta u = G^{+} \cdot (\nu_{\text{des}} - \dot{\omega}^{\text{meas}}), \qquad
u = \mathrm{clip}(u_{\text{prev}} + \Delta u,\ \pm u_{\max}),
\]

   с предварительным ограничением \(\Delta u\) по скорости до \(\pm\dot{u}_{\max} \cdot dt\).
4. **Обновление VFF-RLS.** По \((\Delta u_k, \Delta \dot{\omega}_k)\):

\[
\varepsilon = \Delta \dot{\omega} - \theta^{\top} \Delta u,\qquad
\lambda_k = \mathrm{clip}\bigl(e^{-\|\varepsilon\|^2/\sigma_\varepsilon^2},\ \lambda_{\min},\ \lambda_{\max}\bigr),
\]

   затем стандартная рекурсия RLS по усилению/ковариации с фактором забывания \(\lambda_k\).
5. **Обновление смещения.** Экспоненциальное скользящее среднее невязки между \(\omega\) и его реинтеграцией из \(\dot{\omega}\).

## Быстрый старт

```python
import numpy as np
from tensoraerospace.agent.aa_indi import AAINDIAgent, AAINDIConfig

# Оценка матрицы эффективности управления из on-board модели в точке трима.
G_init = np.array([[-2.0, 0.1, 0.0],
                   [0.05, -1.5, 0.2],
                   [0.0,  0.05, -0.9]])

cfg = AAINDIConfig(
    dt=0.01,
    ref_wn=5.0,
    ref_zeta=0.7,
    u_magnitude_limit=25.0,
    u_rate_limit=200.0,
    vff_forgetting_min=0.9,
    vff_forgetting_max=0.999,
    vff_eps_sensitivity=2.0,
    sensor_cutoff_hz=50.0,
    enable_bias_correction=True,
    G_init=G_init,
    seed=0,
)
agent = AAINDIAgent(n_state=3, n_control=3, config=cfg)

omega = np.zeros(3)
ref = np.array([0.2, -0.1, 0.05])  # задание по угловым скоростям, рад/с

for k in range(500):
    u = agent.predict(omega, ref, k)
    # Шаг объекта (заглушка — подключите свою среду)
    omega = omega + cfg.dt * (G_init @ u)
    metrics = agent.learn(omega, ref, k)
```

!!! tip "Warm-start `G_init` критичен"
    INDI требует разумного \(G\) на первых шагах — при случайной инициализации псевдо-обратная матрица даёт большие значения и привод насыщается раньше, чем VFF-RLS успеет сойтись. Задайте `G_init` из линеаризованной бортовой модели.

## Гиперпараметры

### Эталонная модель

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `ref_wn` | 10.0 | Собственная частота фильтра эталонной модели, рад/с |
| `ref_zeta` | 0.7 | Коэффициент демпфирования |

### Ограничения привода

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `dt` | 0.01 | Шаг управления (с) |
| `u_magnitude_limit` | 25.0 | Жёсткое ограничение по амплитуде на канал (ед. действия среды) |
| `u_rate_limit` | 60.0 | Макс. Δu в секунду на канал |
| `pinv_rcond` | 1e-6 | Порог для `np.linalg.pinv(G)` |
| `G_init` | None | Warm-start формы `(n_state, n_control)` |

### VFF-RLS

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `vff_forgetting_min` | 0.7 | Нижний предел λ — режим быстрой адаптации |
| `vff_forgetting_max` | 0.999 | Верхний предел λ — режим подавления шума |
| `vff_eps_sensitivity` | 1.0 | Норма невязки, при которой λ падает в ~1/e |
| `vff_cov_init` | 1e2 | Начальный масштаб ковариационной матрицы |

### Сенсорный фильтр

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `sensor_cutoff_hz` | 10.0 | Частота среза низкочастотного дифференциатора |
| `bias_forgetting` | 0.99 | Параметр EMA оценщика смещения |
| `enable_bias_correction` | True | Вычитать оценку смещения из ω перед формированием невязки |

## Поддерживаемые окружения

- Любые Gymnasium-среды, чьё наблюдение содержит измеряемые угловые скорости (например, `[alpha, wz]` в `NonlinearLongitudinalF16-v0` после лёгкой подготовки, или полный вектор `[p, q, r]` от 6-DoF объекта).

## Сохранение/загрузка

Тот же API, что и у остальных адаптивных агентов:

```python
run_dir = agent.save("./checkpoints")        # создаёт <date>_AAINDIAgent/
restored = AAINDIAgent.from_pretrained(run_dir)
agent.publish_to_hub("me/my-aaindi", folder_path=run_dir, access_token="hf_...")
```

Сохраняемые артефакты:

- `config.json` — полный `AAINDIConfig` + `n_state` / `n_control`.
- `vff_rls.npz` — `θ` RLS, ковариация `P`, последний `λ`, счётчик обновлений.
- `bias_state.npz` — оценка экспоненциального смещения.
- `deriv_state.npz` — состояние низкочастотного дифференциатора.
- `loop_state.npz` — состояние reference-model, PI-интегратор, последняя команда, кэшированное `ω̇`. Благодаря этому save посреди эпизода восстанавливается бит-в-бит на load (важно, когда `ref_error_kp` / `ref_error_ki` ≠ 0).

## Документация API

::: tensoraerospace.agent.aa_indi.model.AAINDIAgent

::: tensoraerospace.agent.aa_indi.model.AAINDIConfig

::: tensoraerospace.agent.aa_indi.vff_rls.VFFRLSEstimator

::: tensoraerospace.agent.aa_indi.sensor_filter.LowPassDerivative

::: tensoraerospace.agent.aa_indi.sensor_filter.BiasEstimator

## Источники

- Sun et al. *"Active Incremental Nonlinear Dynamic Inversion for Sensor and Actuator Fault Diagnosis and Fault-Tolerant Flight Control"*, TU Delft Aerospace, [research.tudelft.nl](https://research.tudelft.nl/en/publications/active-incremental-nonlinear-dynamic-inversion-for-sensor-and-act/).
- Smeur, Chu, de Croon. *"Adaptive Incremental Nonlinear Dynamic Inversion for Attitude Control of Micro Air Vehicles"*, J. Guid. Control Dyn., 2016.
- Fortescue, Kershenbaum, Ydstie. *"Implementation of Self-Tuning Regulators with Variable Forgetting Factors"*, Automatica, 1981.
