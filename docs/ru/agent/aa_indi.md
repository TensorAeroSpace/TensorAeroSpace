# Active-Adaptive Incremental Nonlinear Dynamic Inversion (AA-INDI)

AA-INDI объединяет инкрементальную инверсию динамики с онлайн-идентификацией эффективности управления. В реализации используются VFF-RLS и дифференциатор с фильтром первого порядка. Сглаживание невязки — эвристика, которая не воспроизводит оценщик отказов датчиков OTSEKF-HOSM из статьи. Точность и восстановление после отказов зависят от возбуждения, настроек, динамики привода и начальной оценки эффективности. См. [NonlinearLongitudinalF16](../model/f16_nonlinear_longitudinal.md).

**Источник**: Atmaca, de Visser, van Kampen (2026), *"Active Incremental Nonlinear Dynamic Inversion for Sensor and Actuator Fault-Tolerant Control"*, TU Delft Aerospace, [research.tudelft.nl](https://research.tudelft.nl/en/publications/active-incremental-nonlinear-dynamic-inversion-for-sensor-and-act/).

## Ключевые идеи

- **INDI-закон**: приращение управления \(\Delta u = G^+ \cdot (\nu_{\text{des}} - \dot{\omega}_{\text{meas}})\) требует только матрицы эффективности управления \(G\), а не полной нелинейной динамики \(f\). Это снижает зависимость от полной модели; ошибки эффективности и задержки по-прежнему влияют на управление.
- **Эталонная модель**: фильтр второго порядка формирует из задающего угловой скорости плавную целевую скорость и её производную \(\nu_{\text{des}} = \dot{\omega}_{\text{ref}}\).
- **VFF-RLS**: фактор забывания \(\lambda_k\) уменьшается к нижнему пределу при росте невязки (быстрая адаптация при отказах/манёврах) и релаксирует к верхнему в спокойном режиме (подавление шума).
- **Сенсорный фильтр**: низкочастотный дифференциатор даёт \(\dot{\omega}\) из сырого \(\omega\), а сглаживание невязки даёт необязательную эвристическую поправку. Постоянное смещение датчика нельзя определить только по невязке реинтеграции его собственного сигнала.

## Отличия от близких методов

| Аспект | INDI | Adaptive INDI | **AA-INDI** |
| --- | --- | --- | --- |
| Эффективность управления \(G\) | Оффлайн / фикс. | Онлайн (базовый RLS) | Онлайн VFF-RLS |
| Отказы датчиков | Не обрабатываются | Не обрабатываются | Эвристика; для постоянного смещения нужен независимый источник информации |
| Адаптация после отказа | Фиксированная эффективность | Обновления RLS | Переменное забывание; восстановление требует проверки |
| Работа с шумом | Фильтрация измерений | Фильтрация и настройка RLS | Согласованные фильтры входа/выхода и настройка VFF |

## Состав AA-INDI

| Компонент | Роль | Реализация |
| --- | --- | --- |
| VFFRLSEstimator | Онлайн-идентификация \(G = \partial \dot{\omega}/\partial u\) с переменным забыванием | `tensoraerospace.agent.aa_indi.VFFRLSEstimator` |
| LowPassDerivative | Причинный дифференциатор (замена HOSM) | `tensoraerospace.agent.aa_indi.LowPassDerivative` |
| BiasEstimator | Экспоненциальное среднее заданной невязки | `tensoraerospace.agent.aa_indi.BiasEstimator` |
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
u = \mathrm{clip}(u_{\text{filtered}} + \Delta u,\ \pm u_{\max}),
\]

   Базовое управление здесь — отфильтрованное фактическое положение привода. Кандидат команды ограничивается относительно предыдущего фактического входа на \(\dot{u}_{\max} dt\).
4. **Обновление VFF-RLS.** По \((\Delta u_k, \Delta \dot{\omega}_k)\):

\[
\varepsilon = \Delta \dot{\omega} - \theta^{\top} \Delta u,\qquad
\varphi_k = \Delta u_k,\qquad K_k = \frac{P_k\varphi_k}{1+\varphi_k^T P_k\varphi_k},\qquad
\lambda_k = \mathrm{clip}\left(1-\frac{\|\varepsilon\|^2}{\sigma_\varepsilon^2(1+\varphi_k^T P_k\varphi_k)},\lambda_{\min},\lambda_{\max}\right),
\]

   затем стандартная рекурсия RLS по усилению/ковариации с фактором забывания \(\lambda_k\).
5. **Обновление смещения.** Экспоненциальное скользящее среднее невязки между \(\omega\) и его реинтеграцией из \(\dot{\omega}\).

## Соответствие оригинальной статье

Усиление и забывание VFF-RLS соответствуют уравнениям (54)–(57) из
[Atmaca et al., AIAA 2026-1743](https://repository.tudelft.nl/file/File_ee9931f5-cf45-45a5-b5a3-0225b0f35da2).
`vff_eps_sensitivity**2` соответствует Σ₀. Верхний предел
`vff_forgetting_max < 1` — расширение библиотеки; значение 1 допускает максимум
из статьи. Ковариация вычисляется в алгебраически эквивалентной форме Joseph,
чтобы избежать потери точности при вычитании. Прежняя экспоненциальная формула
не соответствовала уравнению (55). Старые checkpoint загружаются, но настройки
продолжающейся адаптации требуют повторной проверки.

Агент использует отфильтрованные приращения ускорения и положения привода.
В статье восстанавливаются аэродинамические моменты и оцениваются производные
по управляющим поверхностям; также используется OTSEKF-HOSM. Эти подсистемы
здесь не воспроизведены. Проверка этого класса не устанавливает качество
полной опубликованной архитектуры AA-INDI.

Неконечные отсчёты и переполнение отклоняются до изменения параметров RLS.
В невозбуждаемых направлениях сохраняется заданный закон забывания: численная
защита не устраняет рост ковариации и не доказывает устойчивость.

## Согласование измерений и привода

На каждом шаге вызывайте `predict(measurement, reference, k)`, затем шаг объекта
и `learn(next_measurement, reference, k, applied_action=actual_input)`.
Следующий `predict` должен получить то же измерение, которое передано в `learn`.
Команда и обратная связь используют одинаковые единицы и одинаковое вычитание
триммерного смещения. Без `applied_action` предполагается точное выполнение команды.

`LinearLongitudinalB747` и `LinearLongitudinalLAPAN` возвращают фактический руль
после ограничения скорости в `info["applied_action"]`, в **градусах**.
Для непрерывного сервопривода нужна оценка положения руля за переход;
заданная команда не равна фактическому положению поверхности.

Первый `predict` инициализирует дифференциатор начальным измерением. Обратная
связь привода проходит такой же фильтр, как ускорение; RLS использует их
отфильтрованные приращения. Новые checkpoint сохраняют оба фильтра, предыдущие
измерения и ожидающую команду. Старые checkpoint загружаются, но отсутствующая
история требует повторного прогрева идентификации.

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
| `vff_eps_sensitivity` | 1.0 | Квадратный корень Σ₀ из уравнения (55) |
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

- Atmaca, de Visser, van Kampen (2026). *"Active Incremental Nonlinear Dynamic Inversion for Sensor and Actuator Fault-Tolerant Control"*, TU Delft Aerospace, [research.tudelft.nl](https://research.tudelft.nl/en/publications/active-incremental-nonlinear-dynamic-inversion-for-sensor-and-act/).
- Smeur, Chu, de Croon. *"Adaptive Incremental Nonlinear Dynamic Inversion for Attitude Control of Micro Air Vehicles"*, J. Guid. Control Dyn., 2016.
- Fortescue, Kershenbaum, Ydstie. *"Implementation of Self-Tuning Regulators with Variable Forgetting Factors"*, Automatica, 1981.
