# Adaptive Incremental Dynamic Inversion (AIDI)

AIDI — **отказоустойчивый контроллер полёта**, построенный на основе инкрементальной нелинейной динамической инверсии (Incremental Nonlinear Dynamic Inversion, INDI). Алгоритм адаптирует **матрицу эффективности управления** (control-effectiveness) в режиме онлайн с помощью построчного VFF-RLS, который оценивает мультипликативный масштабирующий множитель \(\Theta\) над известной бортовой моделью \(G_{\text{nominal}}\). Подход не зависит от конкретной модели объекта управления и быстро восстанавливает слежение при потере эффективности руля. См. также нелинейную угловую модель F-16: [Нелинейная угловая модель F-16](../model/f16_nonlinear_angular.md).

**Источник**: Ul Haq, Atmaca & van Kampen, *"Adaptive Incremental Dynamic Inversion for Fault-tolerant Flight Control of a Flying Wing"*, AIAA SciTech 2026, [10.2514/6.2026-1744](https://doi.org/10.2514/6.2026-1744).

## Ключевые идеи

- **Внутренний закон INDI:** \(\Delta u = \tilde{G}^{+} \cdot (\nu_{\text{des}} - \dot{\omega}_{\text{meas}})\), где \(\tilde{G} = \Theta \odot G_{\text{nominal}}\). Достаточно линеаризованной бортовой матрицы эффективности управления; всё остальное поглощается множителем \(\Theta\).
- **VFF на основе информационного содержимого:** \(\lambda_i = 1 - (1 - \phi_i^{\top} K_i)\, \varepsilon_i^2 / \Sigma_0\), где \(\Sigma_0 = \sigma_0^2 N_0\). Соответствует уравнениям 26-27 статьи.
- **Проверка согласованности по осям:** усреднение по столбцам, когда построчные обновления согласованы между собой. Полезно при избыточном отображении управляющих поверхностей на одни и те же оси (по типу Flying-V). По умолчанию `consistency_threshold = 10`, то есть проверка фактически отключена; ужесточайте порог только для действительно избыточных объектов управления.
- **Pseudo-control hedging (PCH):** разрыв \(\nu_{\text{des}} - \dot{\omega}_{\text{meas}}\) подаётся обратно в эталонные модели, чтобы они «замораживались» при насыщении приводов.
- **Протокол бортовой CE:** \(G_{\text{nominal}}(x, u)\) запрашивается на каждом такте у экземпляра `OnboardCEModel` (`F16NonlinearOnboardCE` для F-16, `LinearOnboardCE(B)` для любого объекта управления с известной линеаризацией).

## Архитектура

![Архитектура AIDI: контуры управления, измерения, PCH и адаптация ScalingRLS](../../assets/images/aidi_architecture.png)

На схеме показана текущая реализация `AIDIAgent`: `predict` формирует команду, а `learn` обновляет фильтры и идентификацию по следующему измерению и фактическому управлению. Обратная связь PCH использует запрос ускорения с предыдущего такта. Выход `SpeedController` пока не подключён к управлению тягой.

## Компоненты

| Компонент | Роль | Реализация |
| --- | --- | --- |
| `ScalingRLS` | Построчный VFF-RLS над Θ; маска наблюдаемости + ограничение следа ковариации | `tensoraerospace.agent.aidi.ScalingRLS` |
| `OnboardCEModel` | Протокол, возвращающий \(G_{\text{nominal}}(x, u)\) | `tensoraerospace.agent.aidi.OnboardCEModel` |
| `LinearOnboardCE` | CE с постоянной матрицей | `tensoraerospace.agent.aidi.LinearOnboardCE` |
| `F16NonlinearOnboardCE` | FD-адаптер над угловыми ОДУ F-16; ремап `(wx, wy, wz)` в `(p, q, r)` | `tensoraerospace.agent.aidi.F16NonlinearOnboardCE` |
| `MoorePenroseAllocator` | Псевдоинверсия с защитой от плохой обусловленности | `tensoraerospace.agent.aidi.MoorePenroseAllocator` |
| `PseudoControlHedge` | Сигнал хеджирования + счётчик заморозки по каждой оси | `tensoraerospace.agent.aidi.PseudoControlHedge` |
| `CStarController`, `RollReferenceModel`, `SideslipCompensator`, `SpeedController`, `LinearController` | Блоки внешнего контура | `tensoraerospace.agent.aidi.ref_models` |
| `AIDIAgent` / `AIDIConfig` | Оркестратор и сохранение состояния | `tensoraerospace.agent.aidi.AIDIAgent` |

## Быстрый старт (F-16)

```python
import numpy as np
from tensoraerospace.agent.aidi import AIDIAgent, AIDIConfig, F16NonlinearOnboardCE
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import default_parameters
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16
from tensoraerospace.scripts.benchmark_aidi import _solve_trim

params = default_parameters()
alpha, stabilator = _solve_trim()
x0 = np.zeros(14)
x0[0] = x0[7] = alpha
x0[8] = stabilator
env = NonlinearAngularF16(x0, number_time_steps=1002, dt=0.01,
                          integrator="rk4", airspeed=params.V)
agent = AIDIAgent(3, 3, F16NonlinearOnboardCE(params), AIDIConfig(dt=0.01))
state, _ = env.reset()
agent.reset(initial_action=state[[8, 10, 12]])

def observe(x):
    return {"omega": x[[2, 4, 3]] * [1, 1, -1],  # (p, q, r) = (wx, wz, -wy)
            "alpha": x[0], "beta": x[1], "theta": x[7], "phi": x[5],
            "V": params.V, "state": x.copy()}

ref = {"C_star": 1.0, "phi_cmd": 0.0, "beta_cmd": 0.0, "V_cmd": params.V}
command_rad = agent.predict(observe(state), ref)
next_state, _, terminated, truncated, _ = env.step(np.rad2deg(command_rad))
mean_deflection = (state[[8, 10, 12]] + next_state[[8, 10, 12]]) / 2
metrics = agent.learn(observe(next_state), ref, applied_action=mean_deflection)
```

API сохранения/загрузки и round-trip с Hugging Face у агента такие же, как в `aa_indi`/`et_dhp`/`im_gdhp`.

## Измерения и контур угловых скоростей

- Для штатного состояния F-16 передавайте `(p, q, r) = (wx, wz, -wy)`. Знак рыскания согласован с `F16NonlinearOnboardCE`; уравнения объекта не меняются.
- В начале эпизода вызовите `reset(initial_action=state[[8, 10, 12]])`, передав фактические положения приводов в радианах. Обученные `Theta` и ковариация сохраняются.
- После `predict(obs, refs)` вызывайте ровно один `learn(next_obs, refs, applied_action=...)`. Первое наблюдение используется для инициализации измерителя ускорения.
- `applied_action` — среднее фактическое отклонение привода за переход, в единицах команды. Для F-16 его можно приближённо получить как `(previous_positions + next_positions) / 2`; точность проверяется уменьшением шага. Без обратной связи предполагается идеальное выполнение команды.
- Ускорение и фактическое управление проходят согласованный фильтр первого порядка. Приращение управления добавляется к фильтрованному положению. Ограничитель скорости команды работает относительно предыдущей команды; физические ограничения привода отдельно обеспечивает объект. Фильтр и задержки статьи здесь воспроизведены упрощённо.
- Контур скоростей вычисляет `nu = rate_kp * (omega_des - omega)` в рад/с². Коэффициенты имеют единицы с⁻¹, по умолчанию `(1, 1, 1)`; нулевые коэффициенты отключают обратную связь. При достигнутой постоянной скорости ускорение должно быть нулевым. Старые настройки требуют повторной проверки.
- `learn(..., adapt=False)` сохраняет измерительную историю, не меняя идентификатор. Это диагностический фиксированный контроллер, а не заморозка адаптации в известный момент отказа.

Новые checkpoints сохраняют историю фильтров и незавершённый переход. Старые сохраняют обученные параметры, но заново инициализируют измерительную историю: точное воспроизведение старого ошибочного закона управления не поддерживается.

Для длительных прогонов и многоканальной проверки используйте `scripts/validate_aidi_measurements.py`. Проверяйте завершение эпизода и углы самолёта вместе с ошибками слежения: конечные веса ещё не означают устойчивость.

## Подробный пример

`example/reinforcement_learning/incremental_adp/example_aidi_damage_f16.ipynb` — полный сценарий восстановления при отказе на нелинейной модели F-16: тримминг, базовая траектория, потеря 25 % усиления команды стабилизатора в момент t = 8 с, сравнение прогонов с адаптивной и замороженной (frozen-Θ) идентификацией.

## CLI для бенчмарков

```bash
python -m tensoraerospace.scripts.benchmark_aidi \
    --env f16_nonlinear_angular \
    --baselines frozen \
    --scenarios nominal,stab_50,stab_25,stab_lost,rudder_lost \
    --episodes 5 --steps 1500 \
    --out report.md --csv report.csv
```

Формирует Markdown-таблицу и CSV с RMSE угловых скоростей (рад/с), начиная с t = 2 с. Это проверка удержания скоростей, а не воспроизведение Table 8 или полная оценка ориентации. `stab_25` означает 25% оставшегося усиления команды; аэродинамические коэффициенты при этом не масштабируются. `frozen` отключает идентификацию с начала эпизода и не использует момент отказа. Повторные эпизоды имеют одинаковые детерминированные начальные условия.

## Гиперпараметры

### Внутренний контур и ограничения приводов

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `dt` | 0.01 | Шаг управления (с) |
| `u_magnitude_limit` | `radians(25)` | Ограничение по амплитуде (в тех же единицах, что `u` у `OnboardCEModel`) |
| `u_rate_limit` | `radians(60)` | Максимальное Δu в секунду |
| `pinv_rcond` | 1e-6 | Порог отсечки для `np.linalg.pinv(G)` |
| `cond_threshold` | 1e12 | При превышении `cond(G)` происходит откат к `Δu = 0` |
| `sensor_cutoff_hz` | 15.0 | Частота среза НЧ-фильтра для ω̇ |

### Scaling-RLS

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `rls_lambda_min` | 0.7 | Нижняя граница фактора забывания (быстрая адаптация) |
| `rls_lambda_max` | 0.999 | Верхняя граница фактора забывания (подавление шума) |
| `rls_sigma0` | 1e-3 | СКО шума датчика σ₀, используемое в Σ₀ = σ₀²·N₀ |
| `rls_memory_length` | 100 | Номинальная длина памяти N₀ (отсчётов) |
| `rls_cov_init` | 1.0 | Начальный масштаб P_i |
| `rls_consistency_threshold` | 10.0 | Порог проверки согласованности по осям (≤ 1e-6 для избыточных объектов управления) |

### PCH

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `pch_freeze_after` | 30 | Число тактов насыщения до жёсткой заморозки эталонной скорости |
| `pch_gap_tol` | 1e-3 | Значение `|ν_h|`, ниже которого ось считается отслеженной |

### Внешний контур

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `cstar_kp` / `cstar_ki` | 1.5 / 0.5 | ПИ-коэффициенты C\* |
| `cstar_V_co` | 122.6 | Скорость кроссовера C\* (м/с) |
| `roll_omega_n` / `roll_zeta` | 2.5 / 0.7 | Параметры эталонной модели крена (2-й порядок) |
| `sideslip_kp` / `sideslip_ki` | 1.5 / 0.1 | ПИ-регулятор скольжения |
| `speed_*`, `speed_enabled` | 0 / False | Автомат тяги (по умолчанию выключен) |

## Поддерживаемые окружения

- Любая среда Gymnasium, в которой доступны \((p, q, r)\) и \(\alpha, \beta, \theta, \phi, V\). Опциональное \(n_z\) восстанавливается по \((\alpha, \dot{\alpha}, q, V, \theta, \phi)\), если оно отсутствует.
- Нелинейная угловая среда F-16, подключённая через `F16NonlinearOnboardCE` (ремап осей встроен).
- Любой объект управления с постоянной линеаризованной CE — передавайте `LinearOnboardCE(B)`.

## Сохранение/загрузка

```python
run_dir = agent.save("./checkpoints")           # создаёт <date>_AIDIAgent/
restored = AIDIAgent.from_pretrained(run_dir, onboard_ce=F16NonlinearOnboardCE(...))
agent.publish_to_hub("me/my-aidi", folder_path=run_dir, access_token="hf_...")
```

Сохраняемые артефакты:

- `config.json` — полный `AIDIConfig` плюс `n_state` / `n_control`.
- `scaling_rls.npz` — `theta`, `P`, `last_lambda`, `last_residual`, `num_updates`.
- `outer_state.npz` — интеграторы C\*/скольжения/скорости и состояние эталонной модели крена.
- `pch_state.npz` — сигнал хеджирования, счётчик насыщения, флаги заморозки.
- `deriv_state.npz` — состояние НЧ-дифференциатора.
- `loop_state.npz` — `u_prev`, `omega_prev`, `omega_dot_cached`, последняя команда, последняя `G_nominal`, счётчик шагов.

## Документация API

::: tensoraerospace.agent.aidi.model.AIDIAgent

::: tensoraerospace.agent.aidi.model.AIDIConfig

::: tensoraerospace.agent.aidi.scaling_rls.ScalingRLS

::: tensoraerospace.agent.aidi.onboard_ce.OnboardCEModel

::: tensoraerospace.agent.aidi.onboard_ce.F16NonlinearOnboardCE

::: tensoraerospace.agent.aidi.allocator.MoorePenroseAllocator

::: tensoraerospace.agent.aidi.pch.PseudoControlHedge

## Источники

- Ul Haq, Atmaca, van Kampen. *"Adaptive Incremental Dynamic Inversion for Fault-tolerant Flight Control of a Flying Wing"*, AIAA SciTech 2026, [10.2514/6.2026-1744](https://doi.org/10.2514/6.2026-1744).
- Atmaca, van Kampen. *"Fault Tolerant Control for the Flying-V Using Adaptive Incremental Nonlinear Dynamic Inversion"*, AIAA SciTech 2025, [10.2514/6.2025-0081](https://doi.org/10.2514/6.2025-0081).
- Fortescue, Kershenbaum, Ydstie. *"Implementation of Self-Tuning Regulators with Variable Forgetting Factors"*, Automatica, 1981.
