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

```
                       ┌────────────────────┐
   C*_cmd, φ_cmd,      │  Внешний контур    │
   β_cmd, V_cmd  ───►  │  (C*, крен, β,     │
                       │   скорость, лин.)  │
                       └────────┬───────────┘
                                │ ω_des
   PCH ◄── ω̇_meas ─┐            ▼
                   │   ┌──────────────────┐
                   │   │ Линейный регулятор │ ν
                   │   └──────┬───────────┘
                   │          ▼
                   │   ┌──────────────────┐    G_nominal(x, u)
                   │   │ Внутренний закон │ ◄── OnboardCEModel
                   │   │   AIDI           │
                   │   │ Δu = G̃⁺·(ν−ω̇)  │
                   │   └──────┬───────────┘
                   │          ▼ Δu
                   │   ┌──────────────────┐
                   │   │ Огранич. скор/амп │
                   │   └──────┬───────────┘
                   │          ▼ u
                   │       env.step
                   │          ▼ ω
                   │   ┌──────────────────┐
                   └─◄ │ ω̇ из НЧ-произв.  │
                       └──────┬───────────┘
                              ▼
                       ┌──────────────────┐
                       │ ScalingRLS:      │
                       │ Θ ← Θ + ΔΘ       │
                       │ информац. VFF    │
                       │ проверка согл.   │
                       └──────────────────┘
```

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
import math, numpy as np
from tensoraerospace.agent.aidi import AIDIAgent, AIDIConfig, F16NonlinearOnboardCE
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import default_parameters

agent = AIDIAgent(
    n_state=3, n_control=3,
    onboard_ce=F16NonlinearOnboardCE(default_parameters(), perturb=1e-3),
    config=AIDIConfig(dt=0.01, seed=0),
)

# obs['omega'] в порядке (p, q, r) — у F-16-окружения wy=r и wz=q,
# поэтому требуется переупорядочивание: omega = (obs[2], obs[4], obs[3])
obs = {"omega": np.zeros(3), "alpha": 0.05, "beta": 0.0,
       "theta": 0.0, "phi": 0.0, "V": 200.0, "state": np.zeros(14)}
ref = {"C_star": 1.0, "phi_cmd": 0.0, "beta_cmd": 0.0, "V_cmd": 200.0}

u_rad = agent.predict(obs, references=ref, time_step=0)
# Шаг среды (среда ожидает градусы):
# next_obs, reward, terminated, truncated, info = env.step(np.rad2deg(u_rad))
next_obs = obs  # заглушка для примера; в реальном цикле — результат env.step
metrics = agent.learn(next_obs, references=ref, time_step=0)
```

API сохранения/загрузки и round-trip с Hugging Face у агента такие же, как в `aa_indi`/`et_dhp`/`im_gdhp`.

## Подробный пример

`example/reinforcement_learning/example_aidi_damage_f16.ipynb` — полный сценарий восстановления при отказе на нелинейной модели F-16: тримминг, базовая траектория, потеря 25 % эффективности стабилизатора в момент t = 5 с, сравнение прогонов с адаптивной и замороженной (frozen-Θ) идентификацией.

## CLI для бенчмарков

```bash
python -m tensoraerospace.scripts.benchmark_aidi \
    --env f16_nonlinear_angular \
    --baselines frozen \
    --scenarios nominal,stab_50,stab_25,stab_lost,rudder_lost \
    --episodes 5 --steps 1500 \
    --out report.md --csv report.csv
```

Формирует Markdown-таблицу и CSV с RMSE по осям — аналог Table 8 из статьи, но на F-16.

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
