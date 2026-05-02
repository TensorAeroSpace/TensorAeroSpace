# Adaptive Incremental Dynamic Inversion (AIDI)

AIDI — **отказоустойчивый контроллер полёта** на основе INDI с онлайн-адаптацией матрицы control-effectiveness. Идентифицируется не сама \\(G\\), а мультипликативная поправка \\(\\Theta\\) над известной бортовой моделью \\(G_{\\text{nominal}}\\). Подход model-agnostic и быстро восстанавливает слежение при потере эффективности руля.

**Ссылка**: Ul Haq, Atmaca, van Kampen, *"Adaptive Incremental Dynamic Inversion for Fault-tolerant Flight Control of a Flying Wing"*, AIAA SciTech 2026, [10.2514/6.2026-1744](https://doi.org/10.2514/6.2026-1744).

## Ключевые идеи

- **Внутренний закон INDI:** \\(\\Delta u = \\tilde{G}^{+} \\cdot (\\nu_{\\text{des}} - \\dot{\\omega}_{\\text{meas}})\\), где \\(\\tilde{G} = \\Theta \\odot G_{\\text{nominal}}\\). Достаточно линеаризованной бортовой CE; остальное поглощается \\(\\Theta\\).
- **Информационный VFF:** \\(\\lambda_i = 1 - (1 - \\phi_i^{\\top} K_i)\\, \\varepsilon_i^2 / \\Sigma_0\\), \\(\\Sigma_0 = \\sigma_0^2 N_0\\) — формулы 26-27 статьи.
- **Cross-axis consistency check:** усреднение по строкам, когда per-row обновления согласованы. Полезно при избыточно отображённых поверхностях (Flying V). По умолчанию `consistency_threshold = 10` (выключено); ужесточайте для избыточных объектов.
- **Pseudo-Control Hedging:** разрыв \\(\\nu_{\\text{des}} - \\dot{\\omega}_{\\text{meas}}\\) подаётся обратно в reference models, чтобы они "замораживались" при насыщении.
- **OnboardCEModel-протокол:** \\(G_{\\text{nominal}}(x, u)\\) запрашивается каждый тик; для F-16 — `F16NonlinearOnboardCE` (адаптер с центральной разностью), для любого линейного объекта — `LinearOnboardCE(B)`.

## Компоненты

| Компонент | Роль | Реализация |
| --- | --- | --- |
| `ScalingRLS` | Per-row VFF-RLS над Θ; observability mask + ограничение следа P | `tensoraerospace.agent.aidi.ScalingRLS` |
| `OnboardCEModel` | Протокол, возвращающий \\(G_{\\text{nominal}}(x, u)\\) | `tensoraerospace.agent.aidi.OnboardCEModel` |
| `LinearOnboardCE` | Постоянная матрица CE | `tensoraerospace.agent.aidi.LinearOnboardCE` |
| `F16NonlinearOnboardCE` | FD-адаптер над F-16 angular ODE; ремап `(wx, wy, wz)`→`(p, q, r)` | `tensoraerospace.agent.aidi.F16NonlinearOnboardCE` |
| `MoorePenroseAllocator` | Псевдоинверсия с защитой от плохой обусловленности | `tensoraerospace.agent.aidi.MoorePenroseAllocator` |
| `PseudoControlHedge` | Сигнал хеджирования + per-axis freeze | `tensoraerospace.agent.aidi.PseudoControlHedge` |
| `CStarController`, `RollReferenceModel`, `SideslipCompensator`, `SpeedController`, `LinearController` | Внешний контур | `tensoraerospace.agent.aidi.ref_models` |
| `AIDIAgent` / `AIDIConfig` | Оркестратор + персистентность | `tensoraerospace.agent.aidi.AIDIAgent` |

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

# obs['omega'] в порядке (p, q, r) — у F-16 env wy=r и wz=q, поэтому:
#     omega = (obs[2], obs[4], obs[3])
obs = {"omega": np.zeros(3), "alpha": 0.05, "beta": 0.0,
       "theta": 0.0, "phi": 0.0, "V": 200.0, "state": np.zeros(14)}
ref = {"C_star": 1.0, "phi_cmd": 0.0, "beta_cmd": 0.0, "V_cmd": 200.0}

u_rad = agent.predict(obs, references=ref, time_step=0)
metrics = agent.learn(next_obs, references=ref, time_step=0)
```

API сохранения совпадает с `aa_indi`/`et_dhp`/`im_gdhp`.

## Пример

`example/reinforcement_learning/example_aidi_damage_f16.ipynb` — полный сценарий восстановления при отказе на F-16: тримм, baseline, потеря 25 % эффективности стаба на t = 5 с, сравнение adaptive vs frozen-Θ.

## Бенчмарк CLI

```bash
python -m tensoraerospace.scripts.benchmark_aidi \
    --env f16_nonlinear_angular \
    --baselines frozen \
    --scenarios nominal,stab_50,stab_25,stab_lost,rudder_lost \
    --episodes 5 --steps 1500 \
    --out report.md --csv report.csv
```

Производит Markdown-таблицу + CSV пер-axis RMSE — Table 8 из статьи, но на F-16.

## Источники

- Ul Haq, Atmaca, van Kampen. *"Adaptive Incremental Dynamic Inversion for Fault-tolerant Flight Control of a Flying Wing"*, AIAA SciTech 2026.
- Atmaca, van Kampen. *"Fault Tolerant Control for the Flying-V Using Adaptive Incremental Nonlinear Dynamic Inversion"*, AIAA SciTech 2025.
- Fortescue, Kershenbaum, Ydstie. *"Implementation of Self-Tuning Regulators with Variable Forgetting Factors"*, Automatica, 1981.
