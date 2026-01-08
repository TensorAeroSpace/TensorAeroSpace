# Отказы и среды в проектах HybridRL-FlightControl и RLFC-SACIDHP

## HybridRL-FlightControl
Код: `src/`

### Основные среды
- `envs/citation/citation_env.py` — высокоточная модель Citation (входы по умолчанию `de, da, dr`), базовый класс `BaseEnv`.
- `envs/base_env.py` — общий шаг среды, поддерживает масштабирование действия (`action_scale`), фильтрацию (`filter_action`), выбор награды/наблюдения/треков задачи.
- Наблюдения с шумом: `envs/observations.py` (`sac_attitude_noise`, `noise_states_ref`).

### Смоделированные “отказы” / сценарии деградации
- **Снижение эффективности рулевых поверхностей (LOE)**: реализовано через `action_scale` (скаляр или вектор по каналам) после 10 секунд полёта (`BaseEnv.step`). В экспериментах `exp3_fault` перебираются разные `action_scale`, что эквивалентно LOE по elevator/aileron/rudder.
- **Шум измерений**: в `exp2_noise` используется `sac_attitude_noise` / `noise_states_ref` — добавляется гауссов шум к состояниям в наблюдении.
- **Stall-сценарий** (`tasks/experiment_4_stall.py`, `exp4_stall`): не “поломка привода”, а ухудшенная динамика. В `CitationEnv._check_constraints` эпизод завершается с большим штрафом, если состояние ушло в NaN (прокси выхода из обводного контура / сваливания).

### Ключевые эксперименты
- `exp3_fault.py` — LOE через `action_scale` (fault).
- `exp2_noise.py` — шум в наблюдениях (`sac_attitude_noise`).
- `exp4_stall.py` — сценарий “stall”.

---

## RLFC-SACIDHP
Код: `envs/`, `scripts/`

### Основные среды
- `envs/citation.py` — CitAST (Citation) с отказами/возмущениями (параметры `failure`, `failure_time`, `sensor_noise`, `atm_disturbance`, `control_disturbance`).
- `envs/shortperiod.py` — линейная short-period модель, поддерживает “fault” через изменение аэродинамики.

### Смоделированные отказы/возмущения (Citation)
Включаются после `failure_time`, задаются строкой `failure`:
- **Актуаторы**:
  - `dr_stuck` — заклинивший руль направления (rudder jam, фиксируется на -15°).
  - `da_reduce` — снижение эффективности элерона (aileron LOE, множитель 0.1).
  - `da_limit` — ограничение диапазона элерона (±5°).
  - `de_reduce` / `de_reduce_extreme` — снижение эффективности руля высоты (0.3 / 0.1).
  - `de_limit` — ограничение диапазона руля высоты (±2.5°).
  - `de_invert` — инверсия руля высоты (смена знака).
- **Параметрические/аэродинамические изменения**:
  - `cg_shift` — сдвиг центра масс (`cg = 0.25` м, передаётся в plant).
  - `ht_reduce` — деградация горизонтального оперения (`ht = 0.70`).
  - `icing` — обледенение (`icing = 1`).
- **Возмущения/шум**:
  - `control_disturbance` — добавка к управляющим.
  - `sensor_noise` — гауссов шум измерений (добавляется к `state`).
  - `atm_disturbance` — ступенчатые возмущения по `alpha` (±2.5° в окнах времени).

### Смоделированные отказы/возмущения (ShortPeriod)
- `fault_type` (`de_reduce`, `de_invert`, `cg_shift`) срабатывает на `fault_timestep`:
  - `de_reduce` — уменьшение эффективности руля высоты (пересборка A/B).
  - `de_invert` — инверсия руля высоты (смена знака коэффициентов, пересборка A/B).
  - `cg_shift` — резкий сдвиг CG (пересборка A/B).

### Параметры в скриптах
- `scripts/train_*` / `scripts/evaluate_*` используют поля `failure`, `failure_time`, `sensor_noise` (Citation) или `fault`, `fault_timestep` (ShortPeriod) для включения отказов.

---

## Кратко: соответствие отказов
- **LOE / ограничение диапазона / инверсия / jam**: RLFC (конфиг `failure`) — HybridRL (через `action_scale` после 10 c).
- **Шум сенсоров**: RLFC (`sensor_noise`, `sac_attitude_noise`) — HybridRL (`sac_attitude_noise` / `noise_states_ref`).
- **Сдвиг CG / обледенение / деградация хвоста**: RLFC (`cg_shift`, `icing`, `ht_reduce`) — в HybridRL не задаётся явно (можно эмулировать `action_scale` или отдельной динамикой модели).
- **Сценарий “stall”**: есть в HybridRL (`exp4_stall`), в RLFC — нет отдельного класса, но есть “icing/ht_reduce/de_limit” как аэродинамическая деградация.


