# Моделирование повреждений ЛА в полёте — дизайн

**Дата:** 2026-04-28
**Скоп:** Нелинейная модель F-16 (продольная и угловая 6-DoF)
**Статус:** Design approved, ready for implementation plan

## Мотивация

Тренировка RL-агентов на устойчивость к боевым повреждениям и исследовательская FDM-симуляция требуют возможности менять динамику ЛА в полёте. Целевой сценарий: F-16 в догфайте теряет часть консоли крыла → меняется подъёмная сила, появляется момент крена/рыскания, ЛА становится другим объектом управления.

Текущая библиотека такой возможности не имеет — поиск по ключам `fault`, `failure`, `damage`, `disturbance` ничего не находит.

## Цели

1. **Гибридный кейс RL + FDM** — система одновременно подходит и для тренировки агентов на устойчивость к повреждениям, и для исследовательских прогонов с физически корректным пересчётом параметров.
2. **Триггерные события по времени** — повреждение применяется в заданный момент эпизода (single или multiple events).
3. **Полная таксономия повреждений** — несущие поверхности (включая асимметричные потери), рулевые поверхности (jam / efficiency loss / lost), двигатель (потеря тяги), структурные (изменение массы и ЦМ).
4. **Архитектурная изоляция** — без `damage_profile` существующее поведение не меняется (бит-в-бит); damage-логика сосредоточена в отдельном пакете.

## Не-цели (v1)

- Линейная модель F-16, другие ЛА (B-747, ракета).
- Полный VLM-пересчёт через AVL/Tornado.
- Упругие деформации крыла, изменение профиля от мелких повреждений.
- Каскадные отказы (один отказ запускает другой).
- 3D визуализация повреждённого ЛА.

## Архитектура

### Расположение модулей

Новый пакет `tensoraerospace/aerospacemodel/f16/nonlinear/damage/`:

```
damage/
├── __init__.py
├── geometry.py            — параметрическая геометрия (секции, массы, инерции)
├── presets.py             — встроенная геометрия F-16, готовые сценарии
├── state.py               — DamageState (текущее состояние)
├── events.py              — DamageEvent + DamageProfile
├── manager.py             — DamageManager (триггерит, пересчитывает)
├── recompute.py           — strip-theory пересчёт m, S, b, bA, ЦМ, J*
├── aero_corrections.py    — поправки к Cy/Cx/Cz/Mx/My/Mz
├── controls.py            — jam/loss/efficiency на рулях
├── propulsion.py          — потеря тяги
└── data/
    └── f16_geometry.yaml  — секции F-16 в декларативном виде
```

### Изменения в существующих файлах

- `f16/nonlinear/longitudinal/model.py` и `f16/nonlinear/angular/model.py` — опциональный аргумент `damage_manager` в конструкторе; на каждом шаге `manager.update(t)` и в ODE передаётся текущий `DamageState`.
- `f16/nonlinear/longitudinal/dynamics.py` и `angular/dynamics.py` — хук `apply_damage_corrections(coef, state)` после вычисления базовых коэффициентов.
- `f16/nonlinear/angular/model.py` — расширение на раздельные входы `stab_left` и `stab_right` (для асимметричных отказов стабилизатора).
- `envs/f16/nonlinear_longitudinal.py` и `nonlinear_angular.py` — параметры `damage_profile`, `damage_observable`, `damage_event_callback`.

### Поток данных (один шаг)

```
env.step(action)
  → t_current = self.t
  → damage_manager.update(t_current, t_prev)
        ├── проверяет события в окне
        ├── при срабатывании обновляет DamageState
        └── recompute.apply(state, base_geometry) → новые S, m, J, ЦМ → params
  → action_eff = controls.apply_control_failures(action, state)
  → model.run_step(action_eff, damage_state=state)
        └── integrator(f16_ode, x, u_eff, t, params, state)
              ├── cy_total = get_cy(...) + aero_corrections.delta_cy(α,β,q,state)
              ├── (то же для cx, cz, mx, my, mz)
              ├── thrust_eff = thrust_base · state.engine.thrust_factor
              └── уравнения движения с обновлёнными params
  → obs = (state_vector, опц. damage_state.to_vector())
  → info["damage_state"] = state.snapshot()
```

### Принципы изоляции

- **Базовая модель не знает про damage**: при `damage_manager=None` поведение бит-в-бит идентично текущему (регрессионное тестирование).
- **DamageManager — единственный владелец mutable state**, ODE — чистая функция, читает `DamageState` как read-only.
- **Геометрия отделена от повреждений**: `geometry.py` описывает «здоровый» ЛА, `state.py` описывает что повреждено, `recompute.py` соединяет.
- **Аэро-коррекции — аддитивные/мультипликативные дельты к базовым `.npz` таблицам** (таблицы не пересчитываются).

## Параметрическая геометрия

### Структура секций F-16

```
Wing:        left_root, left_mid, left_tip, right_root, right_mid, right_tip
Hstab:       stab_left, stab_right
Vtail:       vtail
Controls:    elevator_left, elevator_right, rudder, aileron_left, aileron_right,
             flap_left, flap_right
Fuselage:    fuselage_main
```

### Атрибуты секции

```python
@dataclass(frozen=True)
class AeroSection:
    name: str
    side: Literal["left", "right", "center"]
    type: Literal["wing", "stab", "vtail", "control", "fuselage"]

    # Геометрия
    area: float                  # м², проекционная площадь
    span_position: float         # м, расстояние от центральной плоскости
    chord: float                 # м, средняя хорда секции
    sweep: float                 # рад, стреловидность

    # Массово-инерционные
    mass: float                  # кг
    cg_local: tuple[float, float, float]              # м, ЦМ секции
    inertia_local: tuple[float, float, float, float]  # Ixx, Iyy, Izz, Ixz отн. ЦМ секции

    # Аэровклады в коэффициенты
    cl_alpha_contribution: float
    cd0_contribution: float

    # Только для управляющих
    controls_input: Optional[str] = None
    control_effectiveness: float = 1.0
```

### Калибровка

Сумма по секциям должна совпадать с текущими `F16AngularParameters`:

| Параметр | Целевое значение | Способ |
|----------|------------------|--------|
| `S` | 27.87 м² | Σ area по wing |
| `m` | 9295.44 кг | Σ mass по всем секциям |
| `Jx` | 12874.8 кг·м² | Σ через теорему Гюйгенса-Штейнера |
| `Jy` | 75673.6 кг·м² | то же |
| `Jz` | 85552.1 кг·м² | то же |
| `Jxz` | 1331.4 кг·м² | то же |
| `bA` | 3.45 м | площадно-взвешенная хорда |

**Тест-инвариант:** при «нет повреждений» агрегация совпадает с текущими параметрами с точностью `<1%`.

### Хранение

`damage/data/f16_geometry.yaml` — декларативное описание секций. `presets.py` загружает в `BaseGeometry`.

## DamageState, DamageEvent, DamageProfile

### DamageState

```python
@dataclass
class DamageState:
    section_loss: dict[str, float]              # {name: 0.0..1.0}
    control_failures: dict[str, ControlFailure]
    engine: EngineState
    structural: StructuralState

@dataclass
class ControlFailure:
    mode: Literal["healthy", "efficiency_loss", "jam", "free_floating", "lost"]
    efficiency: float = 1.0
    jam_position_rad: float = 0.0

@dataclass
class EngineState:
    thrust_factor: float = 1.0
    hard_failure: bool = False

@dataclass
class StructuralState:
    extra_mass_delta_kg: float = 0.0                            # независимое изменение массы
    extra_cg_shift_m: tuple = (0.0, 0.0, 0.0)                   # дополнительный сдвиг ЦМ (поверх агрегации секций)
    extra_inertia_delta: tuple = (0.0, 0.0, 0.0, 0.0)           # ΔIxx, ΔIyy, ΔIzz, ΔIxz
```

`DamageState` мутабельный, владеется `DamageManager`. Снаружи доступен только read-only через `snapshot()`.

### DamageEvent

```python
@dataclass(frozen=True)
class DamageEvent:
    trigger_time: float
    event_type: Literal["section_loss", "control_failure", "engine_failure", "structural_change"]
    payload: dict
    label: Optional[str] = None
    duration: Optional[float] = None  # None = permanent
```

Примеры payload:

- `section_loss`: `{"section": "left_tip", "loss_fraction": 1.0}`
- `control_failure`: `{"surface": "elevator_left", "mode": "jam", "jam_position_rad": 0.087}`
- `engine_failure`: `{"thrust_factor": 0.4}`
- `structural_change`: `{"mass_delta_kg": -200, "cg_shift_m": (0.3, 0, 0)}`

### DamageProfile

```python
@dataclass
class DamageProfile:
    events: list[DamageEvent]
    seed: Optional[int] = None

    def get_pending_events(self, t_current, t_previous) -> list[DamageEvent]: ...
    @classmethod
    def from_yaml(cls, path: str) -> "DamageProfile": ...
```

### Готовые сценарии (`presets.py`)

| Имя | Описание |
|-----|----------|
| `WING_STRIKE_LEFT_TIP` | t=10s, потеря left_tip полностью |
| `WING_STRIKE_LEFT_HALF` | t=10s, потеря left_tip + 50% left_mid |
| `ELEVATOR_JAM_NEUTRAL` | t=5s, оба руля высоты заклинены в нейтрали |
| `ELEVATOR_JAM_PITCH_UP` | t=5s, заклинены при +10° |
| `RUDDER_LOST` | t=5s, руль направления потерян |
| `ENGINE_FLAMEOUT` | t=5s, thrust_factor=0 |
| `BIRDSTRIKE_COMPOUND` | t=5s, потеря 20% правой консоли + thrust_factor=0.3 |

## Физика пересчёта

### Массово-инерционные параметры

При срабатывании `section_loss`:

1. Для каждой секции `s` с долей потери `f_s`:
   - Эффективная масса: `m_s_eff = m_s · (1 - f_s)`
2. Агрегаты:
   - `m_eff = Σ m_s_eff + structural.extra_mass_delta_kg`
   - `cg_eff = (Σ m_s_eff · cg_local_s) / m_eff + structural.extra_cg_shift_m`
   - `J*_eff` — теорема Гюйгенса-Штейнера от `cg_eff` по сохранившимся секциям
3. `S_eff = Σ area_s · (1 - f_s)` для `type=="wing"`
4. `b_eff = max(span_position_s · (1 - f_s)) для left ⊕ max(span_position_s · (1 - f_s)) для right` (сумма крайних точек обоих полукрыльев — корректно работает и при асимметричной потере)
5. `bA_eff = Σ chord_s · area_s · (1 - f_s) / S_eff` (площадно-взвешенный)

Для permanent событий пересчёт один раз; для temporary (с `duration`) — на каждом шаге, пока активно.

### Аэродинамические поправки (strip-theory)

Дельты к базовым коэффициентам:

Здесь `y_arm_s = span_position_s` — плечо до плоскости симметрии, **со знаком** (отрицательный для левой стороны, положительный для правой); `x_arm_s` — продольное плечо от ЦМ до аэро-центра секции (положительный — впереди ЦМ).

**Силы (нормированы на `q · S_base`):**
```
ΔCy = -Σ_s (cl_alpha_contribution_s · α · f_s)
ΔCx = -Σ_s cd0_contribution_s · f_s + Σ_s 0.05 · f_s · (1 - f_s) · (area_s / S_base)
ΔCz = -Σ_s vtail_contribution_s · β · f_s
```

**Моменты — суммирование вкладов от каждой секции (с её собственным плечом):**
```
ΔMx = -Σ_s (cl_alpha_contribution_s · α · f_s · q · area_s · y_arm_s)
ΔMz = +Σ_s (Δcx_section_s · q · area_s · y_arm_s)        # ΔCx асимметрия → рысканье
ΔMy = -Σ_s (cl_alpha_contribution_s · α · f_s · q · area_s · x_arm_s)
```

где `Δcx_section_s = cd0_contribution_s · f_s + 0.05 · f_s · (1 - f_s)` — локальный вклад секции в ΔCx.

В ODE:
```python
cy_total = get_cy(...) + aero_corrections.delta_cy(α, β, q, damage_state)
mx_total = get_mx(...) + aero_corrections.delta_mx(α, β, q, damage_state)
# и т.д.
```

### Управляющие поверхности

```python
def apply_control_failures(u_command, damage_state) -> u_eff:
    for surface_name, failure in damage_state.control_failures.items():
        idx = SURFACE_TO_INPUT_INDEX[surface_name]
        if failure.mode == "jam":
            u_eff[idx] = failure.jam_position_rad
        elif failure.mode == "efficiency_loss":
            u_eff[idx] *= failure.efficiency
        elif failure.mode in ("lost", "free_floating"):
            u_eff[idx] = 0.0
```

Маппинг `SURFACE_TO_INPUT_INDEX` определяется расширенной угловой моделью с раздельными `stab_left`/`stab_right` входами (см. фаза 0).

### Двигатель

```
T_eff = T_base · damage_state.engine.thrust_factor
если hard_failure: T_eff = 0
```

### Допущения и ограничения

- **Strip-theory — приближение**, точность ~10–20% относительно полного VLM. Достаточно для RL и качественной FDM.
- **Линейность по `f_s`** — потеря половины секции уменьшает её аэровклад в 2 раза.
- **Не моделируется interference** между повреждённым крылом и хвостом.
- **На больших углах атаки** (срыв) — точность дельт падает; срыв в базовых `.npz` таблицах учтён, в дельтах — нет.

## Интеграция в Gym Env

### Расширение конструктора

```python
class NonlinearAngularF16Env(gym.Env):
    def __init__(
        self,
        # ... существующие аргументы ...
        damage_profile: Optional[DamageProfile] = None,
        damage_observable: bool = False,
        damage_event_callback: Optional[Callable[[DamageEvent, DamageState], None]] = None,
    ):
        ...
```

### `step()` (псевдокод)

```python
def step(self, action):
    t_prev = self.t
    self.t += self.dt

    if self.damage_manager:
        triggered = self.damage_manager.update(self.t, t_prev)
        for ev in triggered:
            if self.damage_event_callback:
                self.damage_event_callback(ev, self.damage_manager.state)

    action_eff = action
    if self.damage_manager:
        action_eff = controls.apply_control_failures(action, self.damage_manager.state)

    self.model.run_step(action_eff)

    obs = self._get_observation()
    if self.damage_observable:
        obs = np.concatenate([obs, self.damage_manager.state.to_vector()])

    info = self._get_info()
    if self.damage_manager:
        info["damage_state"] = self.damage_manager.state.snapshot()
        info["damage_events_triggered"] = [
            ev.label or ev.event_type for ev in triggered
        ]

    return obs, reward, terminated, truncated, info
```

### `reset()`

```python
def reset(self, *, seed=None, options=None):
    # ... существующая логика ...
    if self.damage_manager:
        self.damage_manager.reset(seed=seed)
        if options and "damage_profile" in options:
            self.damage_manager.set_profile(options["damage_profile"])
    return obs, info
```

### Рандомизация для RL

```python
class RandomDamageProfileGenerator:
    def __init__(
        self,
        event_types: list[str],
        time_range: tuple[float, float],
        severity_range: tuple[float, float],
        num_events_range: tuple[int, int],
        seed: Optional[int] = None,
    ): ...

    def sample(self) -> DamageProfile: ...
```

### Логирование

- `info["damage_state"]` каждый шаг.
- Опциональное расширение `metrics.TensorBoardSink` для записи `damage/section_loss/<name>`, `damage/engine/thrust_factor` и т.п. как scalar.

### Совместимость

Без `damage_profile` поведение env идентично текущему — все существующие тесты, обучения, скрипты работают без изменений.

## Тестирование

### Регрессионные

- `test_no_damage_bitwise_identical` — `damage_profile=None` даёт траектории, идентичные эталонным снапшотам на 1000 шагов.
- `test_zero_damage_state_identical` — пустой `DamageState` идентичен `None`.

### Калибровка геометрии

- `test_geometry_calibration_matches_params` — сумма по секциям ↔ `F16AngularParameters`, допуск `<1%`.
- `test_cg_position_consistent`.

### Физика повреждений

- `test_symmetric_section_loss_no_roll` — потеря left_tip + right_tip по 50% → нет ΔMx, симметричное падение Cy.
- `test_asymmetric_loss_produces_roll` — потеря left_tip → ΔMx нужного знака, в пределах ±20% от `q · ΔS · y_arm`.
- `test_full_left_wing_loss_dynamics` — отрицательный тест на разумность.
- `test_inertia_recompute_steiner` — пересчёт `J*` согласован с прямым применением теоремы.

### Управляющие поверхности

- `test_jam_overrides_command`
- `test_efficiency_loss_scales_command`
- `test_split_stab_asymmetry` — разные команды на `stab_left`/`stab_right` → ΔMx нужного знака.

### Двигатель и структура

- `test_thrust_factor_scales_force`
- `test_structural_mass_delta`

### События и расписание

- `test_event_triggers_at_correct_time`
- `test_multiple_events_in_one_step`
- `test_inject_event_runtime`

### Env-интеграция

- `test_env_no_damage_baseline`
- `test_env_observation_extended_when_observable`
- `test_env_reset_clears_damage`
- `test_random_profile_generator_seeded`

### Smoke / интеграционные

- `test_demo_scenario_dogfight_wing_strike` — горизонтальный полёт, в `t=10s` теряет left_tip, без управления — стабильная картина с ненулевым креном.

**Целевое покрытие:** 90%+ на новом коде в `damage/`. Покрытие существующего F-16 не должно упасть.

## Фазирование (порядок реализации)

| Фаза | Содержание |
|------|------------|
| **0. Подготовка** | Расширить угловую модель F-16 на раздельные `stab_left`/`stab_right`. Тест: симметричный случай идентичен старому. |
| **1. Geometry + DamageState** | `geometry.py`, `presets.py`, `f16_geometry.yaml`, `DamageState` (без эффектов). Тесты калибровки. |
| **2. Recompute параметров** | `recompute.py` — пересчёт `m, S, b, bA, ЦМ, J*`. Только структурные эффекты, без аэро. Тесты Гюйгенса-Штейнера. |
| **3. Aero corrections** | `aero_corrections.py` — дельты к Cy/Cx/Cz/Mx/My/Mz. Strip-theory вклады. Тесты симметрии/асимметрии. |
| **4. Control failures** | `controls.py` — jam/efficiency_loss/lost. Маппинг surface→input. |
| **5. Engine + Structural** | `propulsion.py`, structural в DamageState. |
| **6. Events + Manager + Profile** | `events.py`, `manager.py`, расписание, `presets.py` сценарии. |
| **7. Env-интеграция** | `nonlinear_angular.py`, `nonlinear_longitudinal.py`. `damage_observable`, callbacks, info. Smoke-тесты. |
| **8. Random generator + примеры** | `RandomDamageProfileGenerator`, обновление `example/` ноутбука с демо dogfight. |
| **9. Docs** | Страница в `docs/` с архитектурной диаграммой, примерами, физическими допущениями. |

Каждая фаза самодостаточна и может быть смержена отдельно.

## Открытые вопросы (для phase-0 plan)

- Точные значения `cl_alpha_contribution` и `cd0_contribution` для секций F-16 — потребуют либо ручного подбора (калибровка под целевой `Cy_α` ~ 4.5/рад), либо однократного прогона через AVL/Tornado offline, либо использование известных корневых/концевых соотношений из NASA TM на F-16.
- Как точно расширять модель на `stab_left`/`stab_right` — менять `control_space` env (breaking change) vs добавлять как опциональный режим (`split_stab=True`). Скорее второе — безопаснее для существующих агентов.
- Формат снапшота `info["damage_state"]` — полный dict или сериализованная dataclass — зависит от привычек репо (надо посмотреть на `info` в текущих env).
