# Моделирование повреждений ЛА

Подсистема повреждений позволяет планировать отказы в ходе симуляции —
потерю законцовки крыла, заклинивание рулевых поверхностей, отказ двигателя,
структурные изменения — и обновлять массу, тензор инерции, аэродинамические
коэффициенты и эффективность рулевых поверхностей в реальном времени.

В настоящее время поддерживается только для **нелинейной модели F-16**
(продольная и 6-DoF угловая).

## Быстрый старт

```python
import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    WING_STRIKE_LEFT_TIP,
)
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

env = NonlinearAngularF16(
    initial_state=np.zeros(14),
    number_time_steps=2000,
    damage_profile=WING_STRIKE_LEFT_TIP,
    split_stab=True,
)
obs, _ = env.reset()
for _ in range(2000):
    obs, r, term, trunc, info = env.step(np.zeros(4))
    if info.get("damage_events_triggered"):
        print(info["damage_events_triggered"])
```

Готовый к запуску пример находится в файле `example/f16_damage_dogfight_demo.py`.

## Встроенные сценарии

| Пресет | Время срабатывания | Эффект |
|--------|--------------------|--------|
| `WING_STRIKE_LEFT_TIP` | t=10 с | Полная потеря левой законцовки крыла |
| `WING_STRIKE_LEFT_HALF` | t=10 с | Левая законцовка + 50% средней секции |
| `ELEVATOR_JAM_NEUTRAL` | t=5 с  | Оба полустабилизатора заклинены в нейтральном положении |
| `ELEVATOR_JAM_PITCH_UP` | t=5 с | Оба заклинены на +10° |
| `RUDDER_LOST` | t=5 с | Руль направления утерян |
| `ENGINE_FLAMEOUT` | t=5 с | Остановка двигателя (тяга = 0) |
| `BIRDSTRIKE_COMPOUND` | t=5 с | 20% правого крыла + 70% потери мощности двигателя |

Импортировать из `tensoraerospace.aerospacemodel.f16.nonlinear.damage`.

## Пользовательские сценарии

```python
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageEvent, DamageProfile,
)

profile = DamageProfile(events=[
    DamageEvent(8.0, "section_loss",
                payload={"section": "right_mid", "loss_fraction": 0.4}),
    DamageEvent(15.0, "engine_failure",
                payload={"thrust_factor": 0.3}),
])
```

Доступные типы событий:

- `section_loss` — payload `{"section": str, "loss_fraction": float в [0,1]}`
- `control_failure` — payload `{"surface": str, "mode": str, ...специфично для режима}`
  Режимы: `"jam"` (с `jam_position_rad`), `"efficiency_loss"` (с `efficiency`), `"lost"`, `"free_floating"`
- `engine_failure` — payload `{"thrust_factor": float, "hard_failure": bool}`
- `structural_change` — payload `{"mass_delta_kg": float, "cg_shift_m": tuple, "inertia_delta": tuple}`

## Случайные профили для обучения с подкреплением

```python
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    RandomDamageProfileGenerator,
)

generator = RandomDamageProfileGenerator(
    event_types=["section_loss", "control_failure"],
    time_range=(5.0, 25.0),
    severity_range=(0.1, 1.0),
    num_events_range=(1, 2),
    seed=42,
)

profile = generator.sample()
obs, info = env.reset(options={"damage_profile": profile})
```

## Наблюдаемые повреждения

По умолчанию агент не наблюдает состояние повреждений — он должен
самостоятельно выявлять ухудшение динамики. Передайте `damage_observable=True`,
чтобы расширить вектор наблюдений долями потерь секций крыла и коэффициентом
тяги двигателя:

```python
env = NonlinearAngularF16(
    initial_state=np.zeros(14),
    number_time_steps=2000,
    damage_profile=profile,
    damage_observable=True,
    split_stab=True,
)
```

Размер вектора наблюдений увеличивается с 14 до `14 + N_sections + 1` элементов.

## Архитектура и физическая модель

Реализация находится в директории
`tensoraerospace/aerospacemodel/f16/nonlinear/damage/`. Проектный документ
расположен по пути
`docs/superpowers/specs/2026-04-28-aircraft-damage-modeling-design.md`.

Ключевые особенности:

- **Параметрический пересчёт геометрии** — при каждом событии повреждения
  масса, площадь крыла, размах, средняя аэродинамическая хорда (MAC), центр
  масс и тензор инерции пересчитываются из секционных вкладов по теореме
  Гюйгенс-Штейнер.
- **Аэродинамические поправки по strip-theory (полосовой теории)** — каждая
  секция вносит пропорциональный вклад в утраченные подъёмную силу, лобовое
  сопротивление и момент при повреждении. Приближённая точность ~10–20 % по
  сравнению с методом вихревой решётки (VLM).
- **Асимметричные повреждения требуют угловой модели 6-DoF** с
  `split_stab=True` (4-компонентное управление: `[stab_left, stab_right, ail, dir]`).
  Симметричные повреждения работают как в продольной, так и в угловой модели.
- **Бит-в-бит идентичный базовый вариант** — без `damage_profile` поведение
  среды байт-в-байт совпадает с неповреждённым базовым вариантом. Существующие
  тесты, обученные агенты и сохранённые траектории работают без изменений.

## Сброс повреждений между эпизодами

`env.reset()` очищает все повреждения и восстанавливает базовые параметры.
Для переопределения профиля в каждом эпизоде:

```python
obs, info = env.reset(options={"damage_profile": new_profile})
```

Это стандартный паттерн для обучения с подкреплением со случайными повреждениями.

## Ограничения

- Линейная модель F-16, B-747 и другие модели пока не поддерживаются.
- Аэродинамические поправки не учитывают влияние скоса потока и срывных
  течений сверх того, что уже закодировано в базовых таблицах данных.
- Упругость крыла / аэроупругие эффекты не моделируются.
- Каскадные отказы (когда одно событие вызывает другое) пока не реализованы;
  их следует явно планировать в профиле повреждений.
