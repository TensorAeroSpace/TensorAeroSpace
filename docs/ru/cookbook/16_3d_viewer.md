# Рецепт 16 — Интерактивный 3D-просмотрщик полётов F-16

Пакет `tensoraerospace.visualization.three_d` превращает завершённый
эпизод в самодостаточный интерактивный WebGL-просмотрщик: параметрическая
геометрия F-16 из 13 секций, мышью можно вращать камеру (orbit),
колесо — зум, анимированный след траектории и live-визуализация
повреждений — всё через тот же `DamageProfile` API, что и в симуляции.

**Связи.** [Моделирование повреждений ЛА](../model/aircraft-damage-modeling.md) ·
**Источник.** `tensoraerospace/visualization/three_d/` ·
**Запускаемый пример.** `example/visualization/example_3d_viewer_f16.py`.

## Быстрый старт

```python
import numpy as np
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageEvent, DamageProfile,
)
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

profile = DamageProfile(events=[
    DamageEvent(20.0, "section_loss",
                payload={"section": "left_tip", "loss_fraction": 1.0}),
])

env = NonlinearAngularF16(
    initial_state=np.zeros(14), number_time_steps=6010,
    dt=0.01, airspeed=200.0, split_stab=True,
    damage_profile=profile,
    render_mode="3d_web",   # НОВОЕ
)
env.reset()
for _ in range(6000):
    env.step(np.zeros(4))

env.render()
```

В обычном Python-скрипте `env.render()` открывает сгенерированный
`flight.html` в браузере по умолчанию. В Jupyter-ноутбуке возвращается
`IPython.display.HTML`, и WebGL-холст встраивается прямо в ячейку.

## Что показывает просмотрщик

* **Меш самолёта.** Параметрическая сборка F-16 из 13 секций на основе
  того же YAML `BaseGeometry`, что используется в физике, — крылья,
  стабилатор, вертикальный хвост, руль, элероны, фюзеляж. Каждая секция —
  именованный `Object3D`, поэтому damage-события адресуют конкретные части.
* **След траектории.** Синяя линия, растущая вдоль инерциального пути
  по мере продвижения анимации.
* **HUD-оверлей (верхний левый угол).** Текущее время, α, β, ωx, ωz
  и метки сработавших damage-событий за последние 1.5 с.
* **Пресеты камеры (внизу).**
  - **Free** — полный орбитальный облёт мышью (по умолчанию).
  - **Chase** — фиксация на 25 м сзади / 8 м выше самолёта,
    следование за его угловым положением.
  - **Top-down** — вид сверху в ортографическом стиле.
* **Таймлайн + скорость.** Скраб в любую точку эпизода; воспроизведение
  на 0.25× / 0.5× / 1× / 2× / 4×.

## Как выглядят повреждения

Просмотрщик читает `flight_log.damage_state_history` (бинарный поиск
на каждый кадр) и применяет три класса эффектов:

| Тип повреждения | Визуал |
|---|---|
| `section_loss` | Цвет секции линейно переходит к красному, прозрачность падает до 0; меш скрывается при `f >= 1.0`. |
| `control_failure` (jam / efficiency_loss / lost / free_floating) | Жёлтый emissive-контур на поражённой поверхности. |
| `engine_failure` | Оранжевый конус выхлопа за фюзеляжем масштабируется по `thrust_factor`; `hard_failure=True` убирает конус. |

Симметричная потеря сохраняет двустороннюю симметрию — оба конца крыла
затухают с одинаковой скоростью. Асимметричная потеря (только одна сторона)
оставляет уцелевшую половину нетронутой, пока другая исчезает — в сочетании
с моментом крена от аэродинамической коррекции это даёт наглядное объяснение
поведения угловой скорости крена после повреждения.

## Программный контроль

Высокоуровневая точка входа — `env.render()`, но можно и вручную собрать HTML:

```python
from tensoraerospace.visualization.three_d import (
    build_flight_log, build_html, save_html, render,
)

log = build_flight_log(env)        # JSON-сериализуемый словарь
html = build_html(log)             # строка с self-contained HTML
path = save_html(log, "flight.html")   # запись на диск
render(env, save_to="flight.html", open_in_browser=False)  # оба варианта
```

Для Jupyter принудительно включить inline-режим:

```python
from tensoraerospace.visualization.three_d import render
render(env, inline=True)   # → IPython.display.HTML
```

## Self-contained вывод

Сгенерированный HTML содержит:
* three.js v0.146 (UMD-бандл, ~600 КБ)
* `OrbitControls` (~26 КБ)
* viewer JS / CSS (~10 КБ)
* JSON с flight log (~50–500 КБ в зависимости от длины эпизода)

Итого на файл: ~700 КБ — ~1 МБ. **Без CDN, без локального сервера, без
сети в рантайме.** Файл можно отправить по почте или закоммитить в docs.

## Скриншот

![3D-просмотрщик F-16 с потерей законцовки](img/16_3d_viewer_screenshot.png)

## Смотрите также

- [Моделирование повреждений ЛА](../model/aircraft-damage-modeling.md)
- [Рецепт 15 — ET-DHP при повреждении ЛА](15_etdhp_damage.md)
- `example/visualization/example_3d_viewer_f16.py` — запускаемый демо-скрипт
