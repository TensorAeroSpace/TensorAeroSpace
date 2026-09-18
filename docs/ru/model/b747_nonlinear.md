# Boeing 747-100 (нелинейная 6-DoF модель)

`tensoraerospace.aerospacemodel.b747.nonlinear` — полная нелинейная
шесть-степенная модель Boeing 747-100, построенная по данным
**NASA CR-2144** (Heffley & Jewell, 1972) — *Aircraft Handling
Qualities Data*, секция IX.

| Параметр | Значение |
|---|---|
| Источник аэродинамики | NASA CR-2144 §IX (Heffley & Jewell 1972) |
| Объём опубликованных данных | 10 trim-точек × 13 longitudinal + 15 lateral non-dimensional derivatives |
| Конфигурации | NOMINAL (cruise), POWER_APPROACH, LANDING |
| Двигатели | 4 × Pratt & Whitney JT9D-7 (188 400 lb T_SLS) |
| Системы координат | NED, body axis, ZYX 321 Euler |
| Поверхности управления | elevator, aileron, rudder + throttle |
| Damage subsystem | per-surface effectiveness + jamming + decay |

![B-747 General Arrangement (NASA CR-2144 Figure IX-2)](img/b747_general_arrangement.png)

## Геометрия и масса (CR-2144 Table IX-3, Figure IX-2)

```
S    = 5500 ft²       (площадь крыла)
b    = 195.68 ft       (размах)
c̄    = 27.31 ft        (средняя аэродинамическая хорда)
c.g. = 0.25 c̄         (положение центра масс)
```

| Конфигурация | W, lb | Iₓ, slug·ft² | I_y | I_z | I_xz |
|---|---:|---:|---:|---:|---:|
| **Nominal** (TOGW − 40% fuel) | 636 600 | 18.2 × 10⁶ | 33.1 × 10⁶ | 49.7 × 10⁶ | 0.97 × 10⁶ |
| **Power Approach** (564 000 lb, 20° flaps) | 564 000 | 13.7 × 10⁶ | 30.5 × 10⁶ | 43.1 × 10⁶ | 0.825 × 10⁶ |
| **Landing** (564 000 lb, 30° flaps, gear down) | 564 000 | 13.7 × 10⁶ | 30.5 × 10⁶ | 43.1 × 10⁶ | 0.825 × 10⁶ |

## Опорная сетка trim-точек

CR-2144 публикует производные на **10 опорных точках** (Table IX-3):

| FC | Конфигурация | h, ft | M | V, ft/s | α₀, deg |
|---|---|---:|---:|---:|---:|
| 1 | LANDING        | 0      | 0.198 | 221 | 8.50 |
| 2 | POWER_APPROACH | 0      | 0.249 | 278 | 5.70 |
| 3 | NOMINAL        | 0      | 0.450 | 502 | 3.10 |
| 4 | NOMINAL        | 0      | 0.650 | 726 | 0.00 |
| 5 | NOMINAL        | 20 000 | 0.500 | 518 | 6.80 |
| 6 | NOMINAL        | 20 000 | 0.650 | 674 | 2.50 |
| 7 | NOMINAL        | 20 000 | 0.800 | 830 | 0.00 |
| 8 | NOMINAL        | 40 000 | 0.700 | 678 | 7.30 |
| 9 | NOMINAL        | 40 000 | 0.800 | 774 | 4.60 |
| 10| NOMINAL        | 40 000 | 0.900 | 871 | 2.40 |

![Огибающая полётов B-747 (NASA CR-2144 Figure IX-1)](img/b747_flight_envelope.png)

Cruise-точки 3..10 заданы регулярной решёткой `(h ∈ {0, 20K, 40K}) ×
(M ∈ {0.45..0.90})`, что позволяет билинейно интерполировать
производные внутри сертификационной огибающей.

## Состояние и управление

**Состояние** (12-D, body axis, NED, ZYX 321 Euler):

```
[u, v, w,           # body velocity, ft/s
 p, q, r,           # body angular rates, rad/s
 φ, θ, ψ,           # Euler angles, rad
 x_e, y_e, z_e]     # NED position, ft  (z_e положительно вниз ⇒ altitude = -z_e)
```

**Управление** (4-D):

```
[δ_e,  δ_a,  δ_r,  δ_T]
   ↓     ↓     ↓     ↓
elevator aileron rudder throttle
 (rad)  (rad)  (rad)   [0, 1]
```

Знаки соответствуют CR-2144 Appendix A: `δ_e > 0` опускает заднюю
кромку (нос вниз); `δ_a > 0` положительный крен; `δ_r > 0` правый
рысканий; throttle линейный, `T = T_SLS · σ(h) · η(M, h) · PLA`.

## Уравнения движения

Стандартный Newton-Euler в body axis:

$$
\dot u = (X_a + T)/m + g_x - (q\,w - r\,v)
$$

$$
\dot v = Y_a/m + g_y - (r\,u - p\,w)
$$

$$
\dot w = Z_a/m + g_z - (p\,v - q\,u)
$$

где гравитация в body-axis:

$$
g_x = -g\sin\theta, \quad g_y = g\cos\theta\sin\varphi, \quad g_z = g\cos\theta\cos\varphi.
$$

Силы $X_a, Y_a, Z_a$ получаются из stability-axis $L, D, Y$
(см. ниже) поворотом на угол атаки α:

$$
X_a = -D\cos\alpha + L\sin\alpha,\quad Z_a = -D\sin\alpha - L\cos\alpha,\quad Y_a = Y.
$$

Угловая динамика — с учётом перекрёстной инерции $I_{xz}$:

$$
\dot p = (I_z\,\bar L + I_{xz}\,\bar N) / \Gamma
$$

$$
\dot q = (M_a - (I_x - I_z)\,p\,r - I_{xz}(p^2 - r^2)) / I_y
$$

$$
\dot r = (I_{xz}\,\bar L + I_x\,\bar N) / \Gamma
$$

где $\Gamma = I_x I_z - I_{xz}^2$, $\bar L = L_a + I_{xz}(p\,q) - (I_z - I_y)\,q\,r$,
$\bar N = N_a - I_{xz}(q\,r) - (I_y - I_x)\,p\,q$.

Кинематика углов Эйлера ZYX 321 и DCM для NED-позиции —
стандартные (см. Stevens-Lewis Appendix B).

## Аэродинамическая модель

Безразмерные коэффициенты строятся **разложением Тейлора вокруг
trim-точки**:

$$
C_L(\alpha, M, q, \dot\alpha, \delta_e) = C_{L_0} + C_{L_\alpha}\,(\alpha - \alpha_0) + C_{L_q}\,\hat q + C_{L_{\dot\alpha}}\,\hat{\dot\alpha} + C_{L_M}\,\Delta M + C_{L_{\delta_e}}\,\delta_e
$$

$$
C_m(\alpha, M, q, \dot\alpha, \delta_e) = C_{m_\alpha}\,(\alpha - \alpha_0) + C_{m_q}\,\hat q + C_{m_{\dot\alpha}}\,\hat{\dot\alpha} + C_{m_M}\,\Delta M + C_{m_{\delta_e}}\,\delta_e
$$

(аналогично для $C_D$, $C_Y$, $C_l$, $C_n$). Здесь
$\hat q = q\bar c/(2V)$, $\hat p = p\,b/(2V)$, $\hat r = r\,b/(2V)$ —
стандартные безразмерные нормировки.

Полные размерные силы и моменты:

$$
L = q_{dyn}\,S\,C_L,\;\; D = q_{dyn}\,S\,C_D,\;\; Y = q_{dyn}\,S\,C_Y
$$

$$
\mathcal L = q_{dyn}\,S\,b\,C_l,\;\; \mathcal M = q_{dyn}\,S\,\bar c\,C_m,\;\; \mathcal N = q_{dyn}\,S\,b\,C_n
$$

где $q_{dyn} = \tfrac12\rho V^2$ — динамическое давление по ISA.

## Двигатель JT9D-7

Упрощённая кривая тяги имеет вид

$$
T = T_{SLS}\,L_\rho(h)\,\max(1-0.49\sqrt{M},\,0.05)\,\mathrm{PLA}_{eff}.
$$

При $h_t=36089$ ft, $\sigma=\rho/\rho_{SL}$ и
$\sigma_t=\sigma(h_t^-)$:

$$
L_\rho(h)=\begin{cases}
\sigma(h)^{0.7}, & h<h_t,\\
\sigma_t^{0.7}\,\rho(h)/\rho(h_t^+), & h\ge h_t.
\end{cases}
$$

Односторонние значения учитывают округление констант в ветвях ISA.
Верхняя ветвь согласована с нижней. Ранее смена степени без согласования
опорного значения создавала искусственный скачок тяги вниз на 30,5%.
Ниже границы закон тяги сохранён; выше тяга примерно на 43,9% больше,
чем в прежней разрывной реализации. Контроллеры для этих высот требуют
повторной оценки. Это исправление согласованности упрощённой кривой,
а не калибровка по измеренным характеристикам двигателя.
Ограничения скорости приводов, постоянные времени и раскрутка двигателя
в данной модели с 12 состояниями не применяются.

## Подсистема повреждений

Per-surface (elevator / aileron / rudder / throttle) effectiveness
$\mu_i \in [0,1]$ + jam-deflection $j_i$ + время-затухание $\tau_i$:

```
u_eff[i] = j_i  if  jam_active[i]  else  μ_i · u_cmd[i]
```

Пять типов событий (`tensoraerospace.aerospacemodel.b747.nonlinear.damage`):

| Событие | Семантика |
|---|---|
| `SurfaceEffectivenessEvent(surface, mu)` | Мгновенная потеря: `μ ← mu` |
| `SurfaceJamEvent(surface, jam_value)` | Поверхность заклинивает на `jam_value` |
| `SurfaceEffectivenessDecay(surface, τ, mu_floor)` | `μ̇ = -(1/τ)(μ − μ_floor)` |
| `EngineFailureEvent(engine_id, thrust_fraction)` | Помоторный отказ (двиг 1..4) с асимметричным yaw-моментом |
| `FlapJamEvent(jammed_config)` | Заклинивание механизации в положении NOMINAL / POWER_APPROACH / LANDING |

### Yaw-момент от асимметричной тяги

Когда `engines_mu[i]` неоднороден по 4 двигателям, модель двигателя
возвращает суммарную тягу +x **и** yaw-момент, рассчитанный по
размаху-координатам двигателей (`±35.8` ft внутренние, `±71.7` ft
внешние):

$$
N_{\text{thrust}} = -\sum_{i=1}^{4} y_i \cdot T_i,\qquad T_i = (T_\text{cluster}/4) \cdot \mu_i
$$

Отказ двигателя на левом крыле → больше тяги остаётся справа → смещение
тяги к +y → `N < 0` → нос разворачивается **влево** (в сторону мёртвого
двигателя), как в реальных V_MC-инцидентах при отказе двигателя.

### Override конфигурации (заклинивание закрылок)

`FlapJamEvent.jammed_config` подменяет выбор аэродинамической конфигурации
внутри `b747_aero`, игнорируя `params.config`. Самолёт сохраняет массу и
моменты инерции исходной конфигурации, но коэффициенты подъёма,
сопротивления и продольного момента берутся из заклиненной — это
канонический сценарий «закрылки застряли на 30° в крейсе».

Готовые пресеты:

* `ELEVATOR_50PCT_LOSS` — 50% потеря elevator при t=5 с (Lu 2019 / Wang 2019)
* `ELEVATOR_JAMMED_NOSE_UP` — Hard-over: elevator заклинило на −2°
* `AILERON_TOTAL_LOSS` — Полная потеря aileron при t=8 с
* `RUDDER_HYDRAULIC_LEAK` — Постепенный износ rudder (τ=8 с, floor=0.3)
* `ENGINE_FLAMEOUT` — Throttle стоп при t=15 с (вся тяга → idle)
* `LEFT_OUTER_ENGINE_FAILURE` — Отказ двигателя №1 в t=10 с (≈ 75% тяги + yaw-момент влево)
* `LEFT_TWO_ENGINES_OUT` — Оба левых двигателя вышли из строя в t=10 с (≈ 50% тяги, максимальная асимметрия)
* `FLAPS_JAMMED_LANDING` — Закрылки заклинило на 30° в t=5 с (высокие CL/CD, низкий V_max)
* `FLAPS_JAMMED_RETRACTED` — Закрылки не выпускаются (застряли в crisis-положении) в t=5 с

## Trim finder

`trim(h, V)` решает три продольных уравнения равновесия методом наименьших
квадратов с ограничениями: газ `[0, 1]`, руль высоты в пределах модели.
Линейные и угловые уравнения масштабируются для решателя; `residual`
возвращает немасштабированную норму всех шести ускорений в связанных осях.
`converged=True` требует успешного поиска и `residual <= tol`.

Начальное приближение газа вне диапазона ограничивается перед поиском.
Малый продольный остаток больше не скрывает ускорение рыскания при отказе
двигателя: для несимметричного установившегося полёта нужен отдельный поиск
боковых управляющих команд. Неуспешное решение не является корректным
начальным условием установившегося полёта.

## Gymnasium env

Среда копирует исходное состояние и ограничивает команды заявленными
пределами до интегрирования и применения поддерживаемых повреждений.
Нечисловые состояния/действия и некорректные настройки времени отвергаются.

Зарегистрирован под id `"NonlinearB747-v0"`. Три способа инициализации:

```python
import gymnasium as gym
import tensoraerospace  # registers env

# 1. По одной из 10 опубликованных trim-точек
env = gym.make("NonlinearB747-v0", flight_condition_id=4, number_time_steps=2000)

# 2. Trim-finder на любой (h, V)
env = gym.make("NonlinearB747-v0", trim_at=(20000.0, 674.0), number_time_steps=2000)

# 3. Произвольное стартовое состояние
env = gym.make("NonlinearB747-v0",
    initial_state=np.array([726, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    number_time_steps=2000)
```

Action-space: либо `"virtual"` (физические единицы), либо
`"normalized"` (для RL: `[-1, 1]^4`).

## Связанные модули

* [Boeing 747-100 (linear)](b747.md) — legacy single-trim-point
  state-space модель (продольный канал).
* [B-747 примеры использования](b747_examples.md) — практические
  рецепты (trim, hover, гимнастика, отказ elevator).
* [Aircraft Damage Modeling](aircraft-damage-modeling.md) — общая
  концепция damage subsystem (F-16 версия).

## Источники

* **NASA CR-2144** — Heffley R.K., Jewell W.F. *Aircraft Handling
  Qualities Data*, Systems Technology Inc., December 1972, §IX.
* Boeing 747-100 Type Certificate Data Sheet A20WE (FAA).
* Mattingly J.D. *Aircraft Engine Design*, AIAA Education Series, 2nd
  ed., 2002 — §8.6.4 (installed-thrust лepсная модель).
* Stevens B.L., Lewis F.L., Johnson E.N. *Aircraft Control and
  Simulation*, Wiley, 3rd ed., 2015 — §3.7 (trim algorithm),
  Appendix B (ZYX 321 kinematics).
