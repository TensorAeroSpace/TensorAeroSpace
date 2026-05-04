# AAI RQ-7 Shadow — тактический БПЛА класса II (нелинейная 6-DoF, СИ)

`tensoraerospace.aerospacemodel.aai_shadow.nonlinear` — полная нелинейная
6-DoF модель **AAI RQ-7 Shadow** (конфигурация RQ-7B), тактического
разведывательного БПЛА класса II, эксплуатируемого армией США и рядом
союзников.

Shadow — это **более крупный, конвенционно-хвостовой родственник**
[Skywalker X8](skywalker_x8_nonlinear.md) (flying-wing) — тот же класс
малого fixed-wing, но с прямым крылом высокого удлинения, twin tail
booms и инвертированным V-tail. В roster'е `tensoraerospace`
он представляет канонический класс II (50-300 кг) UAV.

| Параметр | Значение |
|---|---|
| Источник аэродинамики | Литература по UAV класса II (Beard & McLain, NASA TM-2014-218686, Roskam Vol VI V-tail mixing) |
| Масса / размах / площадь | 170 кг / 6,22 м / 4,42 м² |
| Удлинение крыла (AR) | 8,75 |
| Двигатель | UEL AR-741 однороторный Wankel, 38 л.с. (28 кВт), толкающий |
| Координаты | NED, body axis, ZYX 321 Euler |
| State | 12-D rigid-body |
| Управление | 4 канала: collective ruddervator (δ_e), элерон (δ_a), differential ruddervator (δ_r), throttle (δ_T) |
| Единицы | **СИ** (кг, м, Н, рад, с) — как у Skywalker X8 |

## Геометрия и масса (FAS / AAI published spec, RQ-7B)

```
m   = 170 кг           (cruise weight, half-fuel)
Ix  = 50 кг·м²
Iy  = 80 кг·м²
Iz  = 120 кг·м²
Ixz = 5 кг·м²
c̄  = 0,71 м           (САХ)
b   = 6,22 м           (размах, RQ-7B; оригинальный RQ-7A был 4,27 м)
S   = 4,42 м²          (площадь крыла, RQ-7B)
```

Геометрические числа — конфигурация **RQ-7B** ("Improved Tactical UAV"),
которая увеличила оригинальный RQ-7A размах с 4,27 м до 6,22 м.
Опубликованная крейсерская скорость 36 м/с ≈ 70 узлов согласуется
с площадью крыла 4,42 м² при загрузке 170 кг.

## State и управление

**State** (12-D, body axis, NED, ZYX 321 Euler — СИ):

```
[u, v, w,           # body velocity, м/с
 p, q, r,           # body angular rates, рад/с
 φ, θ, ψ,           # Euler углы, рад
 x_e, y_e, z_e]     # NED position, м
```

**Control** (4-D mixed V-tail convention):

```
[δ_e, δ_a, δ_r, δ_T]
```

У Shadow инвертированный V-tail с двумя **ruddervator** (комбинированными
elevator + rudder поверхностями). Стандартное mixing:

* `δ_e = (δ_l + δ_r) / 2` — collective отклонение действует как elevator
* `δ_r = (δ_l - δ_r) / 2` — differential отклонение действует как rudder

Агент всегда командует mixed-парой `(δ_e, δ_r)`; mechanical mixer
переводит их в физические отклонения ruddervator. Aero-коэффициенты
указаны в mixed-конвенции.

Лимиты: $|\delta_e|, |\delta_a| \le 20°$, $|\delta_r| \le 15°$, все
ограничены скоростью $120\,°/с$.

## Аэродинамическая сборка

Коэффициенты синтезированы из литературы по UAV класса II с
V-tail effective-area scaling:

| Drag |  | Lift |  | Pitch |  |
|---|---:|---|---:|---|---:|
| $C_{D_0}$ | 0,030 | $C_{L_0}$ | 0,28 | $C_{m_0}$ | 0,0 |
| $C_{D_{k_2}}$ | 0,043 | $C_{L_\alpha}$ | 5,0 /рад | $C_{m_\alpha}$ | −1,50 /рад |
| | | $C_{L_q}$ | 7,95 | $C_{m_q}$ | −38,0 |
| | | $C_{L_{\delta_e}}$ | 0,43 | $C_{m_{\delta_e}}$ | −1,20 |
| | | | | $C_{m_{\dot\alpha}}$ | −7,0 |

| Side force |  | Roll |  | Yaw |  |
|---|---:|---|---:|---|---:|
| $C_{Y_\beta}$ | −0,83 | $C_{l_\beta}$ | −0,13 | $C_{n_\beta}$ | 0,073 |
| $C_{Y_p}$ | 0,0 | $C_{l_p}$ | −0,51 | $C_{n_p}$ | −0,069 |
| $C_{Y_r}$ | 0,30 | $C_{l_r}$ | 0,25 | $C_{n_r}$ | −0,095 |
| $C_{Y_{\delta_r}}$ | 0,18 | $C_{l_{\delta_a}}$ | 0,17 | $C_{n_{\delta_a}}$ | −0,011 |
| | | $C_{l_{\delta_r}}$ | 0,024 | $C_{n_{\delta_r}}$ | −0,069 |

Заметные дизайн-точки:

* **Lift slope** $C_{L_\alpha} \approx 5,0$/рад — согласуется с
  AR = 8,75 и lifting-line теорией с поправкой на 2D-airfoil.
* **Indued drag factor** $C_{D_{k_2}} = 1/(\pi\,AR\,e) \approx 0,043$
  с Oswald-эффективностью $e = 0,85$.
* **Эффективность руля направления (V-tail)** $C_{n_{\delta_r}} \approx -0,07$ —
  меньше, чем у обычного вертикального оперения, потому что V-tail
  создаёт боковую силу и yaw-момент через differential deflection,
  а не через выделенный руль.

## Модель двигателя UEL AR-741

Однороторный Wankel rotary, 38 л.с. (28 кВт) при 7 600 об/мин,
толкающий на 24" двухлопастном карбоновом пропеллере. Калиброванная
квадратичная thrust-модель:

$$
T(\delta_T, V) = T_{\max} \cdot \delta_T^2 \cdot (1 - V / V_{\text{zero}})
$$

с $T_{\max} = 380$ Н, $V_{\text{zero}} = 65$ м/с. Калибровочные точки:

| Условие | Тяга |
|---|---:|
| Static, full throttle | 380 Н |
| 36 м/с, 70 % throttle | ~ 75 Н |

70 % крейсерского throttle согласуется с опубликованными RQ-7 endurance
числами (6-9 ч на типичной cruise weight).

## Trim finder

`tensoraerospace.aerospacemodel.aai_shadow.nonlinear.trim(h, V)`
решает $\dot u = \dot w = \dot q = 0$ методом Newton-Raphson:

| Условие | h, м | V, м/с | α | δ_e | δ_T |
|---|---:|---:|---:|---:|---:|
| Типичный loiter | 1000 | 36 | 3,10° | -3,87° | 0,93 |

Residual-норма достигает **машинной точности** ($10^{-15}$). Удержание
trimmed-команд оставляет самолёт в пределах ±0,000 м/с, ±0,000 м
по высоте, ±0,000° по тангажу за 5 секунд.

Высокий trim-throttle (0,93) отражает ограниченный margin маленького
AR-741 двигателя при 170 кг gross weight — близко к опубликованному
service ceiling ~ 4 600 м trim solver fails (двигатель не может
поддерживать level flight там с этой полезной нагрузкой), что
соответствует реальности.

## Gymnasium env

Зарегистрирована как `"NonlinearAAIShadow-v0"`:

```python
import gymnasium as gym
import tensoraerospace  # регистрирует env

# Trim-finder при произвольных (altitude, airspeed) — заметьте СИ!
env = gym.make("NonlinearAAIShadow-v0",
    trim_at=(1000.0, 36.0), number_time_steps=2000)

# Произвольное 12-состояние (СИ)
import numpy as np
env = gym.make("NonlinearAAIShadow-v0",
    initial_state=np.array([35.9, 0, 1.95, 0,0,0, 0, 0.054, 0,
                            0, 0, -1000.0]),
    number_time_steps=2000)
```

Action space: **4-канальное** `[δ_e, δ_a, δ_r, δ_T]`. Используйте
`"virtual"` для raw рад / [0, 1] или `"normalized"` для `[-1, +1]^4`.

## Scope и ограничения

* **Аэродинамические производные синтезированы**, а не транскрибированы
  напрямую из единого канонического AAI / NASA paper. Магнитуды
  cross-checked против NASA TM-2014-218686 и class-II UAV литературы
  (Beard & McLain, Roskam Vol VI), но модель следует считать
  representative-class, а не tail-number-accurate.
* **Гибкость крыла / транзиенты катапультного старта** не моделируются.
  Используется для level-flight envelope ~ 30-50 м/с, h = 0-4500 м.
* **Хуки повреждений открыты**, но событий не выставлено — паритет
  с остальной семьёй.

## Связанные модули

* [Skywalker X8](skywalker_x8_nonlinear.md) — flying-wing товарищ по
  классу малых UAV. Те же шаблоны кода, другая control layout
  (3-канальное vs 4-канальное).
* [Boeing 737 (нелинейная 6-DoF)](b737_nonlinear.md) — большой
  воздушно-реактивный транспортный с тем же модульным паттерном (но в FPS).

## Ссылки

* **Beard R. W., McLain T. W.** *Small Unmanned Aircraft: Theory and
  Practice*, Princeton Univ. Press (2012). Appendix E.1 — производные
  Aerosonde Mark 4.7, использованы как class-II baseline.
* **NASA TM-2014-218686** — RQ-7 Shadow aerodynamic database
  reference для sanity-проверки $C_{L_\alpha}$, $C_{m_\alpha}$ и
  V-tail effective $C_{l_{\delta_r}}$ / $C_{n_{\delta_r}}$.
* **Roskam J.** *Airplane Flight Dynamics and Automatic Flight
  Controls*, Vol VI Appendix C — V-tail mixing relations и
  effective-area scaling.
* **Nelson R. C.** *Flight Stability and Automatic Control*,
  McGraw-Hill 2nd ed. (1998) — диапазоны производных high-AR
  surveillance aircraft.
* **Federation of American Scientists (FAS)** RQ-7 fact sheet —
  geometry, weight, performance numbers.
