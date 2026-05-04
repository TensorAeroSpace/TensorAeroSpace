# Boeing 737 (Нелинейная 6-DoF модель)

`tensoraerospace.aerospacemodel.b737.nonlinear` — полная нелинейная
6-DoF модель семейства Boeing 737 с двумя конфигурациями: оригинальный
737-100/200 (двойной JT8D-9) и 737-NG / 737-800 (двойной CFM56-7B27).

## Происхождение данных

В отличие от 747, F-16 и X-15, у Boeing 737 **нет** единой канонической
NASA-публикации с полным набором безразмерных производных устойчивости.
Использованные численные значения консолидированы из открытых
источников, признанных в академической работе по динамике полёта:

| Источник | Роль |
|---|---|
| **JSBSim 737 model** ([repo](https://github.com/JSBSim-Team/jsbsim/blob/master/aircraft/737/737.xml)) | Основной численный источник — геометрия, массы, инерции, полный набор коэффициентов |
| **Roskam J.** *Airplane Flight Dynamics and Automatic Flight Controls* (1995), Vol VI Appendix B | Оригинальные производные 737-100, которые JSBSim транскрибирует |
| **Hanke C. R.** *The Simulation of a Large Jet Transport Aircraft*, NASA CR-114494 (1971) | Wind-tunnel методология за таблицами Roskam |
| **Cook M. V.** *Flight Dynamics Principles* (3rd ed., Elsevier, 2013), Гл. 11 | Cross-reference 737-100 пример самолёта |
| **NASA TM-86821** *"Mach/CAS control law for the NASA TCV B737"* (1986) | Reference для нелинейной симуляции |
| **FAA TCDS A16WE** | Семейство 737, веса, геометрия, сертифицированные характеристики двигателей |
| **CFM International CFM56-3/-7B fact sheets** | Характеристики двигателей |

JSBSim лицензирован под BSD и построен явно на публично доступной
информации для учебных целей; мы транскрибируем его функциональные
формы коэффициентов как есть.

| Параметр | Значение (737-100) |
|---|---|
| Источник аэродинамики | JSBSim 737-100 / Roskam Vol VI |
| Конфигурации | B737_100, B737_800 |
| Двигатели | 2 × Pratt & Whitney JT8D-9 (737-100) или 2 × CFM56-7B27 (737-800) |
| Координаты | NED, body axis, ZYX 321 Euler |
| Управляющие поверхности | руль высоты, элероны, руль направления + throttle |
| Подсистема повреждений | Хуки открыты (паритет с B-747); событий пока нет |

## Геометрия и массы

```
                      737-100         737-800
S  (площадь крыла)    1171 ft²        1341 ft²
b  (размах)           94.7 ft         117.5 ft
c̄  (САХ)              12.31 ft        12.97 ft
W  (mid-cruise)       100 000 lb      140 000 lb
T_SLS (cluster)       29 000 lbf      54 600 lbf
```

| Конфигурация | W, lb | Iₓ, slug·ft² | I_y | I_z | I_xz |
|---|---:|---:|---:|---:|---:|
| **B737_100** | 100 000 | 562 × 10³ | 1.473 × 10⁶ | 1.894 × 10⁶ | 8.0 × 10³ |
| **B737_800** | 140 000 | 820 × 10³ | 2.300 × 10⁶ | 3.000 × 10⁶ | 12.0 × 10³ |

## Состояние и управление

**State** (12-D, body axis, NED, ZYX 321 Euler):

```
[u, v, w,           # body velocity, ft/s
 p, q, r,           # body angular rates, rad/s
 φ, θ, ψ,           # Euler angles, rad
 x_e, y_e, z_e]     # NED position, ft
```

**Control** (4-D):

```
[δ_e,  δ_a,  δ_r,  δ_T]
   ↓     ↓     ↓     ↓
elevator ail   rud   throttle
 (рад)  (рад) (рад)   [0, 1]
```

Ограничения (JSBSim 737.xml): $|\delta_e| \le 17.2°$, $|\delta_a|, |\delta_r| \le 20.1°$,
все ограничены скоростью $40\,°/с$.

## Уравнения движения

Стандартные Newton-Euler в body axis — идентичны
[B-747 нелинейный](b747_nonlinear.md#уравнения-движения). Два
отличия от 747:

1. **Лёгче планер с меньшими инерциями.** Короткопериодическая и
   Dutch-roll моды короче (~ 1–2 с против 3–4 с у 747).
2. **Двухмоторная конфигурация.** События асимметричной тяги
   используют B-737 spanwise positions $y_1 = -16{,}5$ ft, $y_2 = +16{,}5$ ft
   (737-100 inboard pylons), уже чем outer engines 747 на $\pm 71{,}7$ ft.

## Аэродинамическая сборка (JSBSim functional forms)

Коэффициенты вычисляются на каждой оценке ODE с использованием
**dimensional-decomposition подхода** из JSBSim:

| Коэффициент | Форма |
|---|---|
| $C_L$ | α-таблица (пик $C_L \approx 1{,}45$ при $\alpha = 13°$) + линейный $C_{L_{\delta_e}} = 0.20$ |
| $C_D$ | α-таблица + induced $0.043 C_L^2$ + Mach compressibility таблица + sideslip + elevator drag |
| $C_m$ | $C_{m_\alpha} = -0.6$/рад, Mach-зависимый $C_{m_{\delta_e}}$ (-1.20 до -0.30), демпфирование тангажа + α̇ |
| $C_Y, C_l, C_n$ | Линейные по (β, p̂, r̂, δa, δr); Mach-зависимый $C_{l_{\delta_a}}$ |

α-таблица плавно деградирует за пределами стола (clamped до $C_L = 0.6$
при $\alpha = 35°$), так что интегратор не взрывается, если регулятор
временно перевернёт самолёт. Эффекты земли / конфигурации (flap, gear,
speedbrake) выставлены в исходниках, но не применяются в этом MVP — модель
валидна для **чистого крейсерского режима**.

## Модель двигателя

Двухмоторный кластер с Mach-altitude-derated installed thrust по
Mattingly §8.6.4 (тот же вид, что и в JT9D-модели B-747):

$$
T_{inst}(M, h, \delta_T) = T_{SLS} \cdot \sigma(h)^{n_h} \cdot \eta_{ram}(M) \cdot \mathrm{PLA}_{eff}
$$

с $\eta_{ram}(M) = 1 - 0{,}49\sqrt{M}$ (clamped до 0.05) и
$n_h = 0.7$ ниже тропопаузы / $1.0$ выше. Та же корреляция работает
для JT8D-9 (737-100) и CFM56-7B (737-800) — cross-checked против
FAA TCDS A16WE сертифицированных значений.

| Конфигурация | $T_{SLS}$, lbf | per-engine | Тип двигателя |
|---|---:|---:|---|
| 737-100 | 29 000 | 14 500 | P&W JT8D-9 |
| 737-800 | 54 600 | 27 300 | CFM56-7B27 |

## Trim finder

`tensoraerospace.aerospacemodel.b737.nonlinear.trim(h, V)` решает
$\dot u = \dot w = \dot q = 0$ методом Newton-Raphson, возвращая
trimmed $(\alpha, \delta_e, \delta_T)$. В отличие от X-15, 737 —
обычный транспортный самолёт с воздушно-реактивными двигателями,
которые масштабируются с Mach и высотой — **trim сходится во всём
нормальном диапазоне полёта**:

| Конфигурация | h, ft | M | V, ft/s | α | δ_e | δ_T |
|---|---:|---:|---:|---:|---:|---:|
| B737-100 | 25 000 | 0.74 | 738 | 1.0° | -0.6° | 0.92 |
| B737-800 | 35 000 | 0.83 | 820 | 2.4° | -1.6° | 0.91 |

Residual-нормы достигают $10^{-13}$ — то есть машинная точность.

## Gymnasium env

Зарегистрирована как `"NonlinearB737-v0"`. Два режима инициализации:

```python
import gymnasium as gym
import tensoraerospace  # регистрирует env

# 1. Trim-finder при произвольных (h, V)
env = gym.make("NonlinearB737-v0",
    trim_at=(25_000.0, 738.0), number_time_steps=2000)

# 2. Произвольное начальное состояние
import numpy as np
env = gym.make("NonlinearB737-v0",
    initial_state=np.array([738, 0, 13, 0,0,0, 0, 0.017, 0,
                            0, 0, -25_000]),
    number_time_steps=2000)
```

Для 737-NG / 737-800 передавайте `config=B737Configuration.B737_800`
через конструктор (или прямо при создании `NonlinearB737Env(...)`).

Action-space: либо `"virtual"` (физические единицы), либо `"normalized"`
(для RL: `[-1, 1]^4`).

## Scope и ограничения

* **Механизация не моделируется** — flaps, slats, ground effect, gear,
  speedbrake-приращения выставлены в исходниках, но не применяются. Модель
  валидна для чистого крейса; для approach / landing их нужно
  включить.
* **Хуки повреждений открыты, но событий нет** — паритет с архитектурой
  B-747 (`engines_mu`, `flap_jam_config` и т. д.) подготовлен, но
  конкретных событий не добавлено. Добавить просто, поскольку модель
  двигателя уже принимает `engines_mu` dict.
* **737-NG аэродинамика — масштабированные производные 737-100** —
  Roskam Vol VI отдельно не публикует производные 737-800, поэтому
  dimensional CL/CD/Cm функции оцениваются с новыми reference area /
  span / chord. Допустимо для задач синтеза управления; для
  performance-исследований рекомендуется пересчёт.

## Связанные модули

* [Boeing 747-100 (нелинейный 6-DoF)](b747_nonlinear.md) — более
  тяжёлый 4-моторный воздушно-реактивный с каноническим NASA CR-2144
  derivative bank. Те же шаблоны кода.
* [F-16 (нелинейная продольная)](f16_nonlinear_longitudinal.md) —
  истребитель, средние Mach.
* [X-15 (нелинейный гиперзвук)](x15_nonlinear.md) — исследовательский
  ракетоплан, гиперзвуковой envelope. Те же шаблоны кода, другая
  модель двигателя.

## Ссылки

* **JSBSim** — Berndt J. S. *"JSBSim: An Open Source Flight Dynamics
  Model in C++"*, AIAA Modeling and Simulation Technologies
  Conference, 2004 ([repo](https://github.com/JSBSim-Team/jsbsim)).
* **Roskam J.** *Airplane Flight Dynamics and Automatic Flight
  Controls*, Roskam Aviation, 1995. Vol VI Appendix B.
* **Hanke C. R.** *The Simulation of a Large Jet Transport
  Aircraft*, NASA CR-114494, 1971.
* **Cook M. V.** *Flight Dynamics Principles*, Elsevier 3rd ed.,
  2013, Глава 11.
* **NASA TM-86821** — Bahm C. M., Sivolell P. *"Design and
  verification by nonlinear simulation of a Mach/CAS control law for
  the NASA TCV B737 aircraft"*, 1986
  ([NTRS 19870010857](https://ntrs.nasa.gov/citations/19870010857)).
* **FAA TCDS A16WE** — Boeing 737 type certificate data sheet.
* Mattingly J. D. *Aircraft Engine Design*, AIAA Education Series,
  2nd ed., 2002, §8.6.4 (installed-thrust lapse model).
