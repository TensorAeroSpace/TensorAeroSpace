# F-16 Fighting Falcon — нелинейная 6-DoF угловая динамика

Данный модуль предоставляет **нелинейную 6-DoF угловую** модель F-16 Fighting Falcon, реализованную на чистом Python/NumPy. Модель охватывает полную связанную динамику: продольный, боковой и путевой каналы. Аэродинамические коэффициенты (шесть компонент сил и моментов) интерполируются из таблиц аэродинамической трубы.

![Модель F-16](img/f-16_fighting_falcon.jpg){ width=800 }

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Быстрый старт**

    Запустите угловую модель в пару строк кода.

    [:octicons-arrow-right-24: К примеру](#быстрый-старт)

-   :material-cog-outline: **API модели**

    Документация Python-класса 6-DoF динамики.

    [:octicons-arrow-right-24: К API](#python-api)

-   :material-book-open-variant: **Теория**

    Нелинейные 6-DoF уравнения движения.

    [:octicons-arrow-right-24: К модели](#математическая-модель)

</div>

## Как устроен объект управления

Модель описывает полное угловое движение F-16 с тремя независимыми управляющими поверхностями: стабилизатор (руль высоты), элероны и руль направления.

**Вектор состояния** (14 элементов):

\[
x = \begin{bmatrix}
\alpha & \beta & \omega_x & \omega_y & \omega_z & \gamma & \psi & \theta &
\delta_{\text{stab}} & \dot{\delta}_{\text{stab}} &
\delta_{\text{ail}} & \dot{\delta}_{\text{ail}} &
\delta_{\text{dir}} & \dot{\delta}_{\text{dir}}
\end{bmatrix}^{\top}
\]

**Вектор управления** (3 элемента):

\[
u = \begin{bmatrix}
\delta_{\text{stab,act}} & \delta_{\text{ail,act}} & \delta_{\text{dir,act}}
\end{bmatrix}^{\top}
\]

=== "Переменные состояния"

    | Переменная | Обозначение | Описание |
    |------------|-------------|----------|
    | `alpha` | \(\alpha\) | Угол атаки, рад |
    | `beta` | \(\beta\) | Угол скольжения, рад |
    | `wx` | \(\omega_x\) | Угловая скорость крена, рад/с |
    | `wy` | \(\omega_y\) | Угловая скорость рыскания, рад/с |
    | `wz` | \(\omega_z\) | Угловая скорость тангажа, рад/с |
    | `gamma` | \(\gamma\) | Угол крена, рад |
    | `psi` | \(\psi\) | Угол рыскания, рад |
    | `theta` | \(\theta\) | Угол тангажа, рад |
    | `stab` | \(\delta_{\text{stab}}\) | Положение стабилизатора, рад |
    | `dstab` | \(\dot{\delta}_{\text{stab}}\) | Скорость стабилизатора, рад/с |
    | `ail` | \(\delta_{\text{ail}}\) | Положение элеронов, рад |
    | `dail` | \(\dot{\delta}_{\text{ail}}\) | Скорость элеронов, рад/с |
    | `dir` | \(\delta_{\text{dir}}\) | Положение руля направления, рад |
    | `ddir` | \(\dot{\delta}_{\text{dir}}\) | Скорость руля направления, рад/с |

=== "Переменные управления"

    | Переменная | Обозначение | Описание |
    |------------|-------------|----------|
    | `stab_act` | \(\delta_{\text{stab,act}}\) | Команда на стабилизатор, рад |
    | `ail_act` | \(\delta_{\text{ail,act}}\) | Команда на элероны, рад |
    | `dir_act` | \(\delta_{\text{dir,act}}\) | Команда на руль направления, рад |

=== "Параметры по умолчанию"

    | Параметр | Обозначение | Значение |
    |----------|-------------|----------|
    | Масса самолёта | \(m\) | 9295.44 кг |
    | Длина фюзеляжа | \(l\) | 9.144 м |
    | Площадь крыла | \(S\) | 27.87 м² |
    | Размах крыла | \(b_A\) | 3.45 м |
    | Момент инерции крена | \(J_x\) | 12874.8 кг м² |
    | Момент инерции тангажа | \(J_y\) | 85552.1 кг м² |
    | Момент инерции рыскания | \(J_z\) | 75673.6 кг м² |
    | Центробежный момент инерции | \(J_{xy}\) | 1331.4 кг м² |
    | Высота полёта | \(H\) | 3000 м |
    | Скорость полёта | \(V\) | 120 м/с |

!!! note "О единицах измерения"
    Внутри модели все углы и угловые скорости задаются в радианах.

## Математическая модель {#математическая-модель}

### Аэродинамические силы и моменты

На каждом шаге вычисляются шесть аэродинамических коэффициентов из табличных данных:

\[
C_x, C_y, C_z, m_x, m_y, m_z = f(\alpha, \beta, \delta_{\text{stab}}, \delta_{\text{ail}}, \delta_{\text{dir}}, \delta_{\text{lef}}, \omega_x, \omega_y, \omega_z, V, \ldots)
\]

Силы и моменты в связанной системе координат:

\[
X = -qSC_x, \quad Y = qSC_y, \quad Z = qSC_z
\]

\[
M_x = qSlm_x, \quad M_y = qSlm_y, \quad M_z = qSb_Am_z
\]

с поправкой на положение центра масс:

\[
M_{Ry} = M_y - x_{\text{cg}} Z, \qquad M_{Rz} = M_z + x_{\text{cg}} Y
\]

### Уравнения моментов количества движения

\[
\dot{\omega}_x = \frac{J_y M_{Rx} + J_{xy}(M_{Ry} - h_{Ex}\omega_z) + J_{xy}(J_z - J_x - J_y)\omega_x\omega_z + (J_{xy}^2 + J_y(J_y - J_z))\omega_y\omega_z}{\Gamma}
\]

\[
\dot{\omega}_z = \frac{M_{Rz} + h_{Ex}\omega_y + J_{xy}(\omega_x^2 - \omega_y^2) + (J_x - J_y)\omega_x\omega_y}{J_z}
\]

где \(\Gamma = J_x J_y - J_{xy}^2\).

### Угол атаки и скольжения

\[
\dot{\alpha} = \omega_z + (\omega_y \sin\alpha - \omega_x \cos\alpha)\tan\beta - \frac{Y_a + mg_{ay}}{mV\cos\beta}
\]

\[
\dot{\beta} = \omega_x \sin\alpha + \omega_y \cos\alpha + \frac{Z_a + mg_{az}}{mV}
\]

### Кинематика углов Эйлера

\[
\dot{\gamma} = \omega_x - \cos\gamma \tan\theta \cdot \omega_y + \sin\gamma \tan\theta \cdot \omega_z
\]

\[
\dot{\theta} = \sin\gamma \cdot \omega_y + \cos\gamma \cdot \omega_z
\]

\[
\dot{\psi} = \frac{\cos\gamma}{\cos\theta} \omega_y - \frac{\sin\gamma}{\cos\theta} \omega_z
\]

### Модели приводов

Каждая управляющая поверхность моделируется как колебательное звено второго порядка с ограничениями по положению и скорости:

| Поверхность | Постоянная времени | Демпфирование | Макс. отклонение | Макс. скорость |
|-------------|-------------------|---------------|-----------------|----------------|
| Стабилизатор | 0.03 с | 0.707 | \(\pm 25°\) | \(\pm 60°/\text{s}\) |
| Элероны | 0.02 с | 0.707 | \(\pm 21.5°\) | \(\pm 80°/\text{s}\) |
| Руль направления | 0.03 с | 0.707 | \(\pm 30°\) | \(\pm 120°/\text{s}\) |

### Методы интегрирования

Доступны два численных интегратора:

- **Euler** (по умолчанию) — метод Эйлера первого порядка.
- **RK4** — метод Рунге-Кутты 4-го порядка, более высокая точность при том же шаге.

## Источник данных

1. Stevens & Lewis, "Aircraft Control and Simulation".
2. Аэродинамические таблицы F-16A по данным аэродинамической трубы, сохранённые в формате NumPy `.npz`.

## Быстрый старт {#быстрый-старт}

```python
import numpy as np
from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16

dt = 0.01
number_time_steps = 500

# Состояние (14 элементов): [alpha, beta, wx, wy, wz, gamma, psi, theta,
#                             stab, dstab, ail, dail, dir, ddir]
x0 = np.zeros(14)
x0[0] = np.radians(2.0)  # начальный alpha = 2 градуса

model = AngularF16(
    x0=x0,
    selected_state_output=["alpha", "beta", "wz"],
    dt=dt,
    integrator="rk4",
)

for t in range(number_time_steps - 1):
    u = np.array([np.radians(-2.0), 0.0, 0.0])  # [stab, ail, dir] (рад)
    state_next = model.run_step(u)
```

## Python API

=== "Модель"

    ::: tensoraerospace.aerospacemodel.f16.nonlinear.angular.model.AngularF16

=== "Параметры"

    ::: tensoraerospace.aerospacemodel.f16.nonlinear.angular.params.F16AngularParameters
