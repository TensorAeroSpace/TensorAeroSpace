# Ultrastick‑25e — продольная динамика

БПЛА Ultrastick‑25e — лёгкая экспериментальная платформа. Страница оформлена по аналогии с ELV: быстрый старт, математика, производные и API.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Быстрый старт**

    Запустите среду или модель за минуты.

    [:octicons-arrow-right-24: К примеру](#быстрый-старт)

-   :material-cog-outline: **API модели**

    Документация Python‑класса Ultrastick.

    [:octicons-arrow-right-24: К API](#python-api)

-   :material-gamepad-variant-outline: **Среда Gymnasium**

    Готовая среда для RL‑агентов.

    [:octicons-arrow-right-24: К среде](#python-api)

-   :material-book-open-variant: **Теория**

    Уравнения состояния и численные параметры.

    [:octicons-arrow-right-24: К модели](#математическая-модель)

</div>

## Как устроен объект управления

Модель задана в пространстве состояний:

\[\dot{x} = A x + B u, \quad y = C x + D u\]

Где:

\[
 x = \begin{bmatrix} u & w & \theta & q & h \end{bmatrix}^{\top}, \quad
 u_{in} = \begin{bmatrix} \eta & \delta_t \end{bmatrix}^{\top}
\]

=== "Переменные"

- **u**: продольная скорость, м/с
- **w**: нормальная скорость, м/с
- **θ**: тангаж, рад
- **q**: угловая скорость тангажа, рад/с
- **h**: высота, м
- **η**: отклонение стабилизатора, рад
- **δ_t**: отклонение РУД (тяж), рад

!!! note "О единицах измерения"
    Углы и угловые скорости — в радианах. Методы API позволяют работать в градусах.

## Математическая модель {#математическая-модель}

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

Численные матрицы (пример линеаризации):

\[
\begin{bmatrix}
\dot{u} \\
\dot{w} \\
\dot{\theta} \\
\dot{q} \\
\dot{h}
\end{bmatrix}
=
\begin{bmatrix}
-0.5944 & 0.8008 & -9.791 & -0.8747 & 5.077\times 10^{-5} \\
-0.744 & -7.56 & -0.5294 & 15.72 & -0.000939 \\
0 & 0 & 0 & 1 & 0 \\
1.041 & -7.406 & 0 & -15.81 & -7.284\times 10^{-18} \\
-0.05399 & 0.9985 & -17 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
u \\
w \\
\theta \\
q \\
 h
\end{bmatrix}
 +
\begin{bmatrix}
0.4669 & 0 \\
-2.703 & 0 \\
0 & 0 \\
-133.7 & 0 \\
0 & 0
\end{bmatrix}
\begin{bmatrix}
\eta \\
\delta_t
\end{bmatrix}
\]

### Производные (ключевые)

- \(\theta\dot{} = q\) (элемент A[3,4] = 1)
- Влияние \(\eta\) на уравнение \(q\dot{}\): A[4,*], B[4,1] = −133.7
- Влияние \(\eta\) на уравнения \(u\dot{}\), \(w\dot{}\): B[1,1] = 0.4669, B[2,1] = −2.703

## Источники

1. Ahmed EA, Hafez A, Ouda AN, Ahmed HEH, Abd‑Elkader HM. Modelling of a Small Unmanned Aerial Vehicle. Adv Robot Autom 4:126, 2015.

## Награда

Функция награды по умолчанию возвращает отрицательную абсолютную ошибку отслеживания угла тангажа:

$$r_t = -|\theta(t) - \theta_{\text{ref}}(t)|$$

Чем выше награда (ближе к 0), тем лучше качество отслеживания. Пользовательская функция награды может быть передана через параметр `reward_func`.

## Быстрый старт {#быстрый-старт}

=== "Gymnasium"

```python
import gymnasium as gym 
import numpy as np

from tensoraerospace.envs import LinearLongitudinalUltrastick
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standart import unit_step

dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)
reference_signals = unit_step(degree=5, tp=tp, time_step=10, output_rad=True).reshape(1, -1)

env = gym.make(
    'LinearLongitudinalUltrastick-v0',
    number_time_steps=number_time_steps, 
    initial_state=[[0],[0],[0],[0],[0]],
    reference_signal=reference_signals,
)
state, info = env.reset()
for _ in range(200):
    action = np.array([[1.0, 0.1]])  # [η, δ_t]
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

=== "Только модель"

```python
import numpy as np
from tensoraerospace.aerospacemodel import Ultrastick

dt = 0.01
number_time_steps = 200

x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

model = Ultrastick(
    x0=x0,
    number_time_steps=number_time_steps,
    selected_state_output=["u", "w", "q", "theta", "h"],
    dt=dt,
)

for t in range(number_time_steps - 1):
    u = np.array([1.0, 0.1])  # [η, δ_t]
    x_next = model.run_step(u)
```

## Python API

=== "Модель"

::: tensoraerospace.aerospacemodel.ultrastick.Ultrastick

=== "Среда Gymnasium"

::: tensoraerospace.envs.ultrastick.LinearLongitudinalUltrastick

