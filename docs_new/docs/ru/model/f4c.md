# McDonnell Douglas F‑4C — продольная динамика

F‑4C Phantom II — американский истребитель‑бомбардировщик 3‑го поколения. Страница оформлена по аналогии с ELV: быстрый старт, математика, производные и API.

![Модель F4C](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/QF-4_Holloman_AFB.jpg/1024px-QF-4_Holloman_AFB.jpg){ width=800 }

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Быстрый старт**

    Запустите среду или модель за минуты.

    [:octicons-arrow-right-24: К примеру](#быстрый-старт)

-   :material-cog-outline: **API модели**

    Документация Python‑класса продольной динамики F‑4C.

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
 x = \begin{bmatrix} u & w & q & \theta \end{bmatrix}^{\top}, \quad
 u_{in} = \eta
\]

Типовая структура матриц:

\[
\begin{bmatrix}
\dot{u} \\
\dot{w} \\
\dot{q} \\
\dot{\theta}
\end{bmatrix}
=
\begin{bmatrix}
x_u & x_w & x_q & x_{\theta} \\
z_u & z_w & z_q & z_{\theta} \\
m_u & m_w & m_q & m_{\theta} \\
0 & 0 & 1 & 0
\end{bmatrix}
\begin{bmatrix} u \\ w \\ q \\ \theta \end{bmatrix}
 +
\begin{bmatrix} x_{\eta} \\ z_{\eta} \\ m_{\eta} \\ 0 \end{bmatrix} \eta
\]

=== "Переменные"

    - **u**: продольная скорость, м/с
    - **w**: нормальная скорость, м/с
    - **q**: угловая скорость тангажа, рад/с
    - **θ**: тангаж, рад
    - **η**: управляющее отклонение стабилизатора, рад

=== "Коэффициенты"

    - **x_u, x_w, x_q, x_θ** — частные производные продольной силы \(X\) по \(u, w, q, \theta\)
    - **z_u, z_w, z_q, z_θ** — частные производные нормальной силы \(Z\)
    - **m_u, m_w, m_q, m_θ** — частные производные момента тангажа \(M\)
    - **x_η, z_η, m_η** — производные по управляющему \(\eta\)

!!! note "О единицах измерения"
    Углы и угловые скорости — в радианах. Методы API позволяют получить значения в градусах.

## Математическая модель {#математическая-модель}

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

Численные матрицы (пример линеаризации):

\[
\begin{bmatrix}
\dot{u} \\
\dot{w} \\
\dot{q} \\
\dot{\theta}
\end{bmatrix}
=
\begin{bmatrix}
-0.00679 & 0.00146 & 0 & -32.174 \\
0.0110 & -0.4940 & 1469.7600 & 0 \\
0.003410 & -0.019781184 & -0.4879811 & 0 \\
0 & 0 & 1 & 0 
\end{bmatrix}
\begin{bmatrix}
u \\
w \\
q \\
\theta 
\end{bmatrix}
 +
\begin{bmatrix}
0.0027 \\
-0.0584 \\
-0.0001309 \\
0
\end{bmatrix}
\eta
\]

### Производные (численные значения)

- **Матрица A (производные):**

  | Коэффициент | Значение |
  |-------------|----------|
  | x_u | -0.00679 |
  | x_w | 0.00146 |
  | x_q | 0.0 |
  | x_θ | -32.174 |
  | z_u | 0.0110 |
  | z_w | -0.4940 |
  | z_q | 1469.7600 |
  | z_θ | 0 |
  | m_u | 0.003410 |
  | m_w | -0.019781184 |
  | m_q | -0.4879811 |
  | m_θ | 0 |

- **Вход η (столбец B):**

  | Коэффициент | Значение |
  |-------------|----------|
  | x_η | 0.0027 |
  | z_η | -0.0584 |
  | m_η | -0.0001309 |

!!! tip "Ограничения привода"
    По умолчанию применяются предельные значения управления:

    - Максимальная величина: \(\pm 25^\circ\)
    - Максимальная скорость изменения: \(60^\circ/\text{s}\)

    Внутренние вычисления — в радианах; ограничения переводятся эквивалентно.

## Источники

1. Heffley R. K., Jewell W. F. Aircraft handling qualities data. – NASA, 1972. № AD‑A277031.
2. Etkin B., Reid L. D. Dynamics of flight. – New York : Wiley, 1959. – Т. 2

## Быстрый старт {#быстрый-старт}

=== "Gymnasium"

    ```python
    import gymnasium as gym
    import numpy as np

    from tensoraerospace.envs import LinearLongitudinalF4C
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import unit_step

    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)
    reference_signals = unit_step(degree=5, tp=tp, time_step=10, output_rad=True).reshape(1, -1)

    env = gym.make(
        'LinearLongitudinalF4C-v0',
        number_time_steps=number_time_steps,
        initial_state=[[0],[0],[0],[0]],
        reference_signal=reference_signals,
    )
    state, info = env.reset()
    for _ in range(200):
        action = np.array([[0.1]])
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    ```

=== "Только модель"

    ```python
    import numpy as np
    from tensoraerospace.aerospacemodel import LongitudinalF4C

    dt = 0.01
    number_time_steps = 200

    x0 = np.array([0.0, 0.0, 0.0, 0.0])  # [u, w, q, theta]

    model = LongitudinalF4C(
        x0=x0,
        number_time_steps=number_time_steps,
        selected_state_output=["u", "w", "q", "theta"],
        dt=dt,
    )

    for t in range(number_time_steps - 1):
        u = np.array([[0.05]])  # управление (рад)
        x_next = model.run_step(u)
    ```

## Python API

=== "Модель"

    ::: tensoraerospace.aerospacemodel.f4c.LongitudinalF4C

=== "Среда Gymnasium"

    ::: tensoraerospace.envs.f4c.LinearLongitudinalF4C