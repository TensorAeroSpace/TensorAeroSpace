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
 u_{in} = \delta_e
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
\begin{bmatrix} x_{\delta_e} \\ z_{\delta_e} \\ m_{\delta_e} \\ 0 \end{bmatrix} \delta_e
\]

=== "Переменные"

    - **u**: продольная скорость, м/с
    - **w**: нормальная скорость, м/с
    - **q**: угловая скорость тангажа, рад/с
    - **θ**: тангаж, рад
    - **δₑ**: отклонение руля высоты, рад

=== "Коэффициенты"

    - **x_u, x_w, x_q, x_θ** — частные производные продольной силы \(X\) по \(u, w, q, \theta\)
    - **z_u, z_w, z_q, z_θ** — частные производные нормальной силы \(Z\)
    - **m_u, m_w, m_q, m_θ** — частные производные момента тангажа \(M\)
    - **x_δₑ, z_δₑ, m_δₑ** — производные по управляющему \(\delta_e\)

!!! note "О единицах измерения"
    Углы и угловые скорости — в радианах. Методы API позволяют получить значения в градусах.

## Математическая модель {#математическая-модель}

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

Модель описывается стандартным уравнением состояния:

$$
\dot{x} = Ax + Bu
$$

где:

- **x** — вектор состояния, \(x = [u, w, q, \theta]^{\top}\), представляющий отклонения продольной скорости, вертикальной скорости, скорости тангажа и угла тангажа.
- **u** — вектор управления, в данном случае \(u = [\delta_e]\), где \(\delta_e\) — отклонение руля высоты.
- **A** — матрица состояния (или системная матрица).
- **B** — матрица управления.

### Единицы измерения

Вектор состояния \(x = [u, w, q, \theta]^{\top}\):

- **u, w**: м/с (скорости)
- **q**: рад/с (угловая скорость)
- **θ**: рад (угол)

Вектор управления \(u = [\delta_e]\):

- **δₑ**: рад (отклонение руля высоты)

### Условия полета

Вычисленные матрицы **A** и **B** для F-4C представлены для следующих условий полета:

- **Число Маха**: 0.6
- **Высота**: 35 000 футов

### Численные матрицы

\[
\begin{bmatrix}
\dot{u} \\
\dot{w} \\
\dot{q} \\
\dot{\theta}
\end{bmatrix}
=
\begin{bmatrix}
7.6180 \times 10^{-4} & 4.7612 \times 10^{-3} & 0 & -9.8100 \\
-6.6657 \times 10^{-2} & -2.8567 \times 10^{-1} & 1.8000 \times 10^{2} & 0 \\
1.5124 \times 10^{-3} & -1.0083 \times 10^{-2} & -1.6384 \times 10^{-1} & 0 \\
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
2.6533 \times 10^{-3} \\
-6.8562 \\
-5.4446 \\
0
\end{bmatrix}
\delta_e
\]

### Производные (численные значения)

- **Матрица A (производные):**

  | Коэффициент | Значение |
  |-------------|----------|
  | x_u | 7.6180 × 10⁻⁴ |
  | x_w | 4.7612 × 10⁻³ |
  | x_q | 0.0 |
  | x_θ | -9.8100 |
  | z_u | -6.6657 × 10⁻² |
  | z_w | -2.8567 × 10⁻¹ |
  | z_q | 1.8000 × 10² |
  | z_θ | 0 |
  | m_u | 1.5124 × 10⁻³ |
  | m_w | -1.0083 × 10⁻² |
  | m_q | -1.6384 × 10⁻¹ |
  | m_θ | 0 |

- **Вход δₑ (столбец B):**

  | Коэффициент | Значение |
  |-------------|----------|
  | x_δₑ | 2.6533 × 10⁻³ |
  | z_δₑ | -6.8562 |
  | m_δₑ | -5.4446 |

!!! tip "Ограничения привода"
    По умолчанию применяются предельные значения управления:

    - Максимальная величина: \(\pm 25^\circ\)
    - Максимальная скорость изменения: \(60^\circ/\text{s}\)

    Внутренние вычисления — в радианах; ограничения переводятся эквивалентно.

## Источники

1. Heffley R. K., Jewell W. F. Aircraft handling qualities data. – NASA, 1972. № AD‑A277031.
2. Etkin B., Reid L. D. Dynamics of flight. – New York : Wiley, 1959. – Т. 2

## Награда

Функция награды по умолчанию возвращает отрицательную абсолютную ошибку отслеживания угла тангажа:

$$r_t = -|\theta(t) - \theta_{\text{ref}}(t)|$$

Чем выше награда (ближе к 0), тем лучше качество отслеживания. Пользовательская функция награды может быть передана через параметр `reward_func`.

## Быстрый старт {#быстрый-старт}

=== "Gymnasium"

    ```python
    import gymnasium as gym
    import numpy as np

    from tensoraerospace.envs import LinearLongitudinalF4C
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import unit_step

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