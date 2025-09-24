# БПЛА (UAV) — продольная динамика

Беспилотный летательный аппарат (UAV) — дистанционно управляемое или автономное воздушное судно. Страница оформлена по аналогии с ELV: быстрый старт, математика, производные и API.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Быстрый старт**

    Запустите среду или модель за минуты.

    [:octicons-arrow-right-24: К примеру](#быстрый-старт)

-   :material-cog-outline: **API модели**

    Документация Python‑класса продольной динамики UAV.

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
    Углы и угловые скорости — в радианах. Методы API поддерживают выдачу в градусах.

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
-0.1982 & 0.593 & 1.245 & -9.779 \\
-0.7239 & -3.9848 & 18.7028 & -0.6286 \\
0.3537 & -5.5023 & -5.4722 & 0 \\
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
0.2281 \\
-4.6830  \\
-36.1341 \\
0.0
\end{bmatrix}
\eta
\]

### Производные (численные значения)

- **Матрица A (производные):**

  | Коэффициент | Значение |
  |-------------|----------|
  | x_u | -0.1982 |
  | x_w | 0.593 |
  | x_q | 1.245 |
  | x_θ | -9.779 |
  | z_u | -0.7239 |
  | z_w | -3.9848 |
  | z_q | 18.7028 |
  | z_θ | -0.6286 |
  | m_u | 0.3537 |
  | m_w | -5.5023 |
  | m_q | -5.4722 |
  | m_θ | 0.0 |

- **Вход η (столбец B):**

  | Коэффициент | Значение |
  |-------------|----------|
  | x_η | 0.2281 |
  | z_η | -4.6830 |
  | m_η | -36.1341 |

## Источники

1. A. Rauf, Muhammad Aamir Zafar, Z. Ashraf and H. Akhtar, "Aerodynamic modeling and state-space model extraction of a UAV using DATCOM and Simulink," 2011 3rd International Conference on Computer Research and Development, Shanghai, China, 2011, pp. 88-92, doi: 10.1109/ICCRD.2011.5763860.

## Быстрый старт {#быстрый-старт}

=== "Gymnasium"

    ```python
    import gymnasium as gym 
    import numpy as np

    from tensoraerospace.envs import LinearLongitudinalUAV
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import unit_step

    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)
    reference_signals = unit_step(degree=5, tp=tp, time_step=10, output_rad=True).reshape(1, -1)

    env = gym.make(
        'LinearLongitudinalUAV-v0',
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
    from tensoraerospace.aerospacemodel import LongitudinalUAV

    dt = 0.01
    number_time_steps = 200

    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    model = LongitudinalUAV(
        x0=x0,
        number_time_steps=number_time_steps,
        selected_state_output=["u", "w", "q", "theta"],
        dt=dt,
    )

    for t in range(number_time_steps - 1):
        u = np.array([[0.05]])
        x_next = model.run_step(u)
    ```

## Python API

=== "Модель"

    ::: tensoraerospace.aerospacemodel.uav.LongitudinalUAV

=== "Среда Gymnasium"

    ::: tensoraerospace.envs.uav.LinearLongitudinalUAV