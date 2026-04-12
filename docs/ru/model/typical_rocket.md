# Типичная ракета — продольная динамика

Типовая модель ракеты в продольном канале. Страница оформлена по аналогии с ELV: быстрый старт, математика, таблицы производных и API.

![typical-rocket](img/typical_rocket.png){ width=800 }

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Быстрый старт**

    Запустите среду или модель за минуты.

    [:octicons-arrow-right-24: К примеру](#быстрый-старт)

-   :material-cog-outline: **API модели**

    Документация Python‑класса типовой ракеты.

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
-0.0089 & -0.1474 & 0 & -9.75 \\
-0.0216 & -0.3601 & 5.9470 & 0.01958 \\
0 & -0.0015 & -0.0224 & 0.0006 \\
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
9.748 \\
3.77 \\
-0.034 \\
0.01
\end{bmatrix}
\eta
\]

### Производные (численные значения)

- **Матрица A (производные):**

  | Коэффициент | Значение |
  |-------------|----------|
  | x_u | -0.0089 |
  | x_w | -0.1474 |
  | x_q | 0 |
  | x_θ | -9.75 |
  | z_u | -0.0216 |
  | z_w | -0.3601 |
  | z_q | 5.9470 |
  | z_θ | 0.01958 |
  | m_u | 0.0 |
  | m_w | -0.0015 |
  | m_q | -0.0224 |
  | m_θ | 0.0006 |

- **Вход η (столбец B):**

  | Коэффициент | Значение |
  |-------------|----------|
  | x_η | 9.748 |
  | z_η | 3.77 |
  | m_η | -0.034 |
  | θ_η | 0.01 |

!!! tip "Ограничения привода"
    По умолчанию применяются предельные значения управления:

    - Максимальная величина: \(\pm 25^\circ\)
    - Максимальная скорость изменения: \(60^\circ/\text{s}\)

    Внутренние вычисления — в радианах; ограничения переводятся эквивалентно.

## Источники

1. Arikapalli V. S. N. et al. Missile Longitudinal Dynamics Control Design using Pole Placement and LQR Methods — A Critical Analysis // Defence Science Journal. 2021. 71(5). [Ссылка](https://www.strategicfront.org/forums/attachments/16232-article-text-62198-1-10-20210902-pdf.20806/)

## Награда

Функция награды по умолчанию возвращает отрицательную абсолютную ошибку отслеживания угла тангажа:

$$r_t = -|\theta(t) - \theta_{\text{ref}}(t)|$$

Чем выше награда (ближе к 0), тем лучше качество отслеживания. Пользовательская функция награды может быть передана через параметр `reward_func`.

## Быстрый старт {#быстрый-старт}

=== "Gymnasium"

    ```python
    import gymnasium as gym 
    import numpy as np

    from tensoraerospace.envs import LinearLongitudinalMissileModel
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import unit_step

    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)
    reference_signals = unit_step(degree=5, tp=tp, time_step=10, output_rad=True).reshape(1, -1)

    env = gym.make(
        'LinearLongitudinalMissileModel-v0',
        number_time_steps=number_time_steps, 
        initial_state=[[0],[0],[0],[0]],  # u, w, q, theta
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
    from tensoraerospace.aerospacemodel import MissileModel

    dt = 0.01
    number_time_steps = 200

    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    model = MissileModel(
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

    ::: tensoraerospace.aerospacemodel.rocket.MissileModel

=== "Среда Gymnasium"

    ::: tensoraerospace.envs.rocket.LinearLongitudinalMissileModel
