# Геостационарный спутник (GeoSat) — продольная динамика

Геостационарные спутники — ИСЗ на геостационарной орбите, неподвижные относительно поверхности Земли. Страница оформлена по аналогии с ELV: быстрый старт, математика, таблицы производных и API.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Быстрый старт**

    Запустите среду или модель за минуты.

    [:octicons-arrow-right-24: К примеру](#быстрый-старт)

-   :material-cog-outline: **API модели**

    Документация Python‑класса GeoSat.

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
 x = \begin{bmatrix} \rho & \theta & \omega \end{bmatrix}^{\top}, \quad
 u_{in} = \eta
\]

Типовая структура матриц:

\[
\begin{bmatrix}
\dot{\rho} \\
\dot{\theta} \\
\dot{\omega}
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 & 0 \\
 f_1(\rho, \omega) & 0 & f_2(\rho, \omega) \\
0 & f_3(\omega, r) & 0
\end{bmatrix}
\begin{bmatrix} \rho \\ \theta \\ \omega \end{bmatrix}
 +
\begin{bmatrix} 0 \\ 0 \\ g(r) \end{bmatrix} \eta
\]

=== "Переменные"

    - **ρ**: отношение высоты полёта к радиусу Земли, [-]
    - **θ**: позиция спутника относительно земной СК, рад
    - **ω**: угловая скорость вращения, рад/с
    - **η**: управляющее воздействие (тяга)

=== "Коэффициенты"

    - **f1(ρ, ω) ≈ 0.01036** — производная по ρ
    - **f2(ρ, ω) ≈ 0.7757** — производная по ω в уравнении θ̇
    - **f3(ω, r) ≈ -0.1775** — производная по θ в уравнении ω̇
    - **g(r) ≈ 0.1513** — влияние тяги на ω̇

!!! note "О единицах измерения"
    Углы и угловые скорости — в радианах. Методы API поддерживают выдачу в градусах.

## Математическая модель {#математическая-модель}

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

Численные матрицы (пример линеаризации):

\[
\begin{bmatrix}
\dot{\rho} \\
\dot{\theta} \\
\dot{\omega}
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 & 0 \\
0.01036 & 0 & 0.7757 \\
0 & -0.1775 & 0 
\end{bmatrix}
\begin{bmatrix}
\rho \\
\theta \\
\omega 
\end{bmatrix}
 +
\begin{bmatrix}
0 \\
0 \\
0.1513
\end{bmatrix}
\eta
\]

### Производные (численные значения)

- **Матрица A (производные):**

  | Коэффициент | Значение |
  |-------------|----------|
  | a_ρθ (∂ρ̇/∂θ) | 1.0 |
  | a_θρ (∂θ̇/∂ρ) | 0.01036 |
  | a_θω (∂θ̇/∂ω) | 0.7757 |
  | a_ωθ (∂ω̇/∂θ) | -0.1775 |

- **Вход η (столбец B):**

  | Коэффициент | Значение |
  |-------------|----------|
  | b_η→ω (∂ω̇/∂η) | 0.1513 |

## Источники

1. Tun, Hla & Mon, Lae & Lwin, Kyaw & Naing, Zaw. (2012). Implementation of Communication Satellite Orbit Controller Design Using State Space Techniques. ASEAN Journal on Science and Technology for Development. 29. 29‑49. 10.29037/ajstd.48.

## Награда

Функция награды по умолчанию возвращает отрицательную абсолютную ошибку отслеживания угловой позиции:

$$r_t = -|\theta(t) - \theta_{\text{ref}}(t)|$$

Чем выше награда (ближе к 0), тем лучше качество отслеживания. Пользовательская функция награды может быть передана через параметр `reward_func`.

## Быстрый старт {#быстрый-старт}

=== "Gymnasium"

    ```python
    import gymnasium as gym 
    import numpy as np

    from tensoraerospace.envs import GeoSatEnv
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import unit_step

    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)
    reference_signals = unit_step(degree=5, tp=tp, time_step=10, output_rad=True).reshape(1, -1)

    env = gym.make(
        'GeoSat-v0',
        number_time_steps=number_time_steps, 
        initial_state=[[0],[0],[0]],
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
    from tensoraerospace.aerospacemodel import GeoSat

    dt = 0.01
    number_time_steps = 200

    x0 = np.array([0.0, 0.0, 0.0])  # [rho, theta, omega]

    model = GeoSat(
        x0=x0,
        number_time_steps=number_time_steps,
        selected_state_output=["rho", "theta", "omega"],
        dt=dt,
    )

    for t in range(number_time_steps - 1):
        u = np.array([[0.05]])
        x_next = model.run_step(u)
    ```

## Python API

=== "Модель"

    ::: tensoraerospace.aerospacemodel.geosat.GeoSat

=== "Среда Gymnasium"

    ::: tensoraerospace.envs.geostat.GeoSatEnv
