# Спутник связи (ComSat) — продольная динамика

Искусственный спутник связи — аппарат на орбите для ретрансляции и обработки радиосигналов. Страница оформлена по аналогии с ELV: быстрый старт, математика, таблицы производных и API.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Быстрый старт**

    Запустите среду или модель за минуты.

    [:octicons-arrow-right-24: К примеру](#быстрый-старт)

-   :material-cog-outline: **API модели**

    Документация Python‑класса ComSat.

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
 x = \begin{bmatrix} x_1 \\ x_3 \\ x_4 \end{bmatrix} = \begin{bmatrix} \rho \\ \dot{\rho} \\ \dot{\theta} \end{bmatrix}, \quad
 u = u_2
\]

Линеаризованная система:

\[
\begin{bmatrix}
\dot{x}_1 \\
\dot{x}_3 \\
\dot{x}_4
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 & 0 \\
0.01036 & 0 & 0.7753 \\
0 & -0.01775 & 0
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_3 \\ x_4 \end{bmatrix}
 +
\begin{bmatrix} 0 \\ 0 \\ 0.1513 \end{bmatrix} u_2
\]

=== "Переменные состояния"

    - **x₁ = ρ**: радиальная позиция — расстояние от центра Земли, км
    - **x₃ = ρ̇**: радиальная скорость, м/с
    - **x₄ = θ̇**: угловая скорость, рад/с

=== "Управляющее воздействие"

    - **u₂**: тангенциальная тяга, Н
        - u₂ > 0 — тяга по направлению движения (ускорение спутника)
        - u₂ < 0 — тяга против направления движения (торможение)
        - u₂ = 0 — тяга отсутствует

=== "Коэффициенты системы"

    - **a₁₃ = 1.0** — радиальная позиция изменяется согласно радиальной скорости
    - **a₃₁ = 0.01036** — компонента радиального ускорения от позиции
    - **a₃₄ = 0.7753** — компонента радиального ускорения от угловой скорости
    - **a₄₃ = -0.01775** — компонента углового ускорения от радиальной скорости
    - **b₄ = 0.1513** — влияние тангенциальной тяги на угловое ускорение

!!! note "О единицах измерения"
    Угловые скорости — в радианах. Позиция в км, скорость в м/с. Методы API поддерживают преобразование единиц.

## Математическая модель {#математическая-модель}

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

Численные матрицы (линеаризованная система):

\[
\begin{bmatrix}
\dot{x}_1 \\
\dot{x}_3 \\
\dot{x}_4
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 & 0 \\
0.01036 & 0 & 0.7753 \\
0 & -0.01775 & 0 
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_3 \\
x_4 
\end{bmatrix}
 +
\begin{bmatrix}
0 \\
0 \\
0.1513
\end{bmatrix}
u_2
\]

Развёрнутая форма:
\[
\begin{aligned}
\dot{x}_1 &= x_3 \\
\dot{x}_3 &= 0.01036 \cdot x_1 + 0.7753 \cdot x_4 \\
\dot{x}_4 &= -0.01775 \cdot x_3 + 0.1513 \cdot u_2
\end{aligned}
\]

### Производные (численные значения)

- **Матрица A (производные по состояниям):**

  | Коэффициент | Значение | Физический смысл |
  |-------------|----------|------------------|
  | a₁₃ (∂ẋ₁/∂x₃) | 1.0 | Скорость изменения радиальной позиции = радиальная скорость |
  | a₃₁ (∂ẋ₃/∂x₁) | 0.01036 | Влияние позиции на радиальное ускорение |
  | a₃₄ (∂ẋ₃/∂x₄) | 0.7753 | Влияние угловой скорости на радиальное ускорение |
  | a₄₃ (∂ẋ₄/∂x₃) | -0.01775 | Влияние радиальной скорости на угловое ускорение |

- **Матрица B (управляющее воздействие):**

  | Коэффициент | Значение | Физический смысл |
  |-------------|----------|------------------|
  | b₄ (∂ẋ₄/∂u₂) | 0.1513 | Влияние тангенциальной тяги на угловое ускорение |

!!! tip "Ограничения привода"
    По умолчанию применяются предельные значения управления (внутри модели нормализованы):

    - Максимальная величина: \(\pm 25^\circ\)
    - Максимальная скорость изменения: \(60^\circ/\text{s}\)

    Внутренние вычисления — в радианах; ограничения переводятся эквивалентно.

## Источники

1. Santosh Kumar Choudhary (2015). Design and Analysis of an Optimal Orbit Control for a Communication Satellite. INTERNATIONAL JOURNAL OF COMMUNICATIONS. Volume 9, 2015

## Награда

Функция награды по умолчанию возвращает отрицательную абсолютную ошибку отслеживания радиальной скорости:

$$r_t = -|\dot{\rho}(t) - \dot{\rho}_{\text{ref}}(t)|$$

Чем выше награда (ближе к 0), тем лучше качество отслеживания. Пользовательская функция награды может быть передана через параметр `reward_func`.

## Быстрый старт {#быстрый-старт}

=== "Gymnasium"

    ```python
    import gymnasium as gym
    import numpy as np

    from tensoraerospace.envs import ComSatEnv
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import unit_step

    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)
    # Опорный сигнал для управления угловой скоростью
    reference_signals = unit_step(degree=0.1, tp=tp, time_step=10, output_rad=True).reshape(1, -1)

    env = gym.make(
        'ComSatEnv-v0',
        number_time_steps=number_time_steps,
        initial_state=[[6371.0], [0.0], [0.001]],  # [rho (км), rho_dot (м/с), theta_dot (рад/с)]
        reference_signal=reference_signals,
    )
    state, info = env.reset()
    for _ in range(200):
        action = np.array([[0.1]])  # Тангенциальная тяга u2
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    ```

=== "Только модель"

    ```python
    import numpy as np
    from tensoraerospace.aerospacemodel import ComSat

    dt = 0.01
    number_time_steps = 200

    # Начальное состояние: [rho (км), rho_dot (м/с), theta_dot (рад/с)]
    x0 = np.array([6371.0, 0.0, 0.001])

    model = ComSat(
        x0=x0,
        number_time_steps=number_time_steps,
        selected_state_output=["rho", "rho_dot", "theta_dot"],
        dt=dt,
    )

    for t in range(number_time_steps - 1):
        u = np.array([[0.05]])  # Тангенциальная тяга u2
        x_next = model.run_step(u)
    
    # Получение истории состояний
    rho_history = model.get_state('rho')
    rho_dot_history = model.get_state('rho_dot')
    theta_dot_history = model.get_state('theta_dot')
    ```

## Python API

=== "Модель"

    ::: tensoraerospace.aerospacemodel.comsat.ComSat

=== "Среда Gymnasium"

    ::: tensoraerospace.envs.comsat.ComSatEnv