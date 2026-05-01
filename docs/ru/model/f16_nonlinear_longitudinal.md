# F-16 Fighting Falcon — нелинейная продольная динамика

General Dynamics F-16 Fighting Falcon — американский многофункциональный лёгкий истребитель 4-го поколения. Данный модуль предоставляет **нелинейную** модель продольного канала полёта, реализованную на чистом Python/NumPy. Аэродинамические коэффициенты интерполируются кубическими сплайнами из таблиц аэродинамической трубы, обеспечивая высокоточную динамику в широком диапазоне углов атаки и отклонений управляющих поверхностей. В комплекте идёт среда Gymnasium для обучения агентов управления.

![Модель F-16](img/f-16_fighting_falcon.jpg){ width=800 }

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Быстрый старт**

    Запустите среду или модель в пару строк кода.

    [:octicons-arrow-right-24: К примеру](#быстрый-старт)

-   :material-cog-outline: **API модели**

    Документация Python-класса нелинейной продольной динамики F-16.

    [:octicons-arrow-right-24: К API](#python-api)

-   :material-gamepad-variant-outline: **Среда Gymnasium**

    Готовая среда для обучения агентов управления.

    [:octicons-arrow-right-24: К среде](#python-api)

-   :material-book-open-variant: **Теория**

    Нелинейные уравнения состояния и структура аэродинамических таблиц.

    [:octicons-arrow-right-24: К модели](#математическая-модель)

</div>

## ЛТХ (справочно)

| Параметр | Значение |
|-------------|----------------|
| Модификация | F-16A Block 10 |
| Размах крыла, м | 9.45 |
| Длина самолёта (со штангой ПВД), м | 15.03 |
| Высота самолёта, м | 5.09 |
| Площадь крыла, м² | 27.87 |
| Угол стреловидности, ° | 40.0 |
| Нормальная взлётная масса, кг | 11467 |

## Как устроен объект управления

В отличие от [линейной модели F-16](f16.md), использующей постоянные матрицы коэффициентов \(A\) и \(B\), нелинейная модель вычисляет аэродинамические силы и моменты из табличных данных на каждом шаге. Уравнения движения интегрируются численно (методом Эйлера или RK4).

Вектор состояния и управляющее воздействие:

\[
 x = \begin{bmatrix} \alpha & \omega_z & \delta_{\text{stab}} & \dot{\delta}_{\text{stab}} \end{bmatrix}^{\top}, \quad
 u = \delta_{\text{stab,act}}
\]

=== "Переменные"

    - **\(\alpha\)**: угол атаки, рад
    - **\(\omega_z\)**: угловая скорость тангажа, рад/с
    - **\(\delta_{\text{stab}}\)**: отклонение стабилизатора, рад
    - **\(\dot{\delta}_{\text{stab}}\)**: скорость отклонения стабилизатора, рад/с
    - **\(\delta_{\text{stab,act}}\)**: команда на стабилизатор (управляющее воздействие), рад

=== "Параметры по умолчанию"

    | Параметр | Обозначение | Значение |
    |----------|-------------|----------|
    | Масса самолёта | \(m\) | 9295.44 кг |
    | Площадь крыла | \(S\) | 27.87 м² |
    | Средняя аэродинамическая хорда | \(b_A\) | 3.45 м |
    | Момент инерции тангажа | \(J_z\) | 75673.6 кг м² |
    | Постоянная времени стабилизатора | \(T_{\text{stab}}\) | 0.03 с |
    | Коэффициент демпфирования стабилизатора | \(\xi_{\text{stab}}\) | 0.707 |
    | Высота полёта | \(H\) | 3000 м |
    | Скорость полёта | \(V\) | 150 м/с |
    | Ускорение свободного падения | \(g\) | 9.80665 м/с² |

!!! note "О единицах измерения"
    Внутри модели все углы и угловые скорости задаются в радианах. Среда Gymnasium принимает действия в **градусах** (диапазон \(\pm 25°\)) для совместимости с существующими агентами и преобразует их в радианы внутренне.

## Математическая модель {#математическая-модель}

Модель решает следующую систему нелинейных ОДУ на каждом шаге:

\[
\dot{\alpha} = \omega_z - \frac{R_y - mg}{mV}
\]

\[
\dot{\omega}_z = \frac{M_{Rz}}{J_z}
\]

где аэродинамическая сила и момент:

\[
R_y = qS \cdot C_y(\alpha, \beta, \delta_{\text{stab}}, \delta_{\text{lef}}, \omega_z, V, b_A, \delta_{\text{sb}})
\]

\[
M_{Rz} = qS b_A \cdot m_z(\alpha, \beta, \delta_{\text{stab}}, \delta_{\text{lef}}, \omega_z, V, b_A, \delta_{\text{sb}}) + x_{\text{cg}} \cdot R_y
\]

Скоростной напор \(q\) вычисляется по модели стандартной атмосферы (МСА):

\[
q = \frac{1}{2} \rho(H) V^2, \qquad \rho = \rho_0 \left(\frac{T_0 - LH}{T_0}\right)^{\frac{g}{LR} - 1}
\]

### Аэродинамические таблицы

Коэффициенты \(C_y\) и \(m_z\) не являются постоянными — это многомерные функции, интерполируемые из данных аэродинамической трубы, хранящихся в файлах `.npz`. Таблицы интерполируются кубическими сплайнами (`csaps`).

### Модель привода

Стабилизатор моделируется как колебательное звено второго порядка с ограничениями по положению и скорости:

\[
\ddot{\delta}_{\text{stab}} = \frac{-2 T_{\text{stab}} \xi_{\text{stab}} \dot{\delta}_{\text{stab}} - \delta_{\text{stab}} + \delta_{\text{stab,act}}}{T_{\text{stab}}^2}
\]

!!! tip "Ограничения привода"
    По умолчанию в модели применяются предельные значения управления:

    - Максимальное отклонение стабилизатора: \(\pm 25°\)
    - Максимальная скорость отклонения: \(\pm 60°/\text{s}\)

### Методы интегрирования

Доступны два численных интегратора:

- **Euler** (по умолчанию) — метод Эйлера первого порядка.
- **RK4** — метод Рунге-Кутты 4-го порядка, обеспечивает более высокую точность при том же шаге.

## Источник данных

1. Stevens & Lewis, "Aircraft Control and Simulation".
2. Аэродинамические таблицы F-16A по данным аэродинамической трубы, сохранённые в формате NumPy `.npz`.

## Быстрый старт {#быстрый-старт}

=== "Gymnasium"

    ```python
    import gymnasium as gym
    import numpy as np

    from tensoraerospace.envs import NonlinearLongitudinalF16
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import sinusoid

    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)
    reference_signal = sinusoid(
        degree=3, tp=tp, frequency=0.1, output_rad=True
    ).reshape(1, -1)

    env = gym.make(
        'NonlinearLongitudinalF16-v0',
        number_time_steps=number_time_steps,
        initial_state=np.array([0.0, 0.0]),
        reference_signal=reference_signal,
        dt=dt,
        integrator="euler",
    )

    state, info = env.reset()
    for _ in range(number_time_steps - 1):
        action = np.array([0.0])  # градусы
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    ```

=== "Только модель"

    ```python
    import numpy as np
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import LongitudinalF16

    dt = 0.01
    number_time_steps = 200

    # Состояние: [alpha, wz, stab, dstab] (рад)
    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    model = LongitudinalF16(
        x0=x0,
        selected_state_output=["alpha", "wz"],
        dt=dt,
        integrator="rk4",
    )

    for t in range(number_time_steps - 1):
        u = np.array([np.radians(-2.0)])  # команда на стабилизатор (рад)
        state_next = model.run_step(u)
    ```

## Python API

=== "Модель"

    ::: tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.model.LongitudinalF16

=== "Среда Gymnasium"

    ::: tensoraerospace.envs.f16.nonlinear_longitudinal.NonlinearLongitudinalF16

=== "Параметры"

    ::: tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params.F16LongParameters
