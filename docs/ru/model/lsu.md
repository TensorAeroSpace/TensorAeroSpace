# LSU‑05 NG — продольная динамика

LAPAN Surveillance Aircraft (LSU)‑05 NG — БПЛА для наблюдений и исследований. Страница оформлена по аналогии с ELV: быстрый старт, математика, производные и API.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Быстрый старт**

    Запустите среду или модель за минуты.

    [:octicons-arrow-right-24: К примеру](#быстрый-старт)

-   :material-cog-outline: **API модели**

    Документация Python‑класса продольной динамики LSU‑05.

    [:octicons-arrow-right-24: К API](#python-api)

-   :material-gamepad-variant-outline: **Среда Gymnasium**

    Готовая среда для RL‑агентов.

    [:octicons-arrow-right-24: К среде](#python-api)

-   :material-book-open-variant: **Теория**

    Уравнения состояния и численные параметры.

    [:octicons-arrow-right-24: К модели](#математическая-модель)

</div>

## Как устроен объект управления

\[\dot{x} = A x + B u, \quad y = C x + D u\]

\[
 x = \begin{bmatrix} u & w & q & \theta \end{bmatrix}^{\top}, \quad
 u_{in} = \eta
\]

Типовая структура и численные матрицы:

Объект управления построен в форме пространства состояний — как и
большинство объектов управления в данной библиотеке. Значения матриц
взяты из источников ниже.

## Математическая модель {#математическая-модель}

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

\[
\begin{bmatrix}
\dot{u} \\
\dot{w} \\
\dot{q} \\
\dot{\theta}
\end{bmatrix}
=
\begin{bmatrix}
-0.00271615 & 0.248462 & 0 & -9.81 \\
-0.257616 & -11.3097 & 68.9497 & 0\\
0.0576336 & -7.23232 & -11.3237 & 0 \\
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
1.959083 \\
-73.99448 \\
-188.4752 \\
0.0
\end{bmatrix}
\eta
\]

### Производные (численные значения)

- **Матрица A (производные):**

  | Коэффициент | Значение |
  |-------------|----------|
  | x_u | -0.00271615 |
  | x_w | 0.248462 |
  | x_q | 0 |
  | x_θ | -9.81 |
  | z_u | -0.257616 |
  | z_w | -11.3097 |
  | z_q | 68.9497 |
  | z_θ | 0 |
  | m_u | 0.0576336 |
  | m_w | -7.23232 |
  | m_q | -11.3237 |
  | m_θ | 0 |

- **Вход η (столбец B):**

  | Коэффициент | Значение |
  |-------------|----------|
  | x_η | 1.959083 |
  | z_η | -73.99448 |
  | m_η | -188.4752 |

где

-  :math:`u` Продольная скорость ЛА [м/с]
-  :math:`w` Нормальная скорость ЛА [м/с] 
-  :math:`q` Угловая скорость Тангажа [град/с]
-  :math:`\theta` - Тангаж [град]
-  :math:`\eta` - Угол отклонения стабилизатора [град]
-  :math:`x_u` - частная производная продольной силы по продольной скорости
-  :math:`x_w` - частная производная продольной силы по нормальной скорости
-  :math:`x_q` - частная производная продольной силы по угловой скорости
-  :math:`x_{\theta}` - частная производная продольной силы по углу тангажа
-  :math:`z_u` - частная производная вертикальной силы по продольной скорости
-  :math:`z_w` - частная производная вертикальной силы по нормальной скорости
-  :math:`z_q` - частная производная вертикальной силы по угловой скорости
-  :math:`z_{\theta}` - частная производная вертикальной силы по углу тангажа
-  :math:`m_u` - частная производная момента тангажа по продольной скорости
-  :math:`m_w` - частная производная момента тангажа по нормальной скорости
-  :math:`m_q` - частная производная момента тангажа по угловой скорости
-  :math:`m_{\theta}` - частная производная момента тангажа по углу тангажа

## Источники

1. 2.	Lembaga, D.O., Antariksa, P.D., Septiyana, A., Hidayat, K., Rizaldi, A., Suseno, P.A., Jayanti, E.B., Atmasari, N., Ramadiansyah, M.L., Ramadhan, R.A., Suryo, V.N., Grüter, B., Diepolder, J., Holzapfel, F., Wijaya, Y.G., Dewan, S., Jurnal, P., Dirgantara, T., Wibowo, H., Panas, P., Septanto, H., Harno, A., Syah, N.A., Angkasa, R., Satelit, M.D., Irwanto, H.Y., Avionik, M.E., Hakim, A.N., Utama, A.B., Wahyudi, A.H., Kurniawati, F., Putro, I.E., & Astuti, R.A. STABILITY AND CONTROLLABILITY ANALYSIS ON LINEARIZED DYNAMIC SYSTEM EQUATION OF MOTION OF LSU 05-NG USING KALMAN RANK CONDITION METHOD. - Jurnal Teknologi Dirgantara Vol. 18 No. 2 Desember 2020 : hal 81 – 92 – 2020

## Награда

Функция награды по умолчанию возвращает отрицательную абсолютную ошибку отслеживания угла тангажа:

$$r_t = -|\theta(t) - \theta_{\text{ref}}(t)|$$

Чем выше награда (ближе к 0), тем лучше качество отслеживания. Пользовательская функция награды может быть передана через параметр `reward_func`.

## Быстрый старт {#быстрый-старт}

```python

    import gymnasium as gym 
    import numpy as np
    from tqdm import tqdm

    from tensoraerospace.envs import LinearLongitudinalLAPAN
    from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensoraerospace.signals.standard import unit_step

    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=20, dt=dt) # Временной периуд
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал

    env = gym.make(
        'LinearLongitudinalLAPAN-v0',
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

## Python API

=== "Модель"

    ::: tensoraerospace.aerospacemodel.lapan.LAPAN

=== "Среда Gymnasium"

    ::: tensoraerospace.envs.lapan.LinearLongitudinalLAPAN

