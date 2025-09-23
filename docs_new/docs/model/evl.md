# Ракета-Носитель ELV

ELV(Expendable launch vehicle) - ракета-носитель, предназначенная для выведения полезной нагрузки в космос.

![Expendable launch vehicle](img/evl.png){ width=400 }

## Математическая модель

Объект управления построен в пространстве состояний как и большинство объектов управления в данной библиотеке. Значения матрицы пространства состояний взяты из статьи ниже.

$$\dot{x} = Ax + Bu$$

$$y = Cx + Du$$

Так как объект управления представляет собой объект без внутренних возмущающих процессов выход системы $y$ не учитывается в процессе моделирования поскольку матрицы $C$ и $D$ представляют собой диагональную матрицу и нулевой вектор.

$$\begin{bmatrix}
\dot{w} \\
\dot{q} \\
\dot{\theta} \\
\end{bmatrix}
=
\begin{bmatrix}
z_w & z_q & z_{\theta} \\
m_w & m_q & m_{\theta} \\
0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
w \\
q \\
\theta \\
\end{bmatrix}
+
\begin{bmatrix}
z_{\eta} \\
m_{\eta} \\
0
\end{bmatrix}
\eta$$

Поэтому объект управления представлен в следующем виде

$$\begin{bmatrix}
\dot{w} \\
\dot{q} \\
\dot{\theta} \\
\end{bmatrix}
=
\begin{bmatrix}
-100.85 & 1 & -0.1256 \\
4.7805 & 0 & 0.01958 \\
0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
w \\
q \\
\theta \\
\end{bmatrix}
+
\begin{bmatrix}
0 \\
3.4858 \\
20.42
\end{bmatrix}
\eta$$

где

- $w$ — Нормальная скорость ЛА [м/с]
- $q$ — Угловая скорость Тангажа [град/с]
- $\theta$ — Тангаж [град]
- $\eta$ — Угол отклонения стабилизатора [град]
- $x_w$ — частная производная продольной силы по нормальной скорости
- $x_q$ — частная производная продольной силы по угловой скорости
- $x_{\theta}$ — частная производная продольной силы по углу тангажа
- $z_w$ — частная производная вертикальной силы по нормальной скорости
- $z_q$ — частная производная вертикальной силы по угловой скорости
- $z_{\theta}$ — частная производная вертикальной силы по углу тангажа
- $m_w$ — частная производная момента тангажа по нормальной скорости
- $m_q$ — частная производная момента тангажа по угловой скорости
- $m_{\theta}$ — частная производная момента тангажа по углу тангажа

## Модель

- [`ELVRocket`](../api/tensoraerospace.aerospacemodel.ELVRocket.md)

## Среда моделирования OpenAI Gym

- [`LinearLongitudinalELVRocket`](../api/tensoraerospace.envs.LinearLongitudinalELVRocket.md)

## Источники

1. Aliyu, Bhar & Funmilayo, A. & Okwo, Odooh & Sholiyi, Olusegun. (2019). State-Space Modelling of a Rocket for Optimal Control System Design. Journal of Aircraft and Spacecraft Technology. 3. 128-137. 10.3844/jastsp.2019.128.137. (https://www.researchgate.net/publication/335917723_State-Space_Modelling_of_a_Rocket_for_Optimal_Control_System_Design)
2. Aliyu, Bhar. (2011). Expendable Launch Vehicle Flight Control-Design & Simulation with Matlab/Simulink. (https://www.researchgate.net/publication/301790480_Expendable_Launch_Vehicle_Flight_Control-Design_Simulation_with_MatlabSimulink)

## Пример использования

```python
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from tensoraerospace.envs import LinearLongitudinalELVRocket
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.signals.standart import unit_step

dt = 0.01  # Дискретизация
tp = generate_time_period(tn=20, dt=dt) # Временной период
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp) # Количество временных шагов
reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал

env = gym.make('LinearLongitudinalELVRocket-v0',
           number_time_steps=number_time_steps,
           initial_state=[[0],[0],[0]],
           reference_signal = reference_signals)
env.reset()

observation, reward, done, info = env.step(np.array([[1]]))
```