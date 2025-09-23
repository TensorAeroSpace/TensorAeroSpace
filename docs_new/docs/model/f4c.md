# Самолет McDonnell Douglas F4C

McDonnell Douglas F-4C Phantom II — американский истребитель-бомбардировщик третьего поколения.

![Модель F4C](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/QF-4_Holloman_AFB.jpg/1024px-QF-4_Holloman_AFB.jpg){ width=400 }

## Математическая модель

Объект управления построен в пространстве состояний как и большинство объектов управления в данной библиотеке. Значения матрицы пространства состояний взяты из статьи ниже.

$$\dot{x} = Ax + Bu$$

$$y = Cx + Du$$

Так как объект управления представляет собой объект без внутренних возмущающих процессов выход системы $y$ не учитывается в процессе моделирования поскольку матрицы $C$ и $D$ представляют собой диагональную матрицу и нулевой вектор.

$$\begin{bmatrix}
\dot{u} \\
\dot{\alpha} \\
\dot{q} \\
\dot{\theta} \\
\end{bmatrix}
=
\begin{bmatrix}
x_u & x_{\alpha} & x_q & x_{\theta} \\
z_u & z_{\alpha} & z_q & z_{\theta} \\
m_u & m_{\alpha} & m_q & m_{\theta} \\
0 & 0 & 1 & 0 \\
\end{bmatrix}
\begin{bmatrix}
u \\
\alpha \\
q \\
\theta \\
\end{bmatrix}
+
\begin{bmatrix}
x_{\eta} \\
\alpha_{\eta} \\
m_{\eta} \\
0
\end{bmatrix}
\eta$$

Поэтому объект управления представлен в следующем виде

$$\begin{bmatrix}
\dot{u} \\
\dot{\alpha} \\
\dot{q} \\
\dot{\theta} \\
\end{bmatrix}
=
\begin{bmatrix}
-0.00679 & 0.00146 & 0 & -32.174 \\
0.0110 & -0.4940 & 1469.7600 & 0 \\
0.003410 & -0.019781184 & -0.4879811 & 0 \\
0 & 0 & 1 & 0 \\
\end{bmatrix}
\begin{bmatrix}
u \\
\alpha \\
q \\
\theta \\
\end{bmatrix}
+
\begin{bmatrix}
\\
0.0027 \\
-0.0584 \\
-0.0001309 \\
0
\end{bmatrix}
\eta$$

где

- $u$ — Продольная скорость ЛА [м/с]
- $\alpha$ — угол атаки [град]
- $q$ — Угловая скорость Тангажа [град/с]
- $\theta$ — Тангаж [град]
- $\eta$ — Угол отклонения стабилизатора [град]
- $x_u$ — частная производная продольной силы по продольной скорости
- $x_w$ — частная производная продольной силы по нормальной скорости
- $x_q$ — частная производная продольной силы по угловой скорости
- $x_{\theta}$ — частная производная продольной силы по углу тангажа
- $z_u$ — частная производная вертикальной силы по продольной скорости
- $z_w$ — частная производная вертикальной силы по нормальной скорости
- $z_q$ — частная производная вертикальной силы по угловой скорости
- $z_{\theta}$ — частная производная вертикальной силы по углу тангажа
- $m_u$ — частная производная момента тангажа по продольной скорости
- $m_w$ — частная производная момента тангажа по нормальной скорости
- $m_q$ — частная производная момента тангажа по угловой скорости
- $m_{\theta}$ — частная производная момента тангажа по углу тангажа

## Источники

1. Heffley R. K., Jewell W. F. Aircraft handling qualities data. – NASA, 1972. №.AD-A277031.
2. Etkin B., Reid L. D. Dynamics of flight. – New York : Wiley, 1959. – Т. 2

## Пример использования

```python
import gym
import numpy as np
from tqdm import tqdm

from tensoraerospace.envs import LinearLongitudinalF4C
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.signals.standart import unit_step

dt = 0.01  # Дискретизация
tp = generate_time_period(tn=20, dt=dt) # Временной период
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp) # Количество временных шагов
reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал

env = gym.make('LinearLongitudinalF4C-v0',
           number_time_steps=number_time_steps,
           initial_state=[[0],[0],[0]],
           reference_signal = reference_signals)
env.reset()

observation, reward, done, info = env.step(np.array([[1]]))
```