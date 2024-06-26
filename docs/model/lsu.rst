Модель БПЛА LAPAN Surveillance Aircraft (LSU)-05 NG
===================================================

Беспилотный летательный аппарат (БПЛА) - это дистанционно управляемое и автопилотируемое транспортное средство, способное поддерживать подъемную силу за счет аэродинамических сил, созданный с целью исследований, наблюдений, патрулирования, наблюдения и деятельности поисково-спасательных служб


Математическая модель 
---------------------

Объект управления построен в Пространстве состояний как и большинство объектов управления в данной библиотеке. Значения матрицы простанство состояний взяты из стать ниже.



.. math::
  
  \dot{x}=Ax+Bu

  y=Cx+Du

Так как объект управления предстовляет собой объект без внутренне возмущаю процессов выход системы  :math:`y` не учитывается в процессе моделирования по скольку матрицы  :math:`C` и  :math:`D`` представляют собой диагональную матрицу и нулевой вектор.


.. math::


  \begin{bmatrix}
  \dot{u} \\
  \dot{w} \\
  \dot{q} \\
  \dot{\theta} \\
  \end{bmatrix}
  = 
  \begin{bmatrix}
  x_u & x_w & x_q & x_{\theta} \\
  z_u & z_w & z_q & z_{\theta} \\
  m_u & m_w & m_q & m_{\theta} \\
  0 & 0 & 1 & 0 \\
  \end{bmatrix}
  \begin{bmatrix}
  u \\
  w \\
  q \\
  \theta \\
  \end{bmatrix}
  +
  \begin{bmatrix}
  x_{\eta} \\
  z_{\eta} \\
  m_{\eta} \\
  0
  \end{bmatrix}
  \eta

Поэтому объект управления представлен в следующем виде


.. math::


  \begin{bmatrix}
  \dot{u} \\
  \dot{w} \\
  \dot{q} \\
  \dot{\theta} \\
  \end{bmatrix}
  = 
  \begin{bmatrix}
  -0.00271615 & 0.248462 & 0 & -9.81 \\
  -0.257616 & -11.3097 & 68.9497 & 0\\
  0.0576336 & -7.23232 & -11.3237 & 0 \\
  0 & 0 & 1 & 0 \\
  \end{bmatrix}
  \begin{bmatrix}
  u \\
  w \\
  q \\
  \theta \\
  \end{bmatrix}
  +
  \begin{bmatrix}
  1.959083 \\
  -73.99448 \\
  -188.4752 \\
  0.0
  \end{bmatrix}
  \eta

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



Источники
---------

1. 2.	Lembaga, D.O., Antariksa, P.D., Septiyana, A., Hidayat, K., Rizaldi, A., Suseno, P.A., Jayanti, E.B., Atmasari, N., Ramadiansyah, M.L., Ramadhan, R.A., Suryo, V.N., Grüter, B., Diepolder, J., Holzapfel, F., Wijaya, Y.G., Dewan, S., Jurnal, P., Dirgantara, T., Wibowo, H., Panas, P., Septanto, H., Harno, A., Syah, N.A., Angkasa, R., Satelit, M.D., Irwanto, H.Y., Avionik, M.E., Hakim, A.N., Utama, A.B., Wahyudi, A.H., Kurniawati, F., Putro, I.E., & Astuti, R.A. STABILITY AND CONTROLLABILITY ANALYSIS ON LINEARIZED DYNAMIC SYSTEM EQUATION OF MOTION OF LSU 05-NG USING KALMAN RANK CONDITION METHOD. - Jurnal Teknologi Dirgantara Vol. 18 No. 2 Desember 2020 : hal 81 – 92 – 2020


Пример использования
--------------------

.. code:: python

    import gymnasium as gym 
    import numpy as np
    from tqdm import tqdm

    from tensoraerospace.envs import LinearLongitudinalLAPAN
    from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensoraerospace.signals.standart import unit_step

    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=20, dt=dt) # Временной периуд
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал

    env = gym.make('LinearLongitudinalLAPAN-v0',
               number_time_steps=number_time_steps, 
               initial_state=[[0],[0],[0],[9]],
               reference_signal = reference_signals)
    env.reset() 

    observation, reward, done, info = env.step(np.array([1]))

