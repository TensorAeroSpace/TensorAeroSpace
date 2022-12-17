Модель ракеты
========================================

Ракета - летательный аппарат, двигающийся в пространстве за счёт действия реактивной тяги


.. image:: img/typical_rocket.png
  :width: 400
  :alt: Expendable launch vehicle


Математическая модель 
---------------------

Объект управления построен в Пространстве состояний как и большинство объектов управления в данной библиотеке. Значения матрицы пространство состояний взяты из стать ниже.



.. math::
  
  \dot{x}=Ax+Bu

  y=Cx+Du

Так как объект управления представляет собой объект без внутренне возмущаю процессов выход системы  :math:`y` не учитывается в процессе моделирования по скольку матрицы  :math:`C` и  :math:`D`` представляют собой диагональную матрицу и нулевой вектор.


.. math::


  \begin{bmatrix}
  \dot{u} \\
  \dot{w} \\
  \dot{q} \\
  \dot{\theta} \\
  \end{bmatrix}
  = 
  \begin{bmatrix}
  x_y & x_w & x_q & x_{\theta} \\
  z_y & z_w & z_q & z_{\theta} \\
  m_y & m_w & m_q & m_{\theta} \\
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
  -0.0089 & -0.1474 & 0 & -9.75 \\
  -0.0216 & -0.3601 & 5.9470 & -0.151 \\
  0 & -0.0015 & -0.0224 & 0.0006 \\
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
  9.748 \\
  3.77 \\
  -0.034 \\
  0.01
  \end{bmatrix}
  \eta

где

-  :math:`u` Продольная скорость ЛА [м/с]
-  :math:`w` Нормальная скорость ЛА [м/с] 
-  :math:`q` Угловая скорость Тангажа [град/с]
-  :math:`\theta` - Тангаж [град]
-  :math:`\eta` - Угол отклонения стабилизатора [град]

Модель
------

.. autoclass:: tensorairspace.aerospacemodel.MissileModel
    :members:

Источники
---------

1. Arikapalli V. S. N. et al. Missile Longitudinal Dynamics Control Design using Pole Placement and LQR Methods--A Critical Analysis //Defence Science Journal. – 2021. – Т. 71. – №. 5.