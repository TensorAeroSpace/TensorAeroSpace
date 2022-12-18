Типичная Ракета-Носитель ELV
========================================

ELV(Expendable launch vehicle) - ракета-носитель, предназначенная для выведения полезной нагрузки в космос.

.. image:: img/evl.png
  :width: 400
  :alt: Expendable launch vehicle

Математическая модель 
---------------------

Объект управления построен в Пространстве состояний как и большинство объектов управления в данной библиотеке. Значения матрицы простанство состояний взяты из стать ниже.



.. math::
  
  \dot{x}=Ax+Bu

  y=Cx+Du

Так как объект управления предстовляет собой объект без внутренне возмущаю процессов выход системы  :math:`y` не учитывается в процессе моделирования по скольку матрицы  :math:`C` и  :math:`D`` представляют собой диагональную матрицу и нулевой вектор.


.. math::


  \begin{bmatrix}
  \dot{w} \\
  \dot{q} \\
  \dot{\theta} \\
  \end{bmatrix}
  = 
  \begin{bmatrix}
  z_y & z_w & z_q & z_{\theta} \\
  m_y & m_w & m_q & m_{\theta} \\
  0 & 0 & 1 & 0 \\
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
  \eta

Поэтому объект управления представлен в следующем виде


.. math::


  \begin{bmatrix}
  \dot{w} \\
  \dot{q} \\
  \dot{\theta} \\
  \end{bmatrix}
  = 
  \begin{bmatrix}
  -100.85 & 1 & -0.1256 \\
  4.7805 & 0 & 0.01958  \\
  0 & 0 & 1  \\
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
  \eta

где

-  :math:`w` Нормальная скорость ЛА [м/с] 
-  :math:`q` Угловая скорость Тангажа [град/с]
-  :math:`\theta` - Тангаж [град]
-  :math:`\eta` - Угол отклонения стабилизатора [град]
-  :math:`z_y , z_w , z_q , z_{\theta} ` - частные производные вертикальной силы
-  :math:`m_y , m_w , m_q , m_{\theta} ` - частные производные момента тангажа



Модель
------

.. autoclass:: tensorairspace.aerospacemodel.ELVRocket
    :members:


Среда моделирования OpenAI Gym
------------------------------

.. autoclass:: tensorairspace.envs.LinearLongitudinalELVRocket
    :members:



Источники
---------

1. Aliyu, Bhar & Funmilayo, A. & Okwo, Odooh & Sholiyi, Olusegun. (2019). State-Space Modelling of a Rocket for Optimal Control System Design. Journal of Aircraft and Spacecraft Technology. 3. 128-137. 10.3844/jastsp.2019.128.137.
2. Aliyu, Bhar. (2011). Expendable Launch Vehicle Flight Control-Design & Simulation with Matlab/Simulink. 