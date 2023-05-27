Модель  спутника связи
========================================

Искусственный спутник связи - космический аппарат, выведенный на орбиту в интересах обеспечения ретрансляции и обработки радиосигнала 

Математическая модель 
---------------------

Объект управления построен в Пространстве состояний как и большинство объектов управления в данной библиотеке. Значения матрицы простанство состояний взяты из стать ниже.



.. math::
  
  \dot{x}=Ax+Bu

  y=Cx+Du

Так как объект управления предстовляет собой объект без внутренне возмущаю процессов выход системы  :math:`y` не учитывается в процессе моделирования по скольку матрицы  :math:`C` и  :math:`D`` представляют собой диагональную матрицу и нулевой вектор.


.. math::


  \begin{bmatrix}
  \dot{\rho} \\
  \dot{\theta} \\
  \dot{\omega}
  \end{bmatrix}
  = 
  \begin{bmatrix}
  0 & 1 & 0  \\
  {\omega}^2 + \frac{2}{{\rho}^3} & 0 & 2\rho \omega \\
  0 & \frac{-2\omega}{r} & 0 \\
  \end{bmatrix}
  \begin{bmatrix}
  \rho \\
  \theta \\
  \omega \\
  \end{bmatrix}
  +
  \begin{bmatrix}
  0 \\
  0 \\
  \frac{1}{r} \\
  \end{bmatrix}
  \eta

Поэтому объект управления представлен в следующем виде


.. math::


  \begin{bmatrix}
  \dot{\rho} \\
  \dot{\theta} \\
  \dot{\omega}
  \end{bmatrix}
  = 
  \begin{bmatrix}
    0 & 1 & 0 \\
    0.01036 & 0 & 0.7753 \\
    0 & -0.1774 & 0 \\
  \end{bmatrix}
  \begin{bmatrix}
  \rho \\
  \theta \\
  \omega \\
  \end{bmatrix}
  +
  \begin{bmatrix}
  0 \\
  0  \\
  0.1512\\
  \end{bmatrix}
  \eta

где

-  :math:`\rho` отношение высота полета спутника к радиусу Земли [-]
-  :math:`\theta` позиция спутника относительно земносй системы координат [рад] 
-  :math:`\omega` угловая скорость вращения спутника [рад/с]
-  :math:`r` - высота полета спутника [км]

Источники
---------

1. Santosh Kumar Choudhary (2015). Design and Analysis of an Optimal Orbit Control for a Communication Satellite. INTERNATIONAL JOURNAL OF COMMUNICATIONS. Volume 9, 2015