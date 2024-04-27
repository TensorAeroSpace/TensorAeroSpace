Модель БПЛА Ultrastick-25e
========================================

Беспилотный летательный аппарат (БПЛА) - это дистанционно управляемое и автопилотируемое транспортное средство, способное поддерживать подъемную силу за счет аэродинамических сил, , созданный с целью научных исследований


Математическая модель 
---------------------

Объект управления построен в Пространстве состояний как и большинство объектов управления в данной библиотеке. Значения матрицы простанство состояний взяты из стать ниже.



.. math::
  
  \dot{x}=Ax+Bu

  y=Cx+Du


Объект управления представлен в следующем виде


.. math::


  \begin{bmatrix}
  \dot{u} \\
  \dot{w} \\
  \dot{q} \\
  \dot{\theta} \\
  \dot{h} \\
  \end{bmatrix}
  = 
  \begin{bmatrix}
  -0.5944  &  0.8008  &  -9.791  &  -0.8747  &  5.077*〖10〗^(-5)\\
  -0.744  &  -7.56 & -0.5294 & 15.72 & -0.000939\\
  0 & 0 & 0 & 1 & 0 \\
  1.041 & -7.406 & 0 & -15.81 & -7.284*〖10〗^(-18) \\
  -0.05399 & 0.9985 & -17 & 0 & 0
  \end{bmatrix}
  \begin{bmatrix}
  u \\
  w \\
  q \\
  \theta \\
  h
  \end{bmatrix}
  +
  \begin{bmatrix}
  0.4669 & 0\\
  -2.703 & 0 \\
  0 & 0 \\
  -133.7 & 0 \\
  0 & 0
  \end{bmatrix}
  \begin{matrix}
  \eta \\
  \delta_t
  \end{matrix}

где

-  :math:`u` Продольная скорость ЛА [м/с]
-  :math:`w` Нормальная скорость ЛА [м/с] 
-  :math:`q` Угловая скорость Тангажа [град/с]
-  :math:`\theta` - Тангаж [град]
-  :math:`h` - Высота [м]
-  :math:`\eta` - Угол отклонения стабилизатора [град]
-  :math:`\delta_t` - Угол отклонения ручки управления двигателем [град]



Источники
---------

1. 3.	Ahmed EA, Hafez A, Ouda AN, Ahmed HEH, Abd-Elkader HM  Modelling of a Small Unmanned Aerial Vehicle. Adv Robot Autom 4: 126. doi:10.4172/2168-9695.1000126 - 2015



Пример использования
--------------------

.. code:: python

    import gymnasium as gym 
    import numpy as np
    from tqdm import tqdm

    from tensoraerospace.envs import LinearLongitudinalUltrastick
    from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensoraerospace.signals.standart import unit_step

    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=20, dt=dt) # Временной периуд
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал

    env = gym.make('LinearLongitudinalUltrastick-v0',
               number_time_steps=number_time_steps, 
               initial_state=[[0],[0],[0],[0],[0]],
               reference_signal = reference_signals)
    env.reset() 

    observation, reward, done, info = env.step(np.array([1,5]))

