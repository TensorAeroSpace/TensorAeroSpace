Примеры запуска OpenAI Gym 
==================================================


LinearLongitudinalF16-v0
------------------------
.. code:: ipython3

    import gym 
    import numpy as np
    from tqdm import tqdm
    
    from tensorairspace.envs import LinearLongitudinalF16
    from tensorairspace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensorairspace.signals.standart import unit_step
    from tensorairspace.agent.ihdp.model import IHDPAgent

.. code:: ipython3

    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=20, dt=dt) # Временной периуд
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал

.. code:: ipython3

    env = gym.make('LinearLongitudinalF16-v0',
                   number_time_steps=number_time_steps, 
                   initial_state=[[0],[0],[0],[0]],
                   reference_signal = reference_signals)
    env.reset()

.. code:: ipython3

    observation, reward, done, info = env.step(np.array([[1]]))

.. code:: ipython3

    env.model.store_input




.. parsed-literal::

    array([[1., 0., 0., ..., 0., 0., 0.]])



.. code:: ipython3

    env.reference_signal[0][1]




.. parsed-literal::

    0.0



.. code:: ipython3

    reward




.. parsed-literal::

    array([1.37305984e-05])



LinearLongitudinalB747
----------------------


.. code:: ipython3

    import gym 
    import numpy as np
    from tqdm import tqdm
    
    from tensorairspace.envs import LinearLongitudinalB747
    from tensorairspace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensorairspace.signals.standart import unit_step
    from tensorairspace.agent.ihdp.model import IHDPAgent

.. code:: ipython3

    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=20, dt=dt) # Временной периуд
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал

.. code:: ipython3

    env = gym.make('LinearLongitudinalB747-v0',
                   number_time_steps=number_time_steps, 
                   initial_state=[[0],[0],[0],[0]],
                   reference_signal = reference_signals)
    env.reset()

.. code:: ipython3

    observation, reward, done, info = env.step(np.array([[1]]))

.. code:: ipython3

    env.model.store_input


.. parsed-literal::

    array([[1., 0., 0., ..., 0., 0., 0.]])



.. code:: ipython3

    env.reference_signal[0][1]




.. parsed-literal::

    0.0



.. code:: ipython3

    reward




.. parsed-literal::

    array([2.86790771e-05])

