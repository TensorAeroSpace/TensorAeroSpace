Примеры запуска OpenAI Gym 
==================================================


LinearLongitudinalF16-v0
------------------------
.. code:: python

    import gym 
    import numpy as np
    from tqdm import tqdm
    
    from tensorairspace.envs import LinearLongitudinalF16
    from tensorairspace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensorairspace.signals.standart import unit_step
    from tensorairspace.agent.ihdp.model import IHDPAgent

    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=20, dt=dt) # Временной период
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал

    env = gym.make('LinearLongitudinalF16-v0',
                   number_time_steps=number_time_steps, 
                   initial_state=[[0],[0],[0],[0]],
                   reference_signal = reference_signals)
    env.reset()
    observation, reward, done, info = env.step(np.array([[1]]))


LinearLongitudinalB747-v0
-------------------------


.. code:: python

    import gym 
    import numpy as np
    from tqdm import tqdm
    
    from tensorairspace.envs import LinearLongitudinalB747
    from tensorairspace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensorairspace.signals.standart import unit_step
    from tensorairspace.agent.ihdp.model import IHDPAgent

    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=20, dt=dt) # Временной период
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал


    env = gym.make('LinearLongitudinalB747-v0',
                   number_time_steps=number_time_steps, 
                   initial_state=[[0],[0],[0],[0]],
                   reference_signal = reference_signals)
    env.reset()
    observation, reward, done, info = env.step(np.array([[1]]))



LinearLongitudinalMissileModel-v0
---------------------------------



.. code:: python

    import gym 
    import numpy as np
    from tqdm import tqdm
    
    from tensorairspace.envs import LinearLongitudinalMissileModel
    from tensorairspace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensorairspace.signals.standart import unit_step
    from tensorairspace.agent.ihdp.model import IHDPAgent

    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=20, dt=dt) # Временной период
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал


    env = gym.make('LinearLongitudinalMissileModel-v0',
                   number_time_steps=number_time_steps, 
                   initial_state=[[0],[0],[0],[0]],
                   reference_signal = reference_signals)
    env.reset()
    observation, reward, done, info = env.step(np.array([[1]]))


LinearLongitudinalELVRocket-v0
------------------------------



.. code:: python

    import gym 
    import numpy as np
    from tqdm import tqdm
    
    from tensorairspace.envs import LinearLongitudinalELVRocket
    from tensorairspace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensorairspace.signals.standart import unit_step


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




GeoSat-v0
-----------------------

.. code:: python

    import gym 
    import numpy as np
    from tqdm import tqdm

    from tensorairspace.envs import GeoSatEnv
    from tensorairspace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensorairspace.signals.standart import unit_step

    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=20, dt=dt) # Временной периуд
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал

    env = gym.make('GeoSat-v0',
               number_time_steps=number_time_steps, 
               initial_state=[[0],[0],[0]],
               reference_signal = reference_signals)
    env.reset() 

    observation, reward, done, info = env.step(np.array([[1]]))


ComSatEnv-v0
-------------

.. code:: python

    import gym 
    import numpy as np
    from tqdm import tqdm

    from tensorairspace.envs import ComSatEnv
    from tensorairspace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensorairspace.signals.standart import unit_step

    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=20, dt=dt) # Временной периуд
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал

    env = gym.make('ComSatEnv-v0',
               number_time_steps=number_time_steps, 
               initial_state=[[0],[0],[0]],
               reference_signal = reference_signals)
    env.reset() 

    observation, reward, done, info = env.step(np.array([[1]]))



LinearLongitudinalX15-v0
---------------------------


.. code:: python

    import gym 
    import numpy as np
    from tqdm import tqdm

    from tensorairspace.envs import LinearLongitudinalX15
    from tensorairspace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensorairspace.signals.standart import unit_step

    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=20, dt=dt) # Временной периуд
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал

    env = gym.make('LinearLongitudinalX15-v0',
               number_time_steps=number_time_steps, 
               initial_state=[[0],[0],[0],[0]],
               reference_signal = reference_signals)
    env.reset() 

    observation, reward, done, info = env.step(np.array([[1]]))



LinearLongitudinalF4C-v0
-----------------------------

.. code:: python

    import gym 
    import numpy as np
    from tqdm import tqdm

    from tensorairspace.envs import LinearLongitudinalF4C
    from tensorairspace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensorairspace.signals.standart import unit_step

    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=20, dt=dt) # Временной периуд
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал

    env = gym.make('LinearLongitudinalF4C-v0',
               number_time_steps=number_time_steps, 
               initial_state=[[0],[0],[0]],
               reference_signal = reference_signals)
    env.reset() 

    observation, reward, done, info = env.step(np.array([[1]]))


LinearLongitudinalUAV-v0
-------------------------


.. code:: python

    import gym 
    import numpy as np
    from tqdm import tqdm

    from tensorairspace.envs import LinearLongitudinalUAV
    from tensorairspace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensorairspace.signals.standart import unit_step

    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=20, dt=dt) # Временной периуд
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал

    env = gym.make('LinearLongitudinalUAV-v0',
               number_time_steps=number_time_steps, 
               initial_state=[[0],[0],[0],[0]],
               reference_signal = reference_signals)
    env.reset() 

    observation, reward, done, info = env.step(np.array([[1]]))