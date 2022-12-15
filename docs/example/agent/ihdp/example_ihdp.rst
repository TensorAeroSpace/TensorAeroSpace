Пример использования IHDP вместе с LinearLongitudinalF16-v0
===========================================================


.. code:: ipython3

    import numpy as np
    from tqdm import tqdm
    
    from tensorairspace.envs.f16.linear_longitudial import LinearLongitudinalF16
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

    import gym 
    env = gym.make('LinearLongitudinalF16-v0',
                   number_time_steps=number_time_steps, 
                   initial_state=[[0],[0],[0],[0]],
                   reference_signal = reference_signals,
                  tracking_states=["alpha"])
    env.reset()

.. code:: ipython3

    actor_settings = {
        "start_training": 5,
        "layers": (25, 1), 
        "activations":  ('tanh', 'tanh'), 
        "learning_rate": 2, 
        "learning_rate_exponent_limit": 10,
        "type_PE": "combined",
        "amplitude_3211": 15, 
        "pulse_length_3211": 5/dt, 
        "maximum_input": 25,
        "maximum_q_rate": 20,
        "WB_limits": 30,
        "NN_initial": 120,
        "cascade_actor": False,
        "learning_rate_cascaded":1.2
    }

.. code:: ipython3

    incremental_settings = {
        "number_time_steps": number_time_steps, 
        "dt": dt, 
        "input_magnitude_limits":25, 
        "input_rate_limits":60,
    }

.. code:: ipython3

    critic_settings = {
        "Q_weights": [8], 
        "start_training": -1, 
        "gamma": 0.99, 
        "learning_rate": 15, 
        "learning_rate_exponent_limit": 10,
        "layers": (25,1), 
        "activations": ("tanh", "linear"), 
            "WB_limits": 30,
        "NN_initial": 120,
        "indices_tracking_states": env.indices_tracking_states
    }

.. code:: ipython3

    model = IHDPAgent(actor_settings, critic_settings, incremental_settings, env.tracking_states, env.state_space, env.control_space, number_time_steps, env.indices_tracking_states)

.. code:: ipython3

    xt = np.array([[np.deg2rad(3)], [0]])
    
    for step in tqdm(range(number_time_steps-1)):
        ut = model.predict(xt, reference_signals, step)
        xt, reward, done, info = env.step(np.array(ut))


.. parsed-literal::

    100%|██████████| 2001/2001 [00:24<00:00, 80.92it/s]


.. code:: ipython3

    env.model.plot_transient_process('alpha', tps, reference_signals[0], to_deg=True, figsize=(15,4))




.. image:: img/output_9_0.png

.. code:: ipython3

    env.model.plot_state('wz', tps, reference_signals[0], to_deg=True, figsize=(15,4))


.. image:: img/output_10_1.png
