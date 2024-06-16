Пример использования MPC
===========================================================
.. container:: cell code

   .. code:: python

      import numpy as np
      import gymnasium as gym
      import torch
      from tensoraerospace.envs.f16.linear_longitudial import LinearLongitudinalF16
      from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
      from tensoraerospace.signals.standart import unit_step, sinusoid
      from tensoraerospace.benchmark.function import static_error
      from tensoraerospace.agent.mpc.stochastic import Net
      from tensoraerospace.agent.mpc.gradient import MPCOptimizationAgent
      from tensoraerospace.signals.random import full_random_signal

.. container:: cell code

   .. code:: python

      # Инициализация списка для хранения исторических данных
      hist = []
      dt = 0.1  # Интервал дискретизации времени

      # Генерация временного периода с заданным интервалом дискретизации
      tp = generate_time_period(tn=180, dt=dt) 

      # Конвертация временного периода в секунды
      tps = convert_tp_to_sec_tp(tp, dt=dt)

      # Вычисление общего количества временных шагов
      number_time_steps = len(tp) 

      # Создание заданного сигнала с использованием единичного шага
      # reference_signals = np.reshape(unit_step(degree=0, tp=tp, time_step=20, output_rad=True), [1, -1])
      reference_signals = np.reshape(np.deg2rad(sinusoid(amplitude=0.004, tp=tp, frequency=1)), [1, -1])

      # Создание среды симуляции, задание временных шагов, начального состояния, заданного сигнала и отслеживаемых состояний
      env = gym.make('LinearLongitudinalF16-v0',
                      number_time_steps=number_time_steps, 
                      initial_state=[[0],[0]],
                      reference_signal=reference_signals,
                      state_space = [ "theta", "q",],
                      output_space = ["theta",  "q",],
                      tracking_states=["theta"])

      # Сброс среды к начальному состоянию
      state, info = env.reset()

   .. container:: output stream stderr

      ::

         /Users/asmazaev/Projects/TensorAeroSpace/.venv/lib/python3.11/site-packages/gymnasium/utils/passive_env_checker.py:159: UserWarning: WARN: The obs returned by the `reset()` method is not within the observation space.
           logger.warn(f"{pre} is not within the observation space.")

.. container:: cell code

   .. code:: python

      model = Net(env.action_space.shape[0], env.observation_space.shape[0])

.. container:: cell code

   .. code:: python

      def cost(next_state, action, reference_signals, step):
          # Извлечение состояния системы
          theta, omega_z = np.rad2deg(next_state[0].detach().numpy())
          
          # Получение эталонного сигнала на данном шаге
          theta_ref = np.rad2deg(reference_signals[0][step])
          
          # Расчёт ошибки тангажа
          pitch_error = (theta - theta_ref) ** 2
              
          return (pitch_error ** 2 + 0.1 * omega_z ** 2 + 0.0001 * (action ** 2))

.. container:: cell code

   .. code:: python

      agent = MPCOptimizationAgent(gamma=0.99, action_dim=1, observation_dim=2, model=model, cost_function=cost, env=env, lr=1e-5, criterion=torch.nn.MSELoss())

.. container:: cell code

   .. code:: python

      from tqdm import tqdm

      # Создаем исследовательские сигналы для обучения модели

      exploration_signals = [
          np.reshape(np.deg2rad(sinusoid(amplitude=0.01, tp=tp, frequency=5)), [1, -1]), 
          np.reshape(np.deg2rad(sinusoid(amplitude=0.03, tp=tp, frequency=5)), [1, -1]), 
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.03, tp=tp, frequency=5)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=0.03, tp=tp, frequency=10)), [1, -1]), 
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.03, tp=tp, frequency=10)), [1, -1]), 
          np.reshape(np.deg2rad(sinusoid(amplitude=0.01, tp=tp, frequency=10)), [1, -1]), 
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.01, tp=tp, frequency=10)), [1, -1]), 
          np.reshape(np.deg2rad(sinusoid(amplitude=0.01, tp=tp, frequency=1)), [1, -1]), 
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.01, tp=tp, frequency=1)), [1, -1]), 
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.01, tp=tp, frequency=25)), [1, -1]), 
          np.reshape(np.deg2rad(sinusoid(amplitude=0.01, tp=tp, frequency=25)), [1, -1]), 
          np.reshape(np.deg2rad(sinusoid(amplitude=0.02, tp=tp, frequency=5)), [1, -1]), 

          np.reshape(np.deg2rad(sinusoid(amplitude=0.004, tp=tp, frequency=1)), [1, -1]), 
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.004, tp=tp, frequency=1)), [1, -1]), 
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.008, tp=tp, frequency=1)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=0.008, tp=tp, frequency=1)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.009, tp=tp, frequency=1)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=0.009, tp=tp, frequency=1)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=0.005, tp=tp, frequency=1)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.005, tp=tp, frequency=1)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=0.01, tp=tp, frequency=1)), [1, -1]), 
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.01, tp=tp, frequency=1)), [1, -1]), 
          
          np.reshape(np.deg2rad(sinusoid(amplitude=0.0089, tp=tp, frequency=25)), [1, -1]), 
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.0089, tp=tp, frequency=25)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=0.008, tp=tp, frequency=25)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.008, tp=tp, frequency=25)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.004, tp=tp, frequency=25)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=0.004, tp=tp, frequency=25)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=0.009, tp=tp, frequency=25)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.009, tp=tp, frequency=25)), [1, -1]),
          
          np.reshape(np.deg2rad(sinusoid(amplitude=0.0085, tp=tp, frequency=25)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=-0.0085, tp=tp, frequency=25)), [1, -1]),


          np.reshape(np.deg2rad(sinusoid(amplitude=0.04, tp=tp, frequency=1)), [1, -1]), 

          np.reshape(np.deg2rad(sinusoid(amplitude=0, tp=tp, frequency=0)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=0, tp=tp, frequency=0)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=0.004, tp=tp, frequency=25)), [1, -1]),
          np.reshape(np.deg2rad(sinusoid(amplitude=0.005, tp=tp, frequency=25)), [1, -1]),
          np.reshape(np.deg2rad(unit_step(degree=12, tp=tp, time_step=12)), [1, -1]),
          [None],[None],[None],
      ]
      states = np.array([[0.,    0.]])
      actions = np.array([0])
      next_states = np.array([[0.,   0.]])
      np.random.shuffle(exploration_signals)

      for ref_signal in tqdm(exploration_signals):
          sin_states, sin_actions, sin_next_states = agent.collect_data(num_episodes=40, control_exploration_signal=ref_signal[0])
          states = np.append(states, sin_states, 0)
          actions = np.append(actions, sin_actions, 0)
          next_states = np.append(next_states, sin_next_states, 0)

   .. container:: output stream stderr

      ::

         100%|██████████| 41/41 [01:09<00:00,  1.69s/it]

.. container:: cell code

   .. code:: python

      agent.train_model(states, actions, next_states, epochs=1, batch_size=64)

   .. container:: output stream stderr

      ::

         Loss 2.2726531767602864e-07: 100%|██████████| 1/1 [00:38<00:00, 38.23s/it]

.. container:: cell code

   .. code:: python

      states, actions, next_states = agent.collect_data(num_episodes=10,
                  control_exploration_signal=np.reshape(full_random_signal(0,0.1,180, (-0.5, 0.5), (-5, 5)), [1, -1])[0])

   .. container:: output stream stderr

      ::

         100%|██████████| 10/10 [00:00<00:00, 26.99it/s]

.. container:: cell code

   .. code:: python

      agent.test_network(states, actions, next_states)

   .. container:: output stream stdout

      ::

         Test MSE Loss: 2.5835663109319285e-05

.. container:: cell code

   .. code:: python

      # Инициализация списка для хранения исторических данных
      hist = []
      dt = 0.1  # Интервал дискретизации времени

      # Генерация временного периода с заданным интервалом дискретизации
      tp = generate_time_period(tn=60, dt=dt) 

      # Конвертация временного периода в секунды
      tps = convert_tp_to_sec_tp(tp, dt=dt)

      # Вычисление общего количества временных шагов
      number_time_steps = len(tp) 

      # Создание заданного сигнала с использованием единичного шага
      # reference_signals = np.reshape(unit_step(degree=0, tp=tp, time_step=20, output_rad=True), [1, -1])
      reference_signals = np.reshape(np.deg2rad(sinusoid(amplitude=0.0005, tp=tp, frequency=1)), [1, -1])

      # Создание среды симуляции, задание временных шагов, начального состояния, заданного сигнала и отслеживаемых состояний
      env = gym.make('LinearLongitudinalF16-v0',
                      number_time_steps=number_time_steps, 
                      initial_state=[[0],[0]],
                      reference_signal=reference_signals,
                      state_space = [ "theta", "q",],
                      output_space = [  "theta",  "q",],
                      tracking_states=["theta"])

      # Сброс среды к начальному состоянию
      state, info = env.reset()


      rollout, horizon = 1,1
      for episode in range(1):
          state, info = env.reset()
          episode_reward = 0
          for step in tqdm(range(number_time_steps-2)):
              action, cost = agent.choose_action_ref(state, rollout, horizon, reference_signals, step, optimization_steps=100)
              state, reward, terminated, truncated, info= env.step(action)
              state = state.reshape([1, -1])[0]
              done = terminated or truncated
              episode_reward += reward
              if done:
                  break
          print('rollout: %d, horizon: %d, episode: %d, reward: %d' % (rollout, horizon, episode, episode_reward))

   .. container:: output stream stderr

      ::

         100%|█████████▉| 598/599 [00:12<00:00, 49.60it/s]

   .. container:: output stream stdout

      ::

         rollout: 1, horizon: 1, episode: 0, reward: -1

.. container:: cell code

   .. code:: python

      env.unwrapped.model.plot_transient_process('theta', tps, reference_signals[0], to_deg=True, figsize=(15,4))

   .. container:: output display_data

      .. image:: ./e97b501f4cfa1738d94afd2db40f737178489052.png

.. container:: cell code

   .. code:: python

      env.model.plot_control('ele', tps, to_deg=True, figsize=(15,4))

   .. container:: output stream stderr

      ::

         No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.

   .. container:: output display_data

      .. image:: ./b848f5ee308503f2187ee01927d01166fe3e3dcd.png

.. container:: cell code

   .. code:: python

      env.unwrapped.model.plot_state('theta', tps, figsize=(15,4), to_deg=True)

   .. container:: output display_data

      .. image:: ./94fa6641e66ac11f94f4dfb0d9f27889f4018d59.png

.. container:: cell code

   .. code:: python

      env.unwrapped.model.plot_state('q', tps, figsize=(15,4), to_deg=True)

   .. container:: output display_data

      .. image:: ./b76a9a0c720ecd57ad1cfd5b4dcc4e60f3ae38ae.png

.. container:: cell code

   .. code:: python

      st_e = static_error(reference_signals[0],env.unwrapped.model.get_state("theta", to_deg=True))
      print("Статическая ошибка", st_e, "градуса")

   .. container:: output stream stdout

      ::

         Статическая ошибка -0.26992092291357217 градуса

.. container:: cell code

   .. code:: python

      # Сохраняем агента
      agent.save()
