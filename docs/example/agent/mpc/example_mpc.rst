.. code:: ipython3

    %cd ../

.. code:: ipython3

    import numpy as np
    import gymnasium as gym
    import torch
    from tensoraerospace.envs.f16.linear_longitudial import LinearLongitudinalF16
    from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensoraerospace.signals.standart import unit_step, sinusoid
    from tensoraerospace.benchmark.function import overshoot, settling_time, static_error
    from tqdm import tqdm
    import matplotlib.pyplot as plt

.. code:: ipython3

    # Инициализация списка для хранения исторических данных
    hist = []
    dt = 0.01  # Интервал дискретизации времени
    
    # Генерация временного периода с заданным интервалом дискретизации
    tp = generate_time_period(tn=5, dt=dt) 
    
    # Конвертация временного периода в секунды
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    
    # Вычисление общего количества временных шагов
    number_time_steps = len(tp) 
    
    # Создание заданного сигнала с использованием единичного шага
    # reference_signals = np.reshape(unit_step(degree=0, tp=tp, time_step=20, output_rad=True), [1, -1])
    reference_signals = np.reshape(np.deg2rad(sinusoid(amplitude=0.01, tp=tp, frequency=5)), [1, -1])
    
    # Создание среды симуляции, задание временных шагов, начального состояния, заданного сигнала и отслеживаемых состояний
    env = gym.make('LinearLongitudinalF16-v0',
                   number_time_steps=number_time_steps, 
                   initial_state=[[0],[0],[0],[0]],
                   reference_signal=reference_signals,
                   tracking_states=["alpha"])
    
    # Сброс среды к начальному состоянию
    state, info = env.reset()

.. code:: ipython3

    fig = plt.figure(figsize=(15,5))
    plt.plot(tps, sinusoid(amplitude=0.01, tp=tp, frequency=5))

.. code:: ipython3

    from tensoraerospace.agent.mpc.nn import MPCAgent, Net

.. code:: ipython3

    model = Net()


.. code:: ipython3

    env = gym.make("Pendulum-v1")
    
    seed = 7777777
    np.random.seed(seed)
    random.seed(seed)
    
    def example_cost_function(state, action):
        theta = state[0, 0].item()
        theta_dot = state[0, 1].item()
        return (theta ** 2 + 0.1 * theta_dot ** 2 + 0.001 * (action ** 2))
    
    # Assuming `model`, `env`, and other necessary variables are defined elsewhere


.. code:: ipython3

    def cost(next_state, action, reference_signals, step):
        # Коэффициенты веса для ошибки состояния и управляющего действия
        Q = 10.0  # Вес ошибки состояния
        R = 0.01  # Вес управляющего действия
        
        # Извлечение текущих значений угла атаки и угловой скорости
        alpha, omega = next_state[0].detach().numpy()
        
        # Получение желаемого значения угла атаки на текущем шаге
        alpha_ref = reference_signals[0][step]
        # Расчет ошибки состояния (разница между текущим и желаемым углом атаки)
        state_error = abs(alpha - alpha_ref)
        
        # Расчет стоимости на основе ошибки состояния и управляющего действия
        cost = Q * (state_error**2) 
        return cost
    
    agent = MPCAgent(gamma=0.99, action_dim=1, observation_dim=2, model=model, cost_function=cost)

.. code:: ipython3

    states, actions, next_states = agent.collect_data(env, num_episodes=100)

.. code:: ipython3

    agent.train_model(states, actions, next_states, epochs=20)

.. code:: ipython3

    states, actions, next_states = agent.collect_data(env, num_episodes=10)

.. code:: ipython3

    agent.test_network(states, actions, next_states)

.. code:: ipython3

    rollout, horizon = 50,10
    for episode in range(1):
        state, info = env.reset()
        state = state.reshape([1, -1])[0]
        episode_reward = 0
        for step in tqdm(range(number_time_steps-2)):
            action = agent.choose_action_ref(state, rollout, horizon, reference_signals, step)
            state, reward, terminated, truncated, info= env.step(action)
            state = state.reshape([1, -1])[0]
            done = terminated or truncated
            episode_reward += reward
            if done:
                break
        print('rollout: %d, horizon: %d, episode: %d, reward: %d' % (rollout, horizon, episode, episode_reward))


.. code:: ipython3

    env.model.plot_control('ele', tps, to_deg=True, figsize=(15,4))

.. code:: ipython3

    env.unwrapped.model.plot_transient_process('alpha', tps, reference_signals[0], to_deg=True, figsize=(15,4))
