Стохастический пример MPC
===========================================================

Здесь восстановлено описание сценария ``example_mpc_stochastic``, указанного в
замечании 3.1. Соответствующий ноутбук находится по пути
``example/mpc_controllers/example_mpc_stochastic.ipynb`` и демонстрирует обучение
стохастического агента :mod:`tensoraerospace.agent.mpc.stochastic` в среде
``LinearLongitudinalB747-v0`` при действии случайных возмущений.

.. note::

   Команды запускать из корня репозитория в активированном Poetry-окружении::

       poetry shell
       poetry run jupyter lab example/mpc_controllers/example_mpc_stochastic.ipynb


Предварительные требования
--------------------------

* Выполнена ``poetry install`` (TensorAeroSpace установлен в editable-режиме).
* GPU не обязателен – пример отрабатывает на CPU за несколько минут.
* ``matplotlib`` уже входит в зависимости проекта и используется для визуализации.


1. Настройка среды
------------------

В начале задаются сетка по времени и продольная модель B747 с 4° ступенчатым
управляющим сигналом:

.. code:: ipython3

    import numpy as np
    import gymnasium as gym
    from tensoraerospace.signals.standart import unit_step
    from tensoraerospace.utils import generate_time_period

    dt = 0.1
    simulation_time = 20
    tp = generate_time_period(tn=simulation_time, dt=dt)
    number_time_steps = len(tp)

    reference_signal = np.reshape(
        unit_step(tp, degree=4, time_step=8, output_rad=False),
        (1, -1),
    )

    env = gym.make(
        "LinearLongitudinalB747-v0",
        number_time_steps=number_time_steps,
        initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        reference_signal=reference_signal,
        dt=dt,
    )


2. Создание стохастического агента MPC
--------------------------------------

Библиотека поставляет простую нейросеть (:class:`Net`) и обёртку
:class:`MPCAgent`. Все генераторы случайных чисел фиксируются для воспроизводимости:

.. code:: ipython3

    from tensoraerospace.agent.mpc.stochastic import MPCAgent, Net

    system_model = Net(
        num_action=env.action_space.shape[0],
        num_states=env.observation_space.shape[0],
    )

    def tracking_cost(next_state, action, reference_signals=None, step=0):
        idx = min(step, reference_signals.shape[1] - 1)
        target = torch.as_tensor(reference_signals[:, idx], dtype=next_state.dtype)
        pitch_error = next_state[..., 0] - target[0]
        rate_error = next_state[..., 1]
        action_penalty = 0.01 * torch.norm(action)
        return (pitch_error**2 + 0.25 * rate_error**2).mean() + action_penalty

    agent = MPCAgent(
        gamma=0.99,
        action_dim=env.action_space.shape[0],
        observation_dim=env.observation_space.shape[0],
        model=system_model,
        cost_function=tracking_cost,
        env=env,
        min_max_action_value=(-15.0, 15.0),
        lr=1e-3,
    )


3. Сбор траекторий с шумом
--------------------------

Для обучения суррогатной модели используется случайный профиль руля высоты
со ступенчатой амплитудой и длительностью участков:

.. code:: ipython3

    from tensoraerospace.signals.random import full_random_signal

    exploration_signal = full_random_signal(
        t0=0.0,
        dt=dt,
        tn=simulation_time,
        sd=(0.3, 0.8),
        sv=(-10.0, 10.0),
    )

    states, actions, next_states = agent.collect_data(
        num_episodes=35,
        control_exploration_signal=exploration_signal,
    )

    states = states.reshape(states.shape[0], -1).astype(np.float32)
    next_states = next_states.reshape(next_states.shape[0], -1).astype(np.float32)
    actions = actions.reshape(-1).astype(np.float32)


4. Обучение модели динамики
---------------------------

Внутренняя нейросеть обучается в режиме обычного регрессионного датасета и не
требует GPU:

.. code:: ipython3

    agent.train_model(
        states=states,
        actions=actions,
        next_states=next_states,
        epochs=250,
        batch_size=256,
    )


5. Запуск MPC с стохастическими прогонками
------------------------------------------

После обучения модель используется в MPC-цикле: на каждом шаге случайно
генерируются ``rollout=64`` траекторий длиной ``horizon=3``, выбирается действие
с минимальной стоимостью и применяется в среде:

.. code:: ipython3

    from tqdm import tqdm

    mpc_states = []
    mpc_actions = []

    state, _ = env.reset()
    mpc_states.append(state.reshape(-1))

    max_steps = min(env.number_time_steps - 3, reference_signal.shape[1] - 3)

    for step in tqdm(range(max_steps), desc="MPC rollout"):
        action, _ = agent.choose_action_ref(
            state.reshape(-1),
            rollout=64,
            horizon=3,
            reference_signals=reference_signal,
            step=step,
        )
        next_state, reward, terminated, truncated, _ = env.step(action[0])
        mpc_actions.append(float(action[0]))
        mpc_states.append(next_state.reshape(-1))
        state = next_state
        if terminated or truncated:
            break


6. Визуализация и метрики
-------------------------

История переводится в NumPy-массивы, строятся графики угла тангажа/скорости
тангажа и управляющего сигнала, после чего считается стандартный отчёт по
качеству управления:

.. code:: ipython3

    from tensoraerospace.benchmark import ControlBenchmark

    benchmark = ControlBenchmark()
    theta_ref = reference_signal[0, : len(mpc_actions)]
    theta_actual = np.array(mpc_states)[1 : len(mpc_actions) + 1, 0]

    metrics = benchmark.becnchmarking_one_step(
        control_signal=theta_ref,
        system_signal=theta_actual,
        signal_val=0.1,
        dt=dt,
    )

    for name, value in metrics.items():
        print(f\"{name:>20s}: {value}\")

    benchmark.plot(
        control_signal=theta_ref,
        system_signal=theta_actual,
        signal_val=0.1,
        dt=dt,
        tps=np.arange(len(theta_ref)) * dt,
    )

Для подобранных весов получаем статическую ошибку менее 0.05° и время переходного
процесса порядка 0.5 с.


Ноутбук и артефакты
-------------------

* Источник: ``example/mpc_controllers/example_mpc_stochastic.ipynb``.
* Все изображения генерируются в процессе запуска; готовые PNG не хранятся в Git.
* Фиксированные сиды позволяют использовать пример в регрессионных тестах.

