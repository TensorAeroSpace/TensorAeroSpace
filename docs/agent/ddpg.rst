Deep Deterministic Policy Gradient (DDPG)
================================================================

Deep Deterministic Policy Gradient (DDPG) — это алгоритм обучения с подкреплением, который обучает Q функцию и функцию стратегии.

Что такое DDPG
--------------
DDPG — off-policy актор–критик алгоритм для непрерывных пространств действий. Он сочетает идеи детерминированного градиента политики (DPG) и целевых сетей как в DQN:

- актор (policy network) порождает детерминированное действие в непрерывном пространстве;
- критик (value/Q network) оценивает качество действий;
- обучение ведётся по данным из буфера воспроизведения (Replay Buffer);
- для стабильности применяются целевые сети и «мягкое» обновление параметров.

Когда применять
---------------
Используйте DDPG, когда:

- пространство действий непрерывное (тяга, рулевые поверхности, управление моментами);
- важна выборка опыта (off‑policy) и переобучение на накопленных данных;
- есть физические ограничения по амплитудам и скоростям изменения управляющих воздействий.

Пример из авиации: продольное управление F‑16
--------------------------------------------
В продольном канале самолёта (угол атаки α и угловая скорость тангажа q) управляющее воздействие — непрерывное отклонение стабилизатора. DDPG может обучить стратегию слежения по α с учётом ограничений и динамики самолёта.

Мини‑пример запуска на LinearLongitudinalF16‑v0
-----------------------------------------------
Полный пример см. в ноутбуке ``example/reinforcement_learning/example-ddpg.ipynb``.

.. code-block:: python

    import gymnasium as gym
    import numpy as np
    from tensoraerospace.agent.ddpg.model import DDPG
    from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensoraerospace.signals.standart import unit_step

    # Временная сетка и референс
    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)
    reference_signals = np.reshape(
        unit_step(degree=5, tp=tp, time_step=1000, output_rad=True), [1, -1]
    )

    # Базовая среда F‑16
    env = gym.make(
        'LinearLongitudinalF16-v0',
        number_time_steps=number_time_steps,
        initial_state=[[0], [0], [0]],  # theta, alpha, q
        reference_signal=reference_signals,
        use_reward=True,
        state_space=["theta", "alpha", "q"],
        output_space=["theta", "alpha", "q"],
        control_space=["ele"],
        tracking_states=["alpha"],
    )

    agent = DDPG(env, value_lr=1e-3, policy_lr=1e-4, replay_buffer_size=1_000_000)
    agent.learn(max_frames=12000, max_steps=500, batch_size=128)


Документация
------------

.. autoclass:: tensoraerospace.agent.ddpg.model.DDPG
  :members:


Источники
---------
- `Continuous control with deep reinforcement learning <https://arxiv.org/abs/1509.02971>`_

На каких средах протестили:
--------------------------------------------
- Unity среда
- LinearLongitudinalF16-v0 (пример в репозитории)