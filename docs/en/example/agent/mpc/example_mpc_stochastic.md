Stochastic MPC Example
===========================================================

This page documents the restored `example_mpc_stochastic` workflow referenced in the
Stage 3.1 deliverable. The accompanying notebook lives at
``example/mpc_controllers/example_mpc_stochastic.ipynb`` and demonstrates how to
train the stochastic :mod:`tensoraerospace.agent.mpc.stochastic` controller on the
``LinearLongitudinalB747-v0`` environment with random disturbances.

.. note::

   All commands in this guide assume you run them from the repository root inside
   the Poetry environment::

       poetry shell
       poetry run jupyter lab example/mpc_controllers/example_mpc_stochastic.ipynb


Prerequisites
-------------

* ``poetry install`` (installs TensorAeroSpace in editable mode).
* GPU is optional; the example runs on CPU in a few minutes.
* ``matplotlib`` is part of project dependencies and is used for quick plots.


1. Configure the environment
----------------------------

The notebook first defines the simulation grid and creates the Boeing–747
longitudinal channel with a 4° pitch step reference:

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


2. Instantiate the stochastic MPC agent
---------------------------------------

TensorAeroSpace ships a lightweight feed-forward model (:class:`Net`) plus the
:class:`MPCAgent` wrapper. The notebook keeps all randomness reproducible:

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


3. Collect noisy trajectories
-----------------------------

Exploration data come from a random elevator profile (piece-wise constant in both
frequency and amplitude) to expose the neural surrogate to stochastic behavior:

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


4. Train the internal dynamics model
------------------------------------

The stochastic MPC agent learns a one-step predictor of the environment dynamics.
Training is pure supervised learning and runs entirely on CPU:

.. code:: ipython3

    agent.train_model(
        states=states,
        actions=actions,
        next_states=next_states,
        epochs=250,
        batch_size=256,
    )


5. Run MPC with stochastic rollouts
-----------------------------------

Once the model is trained, the agent launches a rollout loop that samples random
action sequences (``rollout=64``, ``horizon=5``) at each step, evaluates the
tracking cost and applies the best first action:

.. code:: ipython3

    from tqdm import tqdm

    mpc_states = []
    mpc_actions = []

    state, _ = env.reset()
    mpc_states.append(state.reshape(-1))

    MPC_ROLLOUT = 64
    MPC_HORIZON = 5

    max_steps = min(env.unwrapped.number_time_steps - 3, reference_signal.shape[1] - 3)

    for step in tqdm(range(max_steps), desc="MPC rollout"):
        action, _ = agent.choose_action_ref(
            state.reshape(-1),
            rollout=MPC_ROLLOUT,
            horizon=MPC_HORIZON,
            reference_signals=reference_signal,
            step=step,
        )
        # `action` is returned as shape (1, action_dim); env expects (action_dim,)
        action_1d = np.asarray(action, dtype=np.float32).reshape(-1)
        next_state, reward, terminated, truncated, _ = env.step(action_1d)
        mpc_actions.append(float(action_1d[0]))
        mpc_states.append(next_state.reshape(-1))
        state = next_state
        if terminated or truncated:
            break


6. Visualize and benchmark
--------------------------

The notebook converts the history to NumPy arrays, plots pitch/pitch-rate/control
time series and produces the standard control-quality report:

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

    # Optional Plotly visualization
    benchmark.plot(
        control_signal=theta_ref,
        system_signal=theta_actual,
        signal_val=0.1,
        dt=dt,
        tps=np.arange(len(theta_ref)) * dt,
    )

Typical output shows low steady-state error (<0.05°) and settling time under
0.5 s for the chosen weights.


Notebook and assets
-------------------

* Source: ``example/mpc_controllers/example_mpc_stochastic.ipynb``
* Figures are generated on the fly; no static images are committed to the repo.
* The notebook keeps seeds fixed, so regression checks can compare metrics with
  future runs.

