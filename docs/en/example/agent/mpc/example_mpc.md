MPC Usage Example
===========================================================

In this example we show how to use a neural model to control a dynamical system. We will build the model, generate data, train it, and then apply Model Predictive Control (MPC).

Environment Setup
-----------------

First, import the required libraries:

.. code:: ipython3

    import numpy as np
    import gymnasium as gym
    import torch
    from tqdm import tqdm

    from tensoraerospace.envs.f16.linear_longitudial import LinearLongitudinalF16
    from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
    from tensoraerospace.signals.standart import unit_step, sinusoid
    from tensoraerospace.benchmark.function import static_error
    from tensoraerospace.agent.mpc.gradient import MPCOptimizationAgent
    from tensoraerospace.signals.random import full_random_signal
    from tensoraerospace.agent.pid import PID

    dt = 0.01  # Sampling step
    tp = generate_time_period(tn=20, dt=dt)
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp)  # Number of time steps
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1])  # Reference signal

    env = gym.make('LinearLongitudinalB747-v0',
                   number_time_steps=number_time_steps, 
                   initial_state=[[0],[0],[0],[0]],
                   reference_signal = reference_signals)
    state, info = env.reset()

Dynamics Model Creation
-----------------------

We create a neural network to approximate the system dynamics:

.. code:: ipython3

    import torch
    import torch.nn as nn

    class DynamicsModel(nn.Module):
        def __init__(self, state_dim=4, control_dim=1, hidden_dim=64):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim + control_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim)
            )

        def forward(self, x):
            return self.network(x)

Model and Data Initialization
-----------------------------

Initialize the model and define system parameters:

.. code:: ipython3

    from tensoraerospace.agent.mpc.dynamics import DynamicsNN

    state_ranges = [(-10.0, 10.0), (-4.5, 4.5), (-2.3, 2.3), (-15.0, 15.0)]
    
    A = torch.tensor(env.unwrapped.model.A, dtype=torch.float32)
    B = torch.tensor(env.unwrapped.model.B, dtype=torch.float32)
    
    dynamics_nn = DynamicsNN(DynamicsModel(hidden_dim=128))

Generate training data:

.. code:: ipython3

    states, controls, next_states = dynamics_nn.generate_training_data(
        num_samples=300_000,
        state_dim=4,
        control_dim=1,
        state_ranges=state_ranges,
        control_ranges=None,
        control_signals=["sine", "step", "sine_09", "sine_07", 
                         "sine_05_low_freq", "gaussian_noise",
                         "linear_up", "linear_down"],
        A=A,
        B=B)

Model Training
--------------

Train the model on the generated dataset:

.. code:: ipython3

    dynamics_nn.train_and_validate(
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(controls, dtype=torch.float32),
        torch.tensor(next_states, dtype=torch.float32),
        epochs=400,
        batch_size=1024,
        verbose_epoch=20)

```
    Data preparation
    Data loading
    Training started
```

    100%|██████████| 400/400 [16:05<00:00, 2.41s/it]

Applying MPC Control
--------------------

Now we use the trained model for MPC-based control:

.. code:: ipython3

    import matplotlib.pyplot as plt
    from tensoraerospace.agent.mpc.base import AircraftMPC

    mpc = AircraftMPC(dynamics_nn.model, horizon=2, dt=0.1)

Simulation parameters:

.. code:: ipython3

    simulation_time = 20  # Simulation time in seconds
    dt = 0.1
    steps = int(simulation_time / dt)
    
    x0 = np.array([0, 0, 0, 0])
    states = [x0]
    controls = []

Reference trajectory:

.. code:: ipython3

    time = np.arange(steps + mpc.horizon + 1) * dt
    theta_ref = unit_step(degree=2, tp=time, time_step=dt)

Control loop:

.. code:: ipython3

    for i in tqdm(range(steps)):
        current_ref = theta_ref[i:i + mpc.horizon + 1]
        u_opt, predicted_states = mpc.optimize_control(states[-1], current_ref)
        next_states = A @ torch.tensor(states[-1], dtype=torch.float32) + B @ torch.tensor(u_opt)
        
        controls.append(u_opt)
        states.append(next_states.numpy())

Simulation Results Visualization
--------------------------------

.. code:: ipython3

    time_array = np.arange(0, simulation_time, dt)
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_array, [s[3] for s in states[:-1]], label="Actual Theta")
    plt.plot(time_array, theta_ref[:steps], label="Reference Theta")
    plt.ylabel("Theta")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time_array, controls)
    plt.xlabel("Time (s)")
    plt.ylabel("Control (u)")
    
    plt.tight_layout()
    plt.show()

Control Quality Evaluation
--------------------------

.. code:: ipython3

    from tensoraerospace.benchmark import ControlBenchmark
    
    bench = ControlBenchmark()
    
    res = bench.becnchmarking_one_step(
        theta_ref[:-3],
        np.array([float(s[3]) for s in states[:-1]]),
        settling_threshold=1.9,
        dt=dt)

.. code:: ipython3

    print("Steady-state error:", res['static_error'])
    print("Settling time:", res['settling_time'], "s")
    print("Damping degree:", res['damping_degree'])
    print("Overshoot:", res['overshoot'])

```
   Steady-state error:  0.03220049142837533
   Settling time:  0.30000000000000004 s
   Damping degree:  0.0014316554503416693
   Overshoot:  5.013108253479004
```

Visualization of the control benchmark:

.. code:: ipython3

   bench.plot(
       theta_ref[:-3],
       np.array([float(s[3]) for s in states[:-1]]),
       settling_threshold=1.9,
       dt=dt,
       time=time,
       figsize=(15,5))

.. image:: output_10_0.png
```
