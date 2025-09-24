MPC with Transformers Example
=======================================

This walkthrough shows how to train a neural network, implement control with Model Predictive Control (MPC), and evaluate the results using the control benchmark.

1. Build the Dynamics Model
---------------------------
We create a dynamics model using a neural network with transformers. Training data is generated and used to fit the model.

```python

    from tensoraerospace.agent.mpc.transformers import TransformerDynamicsModel
    from tensoraerospace.agent.mpc.dynamics import DynamicsNN
    import torch
    from tensoraerospace.signals.standart import unit_step
    from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
    import numpy as np
    import gymnasium as gym

    # System parameters
    state_dim = 4  # State dimension
    control_dim = 1  # Control dimension
    input_dim = state_dim + control_dim
    output_dim = state_dim

    # Create the model and generate data
    nn_transformers = TransformerDynamicsModel(input_dim, output_dim)
    dynamics_nn = DynamicsNN(nn_transformers)

    dt = 0.01  # Sampling step
    tp = generate_time_period(tn=20, dt=dt)  # Time grid
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp)  # Number of steps

    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1])  # Reference

    # Initialize the Gym environment
    env = gym.make('LinearLongitudinalB747-v0',
                number_time_steps=number_time_steps,
                initial_state=[[0], [0], [0], [0]],
                reference_signal=reference_signals)

    state_ranges = [(-10.0, 10.0), (-4.5, 4.5), (-2.3, 2.3), (-15.0, 15.0)]
    A = torch.tensor(env.unwrapped.model.A, dtype=torch.float32)
    B = torch.tensor(env.unwrapped.model.B, dtype=torch.float32)

    # Generate training data
    states, controls, next_states = dynamics_nn.generate_training_data(
        num_samples=300_000,
        state_dim=4,
        control_dim=1,
        state_ranges=state_ranges,
        control_ranges=None,
        control_signals=["sine", "step", "sine_09", "sine_07", "sine_05_low_freq", "gaussian_noise", "linear_up", "linear_down"],
        A=A,
        B=B
    )

    # Train the model
    dynamics_nn.train_and_validate(
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(controls, dtype=torch.float32),
        torch.tensor(next_states, dtype=torch.float32),
        epochs=400,
        batch_size=1024,
        verbose_epoch=20
    )
```

2. Implement MPC
----------------
After the dynamics model is trained, we run MPC for control.

```python

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from tensoraerospace.agent.mpc.base import AircraftMPC

    # Initialize the MPC controller
    mpc = AircraftMPC(dynamics_nn.model, horizon=2, dt=0.1, iterations=50, learning_rate=10e-6, increment=1e-4)

    # Simulation settings
    simulation_time = 4  # Shorter duration for the example
    dt = 0.1
    steps = int(simulation_time / dt)

    # Initial state
    x0 = np.array([0, 0, 0, 0])
    states = [x0]
    controls = []

    # Reference trajectory
    time = np.arange(steps + mpc.horizon + 1) * dt
    theta_ref = unit_step(degree=2.5, tp=time, time_step=0.02, output_rad=False)

    model_states = [torch.tensor([0., 0., 0., 0.], dtype=torch.float32)]

    # Control loop
    for i in tqdm(range(steps)):
        current_ref = theta_ref[i:i + mpc.horizon + 1]
        
        # Optimize control via MPC
        u_opt, predicted_states = mpc.optimize_control(states[-1], current_ref)
        
        # Update state using the A/B model
        next_states = A @ model_states[i] + B @ torch.tensor(u_opt, dtype=torch.float32)
        
        controls.append(u_opt)
        model_states.append(next_states)
        
        states.append(next_states)
```

3. Visualize the Results
------------------------
We plot:
- Actual system response vs. reference trajectory
- Control input

```python
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
```

.. image:: ./image.png

4. Analyze the Results
----------------------
We evaluate the control performance with `ControlBenchmark`.

```python

    from tensoraerospace.benchmark import ControlBenchmark

    bench = ControlBenchmark()
    res = bench.becnchmarking_one_step(
        theta_ref[:-3],
        np.array([float(s[3]) for s in states[:-1]]),
        settling_threshold=1.9,
        dt=dt
    )

    print("Steady-state error:", res['static_error'])
    print("Settling time:", res['settling_time'], "s")
    print("Damping degree:", res['damping_degree'])
    print("Overshoot:", res['overshoot'])

    # Plot control vs. reference
    bench.plot(
        theta_ref[:-3],
        np.array([float(s[3]) for s in states[:-1]]),
        settling_threshold=0.9,
        dt=dt,
        time=time,
        figsize=(15, 5)
    )
```

```python
    Steady-state error:  0.03743922710418701
    Settling time:  0.2 s
    Damping degree:  0.012593524089427021
    Overshoot:  5.2121734619140625
```

.. image:: ./image-1.png

This example illustrates the end-to-end workflow in TensorAeroSpace: building the dynamics model, implementing control, and analyzing control metrics.

