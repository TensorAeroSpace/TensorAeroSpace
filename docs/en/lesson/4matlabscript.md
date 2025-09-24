# Writing a MATLAB Script

This MATLAB code initializes parameters and initial conditions, runs the Simulink model simulation, retrieves the output data, and manages Simulink models.

```matlab
clear;

% init parameters
[A, B, C, D] = uav1_data();

init = [0.1 0 0 0];
ref_signal = -0.1;

t_s = 0;
t_e = 1200;
dt = 0.1;

sim_out = sim('uav1_model.slx');

y = sim_out.get('yout');

u = y.getElement(1).Values.Data;
w = y.getElement(2).Values.Data;
q = y.getElement(3).Values.Data;
theta = y.getElement(4).Values.Data;
t = y.getElement(5).Values.Data;

bdclose('all');
open('uav1_model.slx');
```

## What the Script Does

1. Clear the workspace

```matlab
clear;
```

Removes all variables from the MATLAB workspace to avoid conflicts with previous data.

2. Initialize model parameters

```matlab
[A, B, C, D] = uav1_data();
```

The `uav1_data` function returns the system matrices (A, B, C, D) describing the dynamics in state-space form.

3. Initial conditions and simulation parameters

```matlab
init = [0.1 0 0 0];
ref_signal = -0.1;

t_s = 0;
t_e = 1200;
dt = 0.1;
```

- `init` — initial state vector.
- `ref_signal` — target control signal.
- `t_s`, `t_e`, `dt` — simulation start/end times and time step.

4. Run the simulation

```matlab
sim_out = sim('uav1_model.slx');
```

Starts the simulation of the `uav1_model.slx` model. The result is stored in `sim_out`.

5. Extract data

```matlab
y = sim_out.get('yout');

u = y.getElement(1).Values.Data;
w = y.getElement(2).Values.Data;
q = y.getElement(3).Values.Data;
theta = y.getElement(4).Values.Data;
t = y.getElement(5).Values.Data;
```

Retrieves control inputs, velocities, pitch angle, and time values from the results.

6. Close/open the model

```matlab
bdclose('all');
open('uav1_model.slx');
```

Closes all open models and opens `uav1_model.slx` for inspection or editing.
