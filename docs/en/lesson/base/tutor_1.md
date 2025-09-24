# Lesson 1: Introduction to Aircraft Control Systems and the State-Space Representation

## 1. Introduction to Aircraft Control Systems (ACS)

Aircraft control systems (ACS) are integrated solutions designed to ensure stability, controllability, and adherence to target flight trajectories. Modern ACS are complex multidimensional automatic control systems that play a key role in the safety and efficiency of both manned and unmanned flights [1].

The core tasks of ACS include:

* **Stabilization:** Maintaining the desired attitude and orientation of the aircraft in the presence of external disturbances (e.g., gusts of wind).
* **Control:** Changing the trajectory and flight parameters according to pilot or autopilot commands.
* **Optimization:** Achieving mission objectives with the best indicators (e.g., minimum fuel consumption or flight time).

Automation hardware used in ACS can be grouped into two categories:

1. **Automatic control and regulation systems:** The "brain" that processes sensor data and generates control signals.
2. **Components, devices, and subsystems of ACS:** Sensors (gyroscopes, accelerometers, GPS receivers), actuators (servo drives), and onboard computers.

## 2. Introduction to the State-Space Method

The **state-space method** is a mathematical framework for describing and analyzing dynamical systems, including ACS. Unlike the classical transfer-function approach, the state-space method offers several advantages:

* **Applicability to nonlinear systems:** It can describe and analyze systems that are not governed by linear differential equations.
* **Handling multidimensional (MIMO) systems:** It is convenient for multiple-input, multiple-output systems, which is typical for modern aircraft.
* **Accounting for initial conditions:** It allows analysis of systems with nonzero initial states.

Key concepts in the state-space method:

* **State variables:** The minimal set of quantities whose current values fully determine the future behavior of the system given known inputs.
* **State vector:** A vector composed of the state variables.
* **State space:** An n-dimensional space whose axes correspond to the state variables. The system's current state is represented as a point in this space.

## 3. Mathematical Description in State Space

### 3.1 Linear Continuous-Time Systems

For a linear continuous-time system with *p* inputs, *q* outputs, and *n* state variables, the state-space model is [2]:

```
ẋ(t) = A(t)x(t) + B(t)u(t)
y(t) = C(t)x(t) + D(t)u(t)
```

where:

* `x(t)` – **state vector** (dimension n x 1)
* `u(t)` – **control vector** (inputs, dimension p x 1)
* `y(t)` – **output vector** (measured signals, dimension q x 1)
* `A(t)` – **system matrix** (n x n), describes the intrinsic dynamics
* `B(t)` – **input matrix** (n x p), shows how inputs affect the state
* `C(t)` – **output matrix** (q x n), maps the state to outputs
* `D(t)` – **feedthrough matrix** (q x p), captures direct input-to-output coupling; often zero

### 3.2 Linear Discrete-Time Systems

For systems operating in discrete time (e.g., digital control), difference equations are used:

```
x(k+1) = Ax(k) + Bu(k)
y(k) = Cx(k) + Du(k)
```

### 3.3 Nonlinear Systems

Nonlinear systems require a more general form:

```
ẋ(t) = f(t, x(t), u(t))
y(t) = h(t, x(t), u(t))
```

Here `f` and `h` are nonlinear vector functions. Analyzing such systems is much more complex and often requires linearization around an operating point.

## 4. References

1. Davydov I.E. Aircraft Control Systems: Electronic textbook. Samara: SSAU, 2013. URL: https://repo.ssau.ru/bitstream/Uchebnye-posobiya/Sistemy-upravleniya-LA-Elektronnyi-resurs-elektron-ucheb-posobie-54206/1/Давыдов%20И.Е.%20Системы%20управления%20ЛА.pdf
2. State space (control theory) // Wikipedia. URL: https://ru.wikipedia.org/wiki/Пространство_состояний_(теория_управления)

## 5. Practical Example in Python

Consider a simple example of constructing and analyzing a state-space system using the `control` library.

```python
import numpy as np
import matplotlib.pyplot as plt
from control import ss, step_response, pole, zero
import control as ctrl

# Example: Simple second-order system (mass-spring-damper)
# Physical parameters
m = 1.0    # mass (kg)
k = 4.0    # spring stiffness (N/m)
c = 1.0    # damping coefficient (N·s/m)

# State variables: x1 = position, x2 = velocity
# Equation of motion: m*x'' + c*x' + k*x = F
# In state space:
# x1' = x2
# x2' = -(k/m)*x1 - (c/m)*x2 + (1/m)*F

# System matrices
A = np.array([[0, 1],
              [-k/m, -c/m]])

B = np.array([[0],
              [1/m]])

C = np.array([[1, 0]])  # measure position only

D = np.array([[0]])

# Create the state-space system
sys = ss(A, B, C, D)

print("State-space system:")
print(f"A = \n{A}")
print(f"B = \n{B}")
print(f"C = \n{C}")
print(f"D = \n{D}")

# Analyze the system poles
poles = pole(sys)
print(f"\nSystem poles: {poles}")

# Stability check
if all(np.real(poles) < 0):
    print("The system is stable")
else:
    print("The system is unstable")

# Step response
t, y = step_response(sys, T=10)

# Plot results
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.title('System step response')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)

# Phase portrait
plt.subplot(2, 1, 2)
# For the phase portrait we need both states
t, y_states = ctrl.step_response(sys, T=10, return_x=True)
plt.plot(y_states[0], y_states[1])
plt.title('Phase portrait')
plt.xlabel('Position x1 (m)')
plt.ylabel('Velocity x2 (m/s)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Example: Conversion from transfer function
# G(s) = 1/(s^2 + s + 4)
from control import tf, ss

# Create the transfer function
num = [1]
den = [1, 1, 4]
G = tf(num, den)

# Convert to state space
sys_from_tf = ss(G)
print(f"\nConversion from transfer function:")
print(f"A = \n{sys_from_tf.A}")
print(f"B = \n{sys_from_tf.B}")
print(f"C = \n{sys_from_tf.C}")
print(f"D = \n{sys_from_tf.D}")
```

### Installing the Required Libraries

```bash
pip install control matplotlib numpy scipy
```

This example demonstrates:

1. Creating a state-space system from physical parameters
2. Analyzing poles and stability
3. Plotting the step response and phase portrait
4. Converting between representations (transfer function ↔ state space)