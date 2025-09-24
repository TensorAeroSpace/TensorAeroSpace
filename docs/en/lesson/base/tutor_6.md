# Lesson 6 — State Observers and the Separation Principle

## 1. The Problem of Incomplete Information

In the previous lesson we synthesized a controller using pole placement. That method assumes the control law `u = -Kx` uses the full state vector `x`. In practice, measuring all states is often:

* **Technically infeasible:** Some states (e.g., angular rates) may not be directly measurable.
* **Economically impractical:** Many sensors add complexity and cost.

Thus, we need the full state `x` for control, but we may only measure the output `y = Cx`.

## 2. The Idea of a State Observer

The solution is to build a **state observer** (state estimator): a model that computes an **estimate** of the state vector, denoted `x̂`.

The idea is simple: run a copy of the real system in parallel and continuously compare the real output `y` with the observer model output `ŷ`, using the difference (error) to correct the model state so that it converges to the true state.

## 3. Luenberger Observer

The most common observer is the **Luenberger observer**, described by:

```
x̂̇ = A x̂ + B u + L (y - ŷ)
```

where:
* `x̂` — estimated state vector.
* `A x̂ + Bu` — model of the real system using the same `A`, `B`, and input `u`.
* `y` — measured output of the real system.
* `ŷ = C x̂` — estimated output from the model.
* `(y - ŷ)` — output estimation error.
* `L` — the **observer gain matrix** to be designed.

The term `L(y - ŷ)` is corrective feedback: if the model output deviates from the real output, it “nudges” `x̂` to reduce the error.

## 4. Observer Error Dynamics

Define the estimation error `e = x - x̂`. We want `e` to converge to zero as fast as possible. Deriving the error dynamics gives:

```
ė = ẋ - x̂̇
ė = (Ax + Bu) - (A x̂ + Bu + L(y - ŷ))
ė = Ax - A x̂ - L(Cx - C x̂)
ė = A(x - x̂) - LC(x - x̂)
ė = (A - LC)e
```

This is an autonomous linear system `ė = (A - LC)e`. The estimation error dynamics depend only on `(A - LC)` and not on the input `u`.

Stability (error convergence) depends on the eigenvalues of `(A - LC)`. We can choose `L` to place these eigenvalues (observer poles) at desired locations.

**Theorem:** If `(A, C)` is completely observable, then the observer poles (eigenvalues of `A - LC`) can be placed arbitrarily by choosing `L` appropriately.

This is the dual of the controller pole placement theorem. Observability is the necessary and sufficient condition to achieve any desired error dynamics.

## 5. Separation Principle

Now we have two separate tasks:

1. **Controller synthesis:** Choose `K` to place the closed-loop poles of `(A - BK)` (assuming `x` is known).
2. **Observer synthesis:** Choose `L` to place the observer poles `(A - LC)` so that `e` decays rapidly.

The **Separation Principle** states these two problems can be solved **independently**. We can:

1. Design the controller as if `x` were known to obtain `K`.
2. Design the observer to obtain a good estimate `x̂`, yielding `L`.
3. Combine them by using `u = -K x̂`.

The eigenvalues of the combined system (controller + observer) are simply the union of the controller poles (eigenvalues of `A - BK`) and the observer poles (eigenvalues of `A - LC`); they do not affect each other.

**Practical tip:** Make the observer 2–5× faster than the plant dynamics, i.e., place observer poles significantly to the left of the controller poles. This ensures `x̂` converges faster than the plant responds, so the controller behaves almost as if it had access to the true state `x`.

## 6. References

1. State observer — Wikipedia. – [URL: https://en.wikipedia.org/wiki/State_observer](https://en.wikipedia.org/wiki/State_observer)
2. State observers. Luenberger observer — T-Flex (RU). – [URL: https://www.tflex.ru/study/au/au_10.php](https://www.tflex.ru/study/au/au_10.php)
3. Control Systems | Observers and Controllers — The University of Manchester. – [URL: https://www.youtube.com/watch?v=FxU-yprh94E](https://www.youtube.com/watch?v=FxU-yprh94E)


## 7. Practical Python Example

We implement a Luenberger observer and demonstrate the separation principle in practice.

```python
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.linalg import eigvals, solve
import sympy as sp

def luenberger_observer_design(A, C, desired_observer_poles):
    """
    Luenberger observer design via pole placement
    """
    n = A.shape[0]
    
    # Observability check
    O = ctrl.obsv(A, C)
    if np.linalg.matrix_rank(O) != n:
        raise ValueError("The system is not fully observable!")
    
    # Duality: pole placement for (A^T, C^T)
    L_T = ctrl.place(A.T, C.T, desired_observer_poles)
    L = L_T.T
    
    return L

def simulate_observer_controller_system():
    """
    Full simulation of a system with controller and observer
    """
    
    print("SYSTEM WITH CONTROLLER AND OBSERVER")
    print("="*60)
    
    # Original system (aircraft-like model)
    A = np.array([[0, 1],
                  [-4, -1]])
    
    B = np.array([[0],
                  [1]])
    
    C = np.array([[1, 0]])  # measure position only (not velocity)
    
    D = np.array([[0]])
    
    print("Original system:")
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"C = \n{C}")
    
    # Controllability and observability
    P = ctrl.ctrb(A, B)
    O = ctrl.obsv(A, C)
    
    controllable = np.linalg.matrix_rank(P) == A.shape[0]
    observable = np.linalg.matrix_rank(O) == A.shape[0]
    
    print(f"\nControllability: {'✓' if controllable else '✗'}")
    print(f"Observability: {'✓' if observable else '✗'}")
    
    if not (controllable and observable):
        print("System not suitable for synthesis!")
        return
    
    # Controller synthesis
    controller_poles = [-2, -3]
    K = ctrl.place(A, B, controller_poles)
    
    print(f"\nController:")
    print(f"Desired controller poles: {controller_poles}")
    print(f"K: {K}")
    
    # Observer synthesis (3× faster)
    observer_poles = [-6, -9]
    L = luenberger_observer_design(A, C, observer_poles)
    
    print(f"\nObserver:")
    print(f"Desired observer poles: {observer_poles}")
    print(f"L: {L}")
    
    # Verify poles
    A_cl = A - B @ K
    A_obs = A - L @ C
    
    actual_controller_poles = eigvals(A_cl)
    actual_observer_poles = eigvals(A_obs)
    
    print(f"\nActual controller poles: {actual_controller_poles}")
    print(f"Actual observer poles: {actual_observer_poles}")
    
    return A, B, C, D, K, L

def simulate_complete_system(A, B, C, D, K, L):
    """
    Simulate the full system: plant + controller + observer
    """
    
    # Augmented system: [x; x_hat]
    n = A.shape[0]
    
    # Augmented A
    A_ext = np.block([
        [A - B @ K,     B @ K],
        [np.zeros((n, n)), A - L @ C]
    ])
    
    # Augmented B (disturbance input)
    B_ext = np.block([
        [B],
        [np.zeros((n, 1))]
    ])
    
    # Augmented C (true output and estimated output)
    C_ext = np.block([
        [C, np.zeros((1, n))],
        [np.zeros((1, n)), C]
    ])
    
    print(f"\n{'='*60}")
    print("FULL SYSTEM SIMULATION")
    print(f"{'='*60}")
    
    # Poles of the augmented system
    poles_extended = eigvals(A_ext)
    print(f"Augmented system poles: {poles_extended}")
    
    # Separation check
    controller_poles = eigvals(A - B @ K)
    observer_poles = eigvals(A - L @ C)
    combined_poles = np.concatenate([controller_poles, observer_poles])
    
    print(f"Controller poles: {controller_poles}")
    print(f"Observer poles: {observer_poles}")
    print(f"Combined poles: {combined_poles}")
    
    separation_check = np.allclose(np.sort(poles_extended), np.sort(combined_poles))
    print(f"Separation principle holds: {'✓' if separation_check else '✗'}")
    
    return A_ext, B_ext, C_ext

def compare_control_scenarios(A, B, C, D, K, L):
    """
    Compare different control scenarios
    """
    
    plt.figure(figsize=(16, 12))
    
    # Time
    t = np.linspace(0, 8, 1000)
    
    # Initial conditions
    x0_true = np.array([1, 0])
    x0_est = np.array([0, 0])
    
    # Scenario 1: Ideal feedback (true state)
    print("Simulating scenario 1: Ideal feedback...")
    A_ideal = A - B @ K
    sys_ideal = ctrl.ss(A_ideal, np.zeros((2, 1)), C, np.zeros((1, 1)))
    
    t1, y1, x1 = ctrl.initial_response(sys_ideal, t, X0=x0_true, return_x=True)
    u1 = -K @ x1
    
    # Scenario 2: Observer only (no control)
    print("Simulating scenario 2: Observer only...")
    
    A_obs_ext = np.block([
        [A, np.zeros((2, 2))],
        [L @ C, A - L @ C]
    ])
    
    C_obs_ext = np.block([
        [C, np.zeros((1, 2))],
        [np.zeros((1, 2)), C]
    ])
    
    sys_obs = ctrl.ss(A_obs_ext, np.zeros((4, 1)), C_obs_ext, np.zeros((2, 1)))
    x0_obs_ext = np.concatenate([x0_true, x0_est])
    
    t2, y2, x2 = ctrl.initial_response(sys_obs, t, X0=x0_obs_ext, return_x=True)
    
    # Scenario 3: Full system (controller + observer)
    print("Simulating scenario 3: Full system...")
    
    def full_system_dynamics(t, state):
        x = state[:2]
        x_hat = state[2:]
        
        y = C @ x
        u = -K @ x_hat
        
        dx_dt = A @ x + B @ u
        dx_hat_dt = A @ x_hat + B @ u + L @ (y - C @ x_hat)
        
        return np.concatenate([dx_dt, dx_hat_dt])
    
    from scipy.integrate import solve_ivp
    
    x0_full = np.concatenate([x0_true, x0_est])
    sol = solve_ivp(full_system_dynamics, [0, t[-1]], x0_full, t_eval=t, rtol=1e-8)
    
    x3 = sol.y[:2, :]
    x_hat3 = sol.y[2:, :]
    y3 = C @ x3
    u3 = -K @ x_hat3
    
    # Plots
    
    # 1. States
    plt.subplot(3, 3, 1)
    plt.plot(t1, x1[0, :], 'b-', label='Ideal FB', linewidth=2)
    plt.plot(t, x3[0, :], 'r--', label='With observer', linewidth=2)
    plt.plot(t2, x2[0, :], 'g:', label='No control', linewidth=2)
    plt.title('State x₁')
    plt.xlabel('Time (s)')
    plt.ylabel('x₁')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 2)
    plt.plot(t1, x1[1, :], 'b-', label='Ideal FB', linewidth=2)
    plt.plot(t, x3[1, :], 'r--', label='With observer', linewidth=2)
    plt.plot(t2, x2[1, :], 'g:', label='No control', linewidth=2)
    plt.title('State x₂')
    plt.xlabel('Time (s)')
    plt.ylabel('x₂')
    plt.legend()
    plt.grid(True)
    
    # 2. Control signals
    plt.subplot(3, 3, 3)
    plt.plot(t1, u1[0, :], 'b-', label='Ideal FB', linewidth=2)
    plt.plot(t, u3[0, :], 'r--', label='With observer', linewidth=2)
    plt.title('Control input')
    plt.xlabel('Time (s)')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    
    # 3. Estimation error
    plt.subplot(3, 3, 4)
    error1 = x3[0, :] - x_hat3[0, :]
    error2 = x3[1, :] - x_hat3[1, :]
    plt.plot(t, error1, 'r-', label='Error x₁')
    plt.plot(t, error2, 'b-', label='Error x₂')
    plt.title('Estimation error')
    plt.xlabel('Time (s)')
    plt.ylabel('x - x̂')
    plt.legend()
    plt.grid(True)
    
    # 4. True vs estimated state
    plt.subplot(3, 3, 5)
    plt.plot(t, x3[0, :], 'b-', label='True x₁', linewidth=2)
    plt.plot(t, x_hat3[0, :], 'r--', label='Estimate x̂₁', linewidth=2)
    plt.title('True vs estimate (x₁)')
    plt.xlabel('Time (s)')
    plt.ylabel('x₁')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 6)
    plt.plot(t, x3[1, :], 'b-', label='True x₂', linewidth=2)
    plt.plot(t, x_hat3[1, :], 'r--', label='Estimate x̂₂', linewidth=2)
    plt.title('True vs estimate (x₂)')
    plt.xlabel('Time (s)')
    plt.ylabel('x₂')
    plt.legend()
    plt.grid(True)
    
    # 5. Phase portraits
    plt.subplot(3, 3, 7)
    plt.plot(x1[0, :], x1[1, :], 'b-', label='Ideal FB', linewidth=2)
    plt.plot(x3[0, :], x3[1, :], 'r--', label='With observer', linewidth=2)
    plt.plot(x0_true[0], x0_true[1], 'ko', markersize=8, label='Initial')
    plt.title('Phase portraits')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.legend()
    plt.grid(True)
    
    # 6. Pole map
    plt.subplot(3, 3, 8)
    
    poles_controller = eigvals(A - B @ K)
    poles_observer = eigvals(A - L @ C)
    poles_original = eigvals(A)
    
    plt.plot(np.real(poles_original), np.imag(poles_original), 'ko', 
             markersize=10, label='Open loop')
    plt.plot(np.real(poles_controller), np.imag(poles_controller), 'bs', 
             markersize=8, label='Controller')
    plt.plot(np.real(poles_observer), np.imag(poles_observer), 'r^', 
             markersize=8, label='Observer')
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.axvspan(-12, 0, alpha=0.2, color='green')
    plt.title('Pole map')
    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.legend()
    plt.grid(True)
    
    # 7. Quality metrics
    plt.subplot(3, 3, 9)
    
    final_value_ideal = x1[0, -1]
    final_value_observer = x3[0, -1]
    
    settling_idx_ideal = np.where(np.abs(x1[0, :] - final_value_ideal) <= 0.02 * abs(final_value_ideal))[0]
    settling_idx_observer = np.where(np.abs(x3[0, :] - final_value_observer) <= 0.02 * abs(final_value_observer))[0]
    
    settling_time_ideal = t1[settling_idx_ideal[0]] if len(settling_idx_ideal) > 0 else t1[-1]
    settling_time_observer = t[settling_idx_observer[0]] if len(settling_idx_observer) > 0 else t[-1]
    
    systems = ['Ideal FB', 'With observer']
    settling_times = [settling_time_ideal, settling_time_observer]
    
    plt.bar(systems, settling_times, alpha=0.7)
    plt.title('Settling time')
    plt.ylabel('Time (s)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Settling time (ideal): {settling_time_ideal:.2f} s")
    print(f"Settling time (observer): {settling_time_observer:.2f} s")
    print(f"Difference: {abs(settling_time_observer - settling_time_ideal):.2f} s")
    
    max_error = np.max(np.abs(error1))
    print(f"Max estimation error: {max_error:.4f}")

def observer_pole_sensitivity():
    """Effect of observer pole placement"""
    
    print(f"\n{'='*60}")
    print("OBSERVER POLE SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    
    A = np.array([[0, 1], [-4, -1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    
    # Fixed controller
    K = ctrl.place(A, B, [-2, -3])
    
    # Observer pole sets
    observer_configs = {
        'Slow': [-1, -1.5],
        'Moderate': [-4, -6],
        'Fast': [-8, -12],
        'Very fast': [-15, -20]
    }
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, (name, poles) in enumerate(observer_configs.items()):
        L = luenberger_observer_design(A, C, poles)
        
        # Simulation
        t = np.linspace(0, 5, 500)
        x0_true = np.array([1, 0])
        x0_est = np.array([0, 0])
        
        def system_dynamics(t, state):
            x = state[:2]
            x_hat = state[2:]
            
            y = C @ x
            u = -K @ x_hat
            
            dx_dt = A @ x + B @ u
            dx_hat_dt = A @ x_hat + B @ u + L @ (y - C @ x_hat)
            
            return np.concatenate([dx_dt, dx_hat_dt])
        
        from scipy.integrate import solve_ivp
        x0_full = np.concatenate([x0_true, x0_est])
        sol = solve_ivp(system_dynamics, [0, t[-1]], x0_full, t_eval=t, rtol=1e-8)
        
        x_true = sol.y[:2, :]
        x_hat = sol.y[2:, :]
        error = x_true - x_hat
        
        # Error plots
        plt.subplot(2, 2, 1)
        plt.plot(t, error[0, :], color=colors[i], label=f'{name} ({poles})')
        
        plt.subplot(2, 2, 2)
        plt.plot(t, error[1, :], color=colors[i], label=f'{name}')
        
        # State plot
        plt.subplot(2, 2, 3)
        plt.plot(t, x_true[0, :], color=colors[i], label=f'{name}')
        
        # Control input
        plt.subplot(2, 2, 4)
        u = -K @ x_hat
        plt.plot(t, u[0, :], color=colors[i], label=f'{name}')
    
    plt.subplot(2, 2, 1)
    plt.title('Estimation error x₁')
    plt.xlabel('Time (s)')
    plt.ylabel('e₁ = x₁ - x̂₁')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.title('Estimation error x₂')
    plt.xlabel('Time (s)')
    plt.ylabel('e₂ = x₂ - x̂₂')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.title('State x₁')
    plt.xlabel('Time (s)')
    plt.ylabel('x₁')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.title('Control input')
    plt.xlabel('Time (s)')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Main program
if __name__ == "__main__":
    # Synthesis
    A, B, C, D, K, L = simulate_observer_controller_system()
    
    # Augmented system analysis
    A_ext, B_ext, C_ext = simulate_complete_system(A, B, C, D, K, L)
    
    # Scenario comparison
    compare_control_scenarios(A, B, C, D, K, L)
    
    # Sensitivity analysis
    observer_pole_sensitivity()
    
    print(f"\n{'='*60}")
    print("KEY TAKEAWAYS")
    print(f"{'='*60}")
    print("1. Separation principle enables independent controller and observer design")
    print("2. Observer should be 3–5× faster than the plant")
    print("3. Excessively fast observer poles can amplify noise")
    print("4. Observer-based control approaches ideal state-feedback performance")
    print("5. Estimation error decays exponentially")
```

This example demonstrates:

1. **Luenberger observer synthesis** using the duality principle
2. **Separation principle** — independent controller and observer design
3. **Comparison of control scenarios** (ideal feedback vs observer-based)
4. **Estimation error analysis** and its dynamics
5. **Impact of observer pole placement** on system performance
6. **Numerical simulation** of the full nonlinear closed-loop system
