# Lesson 5 — State-Space Synthesis: Pole Placement Method

## 1. From Analysis to Synthesis

In previous lessons we learned to analyze systems: represent them in state space, check controllability and observability, and assess stability via the location of poles (eigenvalues of matrix A).

**Synthesis** is the inverse problem. We do not just analyze what exists; we deliberately create a controller that ensures the **desired** behavior of the closed-loop system. In particular, we want stability and specified transient characteristics (speed, oscillatory behavior, etc.).

## 2. State Feedback Control

The main idea in state-space synthesis is to use **state feedback**. We measure (or estimate) all state variables `x(t)` and form the control input `u(t)` based on them.

The control law is:

```
u = -Kx
```

where `K` is the **state-feedback gain matrix**. Our task is to find `K` that yields the desired system properties.

Substitute this law into the state equation of the original system:

```
ẋ = Ax + Bu
ẋ = Ax + B(-Kx) = Ax - BKx
ẋ = (A - BK)x
```

We obtain the **closed-loop system**. Its dynamics are governed by `A_cl = (A - BK)`. Stability and the transient response depend on the eigenvalues of `A_cl`.

## 3. Pole Placement

**Pole placement** consists of selecting `K` so that the eigenvalues of `A_cl = (A - BK)` are at pre-specified, **desired** locations in the complex plane. These desired eigenvalues are the **desired poles** of the closed-loop system.

**Theorem:** If `(A, B)` is completely controllable, then we can place the poles of `(A - BK)` at **any** desired locations by choosing an appropriate `K`.

This is a fundamental result. Controllability is the necessary and sufficient condition for shaping the dynamics of a linear system via state feedback.

### 3.1. How to Choose Desired Poles?

Desired poles determine closed-loop dynamics:

* **Real-axis placement:** The farther left the poles, the faster the decay (a “faster” system).
* **Complex-conjugate poles:** A pair `λ = -σ ± jω` produces oscillations.
    * `σ` (real part) sets the **decay rate**; larger `σ` means faster damping.
    * `ω` (imaginary part) sets the **oscillation frequency**.

Poles are often chosen based on transient requirements, e.g., using standard prototypes like Butterworth that provide a good compromise between speed and overshoot.

### 3.2. Ackermann’s Formula

For single-input systems (scalar `u`) there is a closed-form computation of `K`, known as **Ackermann’s formula**.

1. Choose desired poles `p₁, p₂, ..., pₙ`.
2. Form the desired characteristic polynomial: `α(s) = (s - p₁)(s - p₂)...(s - pₙ)`.
3. Compute the controllability matrix `P = [B, AB, ..., Aⁿ⁻¹B]`.
4. Compute `K` by

    ```
    K = [0 0 ... 1] · P⁻¹ · α(A)
    ```

    where `α(A)` is the matrix polynomial obtained by substituting `A` into `α(s)`.

## 4. Limitations

Pole placement is powerful but requires knowledge of **all** states `x(t)`. In practice, some states may be unmeasurable. This is addressed by **state observers**, covered in the next lesson.

## 5. References

1. Pole placement — Wikipedia. – [URL: https://en.wikipedia.org/wiki/Pole_placement](https://en.wikipedia.org/wiki/Pole_placement)
2. State-Space Methods for Controller Design — University of Michigan. – [URL: https://ctms.engin.umich.edu/CTMS/index.php?example=MotorPosition&section=ControlStateSpace](https://ctms.engin.umich.edu/CTMS/index.php?example=MotorPosition&section=ControlStateSpace)
3. Modal controller synthesis (pole placement) — Habr (RU). – [URL: https://habr.com/ru/post/562152/](https://habr.com/ru/post/562152/)


## 6. Practical Python Example

We will implement pole placement for an aircraft control system and compare different placements.

```python
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.linalg import solve, eigvals
import sympy as sp

def ackermann_formula(A, B, desired_poles):
    """
    Ackermann’s formula to compute the state-feedback gain K
    for single-input (SISO) systems
    """
    n = A.shape[0]
    
    # Controllability check
    P = ctrl.ctrb(A, B)
    if np.linalg.matrix_rank(P) != n:
        raise ValueError("The system is not fully controllable!")
    
    # Desired characteristic polynomial
    # α(s) = (s - p1)(s - p2)...(s - pn)
    poly_coeffs = np.poly(desired_poles)  # polynomial coefficients
    
    # Compute α(A) — matrix polynomial
    alpha_A = np.zeros_like(A)
    for i, coeff in enumerate(poly_coeffs):
        alpha_A += coeff * np.linalg.matrix_power(A, len(poly_coeffs) - 1 - i)
    
    # Ackermann formula: K = [0 0 ... 1] * P^(-1) * α(A)
    e_n = np.zeros((1, n))
    e_n[0, -1] = 1  # vector [0 0 ... 1]
    
    K = e_n @ np.linalg.inv(P) @ alpha_A
    
    return K

def place_poles_comparison():
    """Compare different pole placement choices"""
    
    # Original system (simplified aircraft model)
    A = np.array([[0, 1],
                  [-2, -0.5]])  # lightly damped system
    
    B = np.array([[0],
                  [1]])
    
    C = np.array([[1, 0]])  # measure position
    D = np.array([[0]])
    
    print("ORIGINAL SYSTEM")
    print("="*50)
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    
    # Original poles
    original_poles = eigvals(A)
    print(f"\nOriginal poles: {original_poles}")
    
    # Controllability
    P = ctrl.ctrb(A, B)
    controllable = np.linalg.matrix_rank(P) == A.shape[0]
    print(f"Controllability: {'✓' if controllable else '✗'}")
    
    if not controllable:
        print("System is uncontrollable! Pole placement is impossible.")
        return
    
    # Pole sets
    pole_sets = {
        'Fast decay': [-5, -6],
        'Critical damping': [-3, -3],
        'Oscillatory': [-1+2j, -1-2j],
        'Very fast': [-10, -12]
    }
    
    results = {}
    
    print(f"\n{'='*60}")
    print("CONTROLLER SYNTHESIS")
    print(f"{'='*60}")
    
    for name, poles in pole_sets.items():
        print(f"\n{name}:")
        print(f"Desired poles: {poles}")
        
        # Compute K
        try:
            if len(set(poles)) == 1:  # repeated poles
                K = ctrl.place(A, B, poles)
            else:
                K = ackermann_formula(A, B, poles)
            
            # Verify
            A_cl = A - B @ K
            actual_poles = eigvals(A_cl)
            
            print(f"Gain K: {K}")
            print(f"Actual poles: {actual_poles}")
            
            results[name] = {
                'K': K,
                'poles': actual_poles,
                'A_cl': A_cl
            }
            
        except Exception as e:
            print(f"Error: {e}")
            results[name] = None
    
    return A, B, C, D, results

def simulate_closed_loop_systems(A, B, C, D, results):
    """Simulate closed-loop systems with different controllers"""
    
    # Open-loop system
    sys_open = ctrl.ss(A, B, C, D)
    
    plt.figure(figsize=(15, 12))
    
    # Simulation horizon
    t = np.linspace(0, 5, 1000)
    
    # 1. Step responses
    plt.subplot(3, 2, 1)
    
    # Open-loop
    t_orig, y_orig = ctrl.step_response(sys_open, t)
    plt.plot(t_orig, y_orig, 'k--', linewidth=2, label='Open loop')
    
    # Closed-loop
    colors = ['red', 'blue', 'green', 'orange']
    for i, (name, result) in enumerate(results.items()):
        if result is not None:
            # Closed-loop system
            A_cl = result['A_cl']
            sys_cl = ctrl.ss(A_cl, B, C, D)
            
            try:
                t_cl, y_cl = ctrl.step_response(sys_cl, t)
                plt.plot(t_cl, y_cl, color=colors[i], label=name)
            except:
                print(f"Simulation error for {name}")
    
    plt.title('Step responses')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)
    
    # 2. Pole map
    plt.subplot(3, 2, 2)
    
    # Open-loop poles
    orig_poles = eigvals(A)
    plt.plot(np.real(orig_poles), np.imag(orig_poles), 'ks', 
             markersize=10, label='Open-loop poles')
    
    # Closed-loop poles
    for i, (name, result) in enumerate(results.items()):
        if result is not None:
            poles = result['poles']
            plt.plot(np.real(poles), np.imag(poles), 'o', 
                     color=colors[i], markersize=8, label=name)
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.axvspan(-15, 0, alpha=0.2, color='green', label='Stable region')
    plt.title('Pole map')
    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.legend()
    plt.grid(True)
    
    # 3. Control signals
    plt.subplot(3, 2, 3)
    
    # Initial conditions for demo
    x0 = np.array([1, 0])  # initial offset
    
    for i, (name, result) in enumerate(results.items()):
        if result is not None:
            A_cl = result['A_cl']
            K = result['K']
            
            # Simulation with initial conditions
            sys_cl = ctrl.ss(A_cl, np.zeros((2, 1)), np.eye(2), np.zeros((2, 1)))
            
            try:
                t_sim, x_sim = ctrl.initial_response(sys_cl, t, X0=x0)
                u_sim = -K @ x_sim  # control signal
                plt.plot(t_sim, u_sim[0], color=colors[i], label=name)
            except:
                print(f"Control simulation error for {name}")
    
    plt.title('Control signals')
    plt.xlabel('Time (s)')
    plt.ylabel('u(t)')
    plt.legend()
    plt.grid(True)
    
    # 4. Phase portraits
    plt.subplot(3, 2, 4)
    
    for i, (name, result) in enumerate(results.items()):
        if result is not None:
            A_cl = result['A_cl']
            
            sys_cl = ctrl.ss(A_cl, np.zeros((2, 1)), np.eye(2), np.zeros((2, 1)))
            
            try:
                t_sim, x_sim = ctrl.initial_response(sys_cl, t, X0=x0)
                plt.plot(x_sim[0], x_sim[1], color=colors[i], label=name)
            except:
                print(f"Phase portrait error for {name}")
    
    plt.title('Phase portraits')
    plt.xlabel('x1 (position)')
    plt.ylabel('x2 (velocity)')
    plt.legend()
    plt.grid(True)
    
    # 5. Quality metrics (settling time, overshoot)
    plt.subplot(3, 2, 5)
    
    settling_times = []
    overshoots = []
    names = []
    
    for name, result in results.items():
        if result is not None:
            A_cl = result['A_cl']
            sys_cl = ctrl.ss(A_cl, B, C, D)
            
            try:
                t_step, y_step = ctrl.step_response(sys_cl, t)
                
                # Settling time (2% criterion)
                final_value = y_step[-1]
                settling_idx = np.where(np.abs(y_step - final_value) <= 0.02 * abs(final_value))[0]
                settling_time = t_step[settling_idx[0]] if len(settling_idx) > 0 else t[-1]
                
                # Overshoot
                max_value = np.max(y_step)
                overshoot = (max_value - final_value) / final_value * 100 if final_value != 0 else 0
                
                settling_times.append(settling_time)
                overshoots.append(max(0, overshoot))
                names.append(name)
                
            except:
                pass
    
    x_pos = np.arange(len(names))
    plt.bar(x_pos, settling_times, alpha=0.7)
    plt.title('Settling time')
    plt.xlabel('Controller type')
    plt.ylabel('Time (s)')
    plt.xticks(x_pos, names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6. Overshoot
    plt.subplot(3, 2, 6)
    plt.bar(x_pos, overshoots, alpha=0.7, color='orange')
    plt.title('Overshoot')
    plt.xlabel('Controller type')
    plt.ylabel('Overshoot (%)')
    plt.xticks(x_pos, names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def robust_pole_placement():
    """Demonstrate the effect of model uncertainty on pole placement"""
    
    print(f"\n{'='*60}")
    print("ROBUSTNESS ANALYSIS")
    print(f"{'='*60}")
    
    # Nominal system
    A_nom = np.array([[0, 1],
                      [-2, -0.5]])
    B_nom = np.array([[0], [1]])
    
    # Desired poles
    desired_poles = [-2, -3]
    
    # Controller for nominal system
    K = ackermann_formula(A_nom, B_nom, desired_poles)
    print(f"Controller for nominal system: K = {K}")
    
    # Parameter uncertainty
    uncertainties = np.linspace(-0.5, 0.5, 21)  # ±50% uncertainty
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for delta in uncertainties[::4]:  # every 4th for clarity
        A_pert = A_nom + delta * np.array([[0, 0], [0.5, 0.1]])  # parameter perturbation
        A_cl = A_pert - B_nom @ K
        poles = eigvals(A_cl)
        
        plt.plot(np.real(poles), np.imag(poles), 'o', alpha=0.6, 
                 label=f'δ = {delta:.1f}' if abs(delta) < 0.3 else '')
    
    # Nominal poles
    A_cl_nom = A_nom - B_nom @ K
    poles_nom = eigvals(A_cl_nom)
    plt.plot(np.real(poles_nom), np.imag(poles_nom), 'rs', markersize=10, 
             label='Nominal')
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.title('Sensitivity of poles to uncertainty')
    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.legend()
    plt.grid(True)
    
    # Robust stability vs uncertainty
    plt.subplot(2, 2, 2)
    stable_count = []
    
    for delta in uncertainties:
        A_pert = A_nom + delta * np.array([[0, 0], [0.5, 0.1]])
        A_cl = A_pert - B_nom @ K
        poles = eigvals(A_cl)
        
        # Stability check
        is_stable = all(np.real(pole) < 0 for pole in poles)
        stable_count.append(1 if is_stable else 0)
    
    plt.plot(uncertainties * 100, stable_count, 'bo-')
    plt.title('Robust stability')
    plt.xlabel('Parameter uncertainty (%)')
    plt.ylabel('Stability (1=yes, 0=no)')
    plt.grid(True)
    plt.ylim([-0.1, 1.1])
    
    plt.tight_layout()
    plt.show()

# Main program
if __name__ == "__main__":
    # Synthesis
    A, B, C, D, results = place_poles_comparison()
    
    # Simulation and comparison
    simulate_closed_loop_systems(A, B, C, D, results)
    
    # Robustness analysis
    robust_pole_placement()
    
    print(f"\n{'='*60}")
    print("CONCLUSIONS")
    print(f"{'='*60}")
    print("1. Pole placement allows precise shaping of system dynamics")
    print("2. Faster poles yield faster response but require larger control effort")
    print("3. Complex poles may introduce overshoot")
    print("4. Robustness must be considered under parameter uncertainty")
    print("5. Trade-off between speed and control energy")
```

This example demonstrates:

1. **Implementation of Ackermann’s formula** to compute the feedback gain
2. **Comparison of pole-placement strategies** and their effect on dynamics
3. **Transient quality metrics** (settling time, overshoot)
4. **Result visualization** via step responses, pole maps, and phase portraits
5. **Robustness analysis** — effects of model inaccuracies on controller performance
6. **Practical guidance** for selecting desired poles
