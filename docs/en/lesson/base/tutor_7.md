# Lesson 7 — Practice: Stabilizing Aircraft Longitudinal Motion

## 1. Goal

In this final workshop, we apply everything learned in practice. Our goal is to design an automatic stabilization system (autopilot) for a simplified longitudinal model of a light aircraft. We’ll go from modeling to a full controller-plus-observer synthesis.

## 2. Plant Model

Consider a linearized short-period longitudinal model. State variables are:

* `α` — angle of attack
* `q` — pitch rate

The control input is elevator deflection `δₑ` (delta_e). The measurable output is the pitch angle `θ` (theta), here equal to the integral of `q`.

System matrices (for a particular aircraft and flight condition) can be:

```
A = [[-0.313, 1.0], [-0.0139, -0.426]]
B = [[-0.0322], [0.0]]
C = [[0, 1]]
D = [0]
```

## 3. Step 1: Analyze the Open-Loop System

### 3.1. Stability

First, check the “bare” aircraft (no control) by computing eigenvalues of `A`.

`λ₁,₂ = -0.3695 ± 0.1627j`

Both real parts are negative, so the aircraft is **statically and dynamically stable**. However, the damping (`σ = 0.3695`) is relatively small, which may cause undesirable oscillations.

### 3.2. Controllability and Observability

Check whether the system is controllable and observable.

* **Controllability matrix:** `P = [B, AB]`, `rank(P) = 2` — fully controllable.
* **Observability matrix:** `Q = [C; CA]`, `rank(Q) = 2` — fully observable.

We can proceed with synthesis.

## 4. Step 2: Controller Synthesis (Pole Placement)

We want the system to be more responsive with better damping. Target closed-loop poles:

`p₁,₂ = -1.0 ± 1.0j`

This yields faster decay (`σ = 1.0` vs `0.3695`) and higher oscillation frequency.

1. **Desired characteristic polynomial:** `α(s) = (s - p₁)(s - p₂) = s² + 2s + 2`.
2. **Use Ackermann’s formula** (or equivalent) to find `K`.

Result:

`K = [-26.4, -15.7]`

Control law: `δₑ = -([-26.4, -15.7]) · [α; q] = 26.4 α + 15.7 q`

## 5. Step 3: Observer Synthesis

Assume `α` is not directly measurable; only `q` is measured. We need an observer to estimate `α`.

Make the observer faster than the controller, e.g., 5× faster:

`l₁,₂ = -5.0 ± 5.0j`

1. **Desired observer polynomial:** `α_obs(s) = s² + 10s + 50`.
2. **Use the dual of Ackermann’s formula** to find `L`.

This yields the observer gain `L`.

## 6. Step 4: Assemble the Full System

Full control system consists of:

1. **State observer:**
    `x̂̇ = A x̂ + B δₑ + L (q - q̂)`
2. **Controller:**
    `δₑ = -K x̂`

This autopilot uses measured `q` and commands `δₑ` to stabilize the aircraft with the desired dynamics set by `p₁,₂`.

## 7. Homework

1. Using MATLAB, Python (`control` library), or similar tools, reproduce the computations:
    * Verify stability, controllability, and observability of the open-loop system.
    * Compute `K` for the specified controller poles.
    * Compute `L` for the specified observer poles.
2. Build a Simulink model or write a script to simulate:
    * The open-loop system.
    * The system with controller (true-state feedback).
    * The full system (controller + observer).
3. Compare transients for all three cases under initial perturbation (e.g., initial `α`). Confirm the controller improves behavior vs open-loop, and the full system closely approximates ideal state feedback.

## 8. References

1. Aircraft Pitch: State-Space Methods for Controller Design — University of Michigan. – [URL: https://ctms.engin.umich.edu/CTMS/index.php?example=AircraftPitch&section=ControlStateSpace](https://ctms.engin.umich.edu/CTMS/index.php?example=AircraftPitch&section=ControlStateSpace)
2. Python Control Systems Library. – [URL: https://python-control.readthedocs.io/en/latest/](https://python-control.readthedocs.io/en/latest/)
3. MATLAB Control System Toolbox. – [URL: https://www.mathworks.com/products/control.html](https://www.mathworks.com/products/control.html)


## 9. Full Python Implementation

We implement a complete longitudinal stabilization example with detailed analysis.

```python
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.linalg import eigvals
from scipy.integrate import solve_ivp

class AircraftLongitudinalControl:
    """
    Class for modeling and control of longitudinal aircraft motion
    """
    
    def __init__(self):
        # Short-period longitudinal model
        # States: [alpha, q] - angle of attack and pitch rate
        self.A = np.array([[-0.313, 1.0],
                          [-0.0139, -0.426]])
        
        self.B = np.array([[-0.0322],
                          [0.0]])
        
        self.C = np.array([[0, 1]])  # we measure only the pitch rate q
        
        self.D = np.array([[0]])
        
        # Analysis parameters
        self.n = self.A.shape[0]
        
        print("AIRCRAFT LONGITUDINAL MODEL")
        print("="*50)
        print("States: [α, q]")
        print("α - angle of attack (rad)")
        print("q - pitch rate (rad/s)")
        print("Control: δe - elevator deflection (rad)")
        print("Measurement: q - pitch rate")
        print()
        print(f"A = \n{self.A}")
        print(f"B = \n{self.B}")
        print(f"C = \n{self.C}")
    
    def analyze_open_loop(self):
        """Open-loop analysis"""
        
        print(f"\n{'='*60}")
        print("OPEN-LOOP ANALYSIS")
        print(f"{'='*60}")
        
        # Eigenvalues
        poles = eigvals(self.A)
        print(f"Open-loop poles: {poles}")
        
        # Stability
        stable = all(np.real(pole) < 0 for pole in poles)
        print(f"Stability: {'✓ Stable' if stable else '✗ Unstable'}")
        
        # Oscillation characteristics
        for i, pole in enumerate(poles):
            real_part = np.real(pole)
            imag_part = np.imag(pole)
            
            if abs(imag_part) > 1e-6:
                freq = abs(imag_part)
                damping_ratio = -real_part / abs(pole)
                period = 2 * np.pi / freq
                
                print(f"\nPole {i+1}: {pole:.4f}")
                print(f"  Frequency: {freq:.4f} rad/s")
                print(f"  Period: {period:.2f} s")
                print(f"  Damping ratio: {damping_ratio:.4f}")
                
                if period < 5:
                    print("  Type: Short-period mode")
                else:
                    print("  Type: Long-period mode")
        
        # Controllability and observability
        P = ctrl.ctrb(self.A, self.B)
        O = ctrl.obsv(self.A, self.C)
        
        controllable = np.linalg.matrix_rank(P) == self.n
        observable = np.linalg.matrix_rank(O) == self.n
        
        print(f"\nControllability: {'✓' if controllable else '✗'}")
        print(f"Observability: {'✓' if observable else '✗'}")
        
        if not controllable:
            print("⚠️  System is uncontrollable! Synthesis impossible.")
        if not observable:
            print("⚠️  System is unobservable! Observer impossible.")
        
        return poles, controllable, observable
    
    def design_controller(self, desired_poles=None):
        """Controller synthesis via pole placement"""
        
        if desired_poles is None:
            desired_poles = [-1.0 + 1.0j, -1.0 - 1.0j]  # faster and better damped
        
        print(f"\n{'='*60}")
        print("CONTROLLER SYNTHESIS")
        print(f"{'='*60}")
        
        print(f"Desired poles: {desired_poles}")
        
        # Place poles
        self.K = ctrl.place(self.A, self.B, desired_poles)
        
        # Verify result
        A_cl = self.A - self.B @ self.K
        actual_poles = eigvals(A_cl)
        
        print(f"Feedback gain K: {self.K}")
        print(f"Actual poles: {actual_poles}")
        
        # Control law
        print(f"\nControl law:")
        print(f"δe = -K * [α; q] = -{self.K[0,0]:.3f}*α - {self.K[0,1]:.3f}*q")
        
        return self.K, actual_poles
    
    def design_observer(self, observer_poles=None):
        """State observer synthesis"""
        
        if observer_poles is None:
            # Observer poles 5 times faster than controller poles
            observer_poles = [-5.0 + 5.0j, -5.0 - 5.0j]
        
        print(f"\n{'='*60}")
        print("OBSERVER SYNTHESIS")
        print(f"{'='*60}")
        
        print(f"Desired observer poles: {observer_poles}")
        
        # Observer synthesis (duality)
        L_T = ctrl.place(self.A.T, self.C.T, observer_poles)
        self.L = L_T.T
        
        # Verify result
        A_obs = self.A - self.L @ self.C
        actual_observer_poles = eigvals(A_obs)
        
        print(f"Observer gain L: {self.L}")
        print(f"Actual observer poles: {actual_observer_poles}")
        
        # Observer equation
        print(f"\nObserver equation:")
        print(f"α̂' = {self.A[0,0]:.3f}*α̂ + {self.A[0,1]:.3f}*q̂ + {self.B[0,0]:.3f}*δe + {self.L[0,0]:.3f}*(q - q̂)")
        print(f"q̂' = {self.A[1,0]:.3f}*α̂ + {self.A[1,1]:.3f}*q̂ + {self.B[1,0]:.3f}*δe + {self.L[1,0]:.3f}*(q - q̂)")
        
        return self.L, actual_observer_poles
    
    def simulate_scenarios(self, t_sim=15, disturbance_time=2):
        """Simulate different control scenarios"""
        
        print(f"\n{'='*60}")
        print("SYSTEM SIMULATION")
        print(f"{'='*60}")
        
        t = np.linspace(0, t_sim, 1000)
        
        # Initial conditions
        alpha0 = 0.1  # initial AoA (rad) ≈ 5.7°
        q0 = 0.0      # initial pitch rate
        x0_true = np.array([alpha0, q0])
        
        # Initial estimate (inaccurate)
        x0_est = np.array([0.0, 0.0])
        
        print(f"Initial conditions:")
        print(f"  True state: α₀ = {alpha0:.3f} rad ({np.degrees(alpha0):.1f}°), q₀ = {q0:.3f} rad/s")
        print(f"  Initial estimate: α̂₀ = {x0_est[0]:.3f} rad, q̂₀ = {x0_est[1]:.3f} rad/s")
        
        # Scenario 1: No control
        print("\nSimulating: Open-loop system...")
        sys_open = ctrl.ss(self.A, np.zeros((2, 1)), np.eye(2), np.zeros((2, 1)))
        t1, x1 = ctrl.initial_response(sys_open, t, X0=x0_true, return_x=True)
        
        # Scenario 2: Ideal state feedback
        print("Simulating: Ideal feedback...")
        A_ideal = self.A - self.B @ self.K
        sys_ideal = ctrl.ss(A_ideal, np.zeros((2, 1)), np.eye(2), np.zeros((2, 1)))
        t2, x2 = ctrl.initial_response(sys_ideal, t, X0=x0_true, return_x=True)
        u2 = -self.K @ x2  # control signal
        
        # Scenario 3: Full system (controller + observer)
        print("Simulating: Full system with observer...")
        
        def full_system_dynamics(t, state):
            """Full system dynamics"""
            x = state[:2]      # true state [α, q]
            x_hat = state[2:]  # estimated state [α̂, q̂]
            
            # Measurement (pitch rate only)
            y = self.C @ x
            
            # Control based on estimate
            u = -self.K @ x_hat
            
            # Disturbance (wind gust at disturbance_time)
            disturbance = 0.05 if abs(t - disturbance_time) < 0.1 else 0.0
            
            # True system dynamics
            dx_dt = self.A @ x + self.B @ u + np.array([disturbance, 0])
            
            # Observer dynamics
            dx_hat_dt = self.A @ x_hat + self.B @ u + self.L @ (y - self.C @ x_hat)
            
            return np.concatenate([dx_dt, dx_hat_dt])
        
        # Numerical integration
        x0_full = np.concatenate([x0_true, x0_est])
        sol = solve_ivp(full_system_dynamics, [0, t_sim], x0_full, t_eval=t, rtol=1e-8)
        
        x3 = sol.y[:2, :]
        x_hat3 = sol.y[2:, :]
        u3 = -self.K @ x_hat3
        error3 = x3 - x_hat3
        
        return {
            't': t,
            'open_loop': {'t': t1, 'x': x1},
            'ideal_feedback': {'t': t2, 'x': x2, 'u': u2},
            'full_system': {'t': t, 'x': x3, 'x_hat': x_hat3, 'u': u3, 'error': error3}
        }
    
    def plot_results(self, results):
        """Plot simulation results"""
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        
        t = results['t']
        open_loop = results['open_loop']
        ideal = results['ideal_feedback']
        full = results['full_system']
        
        # 1. Angle of attack
        axes[0, 0].plot(open_loop['t'], np.degrees(open_loop['x'][0, :]), 'k--', 
                       linewidth=2, label='Open loop')
        axes[0, 0].plot(ideal['t'], np.degrees(ideal['x'][0, :]), 'b-', 
                       linewidth=2, label='Ideal FB')
        axes[0, 0].plot(t, np.degrees(full['x'][0, :]), 'r-', 
                       linewidth=2, label='With observer')
        axes[0, 0].set_title('Angle of attack α')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('α (deg)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Pitch rate
        axes[0, 1].plot(open_loop['t'], np.degrees(open_loop['x'][1, :]), 'k--', 
                       linewidth=2, label='Open loop')
        axes[0, 1].plot(ideal['t'], np.degrees(ideal['x'][1, :]), 'b-', 
                       linewidth=2, label='Ideal FB')
        axes[0, 1].plot(t, np.degrees(full['x'][1, :]), 'r-', 
                       linewidth=2, label='With observer')
        axes[0, 1].set_title('Pitch rate q')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('q (deg/s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Control input
        axes[0, 2].plot(ideal['t'], np.degrees(ideal['u'][0, :]), 'b-', 
                       linewidth=2, label='Ideal FB')
        axes[0, 2].plot(t, np.degrees(full['u'][0, :]), 'r-', 
                       linewidth=2, label='With observer')
        axes[0, 2].set_title('Elevator deflection δe')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('δe (deg)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Estimation error — angle of attack
        axes[1, 0].plot(t, np.degrees(full['error'][0, :]), 'g-', linewidth=2)
        axes[1, 0].set_title('Estimation error of α')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('α - α̂ (deg)')
        axes[1, 0].grid(True)
        
        # 5. Estimation error — pitch rate
        axes[1, 1].plot(t, np.degrees(full['error'][1, :]), 'g-', linewidth=2)
        axes[1, 1].set_title('Estimation error of pitch rate')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('q - q̂ (deg/s)')
        axes[1, 1].grid(True)
        
        # 6. True state vs estimate
        axes[1, 2].plot(t, np.degrees(full['x'][0, :]), 'b-', linewidth=2, label='True α')
        axes[1, 2].plot(t, np.degrees(full['x_hat'][0, :]), 'r--', linewidth=2, label='Estimate α̂')
        axes[1, 2].set_title('Comparison: true vs estimate (α)')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('α (deg)')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        # 7. Phase portrait
        axes[2, 0].plot(np.degrees(open_loop['x'][0, :]), np.degrees(open_loop['x'][1, :]), 
                       'k--', linewidth=2, label='Open loop')
        axes[2, 0].plot(np.degrees(ideal['x'][0, :]), np.degrees(ideal['x'][1, :]), 
                       'b-', linewidth=2, label='Ideal FB')
        axes[2, 0].plot(np.degrees(full['x'][0, :]), np.degrees(full['x'][1, :]), 
                       'r-', linewidth=2, label='With observer')
        axes[2, 0].plot(np.degrees(full['x'][0, 0]), np.degrees(full['x'][1, 0]), 
                       'ko', markersize=8, label='Initial point')
        axes[2, 0].set_title('Phase portrait')
        axes[2, 0].set_xlabel('α (deg)')
        axes[2, 0].set_ylabel('q (deg/s)')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # 8. Pole map
        poles_open = eigvals(self.A)
        poles_controller = eigvals(self.A - self.B @ self.K)
        poles_observer = eigvals(self.A - self.L @ self.C)
        
        axes[2, 1].plot(np.real(poles_open), np.imag(poles_open), 'ko', 
                       markersize=10, label='Open loop')
        axes[2, 1].plot(np.real(poles_controller), np.imag(poles_controller), 'bs', 
                       markersize=8, label='Controller')
        axes[2, 1].plot(np.real(poles_observer), np.imag(poles_observer), 'r^', 
                       markersize=8, label='Observer')
        
        axes[2, 1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        axes[2, 1].axvspan(-8, 0, alpha=0.2, color='green', label='Stable region')
        axes[2, 1].set_title('Pole map')
        axes[2, 1].set_xlabel('Real part')
        axes[2, 1].set_ylabel('Imag part')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        # 9. Quality analysis
        # Settling time (2% criterion)
        final_alpha_ideal = ideal['x'][0, -1]
        final_alpha_full = full['x'][0, -1]
        
        settling_idx_ideal = np.where(np.abs(ideal['x'][0, :] - final_alpha_ideal) <= 
                                     0.02 * abs(final_alpha_ideal))[0]
        settling_idx_full = np.where(np.abs(full['x'][0, :] - final_alpha_full) <= 
                                    0.02 * abs(final_alpha_full))[0]
        
        settling_time_ideal = ideal['t'][settling_idx_ideal[0]] if len(settling_idx_ideal) > 0 else ideal['t'][-1]
        settling_time_full = t[settling_idx_full[0]] if len(settling_idx_full) > 0 else t[-1]
        
        systems = ['Ideal FB', 'With observer']
        settling_times = [settling_time_ideal, settling_time_full]
        
        axes[2, 2].bar(systems, settling_times, alpha=0.7, color=['blue', 'red'])
        axes[2, 2].set_title('Settling time')
        axes[2, 2].set_ylabel('Time (s)')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print numerical results
        print(f"\n{'='*60}")
        print("RESULTS ANALYSIS")
        print(f"{'='*60}")
        print(f"Settling time (ideal): {settling_time_ideal:.2f} s")
        print(f"Settling time (observer): {settling_time_full:.2f} s")
        print(f"Difference: {abs(settling_time_full - settling_time_ideal):.2f} s")
        
        max_error_alpha = np.max(np.abs(full['error'][0, :]))
        max_error_q = np.max(np.abs(full['error'][1, :]))
        print(f"Max estimation error α: {np.degrees(max_error_alpha):.3f}°")
        print(f"Max estimation error q: {np.degrees(max_error_q):.3f}°/s")
        
        max_control_ideal = np.max(np.abs(ideal['u'][0, :]))
        max_control_full = np.max(np.abs(full['u'][0, :]))
        print(f"Max elevator deflection (ideal): {np.degrees(max_control_ideal):.1f}°")
        print(f"Max elevator deflection (observer): {np.degrees(max_control_full):.1f}°")

def main():
    """Main function — full example"""
    
    # Create control system object
    aircraft = AircraftLongitudinalControl()
    
    # Analyze open-loop system
    poles, controllable, observable = aircraft.analyze_open_loop()
    
    if not (controllable and observable):
        print("System is not suitable for synthesis!")
        return
    
    # Controller synthesis
    K, controller_poles = aircraft.design_controller()
    
    # Observer synthesis
    L, observer_poles = aircraft.design_observer()
    
    # Simulation
    results = aircraft.simulate_scenarios()
    
    # Plot results
    aircraft.plot_results(results)
    
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    print("✓ Successfully synthesized an automatic control system")
    print("✓ System is stable with good dynamic performance")
    print("✓ Observer provides accurate estimates of unmeasured states")
    print("✓ Observer-based control approaches ideal state feedback")

if __name__ == "__main__":
    main()
```

### Homework

```python
# Additional self-study experiments

def extended_experiments():
    """Additional experiments"""
    
    aircraft = AircraftLongitudinalControl()
    
    # 1. Experiment with different desired poles
    pole_variants = [
        [-0.5 + 0.5j, -0.5 - 0.5j],  # slow
        [-2 + 2j, -2 - 2j],          # fast
        [-1, -2],                     # aperiodic
        [-0.7 + 1.4j, -0.7 - 1.4j]   # lightly damped
    ]
    
    print("Task 1: Study the effect of different controller poles")
    for i, poles in enumerate(pole_variants):
        print(f"Variant {i+1}: {poles}")
        # TODO: Implement synthesis and comparison
    
    # 2. Measurement noise experiment
    print("\nTask 2: Add measurement noise and assess robustness")
    # TODO: Add white noise to q measurements
    
    # 3. Parameter uncertainty experiment
    print("\nTask 3: Study the effect of model uncertainty")
    # TODO: Vary A elements within ±20%
    
    # 4. Compare with a PID controller
    print("\nTask 4: Compare with a classical PID controller")
    # TODO: Implement PID and compare results

# Run these experiments for deeper exploration!
```

This complete example demonstrates:

1. **Object-oriented approach** to control system design
2. **Full development cycle** from analysis to simulation
3. **Detailed visualization** of all aspects of system operation
4. **Practical recommendations** for parameter tuning
5. **Self-study exercises** for reinforcing the material
