# Lesson 3 — Stability Analysis of Control Systems

## 1. Concept of Stability

**Stability** is one of the most important properties of any dynamical system, including aircraft. Simply put, stability is the ability of a system to return to its original equilibrium state after a disturbing influence ceases.

Imagine a ball in different positions:

* **Stable equilibrium:** A ball at the bottom of a bowl. If pushed, it oscillates and returns to the bottom.
* **Unstable equilibrium:** A ball on top of a hill. The slightest push and it rolls off, not returning.
* **Neutral equilibrium:** A ball on a flat horizontal surface. If pushed, it comes to rest at a new position and stays there.

For an aircraft, stability means that after a gust of wind or a brief control deflection, it will autonomously return to the initial flight condition (e.g., straight-and-level flight).

## 2. Types of Stability

There are two primary types of stability:

1. **Static stability:** The ability of a system to generate restoring forces and moments that tend to return it to the equilibrium state immediately after a disturbance. This is the initial tendency to return.
2. **Dynamic stability:** The nature of the transient response as the system returns to equilibrium. A dynamically stable system not only tends to return but does so without growing oscillations; the response decays over time.

A system may be statically stable but dynamically unstable (when undamped or growing oscillations occur about the equilibrium state).

## 3. Aircraft Stability

For analysis convenience, aircraft motion is split into two types:

1. **Longitudinal motion:** Motion in the vertical plane (changes in altitude, pitch angle, speed).
2. **Lateral-directional motion:** Motion in the horizontal plane (changes in heading, roll angle, sideslip angle).

Accordingly, stability is considered separately for each motion type [1]:

* **Longitudinal stability:** The ability to maintain a commanded angle of attack and airspeed.
* **Directional (yaw) stability:** The ability to maintain a commanded flight direction (heading).
* **Lateral (roll) stability:** The ability to restore the original bank angle.

These three combine to determine the overall **lateral-directional stability**.

## 4. Stability Criteria for Linear Systems

For linear systems described in state-space, the key is the **eigenvalues** of the system matrix `A`.

The system `ẋ = Ax` is stable if and only if **all eigenvalues of `A` have strictly negative real parts**.

```
Re(λᵢ) < 0 for all i = 1, ..., n
```

where `λᵢ` are the eigenvalues of matrix `A`.

If at least one eigenvalue has a positive real part, the system is unstable. If some eigenvalues have zero real parts (and none are positive), the system is marginally stable.

### 4.1. Algebraic criteria (Routh–Hurwitz)

These criteria allow us to determine whether the characteristic equation (obtained from `det(A − λI) = 0`) has roots with positive real parts without computing the roots themselves. This is convenient for analytical studies.

### 4.2. Frequency-domain criteria (Nyquist, Mikhailov)

These are based on frequency-response analysis and allow us to infer closed-loop stability from open-loop characteristics. They are especially handy when characteristics are obtained experimentally.

## 5. References

1. Alekseenkov V. Stability of Aircraft (slides). — Aviation College. — [URL: http://taviak.ru/distance/wp-content/uploads/2013/PLA/Ustoi_chivost_letatel_nykh_apparatov(Alekseenkov).pdf](http://taviak.ru/distance/wp-content/uploads/2013/PLA/Ustoi_chivost_letatel_nykh_apparatov(Alekseenkov).pdf)
2. Longitudinal stability of aircraft — Great Russian Encyclopedia. — [URL: https://bigenc.ru/c/prodol-naia-ustoichivost-letatel-nogo-apparata-5e7446](https://bigenc.ru/c/prodol-naia-ustoichivost-letatel-nogo-apparata-5e7446)
3. Stability and controllability of an airplane — Vzletim.ru. — [URL: https://vzletim.ru/upload/iblock/133/aerodynamics09.pdf](https://vzletim.ru/upload/iblock/133/aerodynamics09.pdf)


## 6. Practical Python Example

We will analyze the stability of several systems and demonstrate stability criteria.

```python
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.linalg import eigvals
import sympy as sp

def analyze_stability(A, system_name="System"):
    """Stability analysis by the eigenvalues of matrix A"""
    
    print(f"\n{'='*50}")
    print(f"STABILITY ANALYSIS: {system_name}")
    print(f"{'='*50}")
    
    print("Matrix A:")
    print(A)
    
    # Compute eigenvalues
    eigenvalues = eigvals(A)
    print("\nEigenvalues (poles):")
    
    stable = True
    for i, lam in enumerate(eigenvalues):
        real_part = np.real(lam)
        imag_part = np.imag(lam)
        
        if abs(imag_part) < 1e-10:  # real number
            print(f"λ_{i+1} = {real_part:.4f}")
        else:
            print(f"λ_{i+1} = {real_part:.4f} + {imag_part:.4f}j")
        
        if real_part >= 0:
            stable = False
    
    print(f"\nAll real parts negative: {stable}")
    
    if stable:
        print("✓ SYSTEM IS STABLE")
    else:
        print("✗ SYSTEM IS UNSTABLE")
    
    return eigenvalues, stable

def plot_poles(eigenvalues, title="Pole locations"):
    """Plot pole map on the complex plane"""
    
    plt.figure(figsize=(8, 6))
    
    for lam in eigenvalues:
        plt.plot(np.real(lam), np.imag(lam), 'rx', markersize=10, markeredgewidth=2)
    
    # Imaginary axis (stability boundary)
    y_max = max(abs(np.imag(eigenvalues))) * 1.2 if len(eigenvalues) > 0 else 1
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Stability boundary')
    
    # Shade stable half-plane
    x_min = min(np.real(eigenvalues)) * 1.2 if len(eigenvalues) > 0 else -1
    plt.axvspan(x_min, 0, alpha=0.2, color='green', label='Stable region')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    
    return plt.gcf()

# Example 1: Stable system (mass–spring–damper)
A1 = np.array([[0, 1],
               [-4, -1]])

eigenvals1, stable1 = analyze_stability(A1, "Mass–spring–damper")

# Example 2: Unstable system (inverted pendulum, linearized)
A2 = np.array([[0, 1],
               [1, 0]])  # no damping and an "effective negative spring"

eigenvals2, stable2 = analyze_stability(A2, "Inverted pendulum")

# Example 3: Marginally stable system
A3 = np.array([[0, 1],
               [-1, 0]])  # conservative oscillator

eigenvals3, stable3 = analyze_stability(A3, "Conservative oscillator")

# Example 4: System with complex poles
A4 = np.array([[0, 1, 0],
               [-2, -0.5, 0],
               [0, 0, -3]])

eigenvals4, stable4 = analyze_stability(A4, "Third-order system")

# Pole maps
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
for lam in eigenvals1:
    plt.plot(np.real(lam), np.imag(lam), 'rx', markersize=10, markeredgewidth=2)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.axvspan(-5, 0, alpha=0.2, color='green')
plt.grid(True, alpha=0.3)
plt.title('Stable system')
plt.xlabel('Re(λ)')
plt.ylabel('Im(λ)')

plt.subplot(2, 2, 2)
for lam in eigenvals2:
    plt.plot(np.real(lam), np.imag(lam), 'rx', markersize=10, markeredgewidth=2)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.axvspan(-2, 0, alpha=0.2, color='green')
plt.axvspan(0, 2, alpha=0.2, color='red')
plt.grid(True, alpha=0.3)
plt.title('Unstable system')
plt.xlabel('Re(λ)')
plt.ylabel('Im(λ)')

plt.subplot(2, 2, 3)
for lam in eigenvals3:
    plt.plot(np.real(lam), np.imag(lam), 'rx', markersize=10, markeredgewidth=2)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.axvspan(-2, 0, alpha=0.2, color='green')
plt.grid(True, alpha=0.3)
plt.title('Stability boundary')
plt.xlabel('Re(λ)')
plt.ylabel('Im(λ)')

plt.subplot(2, 2, 4)
for lam in eigenvals4:
    plt.plot(np.real(lam), np.imag(lam), 'rx', markersize=10, markeredgewidth=2)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.axvspan(-4, 0, alpha=0.2, color='green')
plt.grid(True, alpha=0.3)
plt.title('3rd-order system')
plt.xlabel('Re(λ)')
plt.ylabel('Im(λ)')

plt.tight_layout()
plt.show()

# Step-response comparison
print(f"\n{'='*60}")
print("STEP RESPONSE COMPARISON")
print(f"{'='*60}")

# Build systems
sys1 = ctrl.ss(A1, np.array([[0], [1]]), np.array([[1, 0]]), 0)
sys2 = ctrl.ss(A2, np.array([[0], [1]]), np.array([[1, 0]]), 0)
sys3 = ctrl.ss(A3, np.array([[0], [1]]), np.array([[1, 0]]), 0)

# Time axis
t = np.linspace(0, 10, 1000)

plt.figure(figsize=(15, 5))

# Stable system
plt.subplot(1, 3, 1)
try:
    t1, y1 = ctrl.step_response(sys1, t)
    plt.plot(t1, y1)
    plt.title('Stable system\n(decaying response)')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.grid(True)
except:
    plt.text(0.5, 0.5, 'Simulation error', ha='center', va='center')

# Unstable system
plt.subplot(1, 3, 2)
try:
    t2, y2 = ctrl.step_response(sys2, t)
    plt.plot(t2, y2)
    plt.title('Unstable system\n(diverging response)')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.grid(True)
    plt.ylim([-10, 10])  # limit scale for clarity
except:
    plt.text(0.5, 0.5, 'System unstable\n(diverging response)', ha='center', va='center')

# Marginally stable system
plt.subplot(1, 3, 3)
try:
    t3, y3 = ctrl.step_response(sys3, t)
    plt.plot(t3, y3)
    plt.title('Stability boundary\n(persistent oscillations)')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.grid(True)
except:
    plt.text(0.5, 0.5, 'Simulation error', ha='center', va='center')

plt.tight_layout()
plt.show()

# Routh–Hurwitz criterion (symbolic-like procedure)
print(f"\n{'='*60}")
print("ROUTH–HURWITZ CRITERION")
print(f"{'='*60}")

def routh_hurwitz_table(coeffs):
    """Build the Routh–Hurwitz table for a real polynomial"""
    n = len(coeffs)
    table = np.zeros((n, (n+1)//2))
    
    # First two rows
    table[0, :] = coeffs[::2]  # even-indexed coefficients
    if n > 1:
        table[1, :len(coeffs[1::2])] = coeffs[1::2]  # odd-indexed coefficients
    
    # Remaining rows
    for i in range(2, n):
        for j in range(table.shape[1]-1):
            if table[i-1, 0] != 0:
                table[i, j] = (table[i-1, 0] * table[i-2, j+1] - 
                               table[i-2, 0] * table[i-1, j+1]) / table[i-1, 0]
    
    return table

# Example: characteristic polynomial s^3 + 2 s^2 + 3 s + 4
coeffs = [1, 2, 3, 4]  # coefficients from highest to lowest degree
print("Characteristic polynomial: s^3 + 2 s^2 + 3 s + 4")
print("Routh–Hurwitz table:")
routh_table = routh_hurwitz_table(coeffs)
print(routh_table)

# Sign changes in the first column
first_column = routh_table[:, 0]
sign_changes = sum(1 for i in range(len(first_column)-1) 
                   if first_column[i] * first_column[i+1] < 0)
print(f"\nNumber of sign changes in the first column: {sign_changes}")
print(f"Number of right-half-plane poles: {sign_changes}")

if sign_changes == 0:
    print("✓ System is stable by the Routh–Hurwitz criterion")
else:
    print("✗ System is unstable by the Routh–Hurwitz criterion")
```

This example demonstrates:

1. **Stability analysis** via eigenvalues of the system matrix
2. **Pole visualization** on the complex plane with the stable region highlighted
3. **Comparison of step responses** for stable, unstable, and marginally stable systems
4. **Use of the Routh–Hurwitz criterion** for analytical stability assessment
5. **Practical examples** of different types of dynamical systems