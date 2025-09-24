# Lesson 2 — Controllability and Observability

## 1. Introduction to Evaluating Control Systems

After we learn how to represent dynamical systems in state space, two fundamental questions arise:

1. **Controllability:** Can we drive the system from any initial state to any final state within finite time using the available control inputs?
2. **Observability:** Can we infer the internal state of the system by observing only its outputs?

These concepts—**controllability** and **observability**—are key for analyzing and synthesizing control systems. They determine how well we can influence and monitor the system’s behavior. They were first introduced by Rudolf Kalman.

## 2. Controllability

**Definition:** A system is completely controllable if there exists an unrestricted control input `u(t)` that transfers the system from any initial state `x(t₀)` to any final state `x(t₁)` in finite time `t₁ - t₀`.

Intuitively, controllability means we can reach any point in the state space by manipulating the system inputs. If a system is uncontrollable, some states are unreachable regardless of the control actions we apply.

### 2.1 Kalman Controllability Criterion

For linear time-invariant (LTI) systems described by

```
ẋ = Ax + Bu
```

the controllability criterion is based on the rank of the **controllability matrix** `P`:

```
P = [B, AB, A²B, ..., Aⁿ⁻¹B]
```

where `n` is the dimension of the state vector.

**The system is fully controllable if and only if the rank of the controllability matrix equals the state-space dimension:**

```
rank(P) = n
```

If `rank(P) < n`, the system is uncontrollable—there exist states that cannot be reached by any input.

## 3. Observability

**Definition:** A system is completely observable if, from the known inputs `u(t)` and outputs `y(t)` over a finite time interval `t₁ - t₀`, we can uniquely determine the initial state `x(t₀)`.

Observability tells us whether we can understand the internal behavior of the system (its state) by monitoring what we can measure (its outputs). If a system is unobservable, some internal states never affect the outputs and cannot be distinguished.

### 3.1 Kalman Observability Criterion

For LTI systems described by

```
ẋ = Ax + Bu
y = Cx + Du
```

the observability criterion uses the rank of the **observability matrix** `Q`:

```
    [  C  ]
    [  CA ]
Q = [ CA² ]
    [  ... ]
    [CAⁿ⁻¹]
```

**The system is fully observable if and only if the rank of `Q` equals the state-space dimension:**

```
rank(Q) = n
```

If `rank(Q) < n`, the system is unobservable.

## 4. Principle of Duality

There is an important relationship between controllability and observability called the **duality principle**. It states that the system `(A, B)` is controllable if and only if the dual system `(Aᵀ, Cᵀ)` is observable, and vice versa.

In other words, the mathematical tests for controllability and observability mirror each other.

## 5. References

1. Observability (control theory) // Wikipedia. URL: https://ru.wikipedia.org/wiki/Наблюдаемость_(теория_управления)
2. Controllability and observability // Studfile. URL: https://studfile.net/preview/2202942/page:8/
3. Controllability and Observability // Rutgers ECE. URL: http://eceweb1.rutgers.edu/~gajic/psfiles/chap5traCO.pdf

## 6. Practical Example in Python

We will check controllability and observability for the system from the first seminar and examine uncontrollable and unobservable cases.

```python
import numpy as np
import control as ctrl
from scipy.linalg import matrix_rank

def check_controllability(A, B):
    """Check system controllability"""
    n = A.shape[0]

    # Construct the controllability matrix P = [B, AB, A²B, ..., A^(n-1)B]
    P = B.copy()
    AB = A @ B

    for _ in range(1, n):
        P = np.hstack([P, AB])
        AB = A @ AB

    rank_P = matrix_rank(P)

    print("Controllability matrix P:")
    print(P)
    print(f"Rank(P) = {rank_P}")
    print(f"System dimension = {n}")

    if rank_P == n:
        print("✓ The system is FULLY CONTROLLABLE")
        return True
    else:
        print("✗ The system is NOT FULLY CONTROLLABLE")
        return False

def check_observability(A, C):
    """Check system observability"""
    n = A.shape[0]

    # Construct the observability matrix Q = [C; CA; CA²; ...; CA^(n-1)]
    Q = C.copy()
    CA = C @ A

    for _ in range(1, n):
        Q = np.vstack([Q, CA])
        CA = CA @ A

    rank_Q = matrix_rank(Q)

    print("Observability matrix Q:")
    print(Q)
    print(f"Rank(Q) = {rank_Q}")
    print(f"System dimension = {n}")

    if rank_Q == n:
        print("✓ The system is FULLY OBSERVABLE")
        return True
    else:
        print("✗ The system is NOT FULLY OBSERVABLE")
        return False

# Example 1: Controllable and observable system (mass–spring–damper)
print("=" * 60)
print("EXAMPLE 1: Controllable and observable system")
print("=" * 60)

A1 = np.array([[0, 1],
               [-4, -1]])

B1 = np.array([[0],
               [1]])

C1 = np.array([[1, 0]])

print("System 1:")
print(f"A = \n{A1}")
print(f"B = \n{B1}")
print(f"C = \n{C1}")
print()

controllable1 = check_controllability(A1, B1)
print()
observable1 = check_observability(A1, C1)

# Example 2: Uncontrollable system
print("\n" + "=" * 60)
print("EXAMPLE 2: Uncontrollable system")
print("=" * 60)

A2 = np.array([[0, 1, 0],
               [-2, -1, 0],
               [0, 0, -3]])

B2 = np.array([[0],
               [1],
               [0]])  # control does not affect the third state

C2 = np.array([[1, 0, 1]])

print("System 2:")
print(f"A = \n{A2}")
print(f"B = \n{B2}")
print(f"C = \n{C2}")
print()

controllable2 = check_controllability(A2, B2)
print()
observable2 = check_observability(A2, C2)

# Example 3: Unobservable system
print("\n" + "=" * 60)
print("EXAMPLE 3: Unobservable system")
print("=" * 60)

A3 = np.array([[0, 1],
               [-4, -1]])

B3 = np.array([[0],
               [1]])

C3 = np.array([[0, 1]])  # we measure only velocity, not position

print("System 3:")
print(f"A = \n{A3}")
print(f"B = \n{B3}")
print(f"C = \n{C3}")
print()

controllable3 = check_controllability(A3, B3)
print()
observable3 = check_observability(A3, C3)

# Using control library helpers
print("\n" + "=" * 60)
print("USING THE CONTROL LIBRARY")
print("=" * 60)

sys1 = ctrl.ss(A1, B1, C1, 0)
sys2 = ctrl.ss(A2, B2, C2, 0)
sys3 = ctrl.ss(A3, B3, C3, 0)

print("System 1:")
print(f"Controllable: {ctrl.ctrb(A1, B1).shape[1] == matrix_rank(ctrl.ctrb(A1, B1)) == A1.shape[0]}")
print(f"Observable: {ctrl.obsv(A1, C1).shape[0] == matrix_rank(ctrl.obsv(A1, C1)) == A1.shape[0]}")

print("\nSystem 2:")
print(f"Controllable: {ctrl.ctrb(A2, B2).shape[1] == matrix_rank(ctrl.ctrb(A2, B2)) == A2.shape[0]}")
print(f"Observable: {ctrl.obsv(A2, C2).shape[0] == matrix_rank(ctrl.obsv(A2, C2)) == A2.shape[0]}")

print("\nSystem 3:")
print(f"Controllable: {ctrl.ctrb(A3, B3).shape[1] == matrix_rank(ctrl.ctrb(A3, B3)) == A3.shape[0]}")
print(f"Observable: {ctrl.obsv(A3, C3).shape[0] == matrix_rank(ctrl.obsv(A3, C3)) == A3.shape[0]}")

# Demonstrate the duality principle
print("\n" + "=" * 60)
print("DUALITY PRINCIPLE DEMONSTRATION")
print("=" * 60)

A_dual = A1.T
C_dual = B1.T

print("Original system (A1, B1):")
print(f"Controllable: {check_controllability(A1, B1)}")

print("\nDual system (A1^T, B1^T):")
print(f"Observable: {check_observability(A_dual, C_dual)}")
```

This example illustrates:

1. **The controllability test** via the controllability matrix rank
2. **The observability test** via the observability matrix rank
3. **Examples of different system types:** fully controllable/observable, uncontrollable, and unobservable
4. **Use of the control library** helper functions
5. **A demonstration of the duality principle** between controllability and observability


