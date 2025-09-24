Control Theory Basics
======================

.. contents:: On This Page
   :local:
   :depth: 2

What Is a Dynamical System
--------------------------
A dynamical system is an object whose state changes over time under the influence of inputs and internal evolution laws. In engineering, we often describe the system using state-space equations:

.. math::
   \dot{x}(t) = f\big(x(t), u(t), t\big), \qquad y(t) = g\big(x(t), u(t), t\big)

where :math:`x` is the state vector, :math:`u` is the input (control), and :math:`y` is the output (measurable quantity).

Linear Systems and the LTI Model
--------------------------------
A particularly useful special case is the linear time-invariant (LTI) system:

.. math::
   \dot{x}(t) = A x(t) + B u(t), \qquad y(t) = C x(t) + D u(t)

The matrices :math:`A,B,C,D` define the dynamics and measurements. This form is convenient for analyzing stability, observability, and controllability.

Transient and Steady-State Response
-----------------------------------
- The transient response is the portion of the output from the moment the input changes until the system reaches steady state.
- The steady-state response is when the output no longer changes significantly (for a stationary input).

Key quality metrics of the transient response:
- rise time, overshoot, settling time, steady-state error.

Stability (Lyapunov, Briefly)
-----------------------------
A system is stable if small perturbations of the initial conditions lead to small deviations in the trajectory. For LTI systems the criterion is simple: all eigenvalues of matrix :math:`A` must lie in the left half-plane (non-positive real parts; strictly negative for asymptotic stability).

Controllability and Observability
---------------------------------
- Controllability: whether the system can be driven from any initial state to any final state in finite time using admissible control inputs.
- Observability: whether the system state can be reconstructed from measured outputs over a finite time interval.

For the LTI pair :math:`(A,B)`, controllability is checked via the Kalman matrix :math:`\mathcal{C} = [B\ AB\ A^2B\ \dots\ A^{n-1}B]`. Observability is analogous with :math:`\mathcal{O} = [C^T\ (CA)^T\ \dots\ (CA^{n-1})^T]^T`.

Typical Elements and Frequency View (Very Briefly)
--------------------------------------------------
In practice we use approximations with standard elements (first-order lag, integrator, oscillatory). Frequency characteristics (Bode, Nyquist) help tune stability and ensure sufficient margins for PID/MPC controllers.

What to Read Next
-----------------
- Ogata, “Modern Control Engineering”
- Skogestad & Postlethwaite, “Multivariable Feedback Control”

