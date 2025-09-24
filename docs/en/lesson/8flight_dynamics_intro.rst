Introduction to Flight Dynamics
===============================

.. contents:: On This Page
   :local:
   :depth: 2

Aircraft Coordinate Systems and States
--------------------------------------
To describe aircraft motion we use inertial and body-fixed coordinate systems. A typical state vector for a linear longitudinal model is :math:`x = [u\ w\ q\ \theta]^T`, where :math:`u,w` are longitudinal and vertical velocities, :math:`q` is the pitch rate, and :math:`\theta` is the pitch angle.

Linearization and State Matrices
--------------------------------
Nonlinear equations of motion are linearized around operating conditions (trims). The result is an LTI model:

.. math::
   \dot{x} = A x + B u, \qquad y = C x + D u

The structure of :math:`A` reflects aerodynamic derivatives (for example, the sensitivity of forces/moments to speeds/angles/pressure variations).

Modes of Motion (Longitudinal)
------------------------------
- Short-period mode: fast oscillations of angle of attack and pitch, governed by the balance of lift and moment.
- Phugoid mode: slow exchange between potential and kinetic energy with low damping.

Stability Criteria
------------------
Analyzing the eigenvalues of :math:`A` or the root locus reveals the stability and dynamic properties of the modes. Controller tuning (PID/MPC) shifts the spectrum of :math:`A` and changes damping and frequency.

Connection with Simulink/XFLR5
------------------------------
The matrices :math:`A,B,C,D`, obtained from XFLR5 or aerodynamic calculations, can be used directly in the ``State-Space`` block in Simulink for modeling and controller synthesis.

Glossary (Brief)
----------------
- Trim: steady-flight condition for fixed control inputs.
- Aerodynamic derivatives: partial derivatives of forces/moments with respect to state and control variables.
- Stability margin: metrics that characterize the distance to the instability boundary (frequency- or time-domain).

