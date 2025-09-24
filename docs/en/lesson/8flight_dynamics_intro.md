# Introduction to Flight Dynamics

## On This Page

- [Aircraft Coordinate Systems and States](#aircraft-coordinate-systems-and-states)
- [Linearization and State Matrices](#linearization-and-state-matrices)
- [Modes of Motion (Longitudinal)](#modes-of-motion-longitudinal)
- [Stability Criteria](#stability-criteria)
- [Connection with Simulink/XFLR5](#connection-with-simulinkxflr5)
- [Glossary (Brief)](#glossary-brief)

## Aircraft Coordinate Systems and States {#aircraft-coordinate-systems-and-states}

To describe aircraft motion we use inertial and body-fixed coordinate systems. A typical state vector for a linear longitudinal model is $x = [u\ w\ q\ \theta]^T$, where $u, w$ are longitudinal and vertical velocities, $q$ is the pitch rate, and $\theta$ is the pitch angle.

## Linearization and State Matrices {#linearization-and-state-matrices}

Nonlinear equations of motion are linearized around operating points (trim conditions). The result is an LTI model:

$$\dot{x} = A x + B u, \quad y = C x + D u$$

The structure of $A$ reflects aerodynamic derivatives (e.g., the sensitivity of forces/moments to speeds/angles/pressure variations).

## Modes of Motion (Longitudinal) {#modes-of-motion-longitudinal}

- Short-period mode: fast oscillations of angle of attack and pitch, governed by the balance of lift and moments.
- Phugoid mode: slow exchange between potential and kinetic energy with weak damping.

## Stability Criteria {#stability-criteria}

Analyzing the eigenvalues of $A$ or root loci reveals the stability and dynamic properties of the modes. Controller tuning (PID/MPC) shifts the spectrum of $A$ and changes damping and frequency.

## Connection with Simulink/XFLR5 {#connection-with-simulinkxflr5}

The matrices $A, B, C, D$ obtained from XFLR5 or aerodynamic analysis can be used directly in the `State-Space` block in Simulink for modeling and controller synthesis.

## Glossary (Brief) {#glossary-brief}

- Trim: steady flight condition under fixed control inputs.
- Aerodynamic derivatives: partial derivatives of forces/moments with respect to state and control variables.
- Stability margin: metrics that characterize the distance to instability (frequency/time domain).
