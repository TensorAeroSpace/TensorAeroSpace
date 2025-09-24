# Control Theory Basics {#control-theory-basics}

## On This Page {#on-this-page}

- [Control Theory Basics](#control-theory-basics)
- [On This Page](#on-this-page)
- [What Is a Dynamical System](#what-is-a-dynamical-system)
- [Linear Systems and the LTI Model](#linear-systems-and-the-lti-model)
- [Transient and Steady-State Response](#transient-and-steady-state-response)
- [Stability (Lyapunov, Briefly)](#stability-lyapunov-briefly)
- [Controllability and Observability](#controllability-and-observability)
- [Typical Elements and Frequency View (Very Briefly)](#typical-elements-and-frequency-view-very-briefly)
- [What to Read Next](#what-to-read-next)

## What Is a Dynamical System {#what-is-a-dynamical-system}

A dynamical system is an object whose state changes over time under the influence of inputs and internal evolution laws. In engineering, we often use the state-space form:

$$\dot{x}(t) = f(x(t), u(t), t), \quad y(t) = g(x(t), u(t), t)$$

where $x$ is the state vector, $u$ is the input (control), and $y$ is the output (measurable quantity).

## Linear Systems and the LTI Model {#linear-systems-and-the-lti-model}

A particularly useful case is the linear time-invariant (LTI) system:

$$\dot{x}(t) = A x(t) + B u(t), \quad y(t) = C x(t) + D u(t)$$

The matrices $A, B, C, D$ define the dynamics and measurements. This form is convenient for analyzing stability, observability, and controllability.

## Transient and Steady-State Response {#transient-and-steady-state-response}

- The transient response is the portion of the output from the moment the input changes until the system reaches steady state.
- The steady-state response is when the output no longer changes significantly (for a stationary input).

Key metrics of the transient:
- rise time, overshoot, settling time, steady-state error.

## Stability (Lyapunov, Briefly) {#stability-lyapunov-briefly}

A system is stable if small perturbations of the initial conditions lead to small deviations in the trajectory. For LTI systems the criterion is simple: all eigenvalues of matrix $A$ must lie in the left half-plane (non-positive real parts; strictly negative for asymptotic stability).

## Controllability and Observability {#controllability-and-observability}

- Controllability: whether the system can be driven from any initial state to any final state in finite time with admissible control inputs.
- Observability: whether the system state can be reconstructed from measurable outputs over a finite time interval.

For an LTI pair $(A, B)$, controllability is checked via the Kalman matrix $\mathcal{C} = [B\ AB\ A^2B\ \dots\ A^{n-1}B]$. Observability is analogous using $\mathcal{O} = [C^T\ (CA)^T\ \dots\ (CA^{n-1})^T]^T$.

## Typical Elements and Frequency View (Very Briefly) {#typical-elements-and-frequency-view-very-briefly}

In practice we use approximations with standard blocks (first-order lag, integrator, oscillatory). Frequency responses (Bode, Nyquist) help tune stability and reserve margins in PID/MPC controllers.

## What to Read Next {#what-to-read-next}

- Ogata, "Modern Control Engineering"
- Skogestad & Postlethwaite, "Multivariable Feedback Control"
