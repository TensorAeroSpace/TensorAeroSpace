# Communication satellite (ComSat)

ComSat is a **linear model of normalized deviations from a circular orbit**. It is suitable for local control experiments. Its state values, simulation time and input are dimensionless; they are not kilometres, metres per second, seconds or newtons.

## Mathematical model {#mathematical-model}

The model follows the reduced linearization in [Choudhary (2015), *Design and Analysis of an Optimal Orbit Control for a Communication Satellite*](https://www.naun.org/main/NAUN/communications/2015/a102006-085.pdf), equations (14)–(19):

\[
\tau = t/\sqrt{R/g},\qquad \rho = r/R,\qquad u_2 = F_2/(M g).
\]

Here \(R\) is the reference radius, \(g\) the reference gravitational acceleration and \(M\) the satellite mass. A prime denotes differentiation with respect to \(\tau\). The published operating point is approximately \(\rho_0=6.6108\), \(\theta'_0=0.0587\).

\[
x=\begin{bmatrix}\delta\rho\\\delta\rho'\\\delta\theta'\end{bmatrix},\qquad
x'=Ax+Bu_2,
\]

\[
A=\begin{bmatrix}
0&1&0\\
0.01036&0&0.7757\\
0&-0.01775&0
\end{bmatrix},\qquad
B=\begin{bmatrix}0\\0\\0.1513\end{bmatrix},\qquad C=I,\quad D=0.
\]

These coefficients are rounded. They approximate the Jacobian of the normalized orbital equations, rather than defining an exact nonlinear orbit propagator. With zero input the linear invariant \(\delta\theta'+0.01775\delta\rho\) is conserved.

| API state | Meaning |
|---|---|
| `rho` | Normalized radial displacement from the operating orbit |
| `rho_dot` | Derivative of that displacement with respect to normalized time |
| `theta_dot` | Angular-rate deviation with respect to normalized time |

Use small deviations when drawing physical conclusions. The API does not convert SI states or forces into these coordinates. Such conversion requires explicit choices of \(R\), \(g\), \(M\) and the operating orbit.

## Input limits and integration

`ComSat.run_step()` holds the applied input constant during each `dt` interval and advances the linear system using zero-order-hold discretization. `dt` is an increment of \(\tau\).

The legacy simulation limits are \(|u_2|\le25\) and \(|\Delta u_2|\le60\,dt\). They are **numerical settings, not calibrated thruster specifications**; using their full range can leave the region where the linearization represents orbital physics. Slew limiting starts from `initial_control` (zero by default), including the first step. NaN, infinity and commands with more than one element are rejected.

## Environment coordinates and rewards

- `ComSatEnv` uses the three deviations directly and accepts normalized thrust in `[-25, 25]`. Its legacy reward penalizes absolute tracking error. A time limit sets `truncated=True`.
- `ImprovedComSatEnv` exposes `[nominal_rho + delta_rho, delta_rho_prime, delta_theta_prime]` through `state`. It subtracts the offset before passing the state to the linear model. Set `nominal_rho=0` to work directly with deviations. The legacy default `6371.0` is only an external coordinate offset; it does not specify an Earth radius in kilometres.
- Improved observations contain scaled angular-rate tracking error, radial displacement, radial velocity and the previous applied action. Actions in `[-1, 1]` request thrust in `[-25, 25]`. Rewards combine quadratic tracking/state penalties, applied-input and smoothness penalties, and a survival bonus. The input history and reward use the actual slew-limited input.
- `use_initial_action_on_first_step=True` applies `initial_thrust` on the first step. Set it to `False` when the controller must choose the first command.

Observation scales and termination thresholds are numerical configuration choices. They do not certify that a trajectory stays within the physical validity range of the linear model.

## Quick start {#quick-start}

```python
import numpy as np
from tensoraerospace.aerospacemodel.comsat import ComSat
from tensoraerospace.envs.comsat import ImprovedComSatEnv

# Small normalized perturbation; dt is normalized time.
x0 = np.array([0.01, 0.0, 0.001])
model = ComSat(x0, number_time_steps=100, dt=0.1)
trajectory = [model.run_step([0.0]).reshape(-1) for _ in range(100)]

env = ImprovedComSatEnv(
    initial_state=x0,
    reference_signal=np.zeros((1, 101)),
    number_time_steps=101,
    dt=0.1,
    nominal_rho=0.0,
    use_initial_action_on_first_step=False,
)
observation, info = env.reset(seed=11)
observation, reward, terminated, truncated, info = env.step(np.zeros(1))
```

## Python API

::: tensoraerospace.aerospacemodel.comsat.ComSat

::: tensoraerospace.envs.comsat.ComSatEnv

::: tensoraerospace.envs.comsat.ImprovedComSatEnv
