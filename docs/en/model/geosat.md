# Geostationary satellite (GeoSat)

GeoSat implements a three-state **linear model of normalized orbital deviations**. States, input and time are dimensionless. Use it for small perturbations around the operating orbit.

## Mathematical model {#mathematical-model}

The model uses the reduced equations (61), (66) and (70) in [Hla et al. (2012), *Implementation of a Communication Satellite Orbit Controller Design Using State Space Techniques*](https://ajstd.ubd.edu.bn/journal/vol29/iss1/2/).

\[
\tau=t/\sqrt{R/g},\quad \rho=r/R,\quad u_2=F_2/(M g),\qquad
x=[\delta\rho,\delta\rho',\delta\theta']^{\mathsf T}.
\]

A prime denotes differentiation with respect to normalized time \(\tau\). The reference values \(R\), \(g\), and \(M\) must be specified separately before converting to SI units.

\[
x'=Ax+Bu_2,\qquad
A=\begin{bmatrix}0&1&0\\0.01036&0&0.7753\\0&-0.01774&0\end{bmatrix},\quad
B=\begin{bmatrix}0\\0\\0.1512\end{bmatrix},\quad C=I,\quad D=0.
\]

The old Python/MATLAB coefficient `-0.1774` did not match the reduced equations. The corrected `-0.01774` also agrees approximately with the orbital Jacobian entry `-2*omega0/rho0`. The publication contains inconsistent numerical entries in its earlier four-state presentation; the reduced three-state equations are used here.

### State names retained for compatibility

| API name | Physical meaning in this model |
|---|---|
| `rho` | Normalized radial displacement from the operating orbit |
| `theta` | **Radial-velocity deviation**, derivative of `rho` with respect to `tau` |
| `omega` | Angular-rate deviation with respect to `tau` |

`theta` is a legacy name, **not angular position**. The model does not include the angular-position state. Likewise, the legacy control key `ele` denotes normalized tangential thrust, not an elevator angle. Do not use the historical degree-conversion helpers as conversions to physical thrust or SI velocity.

The matrices are rounded approximations of the local orbital Jacobian. At zero input the linear invariant `omega + 0.01774*rho` is conserved. Large deviations require a nonlinear orbital model.

## Inputs and integration

`run_step()` uses exact zero-order-hold discretization of the linear system. `dt` is a normalized-time increment. Legacy numerical limits remain `abs(u2) <= pi*25/180` and `abs(delta_u2) <= (pi*60/180)*dt`. Their origin in degree conversions does not make them angular actuator limits; they are uncalibrated simulation settings.

Slew limiting applies from the first step, relative to `initial_control=0` by default. Invalid nonfinite inputs and commands with the wrong size are rejected before state advancement.

## Gymnasium environment and reward

- `GeoSatEnv` accepts one continuous thrust command within the model's magnitude limits. Its `dt` argument defaults to `0.01`.
- `output_space` selects observation components and order. If omitted, it follows `state_space`. The observation Box matches the selected components.
- Tracking uses the full model state, so tracked components may be omitted from the observation.
- A single reference channel tracks the first entry of `tracking_states`, preserving the legacy objective. Multiple reference channels must match `tracking_states`; the default reward is the negative mean absolute tracking error.
- `reference_signal` may be an array `(channels, T)` or a callable sampled at `i*dt`. Short nonempty array references hold their last value.
- Reaching the episode time limit sets `truncated=True`, `terminated=False`. This preserves the future-value term when training an agent.

## Quick start {#quick-start}

```python
import numpy as np
from tensoraerospace.envs.geosat import GeoSatEnv

env = GeoSatEnv(
    initial_state=[0.01, 0.0, 0.001],
    reference_signal=np.zeros((1, 101)),
    number_time_steps=101,
    dt=0.1,
    tracking_states=["omega"],
    output_space=["rho", "theta", "omega"],
)
observation, info = env.reset(seed=11)
observation, reward, terminated, truncated, info = env.step(np.zeros(1))
```

## Python API

::: tensoraerospace.aerospacemodel.geosat.GeoSat

::: tensoraerospace.envs.geosat.GeoSatEnv
