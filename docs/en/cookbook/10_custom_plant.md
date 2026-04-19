# Recipe 10 — Adding your own plant model

**Goal.** Extend TensorAeroSpace with a new aircraft / rocket / UAV: write the dynamics, wrap it in a Gymnasium env that matches the library's conventions, register it, linearise it for warm-starting adaptive agents.

**Related.** [Recipe 02](02_env_anatomy.md) for the env contract your new model must satisfy.

## The short version

To add a new plant you write **two classes and one registration call**:

1. A **dynamics function** `f(x, u, t, params) -> xdot` — pure NumPy, no Gymnasium dependency.
2. A **Gymnasium env class** that wraps the dynamics, integrates it, and matches the library's `state_space`/`tracking_states`/`reference_signal` contract. In practice you'll subclass `tensoraerospace.envs.base.BaseEnv` (or copy one of the existing envs, e.g. `envs/b747/linear.py`).
3. A `gym.register('<NewPlant>-v0', entry_point='...')` call so `gym.make` can find it.

## 1. Dynamics function

Start with the continuous-time ODE in state-space form. Example — a toy longitudinal model:

```python
import numpy as np

def toy_plant_ode(x: np.ndarray, u: np.ndarray, t: float,
                  params: dict) -> np.ndarray:
    """xdot = f(x, u, t, params). Pure NumPy, no side effects.

    x = [alpha, q], u = [stab]
    alpha_dot = -Z_alpha * alpha + q
    q_dot = M_alpha * alpha + M_q * q + M_delta * stab
    """
    alpha, q = float(x[0]), float(x[1])
    stab = float(u[0])
    Z_alpha = params['Z_alpha']
    M_alpha = params['M_alpha']
    M_q = params['M_q']
    M_delta = params['M_delta']
    return np.array([
        -Z_alpha * alpha + q,
        M_alpha * alpha + M_q * q + M_delta * stab,
    ], dtype=np.float64)
```

Keep `params` as a plain dict — it serialises to JSON easily for reproducibility and checkpoints.

## 2. Gymnasium env

Pick one existing env as template. The linear F-16 and linear B747 envs are the simplest; the nonlinear F-16 is the right template if your dynamics are nonlinear.

The env class must expose at minimum:

```python
class ToyPlantEnv(gym.Env):
    def __init__(
        self,
        number_time_steps: int,
        dt: float = 0.01,
        initial_state: list[list[float]] = [[0.0], [0.0]],
        reference_signal: np.ndarray | None = None,
        state_space: list[str] = ['alpha', 'q'],
        tracking_states: list[str] = ['alpha'],
        output_space: list[str] | None = None,
        use_reward: bool = True,
        reward_func: Callable | None = None,
        integrator: str = 'euler',
        params: dict | None = None,
    ): ...

    def reset(self, *, seed=None, options=None):
        """Return (obs, info)."""

    def step(self, action):
        """Return (obs, reward, terminated, truncated, info)."""

    def get_state(self):
        """Optional helper used by examples — return the current internal state."""
```

Store the full internal state as `self._x` (shape `(n_full_state,)`) and slice it for the observation based on `state_space`. Store `self.ref_signal = reference_signal` so users and agents can read it.

### Integration step

Write a small `_rk4_step` or `_euler_step` method that advances `self._x` by one `dt` using your `toy_plant_ode`. This is the only place where the integrator matters:

```python
def _euler_step(self, u):
    xdot = self.ode(self._x, u, self._t, self.params)
    self._x = self._x + self.dt * xdot
    self._t += self.dt
```

## 3. Registration

At the bottom of your env file or in `tensoraerospace/envs/__init__.py`:

```python
from gymnasium.envs.registration import register

register(
    id='ToyPlant-v0',
    entry_point='my_package.envs:ToyPlantEnv',
    max_episode_steps=10_000,
)
```

After that:

```python
import gymnasium as gym
import tensoraerospace       # keeps your register() call inside the package reachable
env = gym.make('ToyPlant-v0', number_time_steps=2000, ...)
```

## 4. Warm-starting adaptive agents

Once the env works with PID, you can immediately bootstrap any adaptive agent (IHDP / iADP / AA-INDI) **provided you linearise around a trim point**.

### Find the trim

```python
from scipy.optimize import fsolve

def trim_residual(z):
    alpha, stab = z
    return toy_plant_ode(np.array([alpha, 0.0]), np.array([stab]), 0.0, params)

alpha_trim, stab_trim = fsolve(trim_residual, x0=[0.0, 0.0])
```

### Get `G_init` via numerical Jacobian or a short PE excitation

Numerical (one line):

```python
eps = 1e-4
x0 = np.array([alpha_trim, 0.0])
u0 = np.array([stab_trim])
G_cont = (toy_plant_ode(x0, u0 + eps, 0, params) - toy_plant_ode(x0, u0, 0, params)) / eps
G_discrete = G_cont * dt         # what iADP / AA-INDI want
```

Or run a 3-second multi-sine and fit G from `Δx / Δu` (the iADP F-16 notebook uses this).

### Plug into the agent

See [Recipe 06](06_online_adaptive.md) for the common warm-start pattern.

## 5. Testing

Every new env should have at least:

- A **round-trip test** — `env.reset()` → random `env.step()` loop → trajectory is finite, inside `observation_space` bounds.
- A **trim test** — initialised at `(alpha_trim, 0)` with action `stab_trim`, the state stays put (`|x[t+1] - x[t]| < 1e-6`).
- A **reference-signal shape test** — passing a reference of wrong shape raises a clear error.

Copy the patterns from `tests/envs/` for the existing envs.

## 6. Publishing

If your env lives in a separate pip package, make sure the `register()` call runs on `import`. The cleanest idiom is:

```python
# my_package/__init__.py
from . import envs  # triggers registration
```

Users then `import my_package; gym.make('MyPlant-v0', ...)` with no extra ceremony.

If the env is a contribution to TensorAeroSpace proper, file a PR with:

- the dynamics function under `tensoraerospace/aerospacemodel/<your_model>/`,
- the env class under `tensoraerospace/envs/<your_model>/`,
- a registration entry in `tensoraerospace/envs/__init__.py`,
- one or two example notebooks under `example/`,
- a `docs/{en,ru}/model/<your_model>.md` page linked from `mkdocs.yml`.

## Pitfalls

- **Forgetting `reshape(1, -1)` on the reference.** Even for a single-channel tracked state, the env expects a 2-D array. If you get `IndexError: too many indices for array`, this is why.
- **Units mismatch between model and agent.** If your dynamics take radians but your reference signal is in degrees (`output_rad=False`), tracking error explodes. Be explicit in docstrings.
- **Integrator step too large.** Euler with `dt = 0.01` is fine for bandwidth-limited aerospace plants; go to `rk4` for aggressive actuator dynamics (short time constants) or raise `dt` to 0.005.
- **Non-deterministic reset.** If you randomise `initial_state` in `reset()`, accept and use a `seed` argument so experiments are reproducible.

## Where to go next

- **[Recipe 02 — Anatomy of an env](02_env_anatomy.md)** — the contract your new env must satisfy.
- **[Recipe 05 — Deep-RL training end-to-end](05_deep_rl_training.md)** — train an RL agent on your new plant.
- **[Recipe 06 — Online-adaptive agents](06_online_adaptive.md)** — deploy IHDP / iADP / AA-INDI on your new plant.
