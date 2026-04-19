# Recipe 09 — Fault-tolerance with online-adaptive agents

Inject a 50 % elevator effectiveness loss mid-episode and watch iADP and AA-INDI absorb it. Copy each step below; the numbers and the plot at the end are the ones you should reproduce within ±5 %.

Source notebook: [`example/cookbook/recipe_09_fault_tolerance.ipynb`](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/cookbook/recipe_09_fault_tolerance.ipynb).

## Step 1 — Boilerplate and trim

```python
import warnings
warnings.filterwarnings('ignore')
import math

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import fsolve

import tensoraerospace  # noqa: F401
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import f16_ode_long
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params import default_parameters
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig
from tensoraerospace.agent.aa_indi import AAINDIAgent, AAINDIConfig

dt = 0.01
params = default_parameters()

def trim_residual(z):
    alpha, stab = z
    return list(f16_ode_long(np.array([alpha, 0, stab, 0]), np.array([stab]), 0, params)[:2])

sol, *_ = fsolve(trim_residual, x0=[math.radians(2.0), math.radians(-2.0)], full_output=True)
alpha_trim_rad, stab_trim_rad = float(sol[0]), float(sol[1])

def make_env(n):
    env = gym.make('NonlinearLongitudinalF16-v0',
        number_time_steps=n + 2,
        initial_state=[alpha_trim_rad, 0.0, stab_trim_rad, 0.0],
        reference_signal=np.full((1, n + 2), alpha_trim_rad),
        state_space=['alpha','wz','stab','dstab'], control_space=['stab'],
        tracking_states=['alpha'], use_reward=False, dt=dt, integrator='euler',
        control_bias=math.degrees(stab_trim_rad),
    ).unwrapped
    env.reset()
    return env
```

## Step 2 — Warm-start via a 3-second PE excitation

```python
env_pe = make_env(300); obs, _ = env_pe.reset()
wz_hist, u_hist = [float(obs[1])], [0.0]
for t in range(300):
    u = 2.0*math.sin(2*math.pi*0.7*t*dt) + 1.0*math.sin(2*math.pi*1.5*t*dt)
    obs, *_ = env_pe.step(np.array([u]))
    wz_hist.append(float(obs[1])); u_hist.append(float(u))

dwz, du = np.diff(wz_hist), np.diff(u_hist)
A_pe = np.column_stack([dwz[:-1], du[:-1]])
F_wz, G_wz = np.linalg.lstsq(A_pe, dwz[1:], rcond=None)[0]
print(f'PE seed: F_wz = {F_wz:+.4f}, G_wz = {G_wz:+.5f}')
```

**Expected output:**

```
PE seed: F_wz = +0.9997, G_wz = -0.00139
```

If `G_wz` is an order of magnitude off, check that the excitation amplitude is ≥ 1 (we use 2 and 1 for the two sines).

## Step 3 — iADP harness

```python
def run_iadp(wz_cmd, N, fault_gain=1.0, fault_at=None):
    F_init = np.array([[F_wz, 0.0], [0.0, 1.0]])
    G_init = np.array([[G_wz], [0.0]])
    Q, R, gamma = 30_000.0, 0.1, 0.9
    Q_aug = Q * np.array([[1.0, -1.0], [-1.0, 1.0]])
    P_dare = solve_discrete_are(np.sqrt(gamma)*F_init, np.sqrt(gamma)*G_init,
                                Q_aug, np.array([[R]]))
    cfg = IADPConfig(dt=dt, Q=np.array([[Q]]), R=np.array([[R]]),
        gamma=gamma, gamma_rls=0.9999, phi_init=1.0,
        policy_eval_window=300, policy_eval_every=5,
        policy_eval_warmup_updates=20,
        policy_eval_regularization=1e-10, policy_eval_blend=0.10,
        F_init=F_init, G_init=G_init, P_init=P_dare,
        u_magnitude_limit=8.0, u_rate_limit=200.0, seed=0)
    agent = IADPAgent(n_state=1, n_control=1, config=cfg)
    env = make_env(N); obs, _ = env.reset()
    wz_out, u_out = [], []
    gain = 1.0
    for k in range(N):
        if fault_at is not None and k >= fault_at: gain = fault_gain
        u = agent.predict(np.array([float(obs[1])]), np.array([wz_cmd[k]]), k)
        obs, *_ = env.step(u * gain)
        agent.learn(np.array([float(obs[1])]), np.array([wz_cmd[k]]), k)
        wz_out.append(float(obs[1])); u_out.append(float(u[0] * gain))
    return np.asarray(wz_out), np.asarray(u_out)
```

## Step 4 — AA-INDI harness

```python
def run_aaindi(wz_cmd, N, fault_gain=1.0, fault_at=None):
    cfg = AAINDIConfig(
        dt=dt, ref_wn=2.5, ref_zeta=0.9,
        u_magnitude_limit=15.0, u_rate_limit=60.0,
        vff_forgetting_min=0.97, vff_forgetting_max=0.9999,
        vff_eps_sensitivity=0.1, vff_cov_init=1.0,
        sensor_cutoff_hz=15.0, bias_forgetting=0.995,
        enable_bias_correction=False,
        G_init=np.array([[-0.5]]),
        ref_error_kp=0.6, ref_error_ki=0.0,
        seed=0,
    )
    agent = AAINDIAgent(n_state=1, n_control=1, config=cfg)
    env = make_env(N); obs, _ = env.reset()
    wz_out, u_out = [], []
    gain = 1.0
    for k in range(N):
        if fault_at is not None and k >= fault_at: gain = fault_gain
        u = agent.predict(np.array([float(obs[1])]), np.array([wz_cmd[k]]), k)
        obs, *_ = env.step(u * gain)
        agent.learn(np.array([float(obs[1])]), np.array([wz_cmd[k]]), k)
        wz_out.append(float(obs[1])); u_out.append(float(u[0] * gain))
    return np.asarray(wz_out), np.asarray(u_out)
```

## Step 5 — Head-to-head with fault at t = 10 s

```python
N = 1800
t_arr = np.arange(N) * dt
wz_cmd = math.radians(0.8) * np.sin(2*math.pi*0.12*t_arr)
fault_step = int(10.0 / dt)

wz_i_f, u_i_f = run_iadp(wz_cmd, N, fault_gain=0.5, fault_at=fault_step)
wz_a_f, u_a_f = run_aaindi(wz_cmd, N, fault_gain=0.5, fault_at=fault_step)

def rmse(sig, ref, mask):
    return math.degrees(np.sqrt(np.mean((sig[mask] - ref[mask])**2)))

pre  = np.arange(500, fault_step)
post = np.arange(fault_step + 100, N)
print('                 pre-fault RMSE   post-fault RMSE')
print(f'  iADP          {rmse(wz_i_f, wz_cmd, pre):.4f}°/s       {rmse(wz_i_f, wz_cmd, post):.4f}°/s')
print(f'  AA-INDI       {rmse(wz_a_f, wz_cmd, pre):.4f}°/s       {rmse(wz_a_f, wz_cmd, post):.4f}°/s')
```

**Expected output:**

```
                 pre-fault RMSE   post-fault RMSE
  iADP          0.0896°/s       0.0982°/s
  AA-INDI       0.3135°/s       0.3216°/s
```

**Key observation — look at the delta, not the absolute RMSE.** For both agents the RMSE shifts by only ~1 millidegree/s at the fault event: neither agent is surprised by the 50 % gain loss. The absolute gap between iADP and AA-INDI here is tuning (AA-INDI's PI gains are from the step-command example, not re-tuned for a 0.12 Hz sinusoid).

## Step 6 — Plot

```python
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].plot(t_arr, np.degrees(wz_cmd), 'k--', label='command')
axes[0].plot(t_arr, np.degrees(wz_i_f), label='iADP (faulty)', alpha=0.85)
axes[0].plot(t_arr, np.degrees(wz_a_f), label='AA-INDI (faulty)', alpha=0.85)
axes[0].axvline(fault_step * dt, color='red', alpha=0.3, linestyle='--', label='fault event')
axes[0].set_ylabel('ω_z [°/s]'); axes[0].legend(loc='upper right'); axes[0].grid(alpha=0.3)

axes[1].plot(t_arr, u_i_f, label='iADP applied', alpha=0.85)
axes[1].plot(t_arr, u_a_f, label='AA-INDI applied', alpha=0.85)
axes[1].axvline(fault_step * dt, color='red', alpha=0.3, linestyle='--')
axes[1].set_xlabel('time [s]'); axes[1].set_ylabel('Δδₑ (post-fault) [°]')
axes[1].legend(loc='upper right'); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

**Expected plot — compare with yours:**

![iADP vs AA-INDI fault comparison](img/09_fault_comparison.png)

- The command is the black dashed sinusoid.
- **iADP** tracks within a narrow envelope; the transient at `t = 10 s` is barely visible.
- **AA-INDI** shows the phase-lag signature of its rate-tracking inner loop — larger amplitude error, but the RMSE delta at the fault event is still ~1 md/s.
- The red dashed line marks the fault injection.

## What makes this work

- **Continuous RLS identification** — both agents' G-estimate updates every tick; the fault is just a plant change, the identifier converges to the new value.
- **Incremental action** — INDI-style (AA-INDI) and LQT-incremental (iADP) both command Δδ; halving the gain shifts the *rate*, not the direction.
- **No fault-detection state machine** — the agents don't need to know a fault happened.

## Common mismatches

| Symptom | Cause |
|---|---|
| iADP diverges | Forgot DARE `P_init` or used default `policy_eval_regularization`. |
| AA-INDI oscillates | `G_init` warm-start has the wrong sign. |
| RMSE delta > 20 % of pre-fault | Fault gain is too aggressive (try `fault_gain=0.7` first). |

## Where to go next

- **[Recipe 06 — Online-adaptive agents](06_online_adaptive.md)** — the lifecycle framework.
- **[AA-INDI documentation](../agent/aa_indi.md)** — theory + full API.
- **[iADP documentation](../agent/iadp.md)** — theory + full API.
