# UFTC Phase 3 — L4 Distributional SAC + CVaR risk-gate + trim-free reference

**Date:** 2026-05-08
**Status:** Draft, pending implementation plan
**Master spec:** `2026-05-08-uftc-cascade-extension-design.md`
**Predecessors:** Phase 1 MVP, Phase 2 (L1+GLR)

## 1. Scope

Outer-loop, risk-aware reference planner for the UFTC cascade. Adds:

1. **QR distributional critic** — N-quantile representation of return distribution `Z_ψ(s, a)`.
2. **Squashed-Gaussian actor** with **CVaR_α actor objective** — pessimistic update against the lower α-tail of `Z`.
3. **Risk gate β_t** — scalar in `[0, 1]` driving L3 trust-region/lookahead modulation, computed from `var(Z)`, `FDDOutput.severity`, and monitor alarm.
4. **Trim-free longitudinal wrapper** — replaces predefined static trim with adaptive reference computed by the actor when degraded plants are detected.

Gated behind `UFTCConfig.enable_l4_outer`. Default `False` keeps Phase 1+2 behaviour bit-identical.

## 2. Package layout

```
tensoraerospace/agent/uftc/l4/
├── __init__.py
├── dsac.py             # DSACOuter — BaseRLModel
├── critic.py           # QRDistCritic + target-net + QR-Huber loss
├── actor.py            # GaussianActor — squashed-Gaussian with reparam
├── cvar.py             # cvar_alpha_fn + risk_gate
├── replay.py           # PrioritizedReplay with FDD/monitor metadata in transitions
├── trim_free.py        # LongitudinalTrimFreeWrapper
├── train.py            # offline training loop (curriculum over damage presets)
└── README.md
```

## 3. Distributional critic

```python
@dataclass
class CriticConfig:
    n_state: int
    n_action: int
    n_quantiles: int = 32
    hidden_sizes: tuple[int, ...] = (256, 256)
    huber_kappa: float = 1.0

class QRDistCritic(nn.Module):
    def __init__(self, cfg: CriticConfig): ...
    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """returns Z(s,a) of shape (B, N) — N quantile values, ascending in i."""
```

**QR-Huber loss** (Dabney 2018, eq. 10):

```python
def qr_huber_loss(z_pred, z_target, kappa):
    # z_pred, z_target: (B, N)
    tau = (torch.arange(N, device=z_pred.device) + 0.5) / N      # (N,)
    delta = z_target.detach().unsqueeze(1) - z_pred.unsqueeze(2)  # (B, N_pred, N_tgt)
    huber = torch.where(delta.abs() <= kappa,
                        0.5 * delta**2,
                        kappa * (delta.abs() - 0.5 * kappa))
    rho = (tau.view(1, N, 1) - (delta < 0).float()).abs() * huber / kappa
    return rho.mean(dim=2).sum(dim=1).mean()
```

**Target distributional Bellman update**:

```
y = r + γ · Z̄(s', π(s'))         # Z̄ is target-network critic, sampled at action from current actor
loss = qr_huber_loss(Z(s,a), y, kappa)
```

Twin critics (Q1, Q2) — final target uses element-wise minimum of the two quantile vectors per quantile index.

## 4. Actor

```python
@dataclass
class ActorConfig:
    n_state: int
    n_action: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    log_std_min: float = -5.0
    log_std_max: float = 2.0

class GaussianActor(nn.Module):
    def forward(self, s) -> tuple[torch.Tensor, torch.Tensor]:
        # returns (mean, log_std)
    def rsample(self, s) -> tuple[torch.Tensor, torch.Tensor]:
        # returns (a_squashed_in_[-1,1], log_prob_with_tanh_jacobian)
```

`rsample` uses the standard squashed-normal trick (Haarnoja 2018). Action range `[-1, 1]` is rescaled to plant-specific `r̃` range outside the actor.

## 5. CVaR objective

```python
def cvar_alpha_fn(z: torch.Tensor, alpha: float) -> torch.Tensor:
    """z: (B, N). Returns mean of the lowest α-quantile fraction."""
    k = max(1, int(math.floor(alpha * z.size(-1))))
    z_sorted, _ = torch.sort(z, dim=-1)
    return z_sorted[:, :k].mean(dim=-1)

def actor_loss(s_batch, actor, critic, log_alpha, cvar_alpha):
    a, log_prob = actor.rsample(s_batch)
    z = critic(s_batch, a)
    cvar = cvar_alpha_fn(z, cvar_alpha)
    return -(cvar - log_alpha.exp() * log_prob).mean()
```

Gradient flows through `cvar_alpha_fn` because `torch.sort` is differentiable on the sorted values (gradient passes through to the corresponding original positions). Tested explicitly in `test_cvar.py`.

`log_alpha` (entropy temperature) trained as in vanilla SAC:

```python
alpha_loss = (-log_alpha * (log_prob.detach() + target_entropy)).mean()
```

with `target_entropy = -n_action`.

## 6. Risk gate

```python
def risk_gate(z_quantiles: torch.Tensor,
              fdd_severity: float,
              monitor_alarm: str,
              var_target: float = 0.5,
              k_fdd: float = 0.4) -> float:
    var_z = float(z_quantiles.var(dim=-1).mean())
    g_var = float(torch.sigmoid(torch.tensor((var_z - var_target) * 5.0)))
    g_fdd = float(np.clip(k_fdd * fdd_severity, 0.0, 1.0))
    g_alarm = {"OK": 0.0, "WARN": 0.5, "CRITICAL": 1.0}[monitor_alarm]
    return float(min(1.0, max(g_var, g_fdd, g_alarm)))
```

`β_t` is consumed by L3:

```python
# IADPMiddle.predict — Phase 1 fields, extended in Phase 3
effective_lookahead = self.lookahead_dt * (1.0 + 2.0 * beta_t)
effective_trust = trust_radius_nominal + beta_t * (trust_radius_fault - trust_radius_nominal)
```

Higher `β_t` → longer lookahead (smoother `ω_ref`) and wider trust-region (more freedom for L3 to re-fit `F̃,G̃`).

## 7. Replay buffer

```python
@dataclass
class Transition:
    s: np.ndarray
    a_actual: np.ndarray            # u_safe (post-L1)
    r_used: np.ndarray              # r̃_t actually executed
    reward: float
    s_next: np.ndarray
    done: bool
    fdd: FDDOutput
    alarm: str

class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_init: float = 0.4): ...
    def push(self, transition: Transition, priority: float | None = None): ...
    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        # returns (transitions, indices, importance_weights)
    def update_priorities(self, indices, td_errors): ...
```

Storing `a_actual = u_safe` (post-L1) is the **off-policy correction** for Phase 2: actor learns from what the env actually saw, not from `u_indi` that L1 may have clipped.

## 8. DSACOuter (top-level wrapper)

```python
@dataclass
class DSACConfig:
    n_state: int
    n_ref_dim: int
    n_action: int           # for L4 this equals n_ref_dim — actor outputs r̃_t
    cvar_alpha: float = 0.2
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 200_000
    learn_every: int = 1
    update_to_data_ratio: int = 1
    target_entropy: float | None = None       # default = -n_action
    glr_reset_threshold: float = 0.10         # ‖drift_estimate‖ above which we set reset_hint
    eval_mode: bool = True                    # Phase 3 default — see §12 (Operational mode)

class DSACOuter(BaseRLModel):
    def __init__(self, cfg: DSACConfig, *,
                 actor: GaussianActor | None = None,
                 critic: QRDistCritic | None = None,
                 trim_free: LongitudinalTrimFreeWrapper | None = None): ...

    def predict(self, x_obs, base_reference, fdd: FDDOutput,
                monitor_alarm: str = "OK") -> tuple[np.ndarray, float, bool]:
        with torch.no_grad():
            mean, log_std = self.actor(x_obs)
            a = torch.tanh(mean) if self.eval_mode else self.actor.rsample(x_obs)[0]
            r_tilde = self._scale_action_to_reference(a, base_reference)
            z = self.critic(x_obs, a)
            beta_t = risk_gate(z, fdd.severity, monitor_alarm)
            reset_hint = (fdd.fault_kind == "gradual" and
                          fdd.glr_drift_estimate is not None and
                          np.linalg.norm(fdd.glr_drift_estimate) > self.cfg.glr_reset_threshold)
            if self.trim_free:
                r_tilde = self.trim_free.apply(r_tilde, x_obs, base_reference)
            return r_tilde, beta_t, reset_hint

    def learn(self, transition: Transition) -> dict:
        if self._frozen_until and self._step < self._frozen_until:
            return {"frozen": True}
        self.replay.push(transition)
        if len(self.replay) < self.cfg.batch_size:
            return {}
        if self._step % self.cfg.learn_every == 0:
            return self._sgd_step()
        return {}

    def freeze_learning(self, until_step: int) -> None:
        self._frozen_until = until_step

    def degrade_reference_to_hold(self) -> None:
        self._hold_mode = True   # next predict returns base_reference unchanged

    def reset(self): ...
    def save(self, path) -> None: ...
    @classmethod
    def from_pretrained(cls, repo_or_path, ...) -> "DSACOuter": ...
```

`eval_mode` flag controls whether `predict()` uses deterministic mean (for production fly) or stochastic sample (for training). Phase 3 ships `eval_mode=True` as default to disable online learning while flying — see Section 12.

## 9. Trim-free longitudinal wrapper

```python
@dataclass
class LongitudinalTrimFreeConfig:
    V_idx: int
    gamma_idx: int
    alpha_idx: int
    q_idx: int
    enabled: bool = False

class LongitudinalTrimFreeWrapper:
    def __init__(self, cfg: LongitudinalTrimFreeConfig): ...
    def apply(self, r_tilde_actor: np.ndarray, x_obs: np.ndarray,
              base_reference: np.ndarray) -> np.ndarray:
        """Replace alpha_trim/q_trim entries in base_reference with actor output.

        base_reference is expected to specify V_target and gamma_target;
        alpha_target and q_target are *not* trusted under degraded plants
        and come from the actor instead.
        """
        if not self.cfg.enabled:
            return base_reference
        out = base_reference.copy()
        out[self.cfg.alpha_idx] = r_tilde_actor[0]
        out[self.cfg.q_idx]     = r_tilde_actor[1]
        return out
```

When the configured indices match the F-16 longitudinal model, this means: pilot specifies `(V_target, γ_target)` only; L4 fills in `(α_target, q_target)` using its learned policy. For non-aero plants without the four required indices, `enabled=False` and the wrapper passes through.

## 10. Pre-training pipeline

`example/reinforcement_learning/uftc/train_dsac_offline.py`:

```bash
python train_dsac_offline.py \
    --plant f16-nonlinear-angular \
    --presets none:0.3,wing_strike_left_tip:0.1,elevator_jam_neutral:0.15,\
              elevator_jam_pitch_up:0.15,rudder_lost:0.1,engine_flameout:0.1,\
              birdstrike_compound:0.1 \
    --steps 200000 \
    --enable-l1 --enable-glr \
    --out artifacts/dsac/v1/
```

Curriculum: pre-mixed across nominal and 7 damage presets according to weights. First 50K steps — random policy seeding the replay; next 150K steps — on-policy SGD updates.

Reward shaping:

```
r_t = -‖x_t − r̃_t‖²_Q  −  k_u · ‖u_safe − u_indi‖²  −  k_alarm · 1{alarm == CRITICAL}
```

`k_u = 0.05` penalises L1 interventions — the actor learns to stay inside the safe set on its own. `k_alarm = 1.0` aligns with monitor (Phase 4) — actor avoids alarm states even before training the monitor itself.

## 11. UFTCController integration

In `predict()`:

```python
if self.cfg.enable_l4_outer:
    r_eff, beta_t, reset_hint = self.l4.predict(
        x_obs, reference, self._last_fdd, self._monitor_alarm)
else:
    r_eff, beta_t, reset_hint = reference, 0.0, False
self._last_beta = beta_t
self._last_reset_hint = reset_hint
u_iadp, omega_ref = self.middle.predict(x_obs, r_eff, time_step, beta=beta_t)
```

In `learn()` (after `fdd.step` and `middle.learn`):

```python
if self.cfg.enable_l4_outer:
    self.l4.learn(Transition(
        s=x_obs, a_actual=self._last_u_safe, r_used=self._last_r_eff,
        reward=self._last_reward, s_next=next_x_obs, done=done,
        fdd=fdd_out, alarm=self._monitor_alarm))
```

`save()`/`from_pretrained` add `l4/` subdirectory with `actor.pt`, `critic.pt`, `critic_target.pt`, `log_alpha.pt`, optionally `replay.npz`, and `dsac_config.json`.

## 12. Operational mode

Phase 3 ships **eval-only** L4 by default: `DSACConfig.eval_mode = True`. Online updates inside `UFTCController.learn()` are disabled — `learn()` only pushes transitions to replay if `replay_persist=True` is set. Production fly uses pre-trained weights.

Online learning while flying is deferred to Phase 3.1 (separate spec) — it adds nontrivial stability concerns (learning rate vs control bandwidth, replay-priority drift under fault).

## 13. Tests

| File | Coverage |
|---|---|
| `tests/agents/uftc/l4/test_qr_critic.py` | Single-step QR-Huber loss matches reference; loss decreases on stationary dataset; target-net soft-update τ. |
| `tests/agents/uftc/l4/test_qr_critic_monotonicity.py` | After training, predicted quantiles non-decreasing across N — verifies critic learned proper distributional structure. |
| `tests/agents/uftc/l4/test_actor.py` | Squashed-Gaussian log-prob matches torch reference (regression on `torch.distributions.Normal` + tanh Jacobian). |
| `tests/agents/uftc/l4/test_cvar.py` | `cvar_alpha_fn` matches numpy reference; gradient-check via `torch.autograd.gradcheck`. |
| `tests/agents/uftc/l4/test_risk_gate.py` | `β_t` increases monotonically with `var(Z)`, `fdd.severity`, and alarm level. |
| `tests/agents/uftc/l4/test_trim_free.py` | F-16 longitudinal nominal: actor-trim converges to within 5 % of analytical trim after 50K steps; disabled wrapper passes through unchanged. |
| `tests/agents/uftc/l4/test_replay.py` | priority sampling weights correct; transitions store `u_safe` not `u_indi`. |
| `tests/agents/uftc/test_uftc_l4_replay_off_policy.py` | With `enable_l1_shield=True`, replay's `a_actual` matches L1 output (not L2 output) on a 5K-step rollout. |
| `tests/agents/uftc/test_uftc_l4_engine_flameout.py` | F-16 + ENGINE_FLAMEOUT: with pre-trained weights, tracking-RMS not worse than Phase 1 (regression guard); after 50K offline-training episodes, RMS improves ≥ 15 %. |
| `tests/agents/uftc/test_uftc_l4_phase12_invariance.py` | `enable_l4_outer=False` → predict()/learn() bit-identical to Phase 1+2 (Phase 1 invariance compounded). |
| `tests/agents/uftc/test_uftc_l4_freeze_macro.py` | `l4.freeze_learning(until=N)` blocks `learn()` updates; transitions still push to replay; after step N updates resume. |

Coverage target: ≥ 80 % line coverage on `tensoraerospace/agent/uftc/l4/`.

## 14. Known risks

- **Distributional Bellman with squashed actor.** The Bellman target uses `actor.rsample(s')`, then critic at the sampled action. Variance can be high; mitigated by twin-critic min and large `replay_capacity`.
- **CVaR with N=32 quantiles and α=0.2.** k=6 quantiles for the tail mean. Smaller α (e.g. 0.05) requires more quantiles (N=64+) — Phase 3 sticks with α=0.2 unless empirical results force otherwise.
- **Reward shaping coupling.** `k_u` couples reward to L1 — if L1 is buggy, actor sees biased reward. Mitigated by Phase 1+2 bit-invariance regression.

## 15. Out of scope for Phase 3

- Online D-SAC learning while flying (deferred to Phase 3.1).
- IQN / risk-distortion alternative critic (deferred — QR is in the master spec).
- Multi-task curriculum across plants (only F-16 is in scope; spec is plant-agnostic but tests cover F-16).

## 16. References

- Dabney, W. et al. (2018) "Distributional Reinforcement Learning with Quantile Regression," AAAI.
- Haarnoja, T. et al. (2018) "Soft Actor-Critic Algorithms and Applications," ICML.
- Choi, J. et al. (2021) "Risk-Constrained Reinforcement Learning with Percentile Risk Criteria," JMLR — CVaR-RL baseline.
- Schulman, J. et al. (2015) "Trust Region Policy Optimization," ICML — trust-region motivation.
- Master spec: `2026-05-08-uftc-cascade-extension-design.md`.
- Phase 2 sub-spec: `2026-05-08-uftc-l1-hjshield-and-glr-design.md`.
