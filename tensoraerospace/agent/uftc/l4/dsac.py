"""DSACOuter — distributional SAC outer-loop reference planner.

Composes QRDistCritic (twin), GaussianActor, PrioritizedReplay, and a
CVaRₐ actor objective. Phase 3 default is ``eval_mode=True``: predict()
uses the deterministic actor mean and learn() pushes transitions but
does no SGD step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput

from .actor import ActorConfig, GaussianActor
from .critic import CriticConfig, QRDistCritic, qr_huber_loss, soft_update
from .cvar import cvar_alpha_fn, risk_gate
from .replay import PrioritizedReplay, Transition


@dataclass
class DSACConfig:
    n_state: int
    n_ref_dim: int
    n_action: int
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
    target_entropy: float | None = None
    glr_reset_threshold: float = 0.10
    eval_mode: bool = True
    n_quantiles: int = 32
    huber_kappa: float = 1.0
    actor_hidden: tuple[int, ...] = (256, 256)
    critic_hidden: tuple[int, ...] = (256, 256)
    seed: int = 0
    action_scale: float = 1.0


class DSACOuter:
    """Outer-loop risk-aware reference planner (Phase 3)."""

    def __init__(self, cfg: DSACConfig, device: str = "cpu") -> None:
        torch.manual_seed(int(cfg.seed))
        self.cfg = cfg
        self.device = torch.device(device)
        actor_cfg = ActorConfig(n_state=cfg.n_state, n_action=cfg.n_action,
                                hidden_sizes=cfg.actor_hidden)
        critic_cfg = CriticConfig(n_state=cfg.n_state, n_action=cfg.n_action,
                                  n_quantiles=cfg.n_quantiles,
                                  hidden_sizes=cfg.critic_hidden,
                                  huber_kappa=cfg.huber_kappa)
        self.actor = GaussianActor(actor_cfg).to(self.device)
        self.critic1 = QRDistCritic(critic_cfg).to(self.device)
        self.critic2 = QRDistCritic(critic_cfg).to(self.device)
        self.target1 = QRDistCritic(critic_cfg).to(self.device)
        self.target2 = QRDistCritic(critic_cfg).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_c1 = torch.optim.Adam(self.critic1.parameters(), lr=cfg.lr_critic)
        self.opt_c2 = torch.optim.Adam(self.critic2.parameters(), lr=cfg.lr_critic)

        self.target_entropy = (
            float(cfg.target_entropy) if cfg.target_entropy is not None
            else -float(cfg.n_action)
        )
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)

        self.replay = PrioritizedReplay(capacity=cfg.replay_capacity)

        self._frozen_until: int | None = None
        self._hold_mode = False
        self._step = 0

    # ----- predict -----
    def predict(self, x_obs: np.ndarray, base_reference: np.ndarray,
                fdd: FDDOutput, monitor_alarm: str = "OK"
                ) -> tuple[np.ndarray, float, bool]:
        x = torch.tensor(np.asarray(x_obs, dtype=np.float32),
                         dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            if self.cfg.eval_mode:
                a = self.actor.deterministic(x)
            else:
                a, _ = self.actor.rsample(x)
            z = self.critic1(x, a)
        a_np = a.cpu().numpy().reshape(-1) * float(self.cfg.action_scale)
        if self._hold_mode:
            r_tilde = np.asarray(base_reference, dtype=np.float64).copy()
        else:
            r_tilde = (np.asarray(base_reference, dtype=np.float64).copy()
                       + a_np[: int(self.cfg.n_ref_dim)])
        beta_t = risk_gate(z, fdd_severity=fdd.severity,
                           monitor_alarm=monitor_alarm)
        drift = getattr(fdd, "glr_drift_estimate", None)
        reset_hint = bool(drift is not None
                          and float(np.linalg.norm(drift)) > self.cfg.glr_reset_threshold)
        return r_tilde, beta_t, reset_hint

    # ----- learn -----
    def learn(self, t: Transition) -> dict:
        self._step += 1
        if self._frozen_until is not None and self._step < self._frozen_until:
            self.replay.push(t)
            return {"frozen": True, "step": self._step}
        self.replay.push(t)
        if self.cfg.eval_mode:
            return {"eval_mode": True, "step": self._step}
        if len(self.replay) < self.cfg.batch_size:
            return {"warming_up": True, "step": self._step}
        if self._step % self.cfg.learn_every != 0:
            return {"step": self._step}
        return self._sgd_step()

    def _sgd_step(self) -> dict:
        batch, idx, w = self.replay.sample(self.cfg.batch_size)
        s = torch.tensor(np.stack([t.s for t in batch]), dtype=torch.float32, device=self.device)
        a = torch.tensor(np.stack([t.a_actual for t in batch]), dtype=torch.float32, device=self.device)
        r = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.stack([t.s_next for t in batch]), dtype=torch.float32, device=self.device)
        done = torch.tensor([float(t.done) for t in batch], dtype=torch.float32, device=self.device)
        weights = torch.tensor(w, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            a2, logp2 = self.actor.rsample(s2)
            z2 = torch.minimum(self.target1(s2, a2), self.target2(s2, a2))
            y = r.unsqueeze(-1) + self.cfg.gamma * (1.0 - done.unsqueeze(-1)) * (z2 - self.log_alpha.exp() * logp2.unsqueeze(-1))

        z1 = self.critic1(s, a)
        z2_pred = self.critic2(s, a)
        loss_c1 = (qr_huber_loss(z1, y, self.cfg.huber_kappa).unsqueeze(0) * weights).mean()
        loss_c2 = (qr_huber_loss(z2_pred, y, self.cfg.huber_kappa).unsqueeze(0) * weights).mean()
        self.opt_c1.zero_grad(); loss_c1.backward(); self.opt_c1.step()
        self.opt_c2.zero_grad(); loss_c2.backward(); self.opt_c2.step()

        # Actor update against CVaR_alpha of min critic.
        a_pi, logp = self.actor.rsample(s)
        z_pi = torch.minimum(self.critic1(s, a_pi), self.critic2(s, a_pi))
        cvar = cvar_alpha_fn(z_pi, self.cfg.cvar_alpha)
        loss_a = (self.log_alpha.exp() * logp - cvar).mean()
        self.opt_actor.zero_grad(); loss_a.backward(); self.opt_actor.step()

        loss_alpha = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(); loss_alpha.backward(); self.opt_alpha.step()

        soft_update(target=self.target1, source=self.critic1, tau=self.cfg.tau)
        soft_update(target=self.target2, source=self.critic2, tau=self.cfg.tau)

        td = (z1.mean(dim=-1) - y.mean(dim=-1)).abs().detach().cpu().numpy()
        self.replay.update_priorities(idx, td)
        return {"loss_c1": float(loss_c1), "loss_c2": float(loss_c2),
                "loss_a": float(loss_a), "loss_alpha": float(loss_alpha),
                "step": self._step}

    # ----- macro-action sinks (Phase 4 will call these) -----
    def freeze_learning(self, until_step: int) -> None:
        self._frozen_until = int(until_step)

    def degrade_reference_to_hold(self) -> None:
        self._hold_mode = True

    def reset(self) -> None:
        self._hold_mode = False

    # ----- save/load -----
    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), dir_path / "actor.pt")
        torch.save(self.critic1.state_dict(), dir_path / "critic1.pt")
        torch.save(self.critic2.state_dict(), dir_path / "critic2.pt")
        torch.save(self.target1.state_dict(), dir_path / "target1.pt")
        torch.save(self.target2.state_dict(), dir_path / "target2.pt")
        torch.save(self.log_alpha.detach(), dir_path / "log_alpha.pt")
        import json
        (dir_path / "dsac_config.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    @classmethod
    def from_pretrained(cls, dir_path: str | Path, *,
                        cfg: DSACConfig | None = None) -> "DSACOuter":
        dir_path = Path(dir_path)
        if cfg is None:
            import json
            cfg_d = json.loads((dir_path / "dsac_config.json").read_text())
            cfg_d["actor_hidden"] = tuple(cfg_d["actor_hidden"])
            cfg_d["critic_hidden"] = tuple(cfg_d["critic_hidden"])
            cfg = DSACConfig(**cfg_d)
        m = cls(cfg)
        m.actor.load_state_dict(torch.load(dir_path / "actor.pt", map_location="cpu"))
        m.critic1.load_state_dict(torch.load(dir_path / "critic1.pt", map_location="cpu"))
        m.critic2.load_state_dict(torch.load(dir_path / "critic2.pt", map_location="cpu"))
        m.target1.load_state_dict(torch.load(dir_path / "target1.pt", map_location="cpu"))
        m.target2.load_state_dict(torch.load(dir_path / "target2.pt", map_location="cpu"))
        m.log_alpha = torch.tensor(torch.load(dir_path / "log_alpha.pt"),
                                   requires_grad=True, device=m.device)
        m.opt_alpha = torch.optim.Adam([m.log_alpha], lr=cfg.lr_alpha)
        return m
