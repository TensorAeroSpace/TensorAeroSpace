import numpy as np
import torch

from tensoraerospace.agent.mpc.mpc import (
    MPC,
    MPCAgent,
    MPCConstraints,
    MPCStepResponseExtraCostConfig,
    MPCTrackingExtraCostConfig,
    MPCWeights,
)


def _dyn(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    # simple linear-ish dynamics: x_{t+1} = x + [u, 0, 0, 0]
    dx = torch.cat([u, torch.zeros_like(x[..., :3])], dim=-1)
    return x + dx


def test_mpc_solve_shapes():
    horizon = 5
    state_dim = 4
    action_dim = 1
    mpc = MPC(
        dynamics=_dyn,
        state_dim=state_dim,
        action_dim=action_dim,
        horizon=horizon,
        weights=MPCWeights(
            Q_diag=np.ones((state_dim,), dtype=np.float32),
            R_diag=np.ones((action_dim,), dtype=np.float32),
            S_diag=np.ones((action_dim,), dtype=np.float32),
            terminal_weight=1.0,
        ),
        constraints=MPCConstraints(
            u_min=np.array([-1.0], dtype=np.float32),
            u_max=np.array([1.0], dtype=np.float32),
            du_min=np.array([-0.5], dtype=np.float32),
            du_max=np.array([0.5], dtype=np.float32),
        ),
        iters=10,
        lr=0.05,
        optimizer="adam",
        warm_start=False,
        track_best=True,
    )

    x0 = np.zeros((state_dim,), dtype=np.float32)
    x_ref = np.zeros((horizon + 1, state_dim), dtype=np.float32)
    res = mpc.solve(x0=x0, x_ref=x_ref, u_prev=np.array([0.0], dtype=np.float32))

    assert res.u0.shape == (action_dim,)
    assert res.u_seq.shape == (horizon, action_dim)
    assert res.x_seq.shape == (horizon + 1, state_dim)
    assert isinstance(res.final_cost, float)


def test_mpc_agent_tracking_and_step_configs():
    state_dim = 4
    action_dim = 1

    tracking_cfg = MPCTrackingExtraCostConfig(w_du=0.1, w_jerk=0.2)
    step_cfg = MPCStepResponseExtraCostConfig.from_degrees(
        tracked_idx=3, rate_idx=2, dt=0.1
    )

    env_action_low = np.array([-1.0], dtype=np.float32)
    env_action_high = np.array([1.0], dtype=np.float32)

    class _Env:
        def __init__(self):
            self.action_space = type(
                "A", (), {"low": env_action_low, "high": env_action_high, "shape": (1,)}
            )()
            self.observation_space = type("O", (), {"shape": (state_dim,)})()
            self.unwrapped = self
            self.dt = 0.1
            self.number_time_steps = 10
            self.model = type(
                "M", (), {"xt": np.zeros((state_dim,), dtype=np.float32)}
            )()

        def reset(self):
            return np.zeros((state_dim,), dtype=np.float32), {}

        def step(self, action):
            self.model.xt = self.model.xt + np.concatenate(
                [action, np.zeros((state_dim - action_dim,), dtype=np.float32)]
            )
            return (
                self.model.xt.copy(),
                0.0,
                False,
                False,
                {},
            )

    env = _Env()

    agent = MPCAgent(
        env=env,
        horizon=5,
        weights=MPCWeights(
            Q_diag=np.ones((state_dim,), dtype=np.float32),
            R_diag=np.full((action_dim,), 1e-2, dtype=np.float32),
            S_diag=np.full((action_dim,), 1e-1, dtype=np.float32),
        ),
        constraints=MPCConstraints(
            u_min=env_action_low,
            u_max=env_action_high,
        ),
        tracking_type="tracking",
        tracking_config=tracking_cfg,
        step_response_config=step_cfg,
        iters=5,
        mpc_lr=0.05,
        device="cpu",
        seed=0,
    )

    state = np.zeros((state_dim,), dtype=np.float32)
    x_ref = np.zeros((agent.mpc.horizon + 1, state_dim), dtype=np.float32)
    action = agent.select_action(state, x_ref=x_ref)
    assert action.shape == (action_dim,)


def test_mpc_projection_rate_and_mag_clamp():
    horizon = 3
    mpc = MPC(
        dynamics=_dyn,
        state_dim=2,
        action_dim=1,
        horizon=horizon,
        weights=MPCWeights(
            Q_diag=np.ones((2,), dtype=np.float32),
            R_diag=np.ones((1,), dtype=np.float32),
            S_diag=np.ones((1,), dtype=np.float32),
        ),
        constraints=MPCConstraints(
            u_min=np.array([-0.2], dtype=np.float32),
            u_max=np.array([0.2], dtype=np.float32),
            du_min=np.array([-0.1], dtype=np.float32),
            du_max=np.array([0.1], dtype=np.float32),
        ),
        iters=2,
        warm_start=False,
        track_best=False,
    )

    u_raw = torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float32)
    u_prev = torch.tensor([0.0], dtype=torch.float32)
    u_proj = mpc._project_u(u_raw, u_prev)  # type: ignore[arg-type]
    # Should clamp magnitude and rate
    assert torch.allclose(u_proj[0], torch.tensor([0.1]))
    assert torch.all(u_proj <= 0.2 + 1e-6)
    assert torch.all(u_proj >= -0.2 - 1e-6)


def test_tracking_extra_cost_nonzero():
    cfg = MPCTrackingExtraCostConfig(w_du=1.5, w_jerk=2.0)
    agent = MPCAgent(
        env=type(
            "E",
            (),
            {
                "action_space": type("A", (), {"low": [-1.0], "high": [1.0]})(),
                "observation_space": type("O", (), {"shape": (2,)})(),
                "unwrapped": None,
            },
        )(),
        horizon=4,
        weights=MPCWeights(Q_diag=np.ones(2), R_diag=np.ones(1), S_diag=np.ones(1)),
        constraints=MPCConstraints(u_min=[-1.0], u_max=[1.0]),
        tracking_type="tracking",
        tracking_config=cfg,
        iters=2,
    )
    extra = agent._make_tracking_extra_cost(cfg)

    u_seq = torch.tensor([[0.0], [0.2], [0.0], [0.1]], dtype=torch.float32)
    x_seq = torch.zeros((5, 2), dtype=torch.float32)
    cost = extra(x_seq, u_seq, None)
    assert cost.item() > 0.0


def test_step_response_extra_cost_triggers():
    cfg = MPCStepResponseExtraCostConfig.from_degrees(
        tracked_idx=1, rate_idx=None, dt=0.1, ref_change_threshold_deg=0.01
    )
    agent = MPCAgent(
        env=type(
            "E",
            (),
            {
                "action_space": type("A", (), {"low": [-1.0], "high": [1.0]})(),
                "observation_space": type("O", (), {"shape": (2,)})(),
                "unwrapped": None,
            },
        )(),
        horizon=4,
        weights=MPCWeights(Q_diag=np.ones(2), R_diag=np.ones(1), S_diag=np.ones(1)),
        constraints=MPCConstraints(u_min=[-1.0], u_max=[1.0]),
        tracking_type="step_response",
        step_response_config=cfg,
        iters=2,
    )
    extra = agent._make_step_response_extra_cost(cfg)

    u_seq = torch.zeros((4, 1), dtype=torch.float32)
    x_ref = torch.zeros((5, 2), dtype=torch.float32)
    x_ref[2:, 1] = 0.2
    x_seq = torch.zeros((5, 2), dtype=torch.float32)
    x_seq[3:, 1] = 0.4
    cost = extra(x_seq, u_seq, x_ref)
    assert cost.item() > 0.0
