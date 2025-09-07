import numpy as np
import torch

from tensoraerospace.agent.mpc.base import AircraftMPC


class _Dyn:
    def __call__(self, x_u):
        # x_u: [1, state_dim + control_dim]
        # simple stable linear dynamics: next = x + B u
        x = x_u[..., :4]
        u = x_u[..., 4:]
        return x + 0.1 * torch.cat([u, torch.zeros_like(x[..., :3])], dim=-1)


def test_aircraft_mpc_optimize_smoke():
    mpc = AircraftMPC(
        dynamics_model=_Dyn(),
        horizon=3,
        dt=0.1,
        state_dim=4,
        control_dim=1,
        iterations=3,
    )
    x0 = np.zeros(4, dtype=np.float32)
    theta_ref = np.zeros(4, dtype=np.float32)
    u0, traj = mpc.optimize_control(x0, theta_ref)
    assert u0.shape == (1,)
    assert traj.shape == (3, 4)

    # Cost/penalty functions callable
    U = np.zeros((mpc.horizon, mpc.control_dim))
    X = np.zeros((mpc.horizon + 1, mpc.state_dim))
    _ = mpc.cost_function(X, U, np.zeros(mpc.horizon + 1))
    _ = mpc.penalty_function(U)
