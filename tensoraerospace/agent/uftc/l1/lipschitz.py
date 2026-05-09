"""Power-iteration upper bound on the Lipschitz constant of a torch module.

Approximate ``L(f) = sup_x ‖J_f(x)‖_2`` by sampling ``n_starts`` points
from ``sample_fn`` and running ``n_iter`` steps of the power method on
the Jacobian-vector product. Returns the maximum spectral norm seen.
This is an upper bound only when the maximiser falls inside the
sampled distribution; we mitigate with multiple restarts.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def power_iteration_lipschitz(
    model: "torch.nn.Module",
    sample_fn: Callable[[], np.ndarray],
    *,
    n_iter: int = 200,
    n_starts: int = 8,
    dtype=None,
) -> float:
    import torch

    if dtype is None:
        dtype = torch.float32

    L_max = 0.0
    for _ in range(int(n_starts)):
        x_np = sample_fn().astype(np.float64)
        x = torch.tensor(x_np, dtype=dtype, requires_grad=True)

        # Probe output shape so the cotangent matches ``y``.
        y0 = model(x)
        y_shape = y0.shape
        u = torch.randn(y_shape, dtype=dtype)
        u = u / (u.norm() + 1e-12)

        norm = 0.0
        for _ in range(int(n_iter)):
            # Recompute y with create_graph=True so we can differentiate
            # the VJP again to obtain the JVP (double-backward trick).
            y = model(x)
            u_var = u.detach().clone().requires_grad_(True)
            (jt_u,) = torch.autograd.grad(
                y,
                x,
                grad_outputs=u_var,
                retain_graph=True,
                create_graph=True,
            )
            # ``jt_u`` is linear in ``u_var``; differentiating
            # ``<jt_u, w>`` w.r.t. ``u_var`` yields ``J w`` for any w.
            w = jt_u.detach()
            w_norm = float(w.norm())
            if w_norm < 1e-18:
                norm = 0.0
                break
            w = w / w_norm
            (j_w,) = torch.autograd.grad(
                jt_u,
                u_var,
                grad_outputs=w,
                retain_graph=False,
                create_graph=False,
            )
            u_new = j_w.detach()
            norm = float(u_new.norm())
            if norm < 1e-18:
                break
            u = u_new / norm
        L_max = max(L_max, norm)
    return float(L_max)
