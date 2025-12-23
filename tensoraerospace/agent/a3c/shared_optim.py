"""Shared optimizer utilities for multi-process A3C training.

This module provides optimizer implementations or helpers that allow sharing
optimizer state across multiple processes, which is commonly used in A3C-style
training loops.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch


class SharedAdam(torch.optim.Adam):
    """Adam optimizer with shared state for multi-process training.

    This optimizer stores its internal state tensors (step counter, exp_avg,
    exp_avg_sq) in shared memory so multiple worker processes can update a
    single set of parameters consistently.

    Args:
        params: Parameters to optimize.
        lr: Learning rate. Defaults to ``1e-3``.
        betas: Coefficients used for computing running averages of gradient and
            its square. Defaults to ``(0.9, 0.99)``.
        eps: Term added to the denominator for numerical stability. Defaults to
            ``1e-8``.
        weight_decay: Weight decay (L2 penalty). Defaults to ``0``.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        """Initialize shared Adam optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            betas: Beta coefficients for Adam moments.
            eps: Numerical stability term.
            weight_decay: L2 weight decay.
        """
        super(SharedAdam, self).__init__(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        # State initialization
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # PyTorch Adam functional API requires step as a singleton
                # tensor. Use a tensor in shared memory for the step counter
                # as well,
                # so bias-correction terms are consistent across
                # processes.
                state["step"] = torch.zeros(1, dtype=torch.long)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

                # share in memory
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()
