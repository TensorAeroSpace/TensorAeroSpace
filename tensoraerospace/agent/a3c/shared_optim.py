from __future__ import annotations

from typing import Iterable, Tuple

import torch


class SharedAdam(torch.optim.Adam):
    """Разделяемый оптимизатор Adam для многопроцессного обучения.

    Расширяет стандартный оптимизатор Adam для работы в многопроцессной среде,
    где состояние оптимизатора разделяется между процессами.

    Args:
        params: Параметры для оптимизации.
        lr (float): Скорость обучения. По умолчанию 1e-3.
        betas (tuple): Коэффициенты для вычисления скользящих средних градиента
            и его квадрата. По умолчанию (0.9, 0.99).
        eps (float): Термин для численной стабильности. По умолчанию 1e-8.
        weight_decay (float): Коэффициент регуляризации весов. По умолчанию 0.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
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
