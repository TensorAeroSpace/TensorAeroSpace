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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        # State initialization
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # PyTorch Adam functional API requires step as a singleton tensor
                state["step"] = torch.zeros((), dtype=torch.long)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

                # share in memory
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()
