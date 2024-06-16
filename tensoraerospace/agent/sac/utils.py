import numpy as np
import torch


def create_log_gaussian(mean, log_std, t):
    """Вычисляет логарифм плотности вероятности для нормального распределения.

    Аргументы:
        mean (torch.Tensor): Среднее значение распределения.
        log_std (torch.Tensor): Логарифм стандартного отклонения распределения.
        t (torch.Tensor): Входное значение.

    Возвращает:
        log_p (torch.Tensor): Логарифм плотности вероятности.

    """
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    length = mean.shape
    log_z = log_std
    z = length[-1] * torch.log(2 * np.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    """Вычисляет логарифм от суммы экспонент.

    Аргументы:
        inputs (torch.Tensor): Входные данные.
        dim (int): Размерность для вычисления.
        keepdim (bool): Флаг сохранения размерности.

    Возвращает:
        outputs (torch.Tensor): Логарифм от суммы экспонент.

    """
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    """Мягкое обновление параметров модели target по параметрам модели source.

    Аргументы:
        target (torch.nn.Module): Целевая модель.
        source (torch.nn.Module): Исходная модель.
        tau (float): Коэффициент мягкого обновления.

    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    """Жесткое обновление параметров модели target по параметрам модели source.

    Аргументы:
        target (torch.nn.Module): Целевая модель.
        source (torch.nn.Module): Исходная модель.

    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)