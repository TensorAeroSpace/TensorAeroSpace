"""
Утилиты для алгоритма A3C (Asynchronous Advantage Actor-Critic).

Этот модуль содержит вспомогательные функции для реализации алгоритма A3C,
включая функции для обработки данных, инициализации весов, синхронизации
между процессами и записи результатов.
"""

import numpy as np
import torch
from torch import nn


def v_wrap(np_array, dtype=np.float32):
    """Преобразует numpy массив в PyTorch тензор.

    Args:
        np_array (numpy.ndarray): Входной numpy массив.
        dtype (numpy.dtype): Тип данных для преобразования.
            По умолчанию np.float32.

    Returns:
        torch.Tensor: Преобразованный PyTorch тензор.
    """
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    """Инициализирует веса и смещения слоев нейронной сети.

    Args:
        layers (list): Список слоев для инициализации.
    """
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(layer.bias, 0.0)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    """Выполняет синхронизацию между локальной и глобальной сетями.

    Вычисляет градиенты на локальной сети и обновляет глобальную сеть,
    затем копирует параметры глобальной сети в локальную.

    Args:
        opt (torch.optim.Optimizer): Оптимизатор для глобальной сети.
        lnet (torch.nn.Module): Локальная нейронная сеть.
        gnet (torch.nn.Module): Глобальная нейронная сеть.
        done (bool): Флаг завершения эпизода.
        s_ (numpy.ndarray): Следующее состояние.
        bs (list): Буфер состояний.
        ba (list): Буфер действий.
        br (list): Буфер наград.
        gamma (float): Коэффициент дисконтирования.

    Returns:
        dict: Словарь с метриками для логирования
            (loss, value_loss, policy_loss, entropy).
    """
    if done:
        v_s_ = 0.0  # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].detach().cpu().numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:  # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    # Compute forward pass and individual loss components
    s_batch = v_wrap(np.vstack(bs))
    a_batch = (
        v_wrap(np.array(ba), dtype=np.int64)
        if ba[0].dtype == np.int64
        else v_wrap(np.vstack(ba))
    )
    v_t_batch = v_wrap(np.array(buffer_v_target)[:, None])

    lnet.train()
    mu, sigma, values = lnet.forward(s_batch)
    td = v_t_batch - values
    c_loss = td.pow(2)

    base = lnet.distribution(mu, sigma)
    dist = torch.distributions.Independent(base, 1) if lnet.a_dim > 1 else base
    log_prob = dist.log_prob(a_batch)
    entropy = dist.entropy()
    exp_v = log_prob * td.detach().squeeze(-1) + 0.005 * entropy
    a_loss = -exp_v
    total_loss = (a_loss + c_loss.squeeze(-1)).mean()

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    lnet.zero_grad()
    total_loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    # clip gradients for stability before optimizer step
    torch.nn.utils.clip_grad_norm_(gnet.parameters(), max_norm=40.0)
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())

    # Return metrics for logging
    return {
        "loss": total_loss.detach().cpu().item(),
        "value_loss": c_loss.mean().detach().cpu().item(),
        "policy_loss": a_loss.mean().detach().cpu().item(),
        "entropy": entropy.mean().detach().cpu().item(),
    }


def record(global_ep, global_ep_r, ep_r, res_queue, name, writer=None):
    """Записывает результаты эпизода и обновляет глобальные счетчики.

    Args:
        global_ep (multiprocessing.Value): Глобальный счетчик эпизодов.
        global_ep_r (multiprocessing.Value): Глобальная скользящая
            средняя награды.
        ep_r (float): Награда за текущий эпизод.
        res_queue (multiprocessing.Queue): Очередь для результатов.
        name (str): Имя процесса.
        writer (torch.utils.tensorboard.SummaryWriter, optional):
            TensorBoard writer для логирования.
    """
    with global_ep.get_lock():
        global_ep.value += 1
        ep_idx = global_ep.value
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.0:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
        moving_avg = global_ep_r.value

    res_queue.put(moving_avg)

    # Log to TensorBoard if writer is provided
    if writer is not None:
        writer.add_scalar(f"Performance/{name}/episode_reward", ep_r, ep_idx)
        writer.add_scalar(f"Performance/{name}/moving_avg_reward", moving_avg, ep_idx)
