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
        dtype (numpy.dtype): Тип данных для преобразования. По умолчанию np.float32.

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
    """
    if done:
        v_s_ = 0.0  # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:  # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64)
        if ba[0].dtype == np.int64
        else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]),
    )

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    """Записывает результаты эпизода и обновляет глобальные счетчики.

    Args:
        global_ep (multiprocessing.Value): Глобальный счетчик эпизодов.
        global_ep_r (multiprocessing.Value): Глобальная скользящая средняя награды.
        ep_r (float): Награда за текущий эпизод.
        res_queue (multiprocessing.Queue): Очередь для результатов.
        name (str): Имя процесса.
    """
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.0:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:",
        global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )
