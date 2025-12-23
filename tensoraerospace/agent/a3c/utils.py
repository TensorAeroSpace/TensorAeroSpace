"""Utilities for A3C (Asynchronous Advantage Actor-Critic).

This module contains helper functions used by the A3C implementation, such as
tensor conversion helpers, layer initialization, local/global model syncing,
and episodic logging.
"""

import numpy as np
import torch
from torch import nn


def v_wrap(np_array, dtype=np.float32):
    """Convert a NumPy array to a PyTorch tensor.

    Args:
        np_array (numpy.ndarray): Input array.
        dtype (numpy.dtype): Target dtype. Defaults to ``np.float32``.

    Returns:
        torch.Tensor: Converted tensor (shares memory when possible).
    """
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    """Initialize weights and biases for a list of layers.

    Args:
        layers (list): Layers to initialize. Each layer is expected to expose
            ``weight`` and ``bias`` tensors.
    """
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(layer.bias, 0.0)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    """Synchronize local and global networks and perform one optimization step.

    Computes gradients on the local network, applies them to the global network,
    then pulls updated global parameters back into the local network.

    Args:
        opt (torch.optim.Optimizer): Optimizer for the global network.
        lnet (torch.nn.Module): Local network.
        gnet (torch.nn.Module): Global network.
        done (bool): Whether the episode has terminated.
        s_ (numpy.ndarray): Next state.
        bs (list): State buffer.
        ba (list): Action buffer.
        br (list): Reward buffer.
        gamma (float): Discount factor.

    Returns:
        dict: Logging metrics (loss, value_loss, policy_loss, entropy).
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
    """Record episode results and update global counters.

    Args:
        global_ep (multiprocessing.Value): Global episode counter.
        global_ep_r (multiprocessing.Value): Global moving-average reward.
        ep_r (float): Episode reward.
        res_queue (multiprocessing.Queue): Queue used to store the moving average.
        name (str): Worker/process name (used for logging keys).
        writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard
            writer. If not provided, nothing is written.
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
