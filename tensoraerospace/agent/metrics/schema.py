"""Canonical TensorBoard metric names for TensorAeroSpace RL agents.

Every metric written through ``MetricWriter`` (with ``strict=True``)
must use one of the constants defined here. Group prefixes are stable:

    rollout/      — per-episode environment statistics
    loss/         — training losses
    policy/       — policy / action statistics
    value/        — value-function statistics
    diagnostics/  — algorithm-specific diagnostics
    train/        — training progress counters
    eval/         — evaluation episode statistics
    weights/      — network weight histograms
    grads/        — gradient histograms
"""

from __future__ import annotations

import re
from typing import FrozenSet

# ---------------------------------------------------------------------------
# Tier 1 — Mandatory minimum (every RL agent must log)
# ---------------------------------------------------------------------------

ROLLOUT_EPISODE_REWARD = "rollout/episode_reward"
ROLLOUT_EPISODE_LENGTH = "rollout/episode_length"
ROLLOUT_TOTAL_STEPS = "rollout/total_steps"
TRAIN_UPDATES = "train/updates"
TRAIN_LR = "train/lr"

# ---------------------------------------------------------------------------
# Tier 2 — Common (logged when applicable)
# ---------------------------------------------------------------------------

# loss/*
LOSS_ACTOR = "loss/actor"
LOSS_CRITIC = "loss/critic"
LOSS_ENTROPY = "loss/entropy"
LOSS_VALUE = "loss/value"

# policy/*
POLICY_ENTROPY = "policy/entropy"
POLICY_ACTION_STD = "policy/action_std"
POLICY_ACTION_ABS_MEAN = "policy/action_abs_mean"

# value/*
VALUE_MEAN = "value/mean"
VALUE_TD_TARGET = "value/td_target_mean"
VALUE_TD_ERROR_MEAN = "value/td_error_mean"
VALUE_TD_ERROR_MAX = "value/td_error_max"
VALUE_TD_ERROR_MIN = "value/td_error_min"

# diagnostics/*
DIAG_TERMINATED_COUNT = "diagnostics/terminated_count"
DIAG_TRUNCATED_COUNT = "diagnostics/truncated_count"

# eval/*
EVAL_EPISODE_REWARD = "eval/episode_reward"
EVAL_EPISODE_LENGTH = "eval/episode_length"


# ---------------------------------------------------------------------------
# Tier 3 — Per-algorithm extras
# ---------------------------------------------------------------------------

class PPO:
    APPROX_KL = "diagnostics/approx_kl"
    CLIP_FRACTION = "diagnostics/clip_fraction"
    EXPLAINED_VARIANCE = "diagnostics/explained_variance"
    REWARD_MEDIAN = "rollout/episode_reward_median"
    REWARD_P10 = "rollout/episode_reward_p10"
    REWARD_P90 = "rollout/episode_reward_p90"


class SAC:
    LOSS_Q1 = "loss/q1"
    LOSS_Q2 = "loss/q2"
    LOSS_POLICY = "loss/policy"
    LOSS_ALPHA = "loss/alpha"
    ALPHA_VALUE = "policy/alpha"
    Q_MEAN = "value/q_mean"
    LOG_PI_MEAN = "policy/log_pi_mean"
    REPLAY_SIZE = "train/replay_size"


class DSAC:
    # DSAC reuses SAC.* names for the standard losses (loss/q1, loss/q2,
    # loss/policy, loss/alpha, policy/alpha, train/replay_size). Only
    # CAPS regularization terms live here.
    CAPS_SPATIAL = "loss/caps_spatial"
    CAPS_TEMPORAL = "loss/caps_temporal"


class DQN:
    LOSS_Q = "loss/q"
    Q_PRED_SA_MEAN = "value/q_pred_mean"
    Q_TARGET_SA_MEAN = "value/q_target_mean"
    EPSILON = "train/epsilon"
    PER_BETA = "train/per_beta"
    REPLAY_SIZE = "train/replay_size"
    TARGET_UPDATE = "train/target_update"


class DDPG:
    LOSS_POLICY = "loss/policy"
    LOSS_VALUE = "loss/value"
    REPLAY_SIZE = "train/replay_size"


class A2C:
    ADVANTAGE_MEAN = "value/advantage_mean"
    ADVANTAGE_STD = "value/advantage_std"
    ADVANTAGE_NORMALIZED_MEAN = "value/advantage_normalized_mean"
    VALUE_BEFORE_UPDATE = "value/before_update_mean"
    ENTROPY_BETA = "policy/entropy_beta"


class ADP:
    DHP_PHASE_EPISODE = "train/dhp_phase_episode"
    LOSS_ACTOR_HDP = "loss/actor_hdp"
    LOSS_ACTOR_GDHP = "loss/actor_adgdhp"
    LOSS_CRITIC_HDP = "loss/critic_hdp"
    LOSS_CRITIC_GDHP = "loss/critic_gdhp"
    LOSS_CRITIC_LAMBDA = "loss/critic_lambda"


class ADHDP:
    DO_CRITIC = "train/do_critic"
    DO_ACTOR = "train/do_actor"
    ACTION_SAT_FRAC = "policy/action_sat_frac"


class GAIL:
    LOSS_DISCRIMINATOR = "loss/discriminator"
    LOSS_GENERATOR = "loss/generator"
    EXPERT_ACCURACY = "diagnostics/expert_accuracy"
    POLICY_ACCURACY = "diagnostics/policy_accuracy"


# ---------------------------------------------------------------------------
# Histogram / multi-worker prefix conventions
# ---------------------------------------------------------------------------

# Allowed histogram top-level groups.
HISTOGRAM_GROUPS: FrozenSet[str] = frozenset({"weights", "grads"})

# Allowed second-level groups inside weights/<group>/<param> and grads/<group>/<param>.
HISTOGRAM_SUBGROUPS: FrozenSet[str] = frozenset({
    "actor", "critic", "policy", "value", "q1", "q2", "discriminator",
})

# Multi-worker suffix pattern: any registered scalar tag may be suffixed with
# "/worker_<N>" where N is a non-negative integer.
_WORKER_SUFFIX_RE = re.compile(r"/worker_\d+$")


def strip_worker_suffix(tag: str) -> str:
    """Return ``tag`` with any trailing ``/worker_<N>`` stripped."""
    return _WORKER_SUFFIX_RE.sub("", tag)


def _collect_constants(namespace) -> FrozenSet[str]:
    """Return all UPPER_SNAKE_CASE string attributes from a module or class."""
    items = vars(namespace).items()
    return frozenset(
        v
        for k, v in items
        if k.isupper() and isinstance(v, str) and not k.startswith("_")
    )


def _build_registry() -> FrozenSet[str]:
    import sys
    module = sys.modules[__name__]
    parts = [_collect_constants(module)]
    for cls in (PPO, SAC, DSAC, DQN, DDPG, A2C, ADP, ADHDP, GAIL):
        parts.append(_collect_constants(cls))
    out: set[str] = set()
    for p in parts:
        out |= p
    return frozenset(out)


REGISTRY: FrozenSet[str] = _build_registry()


def is_registered_scalar(tag: str) -> bool:
    """Return True if ``tag`` (after stripping multi-worker suffix) is in REGISTRY."""
    return strip_worker_suffix(tag) in REGISTRY


def is_registered_histogram(tag: str) -> bool:
    """Return True if ``tag`` matches ``<top>/<sub>/<param...>`` with allowed groups."""
    parts = tag.split("/")
    if len(parts) < 3:
        return False
    top, sub = parts[0], parts[1]
    return top in HISTOGRAM_GROUPS and sub in HISTOGRAM_SUBGROUPS
