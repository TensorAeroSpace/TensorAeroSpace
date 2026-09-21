"""Native, active parameter paths for the declarative controller profiles.

Environment dimensions, units, initial plant models, RNG seeds and hardware
limits belong to the experiment protocol, not to its optimization variables.
"""

from __future__ import annotations

from .agent import _replace_paths

# These are settings consumed by the interaction loop actually used by a profile.
# For example AIDI.predict_rates bypasses C*/speed guidance and PCH; exposing those
# knobs here would misleadingly offer parameters with no effect on this rollout.
_NATIVE = {
    "aa_indi": {
        "config": (
            "sigma0 enable_sensor_correction forgetting_min covariance_init "
            "acceleration_cutoff_hz rate_feedback pinv_rcond"
        ).split(),
        "config.observer": (
            "imu_std navigation_std initial_bias_std bias_walk_std hosm_gains "
            "drift_scales"
        ).split(),
    },
    "aidi": {
        "config": (
            "pinv_rcond cond_threshold sensor_cutoff_hz rls_lambda_min "
            "rls_lambda_max rls_sigma0 rls_memory_length rls_cov_init "
            "rls_consistency_threshold rate_kp"
        ).split(),
    },
    "iadp": {
        "config": (
            "Q R gamma gamma_rls phi_init policy_eval_window policy_eval_every "
            "policy_eval_iterations policy_eval_warmup_updates "
            "policy_eval_min_samples policy_training_start_step"
        ).split(),
    },
    "imgdhp": {
        "config": (
            "gamma actor_hidden critic_hidden actor_bias_input actor_lr critic_lr "
            "actor_lr_decay critic_lr_decay actor_lr_min critic_lr_min "
            "weight_limit beta_lambda track_Q control_R history_length forgetting "
            "cov_init warmup_steps critic_only_steps critic_updates_per_step "
            "target_update_tau critic_weight_decay obs_scale max_grad_norm "
            "exploration_noise_std optimizer"
        ).split(),
    },
    "et_dhp": {
        "config": (
            "actor_hidden critic_hidden model_hidden actor_lr critic_lr model_lr "
            "model_epochs online_model_fit Q R gamma num_epochs_per_trigger rho "
            "trigger_floor weight_init_scale"
        ).split(),
    },
    "ihdp": {
        "actor_settings": (
            "start_training layers activations learning_rate learning_rate_min "
            "learning_rate_decay type_PE amplitude_3211 pulse_length_3211 "
            "WB_limits"
        ).split(),
        "critic_settings": (
            "Q_weights start_training gamma learning_rate learning_rate_min "
            "learning_rate_decay layers activations WB_limits"
        ).split(),
        "incremental_settings": ["window_size"],
    },
    "mpc": {"weights": "Q_diag R_diag S_diag terminal_weight".split()},
    "hdp": {},
}


def native_parameters(name):
    """Return constructor paths; numeric indices address vector/matrix elements."""
    return sorted(
        f"{prefix}.{field}"
        for prefix, fields in _NATIVE[name].items()
        for field in fields
    )


def is_native_parameter(name, path):
    """Recognize constructor paths and canonical integer indices into array fields."""
    for root in native_parameters(name):
        if path == root:
            return True
        if path.startswith(root + "."):
            return all(
                p.isdigit() and str(int(p)) == p
                for p in path[len(root) + 1 :].split(".")
            )
    return False


def configure(value, params, prefix="config"):
    """Apply native constructor settings atomically, including dataclass checks."""
    updates = {
        k[len(prefix) + 1 :]: params.pop(k)
        for k in list(params)
        if k.startswith(prefix + ".")
    }
    if isinstance(params, ParameterValues):
        fixed = {
            k[len(prefix) + 1 :]: v
            for k, v in params.fixed.items()
            if k.startswith(prefix + ".")
        }
        sampled = {
            k[len(prefix) + 1 :]: v
            for k, v in params.sampled.items()
            if k.startswith(prefix + ".")
        }
        value = _replace_paths(value, fixed) if fixed else value
        return _replace_paths(value, sampled) if sampled else value
    return _replace_paths(value, updates) if updates else value


# Resolve public shortcuts to their native destination so an inactive duplicate
# (e.g. actor_lr plus actor_settings.learning_rate) cannot waste trials.
_ALIASES = {
    "aa_indi": {
        "rate_gain": "config.rate_feedback",
        "cutoff_hz": "config.acceleration_cutoff_hz",
        "covariance_init": "config.covariance_init",
    },
    "aidi": {
        "rate_gain": "config.rate_kp",
        "cutoff_hz": "config.sensor_cutoff_hz",
        "rls_cov_init": "config.rls_cov_init",
    },
    "iadp": {
        "gamma": "config.gamma",
        "forgetting": "config.gamma_rls",
        "control_weight": "config.R",
        "track_weight": "config.Q",
        "phi_init": "config.phi_init",
        "policy_eval_every": "config.policy_eval_every",
        "policy_eval_window": "config.policy_eval_window",
    },
    "imgdhp": {
        "actor_lr": "config.actor_lr",
        "critic_lr": "config.critic_lr",
        "history_length": "config.history_length",
        "actor_hidden": "config.actor_hidden",
        "critic_hidden": "config.critic_hidden",
        "control_weight": "config.control_R",
        "track_weight": "config.track_Q",
        "warmup_steps": "config.warmup_steps",
        "exploration_noise_std": "config.exploration_noise_std",
    },
    "et_dhp": {
        "actor_lr": "config.actor_lr",
        "critic_lr": "config.critic_lr",
        "rho": "config.rho",
        "control_weight": "config.R",
        "track_weight": "config.Q",
        "model_epochs": "config.model_epochs",
        "num_epochs_per_trigger": "config.num_epochs_per_trigger",
        "online_model_fit": "config.online_model_fit",
    },
    "ihdp": {
        "actor_lr": "actor_settings.learning_rate",
        "critic_lr": "critic_settings.learning_rate",
        "gamma": "critic_settings.gamma",
        "track_weight": "critic_settings.Q_weights",
        "excitation_amplitude": "actor_settings.amplitude_3211",
        "actor_hidden": "actor_settings.layers",
        "critic_hidden": "critic_settings.layers",
    },
    "mpc": {
        "track_weight": "weights.Q_diag",
        "control_weight": "weights.R_diag",
        "terminal_weight": "weights.terminal_weight",
    },
    "hdp": {},
}


def destinations(name, key):
    """Resolve a shortcut to the native settings it changes in this controller."""
    if name == "ihdp" and key == "hidden_size":
        return ("actor_settings.layers", "critic_settings.layers")
    if name == "ihdp" and key == "warmup_steps":
        return ("actor_settings.start_training", "critic_settings.start_training")
    return (_ALIASES[name].get(key, key),)


def overlapping(name, left, right):
    """Detect aliases or ancestor paths that would modify the same native setting."""
    return any(
        a == b or a.startswith(b + ".") or b.startswith(a + ".")
        for a in destinations(name, left)
        for b in destinations(name, right)
    )


def validate_search_parameters(name, keys):
    """Reject search dimensions that overwrite one another after alias resolution."""
    keys = list(keys)
    for i, key in enumerate(keys):
        for other in keys[:i]:
            if overlapping(name, key, other):
                raise ValueError(
                    f"Search parameters {key!r} and {other!r} overlap; choose one "
                    f"path to the setting"
                )


class ParameterValues(dict):
    """Keep fixed native templates separate from sampled element overrides."""

    def __init__(self, fixed, sampled):
        super().__init__({**fixed, **sampled})
        self.fixed = fixed
        self.sampled = sampled
