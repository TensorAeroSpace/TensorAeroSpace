"""Native controller profiles and interaction loops for step experiments.

Profiles are starting configurations, not claims of stability or optimality.
Application adapters set units, costs and guidance; they do not replace the
published learning equations implemented by the respective agents.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from optuna.distributions import FloatDistribution as Float
from optuna.distributions import IntDistribution as Int

from ._parameters import ParameterValues, configure, is_native_parameter
from .metrics import TrialRejected


def controller_name(value):
    name = value if isinstance(value, str) else getattr(value, "__name__", "")
    key = name.lower().replace("-", "").replace("_", "").removesuffix("agent")
    aliases = {
        "aaindi": "aa_indi",
        "aidi": "aidi",
        "iadp": "iadp",
        "imgdhp": "imgdhp",
        "etdhp": "et_dhp",
        "ihdp": "ihdp",
        "hdp": "hdp",
        "mpc": "mpc",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown controller {name!r}; choose {list(aliases.values())}"
        )
    return aliases[key]


def default_space(name):
    basics: dict[str, dict[str, Any]] = {
        "aa_indi": {
            "rate_gain": Float(1.0, 12.0, log=True),
            "cutoff_hz": Float(2.0, 15.0, log=True),
        },
        "aidi": {
            "rate_gain": Float(0.5, 10.0, log=True),
            "cutoff_hz": Float(2.0, 20.0, log=True),
        },
        "iadp": {
            "gamma": Float(0.8, 0.999),
            "forgetting": Float(0.99, 1.0),
            "control_weight": Float(0.01, 10.0, log=True),
        },
        "imgdhp": {
            "actor_lr": Float(1e-5, 1e-2, log=True),
            "critic_lr": Float(1e-4, 1e-2, log=True),
            "history_length": Int(2, 8),
        },
        "et_dhp": {
            "actor_lr": Float(1e-5, 1e-2, log=True),
            "critic_lr": Float(1e-5, 1e-2, log=True),
            "rho": Float(0.02, 0.49, log=True),
        },
        "ihdp": {
            "actor_lr": Float(1e-5, 1e-2, log=True),
            "critic_lr": Float(1e-5, 1e-2, log=True),
        },
        "hdp": {
            "actor_lr": Float(1e-5, 1e-2, log=True),
            "critic_lr": Float(1e-5, 1e-2, log=True),
        },
        "mpc": {
            "horizon": Int(10, 60, step=5),
            "control_weight": Float(0.001, 1.0, log=True),
        },
    }
    basic = basics[name]
    extras: dict[str, dict[str, Any]] = {
        "aa_indi": {
            "outer_gain": Float(0.1, 1.0, log=True),
            "covariance_init": Float(0.1, 10.0, log=True),
        },
        "aidi": {
            "outer_gain": Float(0.1, 1.0, log=True),
            "rls_cov_init": Float(0.1, 10.0, log=True),
            "config.rls_sigma0": Float(1e-4, 0.1, log=True),
        },
        "iadp": {
            "track_weight": Float(0.1, 10.0, log=True),
            "phi_init": Float(0.1, 100.0, log=True),
            "policy_eval_every": Int(10, 100, step=10),
        },
        "imgdhp": {
            "track_weight": Float(0.1, 10.0, log=True),
            "control_weight": Float(0.001, 0.1, log=True),
            "config.gamma": Float(0.8, 0.999),
            "config.beta_lambda": Float(0.001, 1.0, log=True),
            "config.forgetting": Float(0.99, 1.0),
        },
        "et_dhp": {
            "track_weight": Float(0.1, 10.0, log=True),
            "control_weight": Float(0.01, 1.0, log=True),
            "config.trigger_floor": Float(1e-5, 0.01, log=True),
            "config.gamma": Float(0.9, 1.0),
        },
        "ihdp": {
            "track_weight": Float(0.1, 10.0, log=True),
            "gamma": Float(0.8, 0.999),
            "hidden_size": Int(4, 32, step=4),
            "excitation_amplitude": Float(0.001, 0.05, log=True),
            "actor_settings.learning_rate_decay": Float(0.995, 1.0),
            "critic_settings.learning_rate_decay": Float(0.995, 1.0),
        },
        "hdp": {
            "gamma": Float(0.9, 0.999),
            "hidden_size": Int(16, 128, step=16),
            "exploration_std": Float(0.001, 0.1, log=True),
        },
        "mpc": {
            "track_weight": Float(0.1, 10.0, log=True),
            "terminal_weight": Float(0.1, 20.0, log=True),
            "lr": Float(0.005, 0.1, log=True),
        },
    }
    extra = extras[name]
    if name == "ihdp":
        basic = {
            "actor_lr": Float(1e-5, 0.5, log=True),
            "critic_lr": Float(1e-5, 0.1, log=True),
        }
    return {**basic, **extra}


# Public profile settings are explicit: no silent misspellings or unit overrides.
_OPTIONS = {
    "aa_indi": {"outer_gain", "rate_limit_deg", "covariance_init"},
    "aidi": {"outer_gain", "rate_limit_deg", "rls_cov_init"},
    "iadp": {"policy_eval_every", "policy_eval_window", "phi_init", "track_weight"},
    "imgdhp": {
        "actor_hidden",
        "critic_hidden",
        "track_weight",
        "control_weight",
        "warmup_steps",
        "exploration_noise_std",
    },
    "et_dhp": {
        "model_epochs",
        "model_samples",
        "track_weight",
        "num_epochs_per_trigger",
        "control_weight",
        "online_model_fit",
    },
    "ihdp": {
        "hidden_size",
        "actor_hidden",
        "critic_hidden",
        "excitation_amplitude",
        "warmup_steps",
        "gamma",
        "track_weight",
    },
    "hdp": {
        "hidden_size",
        "exploration_std",
        "gamma",
        "dhp_w_theta",
        "dhp_w_q",
        "dhp_w_u",
        "dhp_w_du",
        "dhp_use_env_cost",
        "dhp_critic_cycle_episodes",
        "dhp_action_cycle_episodes",
    },
    "mpc": {
        "iters",
        "lr",
        "terminal_weight",
        "track_weight",
        "optimizer",
        "warm_start",
        "track_best",
        "best_check_every",
    },
}


def supported_parameter(name, key):
    return key in (_OPTIONS[name] | set(default_space(name))) or is_native_parameter(
        name, key
    )


def validate_controller(tuner):
    kind, name = tuner.environment.kind, tuner.controller
    unknown = {
        key for key in tuner.controller_options if not supported_parameter(name, key)
    }
    if unknown:
        raise ValueError(f"Unknown {name} controller_options: {sorted(unknown)}")
    if name in ("aa_indi", "aidi"):
        if kind != "boeing" or not set(tuner.reference) <= {
            "phi",
            "theta",
            "psi",
            "p",
            "q",
            "r",
        }:
            raise ValueError(
                f"{name} requires nonlinear B737/B747 with Euler-angle or "
                f"body-rate steps"
            )
        for angle, rate in zip(("phi", "theta", "psi"), ("p", "q", "r")):
            if angle in tuner.reference and rate in tuner.reference:
                raise ValueError(
                    f"Specify either {angle} or {rate} on an axis, not both"
                )
    if (
        kind == "boeing"
        and name not in ("aa_indi", "aidi")
        and not set(tuner.reference)
        <= {"u", "v", "w", "p", "q", "r", "phi", "theta", "psi"}
    ):
        raise ValueError(
            "Boeing state-space profiles support velocities, body rates "
            "and attitudes; position guidance needs a separate profile"
        )
    if name == "hdp" and (kind != "hdp_b747" or list(tuner.reference) != ["theta"]):
        raise ValueError("HDP currently requires ImprovedB747-v0 and a theta step")
    if name != "hdp" and kind == "hdp_b747":
        raise ValueError(
            "ImprovedB747's normalized reward/observation adapter is "
            "currently reserved for HDP; use LinearLongitudinalB747 for other agents"
        )
    if name == "ihdp" and tuner.training_episodes:
        raise ValueError(
            "IHDP currently supports one online episode (training_episodes=0)"
        )


def _controls(physical, tracking):
    if physical.kind != "boeing":
        return list(range(len(physical.bias)))
    # The full virtual command always retains nominal throttle and unused surfaces.
    indices = set()
    if set(tracking) & {"u", "w", "q", "theta", "x_e", "z_e"}:
        indices.add(0)
    if set(tracking) & {"v", "p", "r", "phi", "psi", "y_e"}:
        indices.update((1, 2))
    if set(tracking) & {"u", "x_e", "z_e"}:
        indices.add(3)
    return sorted(indices)


def _healthy_boeing(physical):
    return type(physical.model)(
        x0=physical.initial.copy(), dt=physical.dt, config=physical.env.config
    )


class _InversionRunner:
    def __init__(self, tuner, physical, params):
        self.agent: Any
        from tensoraerospace.agent.aa_indi import (
            AAINDIAgent,
            AAINDIConfig,
            AircraftGeometry,
            FlightMeasurement,
            ObserverConfig,
        )
        from tensoraerospace.agent.aidi import AIDIAgent, AIDIConfig, LinearOnboardCE

        self.physical = physical
        self.name = tuner.controller
        self.tracking = tuple(tuner.reference)
        self.indices = _controls(physical, self.tracking)
        self.outer_gain = params.pop("outer_gain", 0.35)
        self.rate_limit = np.deg2rad(params.pop("rate_limit_deg", 3.0))
        if self.outer_gain <= 0 or self.rate_limit <= 0:
            raise ValueError("outer_gain and rate_limit_deg must be positive")
        model = _healthy_boeing(physical)
        _, B = model.linearize(physical.initial, physical.bias)
        B = B[3:6, self.indices]
        limit = float(
            np.min(np.minimum(-physical.low[self.indices], physical.high[self.indices]))
        )
        gain = params.pop("rate_gain", 3.0)
        cutoff = params.pop("cutoff_hz", 5.0)
        if self.name == "aa_indi":
            geometry = AircraftGeometry.from_parameters(model.param)
            sensor = FlightMeasurement.from_model(
                model, applied_action=physical.bias, surface_indices=self.indices
            )
            derivatives = np.column_stack(
                [
                    geometry.coefficients(
                        np.zeros(3), B[:, j], sensor.density, sensor.airspeed
                    )
                    for j in range(len(self.indices))
                ]
            )
            self.agent = AAINDIAgent(
                configure(
                    AAINDIConfig(
                        geometry=geometry,
                        nominal_derivatives=derivatives,
                        observer=ObserverConfig(
                            dt=physical.dt, gravity=model.param.g_ft_s2 * 0.3048
                        ),
                        covariance_init=params.pop("covariance_init", 1.0),
                        rate_feedback=np.full(3, gain),
                        acceleration_cutoff_hz=cutoff,
                        magnitude_limit=limit,
                        rate_limit=np.deg2rad(20.0),
                    ),
                    params,
                )
            )
        else:
            self.agent = AIDIAgent(
                3,
                len(self.indices),
                LinearOnboardCE(B),
                configure(
                    AIDIConfig(
                        dt=physical.dt,
                        rate_kp=(gain,) * 3,
                        sensor_cutoff_hz=cutoff,
                        u_magnitude_limit=limit,
                        u_rate_limit=np.deg2rad(20.0),
                        rls_cov_init=params.pop("rls_cov_init", 1.0),
                    ),
                    params,
                ),
            )
        if params:
            raise ValueError(f"Unused profile parameters: {params}")

    def _measurement(self, physical):
        from tensoraerospace.agent.aa_indi import FlightMeasurement

        return FlightMeasurement.from_model(
            physical.model,
            applied_action=physical.applied,
            surface_indices=self.indices,
        )

    def _observation(self, physical):
        x = physical.state
        return dict(
            omega=x[3:6].copy(),
            alpha=np.arctan2(x[2], x[0]),
            beta=np.arcsin(np.clip(x[1] / np.linalg.norm(x[:3]), -1, 1)),
            theta=x[7],
            phi=x[6],
            V=np.linalg.norm(x[:3]) * 0.3048,
        )

    def reset(self):
        if self.name == "aa_indi":
            self.agent.reset()
        else:
            self.agent.reset(initial_action=self.physical.bias[self.indices])

    def predict(self, physical, reference, k):
        x = physical.state
        targets = dict(zip(self.tracking, reference[k]))
        euler_rate = self.outer_gain * (
            np.array(
                [
                    targets.get(n, physical.initial[6 + i])
                    for i, n in enumerate(("phi", "theta", "psi"))
                ]
            )
            - x[6:9]
        )
        pd, td, yd = euler_rate
        phi, theta = x[6:8]
        rates = np.array(
            [
                pd - np.sin(theta) * yd,
                np.cos(phi) * td + np.sin(phi) * np.cos(theta) * yd,
                -np.sin(phi) * td + np.cos(phi) * np.cos(theta) * yd,
            ]
        )
        for i, n in enumerate(("p", "q", "r")):
            if n in targets:
                rates[i] = targets[n]
        rates = np.clip(rates, -self.rate_limit, self.rate_limit)
        if self.name == "aa_indi":
            command = self.agent.predict(self._measurement(physical), rates)
        else:
            command = self.agent.predict_rates(self._observation(physical), rates)
        action = physical.bias.copy()
        action[self.indices] = command
        return action

    def learn(self, physical, reference, k):
        if self.name == "aa_indi":
            self.agent.learn(
                self._measurement(physical),
                applied_action=physical.applied[self.indices],
            )
            coefficients = self.agent.identifier.derivatives
        else:
            self.agent.learn(
                self._observation(physical),
                {},
                applied_action=physical.applied[self.indices],
            )
            coefficients = self.agent.rls.theta
        if not np.isfinite(coefficients).all():
            raise TrialRejected("Nonfinite learned control effectiveness")


class _StateRunner:
    def __init__(self, tuner, physical, reference, params, seed):
        self.agent: Any
        self.name, self.physical = tuner.controller, physical
        self.indices = _controls(physical, tuner.reference)
        # Earth-fixed positions drift in level flight and are not regulation
        # states. These profiles control body velocities, rates and attitudes.
        self.state_indices = list(
            range(9 if physical.kind == "boeing" else len(physical.names))
        )
        self.names = [physical.names[i] for i in self.state_indices]
        self.tracking = [self.names.index(n) for n in tuner.reference]
        self.offset = physical.initial[self.state_indices].copy()
        # One degree for angular coordinates, one native unit otherwise.
        self.scale = np.array(
            [
                np.deg2rad(1.0) if u.startswith("rad") else 1.0
                for u in [physical.units[i] for i in self.state_indices]
            ]
        )
        for j, i in enumerate(self.tracking):
            self.scale[i] = abs(reference[-1, j] - reference[0, j])
        self.reference = (
            (reference - self.offset[self.tracking]) / self.scale[self.tracking]
        ).T
        self._initialize_agent(tuner, physical, params, seed)

    def _initialize_agent(self, tuner, physical, params, seed):
        cfg: Any
        n, m, r = len(self.scale), len(self.indices), len(self.tracking)
        self.horizon = params.get("horizon", 20)
        if self.name in ("iadp", "mpc", "et_dhp"):
            A, B = self._nominal_model(physical)
        if self.name == "iadp":
            self.agent = self._make_iadp(A, B, n, m, r, physical, params, seed)
        elif self.name == "imgdhp":
            from tensoraerospace.agent.im_gdhp import IMGDHPAgent, IMGDHPConfig

            cfg = IMGDHPConfig(
                actor_lr=params.pop("actor_lr", 1e-4),
                critic_lr=params.pop("critic_lr", 1e-3),
                history_length=params.pop("history_length", 4),
                identifier_mode="output",
                actor_hidden=params.pop("actor_hidden", (16,)),
                critic_hidden=params.pop("critic_hidden", (16,)),
                warmup_steps=params.pop("warmup_steps", 5),
                track_Q=(params.pop("track_weight", 1.0),) * r,
                control_R=(params.pop("control_weight", 0.01),) * m,
                u_max=1.0,
                exploration_noise_std=params.pop("exploration_noise_std", 0.01),
                seed=seed,
            )
            self.agent = IMGDHPAgent(n, m, r, self.tracking, configure(cfg, params))
        elif self.name == "et_dhp":
            # Augment the regulation state by the known current reference.
            # This makes nominal constant-reference dynamics Markov, instead
            # of training a plant model on an unmodelled changing offset.
            from scipy.linalg import block_diag

            from tensoraerospace.agent.et_dhp import ETDHPAgent, ETDHPConfig

            F = block_diag(A, np.eye(r))
            G = np.vstack((B, np.zeros((r, m))))
            transform = np.eye(n + r)
            transform[self.tracking, n + np.arange(r)] = -1.0
            inverse = np.linalg.inv(transform)
            F, G = transform @ F @ inverse, transform @ G

            def regulation(obs, ref, k):
                return transform @ np.concatenate(
                    (np.asarray(obs).reshape(-1), ref[:, k])
                )

            q = np.full(n + r, 1e-6)
            q[self.tracking] = params.pop("track_weight", 1.0)
            cfg = ETDHPConfig(
                actor_lr=params.pop("actor_lr", 1e-3),
                critic_lr=params.pop("critic_lr", 1e-3),
                rho=params.pop("rho", 0.1),
                Q=tuple(q),
                R=(params.pop("control_weight", 0.1),) * m,
                model_epochs=params.pop("model_epochs", 200),
                num_epochs_per_trigger=params.pop("num_epochs_per_trigger", 2),
                online_model_fit=params.pop("online_model_fit", False),
                seed=seed,
            )
            self.agent = ETDHPAgent(
                n + r, m, state_transform=regulation, config=configure(cfg, params)
            )
            rng = np.random.default_rng(seed)
            samples = params.pop("model_samples", 512)
            xs = rng.uniform(-1.0, 1.0, (samples, n + r))
            us = rng.uniform(-1.0, 1.0, (samples, m))
            self.agent.fit_plant_model(xs, us, xs @ F.T + us @ G.T)
        elif self.name == "ihdp":
            self.agent = _make_ihdp(
                tuner, physical, self.names, self.tracking, self.indices, params, seed
            )
        elif self.name == "mpc":
            import torch

            from tensoraerospace.agent.mpc import MPC, MPCConstraints, MPCWeights

            At, Bt = torch.tensor(A, dtype=torch.float32), torch.tensor(
                B, dtype=torch.float32
            )

            def dynamics(x, u):
                return x @ At.T + u @ Bt.T

            q = np.zeros(n)
            q[self.tracking] = params.pop("track_weight", 1.0)
            self.horizon = params.pop("horizon", 20)
            self.agent = MPC(
                dynamics=dynamics,
                state_dim=n,
                action_dim=m,
                horizon=self.horizon,
                weights=configure(
                    MPCWeights(
                        Q_diag=q,
                        R_diag=np.full(m, params.pop("control_weight", 0.01)),
                        terminal_weight=params.pop("terminal_weight", 1.0),
                        S_diag=np.zeros(m),
                    ),
                    params,
                    "weights",
                ),
                constraints=MPCConstraints(u_min=np.full(m, -1.0), u_max=np.ones(m)),
                iters=params.pop("iters", 30),
                lr=params.pop("lr", 0.05),
                optimizer=params.pop("optimizer", "adam"),
                warm_start=params.pop("warm_start", True),
                track_best=params.pop("track_best", True),
                best_check_every=params.pop("best_check_every", 1),
                seed=seed,
            )
        if params:
            raise ValueError(f"Unused profile parameters: {params}")

    def _make_iadp(self, A, B, n, m, r, physical, params, seed):
        from scipy.linalg import block_diag, solve_discrete_are

        from tensoraerospace.agent.iadp import IADPAgent, IADPConfig

        gamma = params.pop("gamma", 0.98)
        C = np.eye(n)[self.tracking]
        F = block_diag(A, np.eye(r))
        G = np.vstack((B, np.zeros((r, m))))
        E = np.column_stack((C, -np.eye(r)))
        tracking_Q = np.eye(r) * params.pop("track_weight", 1.0)
        R = np.eye(m) * params.pop("control_weight", 0.1)
        cfg = IADPConfig(
            dt=physical.dt,
            Q=tracking_Q,
            R=R,
            gamma=gamma,
            gamma_rls=params.pop("forgetting", 0.995),
            phi_init=params.pop("phi_init", 1.0),
            policy_eval_every=params.pop("policy_eval_every", 50),
            policy_eval_window=params.pop("policy_eval_window", 200),
            n_reference=r,
            output_matrix=C,
            F_init=F,
            G_init=G,
            P_init=None,
            u_magnitude_limit=1.0,
            u_rate_limit=100.0,
            seed=seed,
        )
        cfg = configure(cfg, params)
        Q = E.T @ cfg.Q @ E + np.eye(n + r) * 1e-8
        try:
            cfg.P_init = solve_discrete_are(
                np.sqrt(cfg.gamma) * F, np.sqrt(cfg.gamma) * G, Q, cfg.R
            )
        except np.linalg.LinAlgError as exc:
            raise TrialRejected(
                "No finite nominal value prior for this candidate"
            ) from exc
        return IADPAgent(n, m, config=cfg)

    def _nominal_model(self, physical):
        A, B = physical.nominal_discrete(self.indices)
        A = A[np.ix_(self.state_indices, self.state_indices)]
        B = B[self.state_indices]
        if not np.any(abs(B) > 1e-12):
            raise ValueError(
                "No nominal control effect in observed states; include "
                "actuator states in state_space or use integrator='rk4'"
            )
        A = A * self.scale[None, :] / self.scale[:, None]
        B = B / self.scale[:, None]
        return A, B

    def _state(self, physical):
        return (physical.state[self.state_indices] - self.offset) / self.scale

    def reset(self):
        if self.name != "ihdp":
            self.agent.reset()
        self.previous = np.zeros(len(self.indices))

    def predict(self, physical, reference, k):
        x = self._state(physical)
        if self.name == "mpc":
            target = np.zeros(len(x))
            target[self.tracking] = self.reference[:, k]
            action = self.agent.solve(
                x0=x, x_ref=np.tile(target, (self.horizon + 1, 1)), u_prev=self.previous
            ).u0
        else:
            action = self.agent.predict(
                x.reshape(-1, 1) if self.name == "ihdp" else x, self.reference, k
            )
        self.previous = np.asarray(action).reshape(-1)
        result = physical.bias.copy()
        result[self.indices] += physical.scale[self.indices] * self.previous
        return result

    def learn(self, physical, reference, k):
        if self.name == "et_dhp":
            self.agent.learn(
                self._state(physical),
                self.reference,
                k,
                dt=physical.dt,
                applied_action=(
                    physical.applied[self.indices] - physical.bias[self.indices]
                )
                / physical.scale[self.indices],
            )
        elif self.name in ("iadp", "imgdhp"):
            applied = (
                physical.applied[self.indices] - physical.bias[self.indices]
            ) / physical.scale[self.indices]
            self.agent.learn(
                self._state(physical), self.reference, k, applied_action=applied
            )
        _check_learning_state(self.agent)


def _check_learning_state(agent):
    """Reject a numerical failure even if it occurs on the last update."""
    for name in ("actor", "critic", "plant_model"):
        component = getattr(agent, name, None)
        network = getattr(component, "model", component)
        if network is not None and hasattr(network, "parameters"):
            if any(
                not np.isfinite(p.detach().cpu().numpy()).all()
                for p in network.parameters()
            ):
                raise TrialRejected("Nonfinite learned network weights")
    for component in (
        agent,
        getattr(agent, "rls", None),
        getattr(agent, "incremental_model", None),
    ):
        for name in ("P", "F", "G", "theta", "Phi"):
            value = getattr(component, name, None)
            if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                raise TrialRejected(f"Nonfinite learned {name}")


def _make_ihdp(tuner, physical, names, tracking, indices, params, seed):
    from tensoraerospace.agent.ihdp import IHDPAgent

    n_actions = len(indices)
    hidden = params.pop("hidden_size", 8)
    warmup = params.pop("warmup_steps", 5)
    actor_hidden = tuple(params.pop("actor_hidden", (hidden,)))
    critic_hidden = tuple(params.pop("critic_hidden", (hidden,)))
    actor = dict(
        start_training=warmup,
        layers=(*actor_hidden, n_actions),
        activations=("tanh",) * (len(actor_hidden) + 1),
        learning_rate=params.pop("actor_lr", 1e-3),
        learning_rate_exponent_limit=5,
        type_PE="3211",
        amplitude_3211=params.pop("excitation_amplitude", 0.01),
        pulse_length_3211=5,
        maximum_input=1.0,
        maximum_q_rate=1.0,
        WB_limits=5.0,
        NN_initial=seed,
        cascade_actor=False,
        learning_rate_cascaded=0.001,
        learning_rate_min=0.001,
        learning_rate_decay=0.995,
    )
    critic = dict(
        Q_weights=[params.pop("track_weight", 1.0)] * len(tracking),
        start_training=warmup,
        gamma=params.pop("gamma", 0.95),
        learning_rate=params.pop("critic_lr", 1e-3),
        learning_rate_min=1e-6,
        learning_rate_decay=0.995,
        learning_rate_exponent_limit=5,
        layers=(*critic_hidden, 1),
        activations=("tanh",) * len(critic_hidden) + ("linear",),
        indices_tracking_states=tracking,
        WB_limits=5.0,
        NN_initial=seed,
    )
    incremental = dict(
        number_time_steps=tuner.environment.steps + 1,
        dt=physical.dt,
        input_magnitude_limits=1.0,
        window_size=2 * (len(names) + n_actions),
        input_rate_limits=(
            (np.asarray(physical.model.input_rate_limits) / physical.scale)[indices]
            if physical.kind == "linear"
            else 100.0
        ),
    )
    actor = configure(actor, params, "actor_settings")
    critic = configure(critic, params, "critic_settings")
    incremental = configure(incremental, params, "incremental_settings")
    for settings, output_dim in ((actor, n_actions), (critic, 1)):
        if settings["layers"][-1] != output_dim or len(settings["layers"]) != len(
            settings["activations"]
        ):
            raise ValueError(
                "IHDP layers/activations must preserve the controller's "
                "output dimension"
            )
    return IHDPAgent(
        actor,
        critic,
        incremental,
        list(tuner.reference),
        names,
        [f"action_{i}" for i in indices],
        tuner.environment.steps + 1,
        tracking,
    )


def build_controller(tuner, physical, reference, params, seed):
    settings = ParameterValues(
        copy.deepcopy(tuner.controller_options), copy.deepcopy(params)
    )
    if tuner.controller in ("aa_indi", "aidi"):
        return _InversionRunner(tuner, physical, settings)
    return _StateRunner(tuner, physical, reference, settings, seed)


def run_hdp(tuner, physical, reference, params, seed):
    """Run the native HDP training loop and record its last online episode."""
    from tempfile import TemporaryDirectory

    import gymnasium as gym

    from tensoraerospace.agent.hdp import HDP

    class RecordingEnv(gym.Wrapper):
        def __getattr__(self, name):
            return getattr(self.env, name)

        def reset(self, *, seed=None, options=None):
            physical.reset(seed)
            tuner._check_state(physical)
            self.outputs = [physical.state[[3]].copy()]
            self.actions = []
            return physical.observation.copy(), {}

        def step(self, action):
            observation, reward, terminated, truncated, info = self.env.step(action)
            physical.observation = np.asarray(observation)
            tuner._check_state(physical)
            self.outputs.append(physical.state[[3]].copy())
            self.actions.append(np.asarray(action).copy())
            if (terminated or truncated) and len(
                self.actions
            ) < tuner.environment.steps:
                raise TrialRejected(
                    "HDP environment ended before the requested horizon"
                )
            return observation, reward, terminated, truncated, info

    env = RecordingEnv(physical.env)
    options = {**tuner.controller_options, **params}
    with TemporaryDirectory(prefix="tensoraerospace-hdp-") as log_dir:
        agent = HDP(
            env,
            seed=seed,
            hidden_size=options.pop("hidden_size", 32),
            exploration_std=options.pop("exploration_std", 0.01),
            log_every_updates=1,
            log_dir=log_dir,
            **options,
        )
        try:
            agent.train(
                num_episodes=tuner.training_episodes + 1,
                max_steps=tuner.environment.steps,
                verbose=False,
            )
            _check_learning_state(agent)
            return env.outputs, env.actions
        finally:
            agent.writer.close()
