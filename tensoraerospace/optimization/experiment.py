"""Declarative step experiments: public environment, controller and search settings."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna

from .control import ControlOptimizer, _random_state, _seeds
from .metrics import StepResponseMetric, TrialRejected
from .samplers import make_sampler


@dataclass(frozen=True)
class Step:
    """One step in a named physical state.

    ``amplitude`` is the change, not the final level. ``initial=None`` holds
    the environment's initial state (e.g. trim); otherwise initial is absolute.
    ``at`` is seconds; ``unit`` is native, rad, or deg, validated per state.
    """

    amplitude: float
    at: float = 20.0
    initial: float | None = None
    unit: str = "native"

    def __post_init__(self):
        if (
            not np.isfinite([self.amplitude, self.at]).all()
            or self.amplitude == 0
            or self.at <= 0
        ):
            raise ValueError("A step needs finite nonzero amplitude and at > 0 seconds")
        if self.initial is not None and not np.isfinite(self.initial):
            raise ValueError("initial must be finite or None")
        if self.unit not in ("native", "rad", "deg"):
            raise ValueError("Step.unit must be native, rad or deg")

    def values(self, time, initial, native_unit):
        """Sample the step in native units, using the initial plant state as baseline.

        ``time`` and ``at`` are in seconds. The result is a one-dimensional reference;
        ``amplitude`` specifies the change rather than the final level.
        """
        factor = 1.0
        native_unit = native_unit.removesuffix("/s")
        if self.unit != "native":
            if native_unit not in ("rad", "deg"):
                raise ValueError(f"Cannot express a {native_unit} state in {self.unit}")
            if self.unit != native_unit:
                factor = np.pi / 180 if self.unit == "deg" else 180 / np.pi
        baseline = initial if self.initial is None else self.initial * factor
        return (
            baseline
            + ((time >= self.at) | np.isclose(time, self.at, rtol=1e-12, atol=1e-12))
            * self.amplitude
            * factor
        )


@dataclass
class ControlRun:
    """Complete physical trajectory and benchmark results."""

    time: np.ndarray
    reference: np.ndarray
    output: np.ndarray
    actions: np.ndarray
    metrics: dict
    states: tuple[str, ...]
    units: tuple[str, ...]

    def plot(self, *, baseline=None):
        """Plot commanded states, measured responses and native actuator commands.

        An optional ``baseline`` must have matching states, units, times and references.
        Angular channels are displayed in degrees. Return the Matplotlib figure without
        displaying or saving it.
        """
        import matplotlib.pyplot as plt

        if baseline is not None and (
            baseline.states != self.states
            or baseline.units != self.units
            or not np.array_equal(baseline.time, self.time)
            or not np.array_equal(baseline.reference, self.reference)
        ):
            raise ValueError(
                "Baseline must use the same states, units, time and reference"
            )
        fig, axes = plt.subplots(
            len(self.states) + 1,
            1,
            figsize=(11, 3 * (len(self.states) + 1)),
            sharex=True,
            constrained_layout=True,
        )
        for i, (name, unit) in enumerate(zip(self.states, self.units)):
            factor = 180 / np.pi if unit.startswith("rad") else 1.0
            display_unit = unit.replace("rad", "deg")
            axes[i].step(
                self.time,
                self.reference[:, i] * factor,
                where="post",
                color="#b45309",
                linestyle="--",
                label="Reference",
            )
            if baseline is not None:
                axes[i].plot(
                    self.time,
                    baseline.output[:, i] * factor,
                    color="#64748b",
                    alpha=0.85,
                    label="Baseline",
                )
            axes[i].plot(
                self.time, self.output[:, i] * factor, color="#0369a1", label="Response"
            )
            axes[i].set_ylabel(f"{name} [{display_unit}]")
            axes[i].legend()
            axes[i].grid(alpha=0.2)
        axes[-1].plot(self.time[1:], self.actions)
        axes[-1].set(
            xlabel="Time [s]",
            ylabel="Environment actions",
            title="Commands sent to the environment (native units)",
        )
        axes[-1].grid(alpha=0.2)
        return fig


@dataclass
class TuningResult:
    """Selected parameters with automatic replay and response plotting."""

    optimization: Any
    best_run: ControlRun
    _tuner: Any = field(repr=False)

    @property
    def best_params(self):
        """Return the selected feasible trial's controller parameter mapping."""
        return self.optimization.best_params

    @property
    def best_value(self):
        """Return the minimized aggregate score of the selected feasible trial."""
        return self.optimization.best_value

    @property
    def study(self):
        """Return the Optuna study containing all successful and rejected trials."""
        return self.optimization.study

    def simulate(self, *, seed=0):
        """Replay the selected parameters with a fresh controller and the given seed."""
        return self._tuner.simulate(self.best_params, seed=seed)

    def plot_response(self, *, baseline=None):
        """Plot the selected trajectory against its reference and an optional baseline."""
        return self.best_run.plot(baseline=baseline)

    def plot_history(self):
        """Return the optimization history figure for this result's study."""
        return self.optimization.plot_history()

    def save(self, path):
        """Export parameters and per-case metrics to JSON; no trained weights are saved."""
        self.optimization.save(path)

    def _active_tuner(self):
        """Require this result to belong to the tuner's current study before resuming."""
        if (
            self._tuner.optimizer is None
            or self._tuner.optimizer.study is not self.study
        ):
            raise RuntimeError(
                "This result belongs to an older study; use the current result or "
                "load its saved checkpoint"
            )
        return self._tuner

    def resume(self, n_trials=30, **options):
        """Continue this study with an additional trial budget."""
        return self._active_tuner().resume(n_trials, **options)

    def save_checkpoint(self, path):
        """Save the experiment, history and live sampler for later continuation."""
        self._active_tuner().save_checkpoint(path)


def _validate_experiment_options(reference, duration, training_episodes, torch_threads):
    """Validate named steps, simulation duration and training/thread budgets."""
    if not reference or not all(
        isinstance(k, str) and isinstance(v, Step) for k, v in reference.items()
    ):
        raise ValueError("reference must map physical state names to Step objects")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if (
        isinstance(training_episodes, bool)
        or not isinstance(training_episodes, int)
        or training_episodes < 0
    ):
        raise ValueError("training_episodes must be a nonnegative integer")
    if torch_threads is not None and (
        isinstance(torch_threads, bool)
        or not isinstance(torch_threads, int)
        or torch_threads < 1
    ):
        raise ValueError("torch_threads must be a positive integer or None")


class ControllerTuner:
    """Optimize a controller from environment and step settings, without callbacks.

    Built-in profiles select constructor defaults, search ranges, state/action
    adapters and the native online learning loop. Unsupported combinations fail
    before a search starts. ``available_controllers`` documents compatibility.
    All trials use the same complete horizon and optional training budget.
    """

    def __init__(
        self,
        *,
        env,
        env_kwargs=None,
        reference,
        controller,
        method="tpe",
        metric="cpi",
        duration=40.0,
        controller_options=None,
        search_space=None,
        method_options=None,
        seeds=(0,),
        seed=0,
        training_episodes=0,
        normalize=True,
        constraints=None,
        state_limits=None,
        torch_threads=1,
    ):
        from ._controllers import controller_name, validate_controller
        from ._environment import ExperimentEnvironment

        _validate_experiment_options(
            reference, duration, training_episodes, torch_threads
        )
        self.torch_threads = torch_threads
        self.reference = copy.deepcopy(dict(reference))
        self.controller = controller_name(controller)
        self.controller_options = copy.deepcopy(dict(controller_options or {}))
        self.method, self.metric = method, metric.lower()
        self.duration, self.normalize = float(duration), bool(normalize)
        self.training_episodes = training_episodes
        self.seeds, self.seed = _seeds(seeds), _seeds([seed])[0]
        self.constraints = dict(constraints or {})
        self.state_limits = copy.deepcopy(dict(state_limits or {}))
        self.environment = ExperimentEnvironment(
            env, env_kwargs or {}, self.duration, self.reference
        )
        self.time = np.arange(self.environment.steps + 1) * self.environment.dt
        for step in self.reference.values():
            if step.at >= self.duration or not np.isclose(
                step.at / self.environment.dt, round(step.at / self.environment.dt)
            ):
                raise ValueError("Step times must align with dt and precede duration")
        validate_controller(self)
        self._configure_search_space(search_space)
        self.method_options = dict(method_options or {})
        # Validate sampler settings and parameter paths before any simulation budget.
        make_sampler(method, self.search_space, seed=seed, options=self.method_options)
        self._validate_physical_experiment()
        self.optimizer = None
        self._search_configuration = None
        self._search_n_jobs = None

    def _configure_search_space(self, search_space):
        """Resolve search dimensions and reject aliases that fixed options override."""
        from ._controllers import default_space
        from ._parameters import overlapping, validate_search_parameters

        self.search_space = dict(
            {
                k: v
                for k, v in default_space(self.controller).items()
                if not any(
                    overlapping(self.controller, k, fixed)
                    for fixed in self.controller_options
                )
            }
            if search_space is None
            else search_space
        )
        validate_search_parameters(self.controller, self.search_space)
        # A fixed native setting cannot silently overwrite a sampled shortcut.
        for searched in self.search_space:
            for fixed in self.controller_options:
                if (
                    "." in fixed
                    and "." not in searched
                    and overlapping(self.controller, searched, fixed)
                ):
                    raise ValueError(
                        f"Fixed native parameter {fixed!r} overrides searched alias "
                        f"{searched!r}; use the native path in search_space"
                    )
        if self.controller == "hdp":
            weight_keys = {"dhp_w_theta", "dhp_w_q", "dhp_w_u", "dhp_w_du"}
            if weight_keys & (
                self.controller_options.keys() | self.search_space.keys()
            ):
                if (
                    self.controller_options.get("dhp_use_env_cost") is not False
                    or "dhp_use_env_cost" in self.search_space
                ):
                    raise ValueError(
                        "Set controller_options['dhp_use_env_cost']=False to tune HDP "
                        "utility weights; otherwise environment weights are used"
                    )

    def _validate_physical_experiment(self):
        """Check nominal control effects, metric names and declared state limits."""
        from ._controllers import supported_parameter

        extra = {
            key
            for key in self.search_space
            if not supported_parameter(self.controller, key)
        }
        if extra:
            raise ValueError(
                f"Unknown tuning parameters for {self.controller}: {sorted(extra)}; "
                f"see ControllerTuner.profile({self.controller!r}) for available parameters"
            )
        with self.environment.open(self.seed) as physical:
            reference_values = self._reference_for(physical)
            if self.controller in ("iadp", "mpc", "et_dhp"):
                from ._controllers import _controls

                _, nominal_input = physical.nominal_discrete(
                    _controls(physical, self.reference)
                )
                if not np.any(abs(nominal_input) > 1e-12):
                    raise ValueError(
                        "No nominal input effect in observed states: include actuator states or use integrator='rk4'"
                    )
            known_metrics = set(
                StepResponseMetric(reference_values, self.environment.dt)(
                    reference_values
                )
            )
            channel_metrics = set(
                StepResponseMetric(reference_values[:, 0], self.environment.dt)(
                    reference_values[:, 0]
                )
            )
            known_metrics.update(
                f"{state}.{key}" for state in self.reference for key in channel_metrics
            )
            for key in [self.metric, *self.constraints]:
                if key not in known_metrics:
                    raise ValueError(
                        f"Unknown metric {key!r}; choose from {sorted(known_metrics)}"
                    )
            for name, bounds in self.state_limits.items():
                if (
                    name not in physical.names
                    or len(bounds) != 2
                    or not np.isfinite(bounds).all()
                    or bounds[0] >= bounds[1]
                ):
                    raise ValueError(f"Invalid physical state limit for {name!r}")

    @staticmethod
    def available_controllers():
        """Return canonical controller names and their supported environment families."""
        return {
            "iadp": "Physical linear F16/B747 and nonlinear B737/B747/F16",
            "imgdhp": "Physical linear F16/B747 and nonlinear B737/B747/F16",
            "et_dhp": "Physical linear F16/B747 and nonlinear B737/B747/F16",
            "ihdp": "Physical linear F16/B747 and nonlinear B737/B747/F16; one online episode",
            "mpc": "Physical linear F16/B747 and nonlinear B737/B747/F16; nominal linear MPC",
            "aa_indi": "Nonlinear B737/B747; Euler angles or body rates",
            "aidi": "Nonlinear B737/B747; Euler angles or body rates via native rate boundary",
            "hdp": "ImprovedB747-v0, theta; native HDP training loop",
        }

    @staticmethod
    def profile(controller):
        """List supported tuning parameters and fixed options without building an agent."""
        from ._controllers import _OPTIONS, controller_name, default_space
        from ._parameters import native_parameters

        name = controller_name(controller)
        return {
            "controller": name,
            "search_space": default_space(name),
            "controller_options": sorted(_OPTIONS[name] | set(default_space(name))),
            "tunable_parameters": sorted(
                _OPTIONS[name] | set(default_space(name)) | set(native_parameters(name))
            ),
            "native_parameters": native_parameters(name),
            "compatibility": ControllerTuner.available_controllers()[name],
        }

    def _reference_for(self, physical):
        """Build sample-by-channel steps in the physical environment's native units."""
        values = []
        for name, step in self.reference.items():
            if name not in physical.names:
                raise ValueError(
                    f"Unknown state {name!r}; environment states: {physical.names}"
                )
            i = physical.names.index(name)
            values.append(step.values(self.time, physical.state[i], physical.units[i]))
        reference = np.column_stack(values)
        StepResponseMetric(reference, self.environment.dt, normalize=self.normalize)
        return reference

    def _check_state(self, physical):
        """Reject nonfinite states, envelope violations and user-specified limit
        breaches.
        """
        state = physical.state
        if not np.isfinite(state).all():
            raise TrialRejected("Nonfinite physical state")
        physical.validate_envelope()
        for name, (lower, upper) in self.state_limits.items():
            if not lower <= state[physical.names.index(name)] <= upper:
                raise TrialRejected(f"State {name} left [{lower}, {upper}]")

    def simulate(self, params=None, *, seed=0):
        """Automatically run training and the full scored online episode."""
        from ._controllers import build_controller, run_hdp, supported_parameter

        params = dict(params or {})
        unknown = {
            key for key in params if not supported_parameter(self.controller, key)
        }
        if unknown:
            raise ValueError(f"Unknown trial parameters: {sorted(unknown)}")
        from ._parallel import torch_thread_limit

        with torch_thread_limit(self.torch_threads), _random_state(_seeds([seed])[0]):
            with self.environment.open(seed) as physical:
                reference = self._reference_for(physical)
                physical.set_reference(reference, tuple(self.reference))
                if self.controller == "hdp":
                    outputs, actions = run_hdp(self, physical, reference, params, seed)
                else:
                    runner = build_controller(self, physical, reference, params, seed)
                    for episode in range(self.training_episodes + 1):
                        if episode:
                            physical.reset(seed)
                        runner.reset()
                        self._check_state(physical)
                        indices = [
                            physical.names.index(name) for name in self.reference
                        ]
                        outputs, actions = [physical.state[indices].copy()], []
                        for k in range(self.environment.steps):
                            action = runner.predict(physical, reference, k)
                            if not np.isfinite(action).all():
                                raise TrialRejected("Nonfinite controller action")
                            terminated, truncated = physical.step(action)
                            self._check_state(physical)
                            if (
                                terminated or truncated
                            ) and k + 1 < self.environment.steps:
                                raise TrialRejected(
                                    "Environment ended before the requested horizon"
                                )
                            runner.learn(physical, reference, k)
                            outputs.append(physical.state[indices].copy())
                            actions.append(physical.last_action.copy())
                output, actions = np.asarray(outputs), np.asarray(actions)
                metrics = StepResponseMetric(
                    reference, self.environment.dt, normalize=self.normalize
                )(output)
                for i, name in enumerate(self.reference):
                    for key, value in StepResponseMetric(
                        reference[:, i], self.environment.dt, normalize=self.normalize
                    )(output[:, i]).items():
                        metrics[f"{name}.{key}"] = value
                units = tuple(
                    physical.units[physical.names.index(name)]
                    for name in self.reference
                )
        return ControlRun(
            self.time.copy(),
            reference,
            output,
            actions,
            metrics,
            tuple(self.reference),
            units,
        )

    def diagnostics(self):
        """Inspect trial failures and observed constraint values, including after an error."""
        if self.optimizer is None:
            raise RuntimeError("Call optimize() before requesting search diagnostics")
        return self.optimizer.diagnostics()

    def _configuration(self):
        """All settings that determine comparable evaluations and sampler draws."""
        return {
            "environment": vars(self.environment),
            "reference": self.reference,
            "controller": self.controller,
            "controller_options": self.controller_options,
            "search_space": self.search_space,
            "method": self.method,
            "method_options": self.method_options,
            "metric": self.metric,
            "normalize": self.normalize,
            "constraints": self.constraints,
            "state_limits": self.state_limits,
            "duration": self.duration,
            "time": self.time,
            "training_episodes": self.training_episodes,
            "seeds": self.seeds,
            "seed": self.seed,
            "torch_threads": self.torch_threads,
        }

    def _check_continuation(self):
        """Require unchanged experiment settings and no running trials before resuming."""
        from ._continuation import same_setting

        if self.optimizer is None or self._search_configuration is None:
            raise RuntimeError("Call optimize() before resume() or save_checkpoint()")
        changed = [
            key
            for key, value in self._configuration().items()
            if not same_setting(value, self._search_configuration[key])
        ]
        if changed:
            raise ValueError(
                f"Cannot continue: experiment settings changed: {', '.join(changed)}. "
                "Restore the original settings or call optimize() for a new study."
            )
        if any(
            trial.state == optuna.trial.TrialState.RUNNING
            for trial in self.optimizer.study.trials
        ):
            raise RuntimeError("Wait for the active search to stop before continuing")
        return self.optimizer

    def _validate_n_jobs(self, n_jobs):
        """Validate the worker count and enforce serial proposals for annealing."""
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer")
        if n_jobs > 1 and self.method == "annealing":
            raise ValueError(
                "Annealing requires n_jobs=1: each proposal depends on the previous "
                "result; use tpe, genetic or random for parallel search"
            )

    def _make_optimizer(self, n_jobs, *, study=None):
        """Build a serial or process-based evaluator, optionally reusing a study."""

        def objective(params, seed):
            """Return metrics from one complete seeded controller experiment."""
            return self.simulate(params, seed=seed).metrics

        if study is None:
            options = dict(self.method_options)
            if n_jobs > 1 and self.method == "tpe":
                # Avoid sampling near candidates already in the current batch.
                options.setdefault("constant_liar", True)
            sampler = make_sampler(
                self.method, self.search_space, seed=self.seed, options=options
            )
        else:
            # Keep RNG, annealing temperature/accepted point and sampler options.
            sampler = study.sampler
        optimizer_class: type[ControlOptimizer] = ControlOptimizer
        parallel_options = {}
        if n_jobs > 1:
            from ._parallel import ParallelControllerOptimizer

            optimizer_class = ParallelControllerOptimizer
            parallel_options = {"tuner": self, "n_jobs": n_jobs}
        optimizer = optimizer_class(
            self.search_space,
            objective,
            metric=self.metric,
            constraints=self.constraints,
            seeds=self.seeds,
            seed=self.seed,
            sampler=sampler,
            # Annealing/genetic fitness needs comparable complete-case scores.
            pruner=optuna.pruners.NopPruner(),
            **parallel_options,
        )
        if study is not None:
            optimizer.study = study
        optimizer.study.set_user_attr(
            "execution", {"n_jobs": n_jobs, "torch_threads": self.torch_threads}
        )
        return optimizer

    def _run_search(self, n_trials, **options):
        """Run or continue the configured search and replay its best feasible
        parameters.
        """
        assert self.optimizer is not None
        result = self.optimizer.optimize(n_trials, **options)
        return TuningResult(
            result, self.simulate(result.best_params, seed=self.seeds[0]), self
        )

    def optimize(
        self,
        n_trials=30,
        *,
        timeout=None,
        target=None,
        patience=None,
        initial_params=None,
        show_progress_bar=True,
        n_jobs=1,
    ):
        """Start a new study, search, then replay the winner; use resume to continue.

        ``show_progress_bar=False`` disables the notebook/terminal progress bar.
        ETA estimates the remaining search budget after completed trials; the
        automatic replay of the selected controller follows the search.
        ``n_jobs>1`` evaluates reproducible batches in isolated CPU processes
        (TPE, genetic or random). Running batches finish before early stopping;
        a batch uses up to n_jobs candidates. Annealing requires n_jobs=1.
        """
        self._validate_n_jobs(n_jobs)
        self.optimizer = self._make_optimizer(n_jobs)
        self._search_configuration = copy.deepcopy(self._configuration())
        self._search_n_jobs = n_jobs
        return self._run_search(
            n_trials,
            timeout=timeout,
            target=target,
            patience=patience,
            initial_params=initial_params,
            show_progress_bar=show_progress_bar,
        )

    def resume(
        self,
        n_trials=30,
        *,
        timeout=None,
        target=None,
        patience=None,
        initial_params=None,
        show_progress_bar=True,
        n_jobs=None,
    ):
        """Add trials to the existing study without resetting history or sampler.

        ``n_trials`` is an additional budget. ``n_jobs=None`` keeps the previous
        worker count. Target considers the best of all trials; patience/timeout
        start a new budget for this call. The experiment must remain unchanged.
        This continues parameter search, not a selected policy's neural weights.
        """
        optimizer = self._check_continuation()
        jobs = self._search_n_jobs if n_jobs is None else n_jobs
        self._validate_n_jobs(jobs)
        if jobs != self._search_n_jobs:
            self.optimizer = self._make_optimizer(jobs, study=optimizer.study)
            self._search_n_jobs = jobs
        return self._run_search(
            n_trials,
            timeout=timeout,
            target=target,
            patience=patience,
            initial_params=initial_params,
            show_progress_bar=show_progress_bar,
        )

    def find_best_response(
        self,
        n_trials=30,
        *,
        timeout=None,
        patience=None,
        initial_params=None,
        show_progress_bar=True,
        n_jobs=None,
    ):
        """Find the lowest observed response metric within an additional budget.

        Starts a study on the first call and resumes it thereafter. There is no
        stopping target: the configured metric (CPI by default) is minimized
        until the trial/time budget or optional patience is exhausted. Existing
        constraints and physical checks remain mandatory. If every candidate is
        rejected, diagnostics remain available and no infeasible result wins.

        ``n_jobs=None`` selects one worker initially, then retains the existing
        worker count. Use optimize() to deliberately start a fresh search. This
        returns the best tested feasible response, not a global-optimum claim.
        """
        options = dict(
            timeout=timeout,
            patience=patience,
            initial_params=initial_params,
            show_progress_bar=show_progress_bar,
        )
        if self.optimizer is None:
            return self.optimize(
                n_trials, n_jobs=1 if n_jobs is None else n_jobs, **options
            )
        return self.resume(n_trials, n_jobs=n_jobs, **options)

    def save_checkpoint(self, path):
        """Atomically save this stopped search, including the sampler and RNG.

        Unlike save()'s JSON report, this pickle checkpoint can resume after a
        kernel restart. Load only your own trusted files in a compatible SDK /
        Python / Optuna environment. Trained policy weights are not stored.
        """
        from ._continuation import write_checkpoint

        optimizer = self._check_continuation()
        experiment = copy.copy(self)
        experiment.optimizer = None  # Closures/process executors are not serialized.
        write_checkpoint(
            path,
            {
                "format": "tensoraerospace.controller-search",
                "version": 1,
                "tuner": experiment,
                "study": optimizer.study,
            },
        )

    @classmethod
    def load_checkpoint(cls, path):
        """Restore a trusted pickle checkpoint without running any simulations."""
        import pickle
        from pathlib import Path

        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        if (
            not isinstance(payload, dict)
            or payload.get("format") != "tensoraerospace.controller-search"
            or payload.get("version") != 1
            or not isinstance(payload.get("tuner"), cls)
            or not isinstance(payload.get("study"), optuna.Study)
        ):
            raise ValueError("Not a supported ControllerTuner checkpoint")
        tuner = payload["tuner"]
        tuner._validate_n_jobs(tuner._search_n_jobs)
        tuner.optimizer = tuner._make_optimizer(
            tuner._search_n_jobs, study=payload["study"]
        )
        tuner._check_continuation()
        return tuner
