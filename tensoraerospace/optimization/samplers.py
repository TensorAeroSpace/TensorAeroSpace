"""Selectable search algorithms for controller tuning.

Annealing uses Metropolis acceptance with geometric cooling. Genetic search
uses Optuna's NSGA-II, including its single-objective specialization. Both
operate on the same evaluated controller trials as TPE and random search.
"""

from __future__ import annotations

import math

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, IntDistribution
from optuna.trial import TrialState


class AnnealingSampler(optuna.samplers.BaseSampler):
    """Serial simulated annealing on a fixed mixed parameter space.

    Positive log distributions are explored in log coordinates. Categorical
    mutations select a new category; numeric proposals reflect at the bounds.
    Rejected/failed trials cannot become the chain's current state. Continuing
    the same sampler preserves its accepted point, temperature and RNG. A new
    sampler attached to existing history starts at the best completed point.
    """

    def __init__(self, space, *, seed=0, temperature=1.0, cooling=0.95, step_size=0.2):
        if not np.isfinite([temperature, cooling, step_size]).all():
            raise ValueError("Annealing settings must be finite")
        if temperature <= 0 or not 0 < cooling <= 1 or not 0 < step_size <= 1:
            raise ValueError("Need temperature > 0, cooling and step_size in (0, 1]")
        self.space = dict(space)
        self.rng = np.random.default_rng(seed)
        self.temperature = float(temperature)
        self.cooling, self.step_size = float(cooling), float(step_size)
        self.current = None
        self.current_value = float("inf")
        self._initialized = False
        self._independent = optuna.samplers.RandomSampler(seed=seed)

    def infer_relative_search_space(self, study, trial):
        """Expose nonconstant dimensions for joint annealing proposals."""
        return {k: d for k, d in self.space.items() if not d.single()}

    def sample_independent(self, study, trial, param_name, param_distribution):
        """Draw a parameter with the seeded fallback random sampler."""
        return self._independent.sample_independent(
            study, trial, param_name, param_distribution
        )

    def sample_relative(self, study, trial, search_space):
        """Propose a bounded neighbor of the accepted point, using log scales as needed."""
        if study.direction != optuna.study.StudyDirection.MINIMIZE:
            raise ValueError("AnnealingSampler supports minimization")
        if not self._initialized:
            complete = [
                t
                for t in study.trials
                if t.state == TrialState.COMPLETE
                and t.value is not None
                and np.isfinite(t.value)
            ]
            if complete:
                best = min(complete, key=lambda t: t.value)
                self.current, self.current_value = dict(best.params), best.value
            self._initialized = True
        if self.current is None:
            return {
                name: self.sample_independent(study, trial, name, d)
                for name, d in search_space.items()
            }
        result = {}
        for name, distribution in search_space.items():
            if isinstance(distribution, CategoricalDistribution):
                choices = distribution.choices
                result[name] = choices[int(self.rng.integers(len(choices)))]
                continue
            low, high = distribution.low, distribution.high
            current = self.current[name]
            if distribution.log:
                low, high, current = np.log([low, high, current])
            coordinate = (current - low) / (high - low)
            proposed = (coordinate + self.rng.normal(0.0, self.step_size)) % 2.0
            proposed = proposed if proposed <= 1.0 else 2.0 - proposed
            value = low + proposed * (high - low)
            if distribution.log:
                value = float(np.exp(value))
            if distribution.step is not None:
                value = (
                    distribution.low
                    + round((value - distribution.low) / distribution.step)
                    * distribution.step
                )
            value = float(np.clip(value, distribution.low, distribution.high))
            result[name] = (
                int(value) if isinstance(distribution, IntDistribution) else value
            )
        return result

    def after_trial(self, study, trial, state, values):
        """Apply Metropolis acceptance to a finite completed trial, then cool the chain."""
        if (
            state == TrialState.COMPLETE
            and values is not None
            and np.isfinite(values[0])
        ):
            value = float(values[0])
            difference = self.current_value - value
            if difference >= 0 or self.rng.random() < math.exp(
                max(-745.0, difference / self.temperature)
            ):
                self.current, self.current_value = dict(trial.params), value
        self.temperature = max(
            float(np.finfo(float).tiny), self.temperature * self.cooling
        )


def make_sampler(method, space, *, seed=0, options=None):
    """Build the selected algorithm; misspelled methods/options fail immediately."""
    options = dict(options or {})
    if method == "annealing":
        return AnnealingSampler(space, seed=seed, **options)
    if method == "genetic":
        options.setdefault("population_size", 8)
        return optuna.samplers.NSGAIISampler(seed=seed, **options)
    if method == "tpe":
        options.setdefault("n_startup_trials", 5)
        return optuna.samplers.TPESampler(seed=seed, **options)
    if method == "random":
        return optuna.samplers.RandomSampler(seed=seed, **options)
    raise ValueError("method must be 'annealing', 'genetic', 'tpe' or 'random'")
