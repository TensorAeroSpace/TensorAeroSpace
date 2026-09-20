"""Notebook/terminal progress for controller search."""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager

from optuna.trial import TrialState
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}, ETA {remaining}{postfix}]"


class _SearchProgress:
    def __init__(self, bar, metric, best_trial):
        self.bar = bar
        self.metric = metric
        self.best_trial = best_trial
        self.completed = 0
        self.rejected = 0
        self.reason = "finished"
        self._refresh_best(refresh=True)

    def _refresh_best(self, *, refresh=False):
        if self.bar is None:
            return
        if self.best_trial is None:
            best = f"best {self.metric}=-- (no feasible trial)"
        else:
            best = f"best {self.metric}={self.best_trial.value:.6g} (#{self.best_trial.number})"
        self.bar.set_postfix_str(f"{best}, rejected={self.rejected}", refresh=refresh)

    def update(self, study, trial):
        """Optuna callback; only fully feasible trials may update the best score."""
        self.completed += 1
        self.rejected += trial.state == TrialState.PRUNED
        if (
            trial.state == TrialState.COMPLETE
            and trial.value is not None
            and math.isfinite(trial.value)
            and (self.best_trial is None or trial.value < self.best_trial.value)
        ):
            self.best_trial = trial
        if self.bar is not None:
            self._refresh_best()
            self.bar.update(1)

    def close(self):
        if self.bar is not None:
            # Keep the actual count on early stop, with no stale remaining ETA.
            self.bar.bar_format = _BAR_FORMAT.replace("{remaining}", "00:00")
            self.bar.set_description_str(f"Search {self.reason}", refresh=False)
            self.bar.refresh()
            self.bar.close()
            if hasattr(self.bar, "container"):
                # tqdm's notebook backend assumes n < total means an error.
                # A valid target/patience/timeout stop keeps its actual count.
                self.bar.container.children[1].bar_style = (
                    "danger" if self.reason in ("failed", "interrupted") else "success"
                )


@contextmanager
def search_progress(*, total, metric, best_trial=None, enabled=True):
    """Use tqdm auto-detection; restore log handlers and close on every exit."""
    if not enabled:
        yield _SearchProgress(None, metric, best_trial)
        return
    # Forward Optuna console messages through tqdm so they do not overwrite it.
    # No global verbosity changes; existing handlers are restored on exit.
    with logging_redirect_tqdm(loggers=[logging.getLogger("optuna")], tqdm_class=tqdm):
        bar = tqdm(
            total=total,
            desc="Searching",
            unit="trial",
            dynamic_ncols=True,
            miniters=1,
            bar_format=_BAR_FORMAT,
        )
        progress = _SearchProgress(bar, metric, best_trial)
        try:
            yield progress
        except KeyboardInterrupt:
            progress.reason = "interrupted"
            raise
        except BaseException:
            progress.reason = "failed"
            raise
        finally:
            progress.close()
