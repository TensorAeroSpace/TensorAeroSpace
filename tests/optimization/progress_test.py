"""Search UI contracts: measured ETA, feasible best, cleanup and API forwarding."""

import io
import logging

import pytest
from tqdm import std

from tensoraerospace.optimization import ControlOptimizer, Float, _progress
from tensoraerospace.optimization.agent import OptimizableAgent


@pytest.fixture
def rendered_progress(monkeypatch):
    stream = io.StringIO()
    now = [100.0]
    bars = []
    monkeypatch.setattr(std, "time", lambda: now[0])

    class RecordingBar(std.tqdm):
        def __init__(self, *args, **kwargs):
            self.frames = []
            kwargs.update(file=stream, mininterval=0.0, miniters=1)
            super().__init__(*args, **kwargs)
            bars.append(self)

        def display(self, *args, **kwargs):
            self.frames.append(str(self))
            return super().display(*args, **kwargs)

    monkeypatch.setattr(_progress, "tqdm", RecordingBar)
    return stream, now, bars


def test_bar_shows_remaining_time_and_only_feasible_best(rendered_progress):
    stream, now, bars = rendered_progress
    values = iter([3.0, 0.1, 2.0])

    def evaluate(params, seed):
        now[0] += 2.0
        value = next(values)
        return {"cpi": value, "tail": 1.0 if value == 0.1 else 0.0}

    optimizer = ControlOptimizer(
        {"x": Float(0, 1)}, evaluate, seeds=[0], constraints={"tail": 0.1}
    )
    result = optimizer.optimize(3)
    bar = bars[0]
    assert "1/3 [00:02, ETA 00:04" in stream.getvalue()
    rejected_frame = next(frame for frame in bar.frames if "2/3 [" in frame)
    assert "best cpi=3" in rejected_frame
    assert "rejected=1" in rejected_frame
    assert "best cpi=0.1" not in stream.getvalue()
    assert "best cpi=2 (#2)" in stream.getvalue()
    assert bar.n == bar.total == 3
    assert bar.disable  # closed
    assert result.best_value == 2.0


def test_no_feasible_trial_is_explicit_in_progress(rendered_progress):
    stream, _, bars = rendered_progress
    optimizer = ControlOptimizer(
        {"x": Float(0, 1)},
        lambda p, s: {"cpi": 3.4},
        seeds=[0],
        constraints={"cpi": 0.6},
    )
    with pytest.raises(RuntimeError, match="No feasible"):
        optimizer.optimize(2)
    assert "no feasible trial" in stream.getvalue()
    assert "rejected=2" in stream.getvalue()
    assert "best cpi=3.4" not in stream.getvalue()
    assert bars[0].disable


@pytest.mark.parametrize(
    "options,reason,expected",
    [
        ({"target": 1.0}, "target reached", 1),
        ({"patience": 1}, "patience", 2),
    ],
)
def test_early_stop_keeps_actual_trial_count_and_clears_eta(
    rendered_progress, options, reason, expected
):
    stream, now, bars = rendered_progress

    def evaluate(p, s):
        now[0] += 2.0
        return 1.0

    optimizer = ControlOptimizer({"x": Float(0, 1)}, evaluate, seeds=[0])
    optimizer.optimize(10, **options)
    final = bars[0].frames[-1]
    assert bars[0].n == expected
    assert bars[0].total == 10
    assert f"stopped ({reason})" in final
    assert "ETA 00:00" in final


def test_timeout_keeps_actual_count(rendered_progress, monkeypatch):
    _, _, bars = rendered_progress
    optimizer = ControlOptimizer({"x": Float(0, 1)}, lambda p, s: 1.0, seeds=[0])
    # Exercise the public timeout path deterministically without sleeping.
    run = optimizer.run_optimization

    def short_budget(func, n_trials, **options):
        assert options["timeout"] == 2.0
        run(func, 1, **options)

    monkeypatch.setattr(optimizer, "run_optimization", short_budget)
    optimizer.optimize(10, timeout=2.0)
    assert bars[0].n == 1
    assert "stopped (timeout)" in bars[0].frames[-1]
    assert "ETA 00:00" in bars[0].frames[-1]


@pytest.mark.parametrize("error", [ValueError("controller bug"), KeyboardInterrupt()])
def test_progress_closes_and_restores_logging_on_error(rendered_progress, error):
    _, _, bars = rendered_progress
    logger = logging.getLogger("optuna")
    handlers = list(logger.handlers)
    level = logger.level

    def evaluate(p, s):
        raise error

    optimizer = ControlOptimizer({"x": Float(0, 1)}, evaluate, seeds=[0])
    with pytest.raises(type(error)):
        optimizer.optimize(3)
    assert bars[0].disable
    assert (
        "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    ) in bars[0].frames[-1]
    assert logger.handlers == handlers
    assert logger.level == level


def test_progress_can_be_disabled_without_creating_bar(rendered_progress):
    stream, _, bars = rendered_progress
    optimizer = ControlOptimizer({"x": Float(0, 1)}, lambda p, s: 1.0, seeds=[0])
    optimizer.optimize(2, show_progress_bar=False)
    assert bars == []
    assert stream.getvalue() == ""


def test_resumed_search_shows_prior_best_but_counts_only_new_trials(rendered_progress):
    stream, _, bars = rendered_progress
    values = iter([1.0, 2.0, 3.0])
    optimizer = ControlOptimizer(
        {"x": Float(0, 1)}, lambda p, s: next(values), seeds=[0]
    )
    optimizer.optimize(1, show_progress_bar=False)
    optimizer.optimize(2)
    assert "best cpi=1 (#0)" in bars[0].frames[1]
    assert bars[0].n == bars[0].total == 2
    assert len(optimizer.study.trials) == 3


def test_agent_class_forwards_progress_option(rendered_progress):
    _, _, bars = rendered_progress

    class Agent(OptimizableAgent):
        def __init__(self, gain):
            self.gain = gain

    result = Agent.optimize(
        {"gain": Float(0, 1)},
        lambda a, s: a.gain,
        agent_kwargs={},
        seeds=[0],
        n_trials=2,
        show_progress_bar=False,
    )
    assert len(result.study.trials) == 2
    assert bars == []


def test_notebook_target_stop_keeps_count_and_success_style(monkeypatch):
    pytest.importorskip("ipywidgets")
    from tqdm.notebook import tqdm as notebook_tqdm

    bars = []

    class RecordingNotebookBar(notebook_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["display"] = False
            super().__init__(*args, **kwargs)
            bars.append(self)

    monkeypatch.setattr(_progress, "tqdm", RecordingNotebookBar)
    optimizer = ControlOptimizer({"x": Float(0, 1)}, lambda p, s: 1.0, seeds=[0])
    optimizer.optimize(10, target=1.0)
    assert bars[0].n == 1 and bars[0].total == 10
    assert bars[0].container.children[1].bar_style == "success"
    assert "target reached" in bars[0].container.children[0].value.replace(
        "\u2007", " "
    )
    assert "best cpi=1" in bars[0].container.children[2].value.replace("\u2007", " ")
