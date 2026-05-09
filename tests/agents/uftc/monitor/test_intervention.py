"""MacroActionDispatcher invokes correct method on each layer."""

from __future__ import annotations

from dataclasses import dataclass, field

from tensoraerospace.agent.uftc.monitor.intervention import (
    MacroAction,
    MacroActionDispatcher,
)


@dataclass
class _StubL3:
    reset_calls: list[float] = field(default_factory=list)

    def force_reset(self, severity_hint: float) -> None:
        self.reset_calls.append(float(severity_hint))


@dataclass
class _StubL4:
    freeze_until: int | None = None
    degrade_calls: int = 0

    def freeze_learning(self, until_step: int) -> None:
        self.freeze_until = int(until_step)

    def degrade_reference_to_hold(self) -> None:
        self.degrade_calls += 1


@dataclass
class _StubL1:
    hold_calls: int = 0

    def request_actuator_hold(self) -> None:
        self.hold_calls += 1


def test_dispatch_calls_correct_methods() -> None:
    l3, l4, l1 = _StubL3(), _StubL4(), _StubL1()
    d = MacroActionDispatcher(l3=l3, l4=l4, l1=l1)
    d.dispatch(
        [
            MacroAction("freeze_l4_learning", {"duration": 100}),
            MacroAction("force_rls_reset", {"severity": 0.7}),
            MacroAction("degrade_reference_to_hold"),
            MacroAction("request_actuator_hold"),
        ],
        current_step=42,
    )
    assert l4.freeze_until == 142
    assert l3.reset_calls == [0.7]
    assert l4.degrade_calls == 1
    assert l1.hold_calls == 1


def test_dispatch_swallows_layer_exceptions() -> None:
    class _BoomL3:
        def force_reset(self, severity_hint: float) -> None:
            raise RuntimeError("nope")

    d = MacroActionDispatcher(l3=_BoomL3(), l4=None, l1=None)
    diag = d.dispatch([MacroAction("force_rls_reset")], current_step=0)
    # No exception bubbles up; nothing recorded for force_rls_reset.
    assert "force_rls_reset" not in diag


def test_dispatch_with_missing_layers_is_noop() -> None:
    d = MacroActionDispatcher(l3=None, l4=None, l1=None)
    diag = d.dispatch(
        [
            MacroAction("freeze_l4_learning", {"duration": 1}),
            MacroAction("request_actuator_hold"),
        ],
        current_step=0,
    )
    assert diag == {}
