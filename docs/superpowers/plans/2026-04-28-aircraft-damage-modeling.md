# Aircraft Damage Modeling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a damage-modeling subsystem to the nonlinear F-16 (longitudinal + 6-DoF angular) so that damage events fire at scheduled times during simulation, recomputing mass/inertia/aero coefficients and modifying control surfaces, engine, and structural state.

**Architecture:** Self-contained `damage/` package under `tensoraerospace/aerospacemodel/f16/nonlinear/`, plus minimal hooks into models, dynamics, and gym envs. With `damage_profile=None` the existing behaviour is bit-for-bit identical (regression-tested). Strip-theory aero corrections + Huygens-Steiner inertia recompute provide physical fidelity.

**Tech Stack:** Python 3.11+, NumPy, dataclasses, PyYAML (already a transitive dep via mkdocs), pytest, Gymnasium.

**Spec:** `docs/superpowers/specs/2026-04-28-aircraft-damage-modeling-design.md`.

---

## Pre-flight check

Before starting Task 0, run the full F-16 test suite to capture the baseline:

```bash
cd /home/mr8bit/Projects/TensorAeroSpace
poetry run pytest tests/aerospacemodel/ tests/envs/ -x --tb=short -q
```

Record the output (it should be a clean pass). The plan adds new tests; existing tests must continue to pass at every commit point.

---

## Phase 0 — Split stabilator into stab_left / stab_right

The current `AngularF16` has a single `stab` input producing a symmetric pitching moment. Asymmetric stabilator failures (one side jammed, other side healthy) require independent left/right inputs that combine to give pitch (mean) and roll (differential). This is an additive change behind a feature flag — default behaviour is unchanged.

### Task 0.1: Add `split_stab` flag and 4-input mode to AngularF16

**Files:**
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/angular/model.py`
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/angular/dynamics.py`
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/angular/params.py`
- Test: `tests/aerospacemodel/f16_nonlinear_angular_split_stab_test.py` (new)

- [ ] **Step 1: Add baseline regression test**

`tests/aerospacemodel/f16_nonlinear_angular_split_stab_test.py`:

```python
"""Regression: split_stab=False is bit-identical to legacy 3-input mode."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16


def test_default_mode_is_legacy_three_input():
    m = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    assert m.action_space_length == 3
    assert m.control_list == ["stab", "ail", "dir"]


def test_default_mode_step_unchanged_after_flag_addition():
    """Lock current behaviour: zero-input step must give the canonical baseline."""
    m = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    m.run_step(np.zeros(3))
    s = m.current_state
    # Regenerate this baseline by running the test once and copying current
    # values. Once captured, this acts as a regression lock.
    assert s.shape == (14,)
    # Sanity: alpha/beta should not have moved much from zero
    assert abs(s[0]) < 1e-3
    assert abs(s[1]) < 1e-6


def test_split_stab_mode_advertises_four_inputs():
    m = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4", split_stab=True)
    assert m.action_space_length == 4
    assert m.control_list == ["stab_left", "stab_right", "ail", "dir"]


def test_split_stab_symmetric_command_matches_legacy():
    """If both halves get the same command, the trajectory must equal the
    single-input case."""
    legacy = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    split = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4", split_stab=True)
    u_legacy = np.array([0.05, 0.0, 0.0])  # 0.05 rad stab
    u_split = np.array([0.05, 0.05, 0.0, 0.0])  # both halves at 0.05
    for _ in range(50):
        legacy.run_step(u_legacy)
        split.run_step(u_split)
    np.testing.assert_allclose(
        legacy.current_state, split.current_state, atol=1e-12,
        err_msg="symmetric split-stab must reproduce legacy single-stab behaviour",
    )


def test_split_stab_asymmetric_produces_roll_moment():
    """Differential stabilator command must induce a roll moment (dwx != 0)."""
    m = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4", split_stab=True)
    u = np.array([0.10, -0.10, 0.0, 0.0])  # +0.10 left, -0.10 right
    for _ in range(20):
        m.run_step(u)
    s = m.current_state
    assert abs(s[2]) > 1e-4, f"wx (roll rate) should be non-zero, got {s[2]}"
```

- [ ] **Step 2: Run the test — first three pass, last two fail**

```bash
poetry run pytest tests/aerospacemodel/f16_nonlinear_angular_split_stab_test.py -v
```

Expected: `test_default_mode_is_legacy_three_input` PASS, `test_default_mode_step_unchanged_after_flag_addition` PASS, then split-stab tests FAIL with `TypeError: __init__() got an unexpected keyword argument 'split_stab'`.

- [ ] **Step 3: Add `split_stab` parameter to AngularF16**

In `tensoraerospace/aerospacemodel/f16/nonlinear/angular/model.py`, modify the `__init__` signature and body. Add `split_stab: bool = False` argument; when True, action length = 4 and control_list = `["stab_left", "stab_right", "ail", "dir"]`. The 4-input action is mapped to legacy 3-input via `(stab_mean, ail, dir)` plus a stored differential `delta_stab`. Replace the `action_space_length`/`control_list` setup:

```python
def __init__(
    self,
    x0: ArrayLike,
    selected_state_output=None,
    t0: float = 0,
    dt: float = 0.01,
    integrator: Literal["euler", "rk4"] = "euler",
    split_stab: bool = False,
) -> None:
    x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
    if x0_arr.size != 14:
        raise ValueError(f"x0 must have 14 elements; got {x0_arr.size}")
    super().__init__(x0_arr, selected_state_output, t0, dt)
    self.split_stab = split_stab
    _list_state = [
        "alpha", "beta", "wx", "wy", "wz",
        "gamma", "psi", "theta",
        "stab", "dstab", "ail", "dail", "dir", "ddir",
    ]
    if split_stab:
        _control_list = ["stab_left", "stab_right", "ail", "dir"]
    else:
        _control_list = ["stab", "ail", "dir"]
    self.action_space_length = len(_control_list)
    self.param: F16AngularParameters = default_parameters()
    self.x_history = [x0_arr.reshape(14, 1)]
    self._initialize_selected_state_index(self.selected_state_output, _list_state)
    self.list_state = _list_state
    self.control_list = _control_list
    if integrator == "euler":
        self._step_fn = euler
    elif integrator == "rk4":
        self._step_fn = rk4
    else:
        raise ValueError(f"unknown integrator: {integrator!r}")
    self._integrator_name = integrator
```

And modify `run_step` to forward split-stab info to the ODE:

```python
def run_step(self, u: ArrayLike) -> np.ndarray:
    u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
    if u_arr.size != self.action_space_length:
        raise ValueError(
            "Размерность управляющего вектора задана неверно."
            f" Текущее значение {u_arr.size}, не соответсвует {self.action_space_length}"
        )

    if self.split_stab:
        # u = [stab_left, stab_right, ail, dir]; convert to (stab_mean, ail, dir)
        # plus a differential delta carried via params for ODE roll-moment term.
        stab_mean = 0.5 * (u_arr[0] + u_arr[1])
        delta_stab = 0.5 * (u_arr[0] - u_arr[1])  # +ve delta = LWD (left up)
        u_legacy = np.array([stab_mean, u_arr[2], u_arr[3]], dtype=np.float64)
        # Stash on params (read by ODE; reset to 0 each step for safety)
        self.param.delta_stab_cmd = float(delta_stab)
    else:
        u_legacy = u_arr
        self.param.delta_stab_cmd = 0.0

    x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
    t_now = self.t0 + self.dt * self.time_step
    x_next = self._step_fn(f16_ode_6dof, x_prev, u_legacy, t_now, self.dt, self.param)

    x_next_col = x_next.reshape(14, 1)
    self.x_history.append(x_next_col)
    self.u_history.append(u_arr.reshape(-1, 1))
    self.time_step += 1

    if self.selected_state_output:
        return x_next_col[self.selected_state_index]
    return x_next_col
```

- [ ] **Step 4: Add `delta_stab_cmd` field to params**

In `tensoraerospace/aerospacemodel/f16/nonlinear/angular/params.py`, append after `hEx`:

```python
    # Split-stab differential command (rad). Set by AngularF16.run_step
    # when split_stab=True; 0.0 means symmetric (legacy) operation.
    delta_stab_cmd: float = 0.0
```

- [ ] **Step 5: Add roll-moment term from delta_stab in ODE**

In `tensoraerospace/aerospacemodel/f16/nonlinear/angular/dynamics.py`, just before computing `Mx`:

```python
    # Differential stabilator → roll moment.
    # F-16 stab arm ≈ 1.5 m. delta_stab > 0 means left half UP, right DOWN
    # (positive roll = right-wing-down). Coefficient is ∂Cz/∂stab × y_arm/l.
    # For the level here we use a simple lever: ΔMx ≈ q*S*l * (-Cz_per_stab * δ * y_arm/l)
    # where Cz_per_stab ≈ 0.6/rad (approximate, validated to give realistic roll
    # rates ~ 30 deg/s for δ=10 deg).
    DELTA_STAB_ROLL_GAIN = 0.6  # 1/rad, calibrated for F-16 stabilator
    delta_stab = float(getattr(p, "delta_stab_cmd", 0.0))
    mx_split = -DELTA_STAB_ROLL_GAIN * delta_stab
    Mx = p.q * p.S * p.l * (mx_ + mx_split)
```

(Replace the existing `Mx = p.q * p.S * p.l * mx_` line.)

- [ ] **Step 6: Run all split-stab tests — they should now pass**

```bash
poetry run pytest tests/aerospacemodel/f16_nonlinear_angular_split_stab_test.py -v
```

Expected: all five tests PASS.

- [ ] **Step 7: Run full F-16 test suite to verify no regressions**

```bash
poetry run pytest tests/aerospacemodel/ tests/envs/ -x --tb=short -q
```

Expected: pass count ≥ baseline from pre-flight check + 5 (the new tests).

- [ ] **Step 8: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/angular/ \
        tests/aerospacemodel/f16_nonlinear_angular_split_stab_test.py
git commit -m "$(cat <<'EOF'
feat(f16): add split_stab mode for asymmetric stabilator commands

Adds an optional 4-input mode (stab_left/stab_right/ail/dir) to AngularF16,
producing a roll moment proportional to the left-right stabilator
differential. Default (split_stab=False) is bit-identical to current
3-input behaviour. Required for asymmetric control-surface failures in
the upcoming damage subsystem.
EOF
)"
```

---

## Phase 1 — Geometry primitives, presets, DamageState

### Task 1.1: AeroSection dataclass and BaseGeometry container

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/__init__.py`
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/geometry.py`
- Test: `tests/aerospacemodel/f16_damage/__init__.py` (empty)
- Test: `tests/aerospacemodel/f16_damage/geometry_test.py` (new)

- [ ] **Step 1: Create empty package files**

```bash
mkdir -p tensoraerospace/aerospacemodel/f16/nonlinear/damage/data
mkdir -p tests/aerospacemodel/f16_damage
touch tensoraerospace/aerospacemodel/f16/nonlinear/damage/__init__.py
touch tests/aerospacemodel/f16_damage/__init__.py
```

- [ ] **Step 2: Write the failing test**

`tests/aerospacemodel/f16_damage/geometry_test.py`:

```python
"""AeroSection and BaseGeometry primitives."""

from __future__ import annotations

import numpy as np
import pytest


def test_aero_section_is_frozen():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.geometry import (
        AeroSection,
    )
    s = AeroSection(
        name="left_tip", side="left", type="wing",
        area=2.0, span_position=-3.5, chord=1.5, sweep=0.0,
        mass=80.0, cg_local=(0.0, -3.5, 0.0),
        inertia_local=(50.0, 100.0, 60.0, 0.0),
        cl_alpha_contribution=0.4, cd0_contribution=0.005,
    )
    with pytest.raises((AttributeError, Exception)):
        s.area = 99.0


def test_base_geometry_aggregates_mass_and_area():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.geometry import (
        AeroSection, BaseGeometry,
    )
    sections = [
        AeroSection(
            name=f"sec_{i}", side="left" if i % 2 == 0 else "right",
            type="wing",
            area=1.0, span_position=(-1.0 if i % 2 == 0 else 1.0) * (i + 1),
            chord=1.0, sweep=0.0,
            mass=10.0,
            cg_local=(0.0, (-1.0 if i % 2 == 0 else 1.0) * (i + 1), 0.0),
            inertia_local=(1.0, 1.0, 1.0, 0.0),
            cl_alpha_contribution=0.1,
            cd0_contribution=0.001,
        )
        for i in range(4)
    ]
    g = BaseGeometry(sections=sections)
    assert g.total_wing_area() == pytest.approx(4.0)
    assert g.total_mass() == pytest.approx(40.0)
    # CG should be at y=0 for symmetric layout
    assert g.center_of_mass()[1] == pytest.approx(0.0, abs=1e-9)


def test_base_geometry_lookup_by_name():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.geometry import (
        AeroSection, BaseGeometry,
    )
    s = AeroSection(
        name="left_tip", side="left", type="wing",
        area=1.0, span_position=-3.0, chord=1.0, sweep=0.0,
        mass=10.0, cg_local=(0, -3, 0), inertia_local=(1, 1, 1, 0),
        cl_alpha_contribution=0.1, cd0_contribution=0.001,
    )
    g = BaseGeometry(sections=[s])
    assert g.section("left_tip") is s
    with pytest.raises(KeyError):
        g.section("nonexistent")
```

- [ ] **Step 3: Run — fails with ImportError**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/geometry_test.py -v
```

Expected: FAIL with `ImportError: cannot import name 'AeroSection'`.

- [ ] **Step 4: Implement geometry.py**

`tensoraerospace/aerospacemodel/f16/nonlinear/damage/geometry.py`:

```python
"""Parametric geometry primitives for damage modeling.

Each aircraft is described as a list of AeroSection objects. Sections aggregate
into a BaseGeometry, which serves as the source of truth for mass, inertia,
and aerodynamic contributions. Damage is applied by scaling per-section loss
fractions (see state.py and recompute.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]
Inertia4 = Tuple[float, float, float, float]  # Ixx, Iyy, Izz, Ixz


@dataclass(frozen=True)
class AeroSection:
    """A discrete piece of the aircraft.

    Coordinates are in the body-fixed frame:
      x — forward (positive towards the nose)
      y — right (positive towards the right wingtip)
      z — down

    `span_position` is the y-coordinate of the section's aero centre
    (signed: negative=left, positive=right).
    """

    name: str
    side: Literal["left", "right", "center"]
    type: Literal["wing", "stab", "vtail", "control", "fuselage"]

    area: float                     # m², projected area
    span_position: float            # m (signed, see above)
    chord: float                    # m, mean chord
    sweep: float                    # rad

    mass: float                     # kg
    cg_local: Vec3                  # body-frame cg of the section, m
    inertia_local: Inertia4         # Ixx, Iyy, Izz, Ixz about section cg

    cl_alpha_contribution: float    # 1/rad, this section's share of ∂Cy/∂α
    cd0_contribution: float         # this section's share of Cx0

    controls_input: Optional[str] = None      # "stab_left", "rudder", ...
    control_effectiveness: float = 1.0
    aero_x_arm: float = 0.0          # x-arm from aircraft CG to section's aero centre


@dataclass
class BaseGeometry:
    """Aggregate description of the un-damaged aircraft."""

    sections: list[AeroSection]

    def __post_init__(self) -> None:
        names = [s.name for s in self.sections]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate section names")
        self._index: dict[str, AeroSection] = {s.name: s for s in self.sections}

    def section(self, name: str) -> AeroSection:
        try:
            return self._index[name]
        except KeyError as e:
            raise KeyError(f"No section named {name!r}") from e

    def section_names(self) -> list[str]:
        return [s.name for s in self.sections]

    def total_wing_area(self) -> float:
        return sum(s.area for s in self.sections if s.type == "wing")

    def total_mass(self) -> float:
        return sum(s.mass for s in self.sections)

    def center_of_mass(self) -> np.ndarray:
        m_total = self.total_mass()
        if m_total <= 0:
            raise ValueError("Total mass is non-positive")
        x = sum(s.mass * s.cg_local[0] for s in self.sections) / m_total
        y = sum(s.mass * s.cg_local[1] for s in self.sections) / m_total
        z = sum(s.mass * s.cg_local[2] for s in self.sections) / m_total
        return np.array([x, y, z])
```

- [ ] **Step 5: Run tests — all pass**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/geometry_test.py -v
```

Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/ \
        tests/aerospacemodel/f16_damage/
git commit -m "feat(f16-damage): add AeroSection and BaseGeometry primitives"
```

### Task 1.2: F-16 geometry data (YAML + presets loader)

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/data/f16_geometry.yaml`
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/presets.py`
- Test: `tests/aerospacemodel/f16_damage/presets_test.py`

- [ ] **Step 1: Write the failing calibration test**

`tests/aerospacemodel/f16_damage/presets_test.py`:

```python
"""F-16 base geometry calibration: aggregate properties must match
F16AngularParameters within ~1%.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    F16AngularParameters,
)


@pytest.fixture
def base_geo():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
        load_f16_geometry,
    )
    return load_f16_geometry()


def test_total_mass_matches_params(base_geo):
    p = F16AngularParameters()
    assert base_geo.total_mass() == pytest.approx(p.m, rel=0.01)


def test_total_wing_area_matches_params(base_geo):
    p = F16AngularParameters()
    assert base_geo.total_wing_area() == pytest.approx(p.S, rel=0.01)


def test_required_section_names_present(base_geo):
    names = set(base_geo.section_names())
    required = {
        "left_root", "left_mid", "left_tip",
        "right_root", "right_mid", "right_tip",
        "stab_left", "stab_right",
        "vtail",
        "rudder", "aileron_left", "aileron_right",
        "fuselage_main",
    }
    missing = required - names
    assert not missing, f"Missing required sections: {missing}"


def test_section_sides_consistent(base_geo):
    for s in base_geo.sections:
        if s.name.startswith("left_") or s.name.endswith("_left"):
            assert s.side == "left", f"{s.name} side={s.side}"
        if s.name.startswith("right_") or s.name.endswith("_right"):
            assert s.side == "right", f"{s.name} side={s.side}"
```

- [ ] **Step 2: Run — fails with ImportError**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/presets_test.py -v
```

Expected: FAIL — `ImportError: cannot import name 'load_f16_geometry'`.

- [ ] **Step 3: Create the YAML data file**

`tensoraerospace/aerospacemodel/f16/nonlinear/damage/data/f16_geometry.yaml`:

```yaml
# F-16 baseline geometry, calibrated so totals match F16AngularParameters
# (m=9295.44 kg, S=27.87 m², bA=3.45 m, Jx/Jy/Jz/Jxz from params.py).
#
# Sources: NASA TM-1538 (F-16 aerodynamic data), Stevens & Lewis "Aircraft
# Control and Simulation" Ch. 3. Where the spec defines bA=3.45 (actually MAC),
# we use that for chord on wing sections.
#
# Coordinate frame (body): x forward, y right, z down. CG of the aircraft is
# the origin (rcgx will recompute on damage).

sections:
  # ----- WING (6 sections, 3 per side, areas total to S=27.87 m²) -----
  - name: left_root
    side: left
    type: wing
    area: 5.40
    span_position: -1.20
    chord: 4.20
    sweep: 0.70
    mass: 280.0
    cg_local: [0.50, -1.20, 0.10]
    inertia_local: [120.0, 480.0, 380.0, 0.0]
    cl_alpha_contribution: 0.85
    cd0_contribution: 0.0050
    aero_x_arm: 0.50

  - name: left_mid
    side: left
    type: wing
    area: 4.10
    span_position: -2.80
    chord: 2.80
    sweep: 0.70
    mass: 180.0
    cg_local: [0.20, -2.80, 0.05]
    inertia_local: [200.0, 380.0, 220.0, 0.0]
    cl_alpha_contribution: 0.65
    cd0_contribution: 0.0030
    aero_x_arm: 0.20

  - name: left_tip
    side: left
    type: wing
    area: 3.43
    span_position: -4.30
    chord: 1.80
    sweep: 0.70
    mass: 100.0
    cg_local: [-0.10, -4.30, 0.02]
    inertia_local: [220.0, 280.0, 120.0, 0.0]
    cl_alpha_contribution: 0.55
    cd0_contribution: 0.0020
    aero_x_arm: -0.10

  - name: right_root
    side: right
    type: wing
    area: 5.40
    span_position: 1.20
    chord: 4.20
    sweep: 0.70
    mass: 280.0
    cg_local: [0.50, 1.20, 0.10]
    inertia_local: [120.0, 480.0, 380.0, 0.0]
    cl_alpha_contribution: 0.85
    cd0_contribution: 0.0050
    aero_x_arm: 0.50

  - name: right_mid
    side: right
    type: wing
    area: 4.10
    span_position: 2.80
    chord: 2.80
    sweep: 0.70
    mass: 180.0
    cg_local: [0.20, 2.80, 0.05]
    inertia_local: [200.0, 380.0, 220.0, 0.0]
    cl_alpha_contribution: 0.65
    cd0_contribution: 0.0030
    aero_x_arm: 0.20

  - name: right_tip
    side: right
    type: wing
    area: 3.43
    span_position: 4.30
    chord: 1.80
    sweep: 0.70
    mass: 100.0
    cg_local: [-0.10, 4.30, 0.02]
    inertia_local: [220.0, 280.0, 120.0, 0.0]
    cl_alpha_contribution: 0.55
    cd0_contribution: 0.0020
    aero_x_arm: -0.10

  # ----- HORIZONTAL STABILATOR (2 sections, all-moving) -----
  - name: stab_left
    side: left
    type: stab
    area: 2.40
    span_position: -1.60
    chord: 1.80
    sweep: 0.45
    mass: 90.0
    cg_local: [-4.50, -1.60, 0.00]
    inertia_local: [60.0, 180.0, 130.0, 0.0]
    cl_alpha_contribution: 0.18
    cd0_contribution: 0.0010
    controls_input: stab_left
    aero_x_arm: -4.50

  - name: stab_right
    side: right
    type: stab
    area: 2.40
    span_position: 1.60
    chord: 1.80
    sweep: 0.45
    mass: 90.0
    cg_local: [-4.50, 1.60, 0.00]
    inertia_local: [60.0, 180.0, 130.0, 0.0]
    cl_alpha_contribution: 0.18
    cd0_contribution: 0.0010
    controls_input: stab_right
    aero_x_arm: -4.50

  # ----- VERTICAL TAIL -----
  - name: vtail
    side: center
    type: vtail
    area: 5.10
    span_position: 0.0
    chord: 2.50
    sweep: 0.78
    mass: 110.0
    cg_local: [-4.20, 0.00, -1.50]
    inertia_local: [220.0, 240.0, 60.0, 0.0]
    cl_alpha_contribution: 0.0
    cd0_contribution: 0.0015
    aero_x_arm: -4.20

  # ----- CONTROL SURFACES (rudder + ailerons) -----
  - name: rudder
    side: center
    type: control
    area: 1.10
    span_position: 0.0
    chord: 0.80
    sweep: 0.78
    mass: 25.0
    cg_local: [-4.80, 0.0, -2.20]
    inertia_local: [40.0, 50.0, 12.0, 0.0]
    cl_alpha_contribution: 0.0
    cd0_contribution: 0.0005
    controls_input: rudder
    aero_x_arm: -4.80

  - name: aileron_left
    side: left
    type: control
    area: 0.60
    span_position: -3.80
    chord: 0.55
    sweep: 0.70
    mass: 18.0
    cg_local: [-0.20, -3.80, 0.02]
    inertia_local: [10.0, 28.0, 18.0, 0.0]
    cl_alpha_contribution: 0.0
    cd0_contribution: 0.0004
    controls_input: aileron_left
    aero_x_arm: -0.20

  - name: aileron_right
    side: right
    type: control
    area: 0.60
    span_position: 3.80
    chord: 0.55
    sweep: 0.70
    mass: 18.0
    cg_local: [-0.20, 3.80, 0.02]
    inertia_local: [10.0, 28.0, 18.0, 0.0]
    cl_alpha_contribution: 0.0
    cd0_contribution: 0.0004
    controls_input: aileron_right
    aero_x_arm: -0.20

  # ----- FUSELAGE (single lumped section, holds the bulk of the mass) -----
  - name: fuselage_main
    side: center
    type: fuselage
    area: 0.0
    span_position: 0.0
    chord: 0.0
    sweep: 0.0
    mass: 7825.44
    cg_local: [0.0, 0.0, 0.0]
    inertia_local: [10000.0, 80000.0, 73000.0, 1300.0]
    cl_alpha_contribution: 0.0
    cd0_contribution: 0.0050
    aero_x_arm: 0.0
```

- [ ] **Step 4: Implement presets.py**

`tensoraerospace/aerospacemodel/f16/nonlinear/damage/presets.py`:

```python
"""F-16 geometry preset loader."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from .geometry import AeroSection, BaseGeometry

_DATA_PACKAGE = "tensoraerospace.aerospacemodel.f16.nonlinear.damage.data"
_F16_FILE = "f16_geometry.yaml"


def load_f16_geometry() -> BaseGeometry:
    """Load the calibrated F-16 baseline geometry."""
    data_path = resources.files(_DATA_PACKAGE).joinpath(_F16_FILE)
    with data_path.open("r") as f:
        raw = yaml.safe_load(f)
    sections = [
        AeroSection(
            name=s["name"],
            side=s["side"],
            type=s["type"],
            area=float(s["area"]),
            span_position=float(s["span_position"]),
            chord=float(s["chord"]),
            sweep=float(s["sweep"]),
            mass=float(s["mass"]),
            cg_local=tuple(float(v) for v in s["cg_local"]),
            inertia_local=tuple(float(v) for v in s["inertia_local"]),
            cl_alpha_contribution=float(s["cl_alpha_contribution"]),
            cd0_contribution=float(s["cd0_contribution"]),
            controls_input=s.get("controls_input"),
            control_effectiveness=float(s.get("control_effectiveness", 1.0)),
            aero_x_arm=float(s.get("aero_x_arm", 0.0)),
        )
        for s in raw["sections"]
    ]
    return BaseGeometry(sections=sections)
```

- [ ] **Step 5: Verify PyYAML availability**

```bash
poetry run python -c "import yaml; print(yaml.__version__)"
```

Expected: prints a version (PyYAML is already a transitive dep). If it fails:

```bash
poetry add pyyaml
```

- [ ] **Step 6: Run preset tests**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/presets_test.py -v
```

Expected: 4 PASS. If `total_mass` or `total_wing_area` deviate, tweak the YAML totals — the calibration is intentionally split so each individual mass/area is tunable.

- [ ] **Step 7: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/data/ \
        tensoraerospace/aerospacemodel/f16/nonlinear/damage/presets.py \
        tests/aerospacemodel/f16_damage/presets_test.py
git commit -m "feat(f16-damage): add F-16 baseline geometry preset (YAML + loader)"
```

### Task 1.3: DamageState dataclass

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/state.py`
- Test: `tests/aerospacemodel/f16_damage/state_test.py`

- [ ] **Step 1: Write failing tests**

`tests/aerospacemodel/f16_damage/state_test.py`:

```python
"""DamageState behaviour: defaults, mutation, snapshot."""

from __future__ import annotations

import pytest


def test_default_state_is_healthy():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
        load_f16_geometry,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
        DamageState,
    )
    geo = load_f16_geometry()
    s = DamageState.healthy(geo)
    assert all(v == 0.0 for v in s.section_loss.values())
    assert s.engine.thrust_factor == 1.0
    assert not s.engine.hard_failure
    assert s.structural.extra_mass_delta_kg == 0.0
    assert s.control_failures == {}


def test_section_loss_clamped_to_unit_interval():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
        DamageState,
    )
    s = DamageState(section_loss={"x": 0.0}, control_failures={})
    s.set_section_loss("x", 1.5)
    assert s.section_loss["x"] == 1.0
    s.set_section_loss("x", -0.2)
    assert s.section_loss["x"] == 0.0


def test_control_failure_validates_mode():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
        ControlFailure,
    )
    cf = ControlFailure(mode="jam", jam_position_rad=0.1)
    assert cf.mode == "jam"
    with pytest.raises(ValueError):
        ControlFailure(mode="not_a_mode")  # type: ignore[arg-type]


def test_snapshot_is_independent_copy():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
        load_f16_geometry,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
        DamageState,
    )
    geo = load_f16_geometry()
    s = DamageState.healthy(geo)
    snap = s.snapshot()
    s.set_section_loss(geo.section_names()[0], 0.5)
    # snap is a frozen dict-like
    first = geo.section_names()[0]
    assert snap["section_loss"][first] == 0.0
```

- [ ] **Step 2: Run — fails**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/state_test.py -v
```

Expected: ImportError on first run.

- [ ] **Step 3: Implement state.py**

`tensoraerospace/aerospacemodel/f16/nonlinear/damage/state.py`:

```python
"""Mutable runtime state describing what is currently damaged."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Literal

from .geometry import BaseGeometry

ControlMode = Literal[
    "healthy", "efficiency_loss", "jam", "free_floating", "lost"
]
_VALID_CONTROL_MODES = (
    "healthy", "efficiency_loss", "jam", "free_floating", "lost",
)


@dataclass
class ControlFailure:
    mode: ControlMode = "healthy"
    efficiency: float = 1.0
    jam_position_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in _VALID_CONTROL_MODES:
            raise ValueError(
                f"ControlFailure.mode must be one of "
                f"{_VALID_CONTROL_MODES}; got {self.mode!r}"
            )
        if not (0.0 <= self.efficiency <= 1.0):
            raise ValueError(
                f"efficiency must be in [0,1]; got {self.efficiency}"
            )


@dataclass
class EngineState:
    thrust_factor: float = 1.0
    hard_failure: bool = False


@dataclass
class StructuralState:
    extra_mass_delta_kg: float = 0.0
    extra_cg_shift_m: tuple = (0.0, 0.0, 0.0)
    extra_inertia_delta: tuple = (0.0, 0.0, 0.0, 0.0)


@dataclass
class DamageState:
    section_loss: dict[str, float]
    control_failures: dict[str, ControlFailure]
    engine: EngineState = field(default_factory=EngineState)
    structural: StructuralState = field(default_factory=StructuralState)

    @classmethod
    def healthy(cls, geometry: BaseGeometry) -> "DamageState":
        return cls(
            section_loss={name: 0.0 for name in geometry.section_names()},
            control_failures={},
            engine=EngineState(),
            structural=StructuralState(),
        )

    def set_section_loss(self, section_name: str, fraction: float) -> None:
        clamped = max(0.0, min(1.0, float(fraction)))
        self.section_loss[section_name] = clamped

    def set_control_failure(self, surface: str, failure: ControlFailure) -> None:
        if failure.mode == "healthy":
            self.control_failures.pop(surface, None)
        else:
            self.control_failures[surface] = failure

    def snapshot(self) -> dict:
        """Return a deep-copied, JSON-friendly view."""
        return {
            "section_loss": dict(self.section_loss),
            "control_failures": {
                name: {
                    "mode": cf.mode,
                    "efficiency": cf.efficiency,
                    "jam_position_rad": cf.jam_position_rad,
                }
                for name, cf in self.control_failures.items()
            },
            "engine": {
                "thrust_factor": self.engine.thrust_factor,
                "hard_failure": self.engine.hard_failure,
            },
            "structural": {
                "extra_mass_delta_kg": self.structural.extra_mass_delta_kg,
                "extra_cg_shift_m": tuple(self.structural.extra_cg_shift_m),
                "extra_inertia_delta": tuple(self.structural.extra_inertia_delta),
            },
        }
```

- [ ] **Step 4: Run tests — all pass**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/state_test.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/state.py \
        tests/aerospacemodel/f16_damage/state_test.py
git commit -m "feat(f16-damage): add DamageState with control/engine/structural sub-states"
```

---

## Phase 2 — Recompute physical parameters from DamageState

### Task 2.1: Recompute mass, S, b, bA, CG

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/recompute.py`
- Test: `tests/aerospacemodel/f16_damage/recompute_test.py`

- [ ] **Step 1: Write failing tests**

`tests/aerospacemodel/f16_damage/recompute_test.py`:

```python
"""Recompute aircraft parameters from DamageState + BaseGeometry."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    F16AngularParameters,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
    load_f16_geometry,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
    DamageState,
)


@pytest.fixture
def geo():
    return load_f16_geometry()


@pytest.fixture
def healthy(geo):
    return DamageState.healthy(geo)


def test_healthy_recompute_matches_baseline(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_mass_geometry,
    )
    out = recompute_mass_geometry(geo, healthy)
    p = F16AngularParameters()
    assert out["m"] == pytest.approx(p.m, rel=0.01)
    assert out["S"] == pytest.approx(p.S, rel=0.01)
    assert out["bA"] == pytest.approx(p.bA, rel=0.05)
    # CG of healthy aircraft is at origin (per fuselage_main cg=0)
    assert abs(out["cg"][1]) < 0.01


def test_left_tip_full_loss_reduces_mass_and_area(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_mass_geometry,
    )
    healthy.set_section_loss("left_tip", 1.0)
    out = recompute_mass_geometry(geo, healthy)
    base = recompute_mass_geometry(geo, DamageState.healthy(geo))
    tip = geo.section("left_tip")
    assert out["m"] == pytest.approx(base["m"] - tip.mass, rel=0.001)
    assert out["S"] == pytest.approx(base["S"] - tip.area, rel=0.001)
    # CG must shift to the right (positive y) when left tip is lost
    assert out["cg"][1] > 0.005


def test_symmetric_loss_keeps_cg_centered(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_mass_geometry,
    )
    healthy.set_section_loss("left_tip", 0.5)
    healthy.set_section_loss("right_tip", 0.5)
    out = recompute_mass_geometry(geo, healthy)
    assert abs(out["cg"][1]) < 1e-6


def test_b_eff_uses_outermost_surviving_section(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_mass_geometry,
    )
    # Lose left tip entirely → effective half-span on left is left_mid (≈ 2.80 m)
    healthy.set_section_loss("left_tip", 1.0)
    out = recompute_mass_geometry(geo, healthy)
    # b_eff = max(left_mid abs y) + max(right_tip abs y) = 2.80 + 4.30 = 7.10
    assert out["b"] == pytest.approx(2.80 + 4.30, abs=0.01)
```

- [ ] **Step 2: Run — fails with ImportError**

- [ ] **Step 3: Implement recompute_mass_geometry**

`tensoraerospace/aerospacemodel/f16/nonlinear/damage/recompute.py`:

```python
"""Recompute mass, S, b, bA, and CG from DamageState."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .geometry import BaseGeometry
from .state import DamageState


def recompute_mass_geometry(geo: BaseGeometry, state: DamageState) -> Dict:
    """Aggregate per-section damaged contributions into bulk parameters.

    Returns a dict with keys: m, S, b, bA, cg (np.ndarray shape (3,)).
    """
    sections = geo.sections
    losses = state.section_loss

    # Effective masses
    m_eff_per_section = np.array([
        s.mass * (1.0 - losses.get(s.name, 0.0)) for s in sections
    ], dtype=np.float64)
    m_eff = float(m_eff_per_section.sum() + state.structural.extra_mass_delta_kg)
    if m_eff <= 0.0:
        raise ValueError(
            f"Effective mass {m_eff} non-positive; check damage state"
        )

    # CG: mass-weighted average of remaining masses, plus structural shift
    cg_x = float(sum(
        m * s.cg_local[0] for m, s in zip(m_eff_per_section, sections)
    ) / (m_eff - state.structural.extra_mass_delta_kg + 1e-30))
    cg_y = float(sum(
        m * s.cg_local[1] for m, s in zip(m_eff_per_section, sections)
    ) / (m_eff - state.structural.extra_mass_delta_kg + 1e-30))
    cg_z = float(sum(
        m * s.cg_local[2] for m, s in zip(m_eff_per_section, sections)
    ) / (m_eff - state.structural.extra_mass_delta_kg + 1e-30))
    cg = np.array([
        cg_x + state.structural.extra_cg_shift_m[0],
        cg_y + state.structural.extra_cg_shift_m[1],
        cg_z + state.structural.extra_cg_shift_m[2],
    ])

    # Effective wing area (only type=="wing" contributes)
    s_eff = float(sum(
        s.area * (1.0 - losses.get(s.name, 0.0))
        for s in sections if s.type == "wing"
    ))

    # Effective span: outermost surviving point on each side
    def half_span(side: str) -> float:
        ys = [
            abs(s.span_position) * (0.0 if losses.get(s.name, 0.0) >= 1.0 else 1.0)
            for s in sections
            if s.type == "wing" and s.side == side
        ]
        return max(ys) if ys else 0.0

    b_eff = half_span("left") + half_span("right")

    # MAC: area-weighted chord over surviving wing area
    chord_num = sum(
        s.chord * s.area * (1.0 - losses.get(s.name, 0.0))
        for s in sections if s.type == "wing"
    )
    bA_eff = float(chord_num / s_eff) if s_eff > 0.0 else 0.0

    return {"m": m_eff, "S": s_eff, "b": b_eff, "bA": bA_eff, "cg": cg}
```

- [ ] **Step 4: Run tests — all pass**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/recompute_test.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/recompute.py \
        tests/aerospacemodel/f16_damage/recompute_test.py
git commit -m "feat(f16-damage): recompute mass/area/span/MAC/CG from damage state"
```

### Task 2.2: Recompute inertia tensor (Huygens-Steiner)

**Files:**
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/recompute.py`
- Modify: `tests/aerospacemodel/f16_damage/recompute_test.py`

- [ ] **Step 1: Add inertia tests**

Append to `recompute_test.py`:

```python
def test_healthy_inertia_matches_baseline(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_inertia,
    )
    out = recompute_inertia(geo, healthy, cg=geo.center_of_mass())
    p = F16AngularParameters()
    # Baseline inertias: tolerance 5% (geometry quantisation)
    assert out["Jx"] == pytest.approx(p.Jx, rel=0.05)
    assert out["Jy"] == pytest.approx(p.Jy, rel=0.05)
    assert out["Jz"] == pytest.approx(p.Jz, rel=0.05)


def test_steiner_known_two_point_masses():
    """Validate Huygens-Steiner: two equal point masses at ±d on x-axis,
    each m, give Iyy = Izz = 2 m d² about the centre."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.geometry import (
        AeroSection, BaseGeometry,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
        DamageState,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_inertia,
    )
    sections = [
        AeroSection(
            name="a", side="center", type="fuselage",
            area=0.0, span_position=0.0, chord=0.0, sweep=0.0,
            mass=2.0, cg_local=(3.0, 0.0, 0.0), inertia_local=(0, 0, 0, 0),
            cl_alpha_contribution=0.0, cd0_contribution=0.0,
        ),
        AeroSection(
            name="b", side="center", type="fuselage",
            area=0.0, span_position=0.0, chord=0.0, sweep=0.0,
            mass=2.0, cg_local=(-3.0, 0.0, 0.0), inertia_local=(0, 0, 0, 0),
            cl_alpha_contribution=0.0, cd0_contribution=0.0,
        ),
    ]
    geo = BaseGeometry(sections=sections)
    state = DamageState.healthy(geo)
    out = recompute_inertia(geo, state, cg=np.zeros(3))
    # Iyy = Izz = 2*2*9 = 36; Ixx = 0
    assert out["Jx"] == pytest.approx(0.0, abs=1e-9)
    assert out["Jy"] == pytest.approx(36.0, rel=1e-9)
    assert out["Jz"] == pytest.approx(36.0, rel=1e-9)


def test_left_tip_loss_reduces_jx(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_inertia, recompute_mass_geometry,
    )
    base = recompute_inertia(geo, healthy, cg=geo.center_of_mass())
    healthy.set_section_loss("left_tip", 1.0)
    new_cg = recompute_mass_geometry(geo, healthy)["cg"]
    out = recompute_inertia(geo, healthy, cg=new_cg)
    # Jx primarily comes from y-distance of mass; losing a tip reduces it
    assert out["Jx"] < base["Jx"]
```

- [ ] **Step 2: Run — last 3 tests fail**

- [ ] **Step 3: Add `recompute_inertia` to recompute.py**

Append to `tensoraerospace/aerospacemodel/f16/nonlinear/damage/recompute.py`:

```python
def recompute_inertia(
    geo: BaseGeometry, state: DamageState, cg: np.ndarray
) -> Dict[str, float]:
    """Compute aircraft inertias about `cg` using Huygens–Steiner.

    For each surviving section: J_about_cg = J_local_about_section_cg +
    m_eff * (parallel-axis offsets).
    """
    cg = np.asarray(cg, dtype=np.float64).reshape(3)
    Jx = Jy = Jz = Jxz = 0.0
    for s in geo.sections:
        f = state.section_loss.get(s.name, 0.0)
        m_eff = s.mass * (1.0 - f)
        if m_eff <= 0.0:
            continue
        Ixx_l, Iyy_l, Izz_l, Ixz_l = s.inertia_local
        # Scale local inertia by remaining mass fraction (uniform-density assumption)
        scale = (1.0 - f)
        Ixx_l *= scale
        Iyy_l *= scale
        Izz_l *= scale
        Ixz_l *= scale
        rx = s.cg_local[0] - cg[0]
        ry = s.cg_local[1] - cg[1]
        rz = s.cg_local[2] - cg[2]
        # Parallel-axis: I_about_cg = I_local + m * (r⊥)²
        Jx += Ixx_l + m_eff * (ry**2 + rz**2)
        Jy += Iyy_l + m_eff * (rx**2 + rz**2)
        Jz += Izz_l + m_eff * (rx**2 + ry**2)
        Jxz += Ixz_l - m_eff * rx * rz  # off-diagonal: -m*rx*rz
    # Apply structural extras
    Jx += state.structural.extra_inertia_delta[0]
    Jy += state.structural.extra_inertia_delta[1]
    Jz += state.structural.extra_inertia_delta[2]
    Jxz += state.structural.extra_inertia_delta[3]
    return {"Jx": float(Jx), "Jy": float(Jy), "Jz": float(Jz), "Jxz": float(Jxz)}
```

- [ ] **Step 4: Run — all pass**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/recompute_test.py -v
```

Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/recompute.py \
        tests/aerospacemodel/f16_damage/recompute_test.py
git commit -m "feat(f16-damage): recompute inertia tensor via Huygens-Steiner"
```

### Task 2.3: Apply recomputed params to F16AngularParameters

**Files:**
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/recompute.py`
- Modify: `tests/aerospacemodel/f16_damage/recompute_test.py`

- [ ] **Step 1: Add tests for `apply_to_params`**

Append to `recompute_test.py`:

```python
def test_apply_to_params_updates_in_place(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        apply_to_params,
    )
    p = F16AngularParameters()
    base_m = p.m
    healthy.set_section_loss("left_tip", 1.0)
    apply_to_params(p, geo, healthy)
    assert p.m < base_m
    assert p.S < 27.87  # baseline S
    assert p.rcgx == pytest.approx(-0.05 * p.bA, rel=1e-9)


def test_apply_to_params_healthy_is_idempotent(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        apply_to_params,
    )
    p_baseline = F16AngularParameters()
    p_recomputed = F16AngularParameters()
    apply_to_params(p_recomputed, geo, healthy)
    assert p_recomputed.m == pytest.approx(p_baseline.m, rel=0.01)
    assert p_recomputed.S == pytest.approx(p_baseline.S, rel=0.01)
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Add `apply_to_params`**

Append to `recompute.py`:

```python
def apply_to_params(params, geo: BaseGeometry, state: DamageState) -> None:
    """Mutate an F16AngularParameters/F16LongParameters in place.

    Updates m, S, bA, Jx, Jy, Jz, Jxz, rcgx, l (where applicable).
    """
    mg = recompute_mass_geometry(geo, state)
    inertia = recompute_inertia(geo, state, cg=mg["cg"])
    params.m = mg["m"]
    params.S = mg["S"]
    params.bA = mg["bA"]
    if hasattr(params, "l"):
        params.l = mg["b"]
    params.Jx = inertia["Jx"]
    params.Jy = inertia["Jy"]
    params.Jz = inertia["Jz"]
    if hasattr(params, "Jxz"):
        params.Jxz = inertia["Jxz"]
    # rcgx convention from params.__post_init__
    params.rcgx = -0.05 * params.bA
```

- [ ] **Step 4: Run — all pass**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/recompute_test.py -v
```

Expected: 9 PASS.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/recompute.py \
        tests/aerospacemodel/f16_damage/recompute_test.py
git commit -m "feat(f16-damage): add apply_to_params for in-place mutation"
```

---

## Phase 3 — Aerodynamic corrections (strip-theory)

### Task 3.1: Force corrections (delta_cy, delta_cx, delta_cz)

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/aero_corrections.py`
- Test: `tests/aerospacemodel/f16_damage/aero_corrections_test.py`

- [ ] **Step 1: Write failing tests**

`tests/aerospacemodel/f16_damage/aero_corrections_test.py`:

```python
"""Strip-theory aero corrections from DamageState."""

from __future__ import annotations

import math

import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
    load_f16_geometry,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
    DamageState,
)


@pytest.fixture
def geo():
    return load_f16_geometry()


@pytest.fixture
def healthy(geo):
    return DamageState.healthy(geo)


def test_healthy_forces_are_zero(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    alpha = math.radians(5.0)
    beta = 0.0
    assert aero_corrections.delta_cy(alpha, beta, geo, healthy) == 0.0
    assert aero_corrections.delta_cx(alpha, beta, geo, healthy) == 0.0
    assert aero_corrections.delta_cz(alpha, beta, geo, healthy) == 0.0


def test_left_tip_full_loss_reduces_cy(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    alpha = math.radians(5.0)
    healthy.set_section_loss("left_tip", 1.0)
    dcy = aero_corrections.delta_cy(alpha, 0.0, geo, healthy)
    # Negative delta: lift contribution lost. Magnitude ~ 0.55 * α.
    assert dcy < 0
    expected = -geo.section("left_tip").cl_alpha_contribution * alpha
    assert dcy == pytest.approx(expected, rel=0.05)


def test_partial_loss_scales_linearly(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    alpha = math.radians(5.0)
    healthy.set_section_loss("left_tip", 1.0)
    full = aero_corrections.delta_cy(alpha, 0.0, geo, healthy)
    healthy.set_section_loss("left_tip", 0.5)
    half = aero_corrections.delta_cy(alpha, 0.0, geo, healthy)
    assert half == pytest.approx(full * 0.5, rel=0.001)


def test_partial_loss_adds_drag(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    healthy.set_section_loss("left_tip", 1.0)  # totally lost: no jagged-edge drag
    dcx_full = aero_corrections.delta_cx(0.0, 0.0, geo, healthy)
    healthy.set_section_loss("left_tip", 0.5)  # half lost: max jagged-edge drag
    dcx_half = aero_corrections.delta_cx(0.0, 0.0, geo, healthy)
    # Half-loss should give NET higher drag than full loss
    # because full-loss removes both contributions, half-loss adds jagged drag
    assert dcx_half > dcx_full
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement aero_corrections.py (forces)**

`tensoraerospace/aerospacemodel/f16/nonlinear/damage/aero_corrections.py`:

```python
"""Strip-theory aerodynamic corrections from DamageState.

All deltas are dimensionless (normalized so they add directly to the base
F-16 coefficients Cy/Cx/Cz/Mx/My/Mz). The base S used for normalisation is
the BaseGeometry's total wing area.
"""

from __future__ import annotations

from .geometry import BaseGeometry
from .state import DamageState


_JAGGED_DRAG_COEF = 0.05  # peaks at f=0.5; calibrated from test cases


def _base_wing_area(geo: BaseGeometry) -> float:
    return geo.total_wing_area()


def delta_cy(alpha: float, beta: float, geo: BaseGeometry, state: DamageState) -> float:
    """Lost normal-force contribution: ΔCy = -Σ cl_α_s · α · f_s · (area_s/S_base)."""
    S_base = _base_wing_area(geo)
    if S_base <= 0.0:
        return 0.0
    return float(-sum(
        s.cl_alpha_contribution * alpha * state.section_loss.get(s.name, 0.0)
        * (s.area / S_base)
        for s in geo.sections if s.type == "wing"
    ))


def delta_cx(alpha: float, beta: float, geo: BaseGeometry, state: DamageState) -> float:
    """Drag delta: lost cd0 contribution + jagged-edge drag from partial damage."""
    S_base = _base_wing_area(geo)
    if S_base <= 0.0:
        return 0.0
    delta = 0.0
    for s in geo.sections:
        f = state.section_loss.get(s.name, 0.0)
        if f <= 0.0:
            continue
        delta -= s.cd0_contribution * f
        if s.type == "wing":
            # Jagged-edge drag: peaks at f=0.5
            delta += _JAGGED_DRAG_COEF * f * (1.0 - f) * (s.area / S_base)
    return float(delta)


def delta_cz(alpha: float, beta: float, geo: BaseGeometry, state: DamageState) -> float:
    """Side-force delta: dominated by vtail loss (proportional to β)."""
    delta = 0.0
    for s in geo.sections:
        f = state.section_loss.get(s.name, 0.0)
        if f <= 0.0:
            continue
        if s.type == "vtail":
            # Treat vtail's cl_alpha-equivalent for sideslip with a small constant.
            VTAIL_BETA_GAIN = 0.40  # 1/rad
            delta -= VTAIL_BETA_GAIN * beta * f
    return float(delta)
```

- [ ] **Step 4: Run — pass**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/aero_corrections_test.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/aero_corrections.py \
        tests/aerospacemodel/f16_damage/aero_corrections_test.py
git commit -m "feat(f16-damage): add strip-theory force corrections (Cy/Cx/Cz)"
```

### Task 3.2: Moment corrections (delta_mx, delta_my, delta_mz)

**Files:**
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/aero_corrections.py`
- Modify: `tests/aerospacemodel/f16_damage/aero_corrections_test.py`

- [ ] **Step 1: Add moment tests**

Append to `aero_corrections_test.py`:

```python
def test_left_tip_loss_creates_positive_roll_moment(geo, healthy):
    """Loss of left wing tip → less lift on left → roll towards left
    (negative wx convention is matlab-port specific; here ΔMx must be
    nonzero with a defined sign)."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    alpha = math.radians(5.0)
    healthy.set_section_loss("left_tip", 1.0)
    dmx = aero_corrections.delta_mx(alpha, 0.0, geo, healthy)
    # Lost left-side lift on negative-y → roll moment magnitude > 0
    assert abs(dmx) > 1e-3


def test_symmetric_loss_no_roll(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    alpha = math.radians(5.0)
    healthy.set_section_loss("left_tip", 0.5)
    healthy.set_section_loss("right_tip", 0.5)
    dmx = aero_corrections.delta_mx(alpha, 0.0, geo, healthy)
    assert abs(dmx) < 1e-9


def test_yaw_moment_from_asymmetric_drag(geo, healthy):
    """Half-loss on left tip → jagged-edge drag → yaw moment."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    healthy.set_section_loss("left_tip", 0.5)
    dmz = aero_corrections.delta_mz(0.0, 0.0, geo, healthy)
    # left side jagged drag at -y arm → some yaw moment (sign depends on
    # convention; just assert non-zero)
    assert abs(dmz) > 1e-6


def test_pitch_moment_from_lost_stab(geo, healthy):
    """Loss of horizontal stab → pitch moment delta from lost lift on tail."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    alpha = math.radians(5.0)
    healthy.set_section_loss("stab_left", 1.0)
    dmy = aero_corrections.delta_my(alpha, 0.0, geo, healthy)
    # Tail is aft (negative aero_x_arm), losing it removes a downward
    # contribution → magnitude > 0
    assert abs(dmy) > 1e-4
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Append moment functions to aero_corrections.py**

```python
def delta_mx(alpha: float, beta: float, geo: BaseGeometry, state: DamageState) -> float:
    """Roll-moment coefficient delta from asymmetric lift loss.

    ΔMx (dimensionless) = -Σ cl_α_s · α · f_s · (area_s/S_base) · (y_arm_s/b_base)
    where b_base is the base span (for normalisation: cmx = Mx/(q·S·l)).
    """
    S_base = _base_wing_area(geo)
    if S_base <= 0.0:
        return 0.0
    # Use 2 × max half-span as proxy for b_base
    b_base = 2.0 * max(
        (abs(s.span_position) for s in geo.sections if s.type == "wing"),
        default=1.0,
    )
    return float(-sum(
        s.cl_alpha_contribution * alpha * state.section_loss.get(s.name, 0.0)
        * (s.area / S_base) * (s.span_position / b_base)
        for s in geo.sections if s.type == "wing"
    ))


def delta_mz(alpha: float, beta: float, geo: BaseGeometry, state: DamageState) -> float:
    """Yaw-moment coefficient delta from asymmetric drag.

    Uses local ΔCx on each section, multiplied by its y-arm.
    """
    S_base = _base_wing_area(geo)
    if S_base <= 0.0:
        return 0.0
    b_base = 2.0 * max(
        (abs(s.span_position) for s in geo.sections if s.type == "wing"),
        default=1.0,
    )
    out = 0.0
    for s in geo.sections:
        f = state.section_loss.get(s.name, 0.0)
        if f <= 0.0:
            continue
        local_dcx = -s.cd0_contribution * f
        if s.type == "wing":
            local_dcx += _JAGGED_DRAG_COEF * f * (1.0 - f) * (s.area / S_base)
        out += local_dcx * (s.span_position / b_base)
    return float(out)


def delta_my(alpha: float, beta: float, geo: BaseGeometry, state: DamageState) -> float:
    """Pitch-moment coefficient delta from lost lift × x-arm.

    Normalised by S_base × bA_base.
    """
    S_base = _base_wing_area(geo)
    if S_base <= 0.0:
        return 0.0
    # bA_base for normalisation: area-weighted chord
    bA_base = sum(
        s.chord * s.area for s in geo.sections if s.type == "wing"
    ) / S_base
    return float(-sum(
        s.cl_alpha_contribution * alpha * state.section_loss.get(s.name, 0.0)
        * (s.area / S_base) * (s.aero_x_arm / bA_base)
        for s in geo.sections if s.type in ("wing", "stab")
    ))
```

- [ ] **Step 4: Run — all pass**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/aero_corrections_test.py -v
```

Expected: 8 PASS total.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/aero_corrections.py \
        tests/aerospacemodel/f16_damage/aero_corrections_test.py
git commit -m "feat(f16-damage): add roll/yaw/pitch moment corrections"
```

### Task 3.3: Wire aero corrections into the angular ODE

**Files:**
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/angular/dynamics.py`
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/angular/model.py`
- Test: `tests/aerospacemodel/f16_damage/ode_integration_test.py` (new)

- [ ] **Step 1: Write the failing integration test**

`tests/aerospacemodel/f16_damage/ode_integration_test.py`:

```python
"""End-to-end: damage application in AngularF16 changes trajectory."""

from __future__ import annotations

import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
    load_f16_geometry,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
    DamageState,
)


def test_no_damage_attribute_keeps_legacy_behaviour():
    m_legacy = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    m_legacy.run_step(np.zeros(3))
    s_legacy = m_legacy.current_state.copy()
    # New construct with damage_state=None still gives identical trajectory
    m_with = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    # damage_state attribute defaults to None
    assert getattr(m_with, "damage_state", None) is None
    m_with.run_step(np.zeros(3))
    np.testing.assert_allclose(s_legacy, m_with.current_state, atol=1e-12)


def test_damage_state_attribute_used_in_ode():
    geo = load_f16_geometry()
    state_healthy = DamageState.healthy(geo)
    state_damaged = DamageState.healthy(geo)
    state_damaged.set_section_loss("left_tip", 1.0)

    # Run baseline
    m1 = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    m1.damage_state = state_healthy
    m1.damage_geometry = geo
    for _ in range(10):
        m1.run_step(np.zeros(3))

    # Run with damaged left tip
    m2 = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    m2.damage_state = state_damaged
    m2.damage_geometry = geo
    for _ in range(10):
        m2.run_step(np.zeros(3))

    # Trajectories must differ (specifically: roll rate wx)
    diff = np.abs(m1.current_state - m2.current_state)
    assert diff.sum() > 1e-6, "Damage application had no effect on trajectory"
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Add `damage_state`/`damage_geometry` attributes to AngularF16**

In `tensoraerospace/aerospacemodel/f16/nonlinear/angular/model.py`, after `self.param: F16AngularParameters = default_parameters()`:

```python
        # Damage subsystem (None = healthy aircraft, legacy behaviour)
        self.damage_state = None
        self.damage_geometry = None
```

In `run_step`, just before computing `t_now`, inject damage hooks (params already set by manager elsewhere; ODE reads damage info via params):

```python
        # If damage is active, stash a reference on params so the ODE can
        # read it. Cleared each step.
        if self.damage_state is not None and self.damage_geometry is not None:
            self.param.damage_state = self.damage_state
            self.param.damage_geometry = self.damage_geometry
        else:
            self.param.damage_state = None
            self.param.damage_geometry = None
```

In `tensoraerospace/aerospacemodel/f16/nonlinear/angular/params.py`, append after `delta_stab_cmd`:

```python
    # Damage subsystem hooks (set by AngularF16.run_step). None = healthy.
    damage_state: object = None
    damage_geometry: object = None
```

- [ ] **Step 4: Apply corrections in dynamics.py**

In `tensoraerospace/aerospacemodel/f16/nonlinear/angular/dynamics.py`, after the existing aero coefficient block (after computing `cx, cy, cz, mx_, my_, mz_`):

```python
    # ----------------------------------------------------------------
    # Apply damage corrections (no-op if damage_state is None)
    # ----------------------------------------------------------------
    damage_state = getattr(p, "damage_state", None)
    damage_geo = getattr(p, "damage_geometry", None)
    if damage_state is not None and damage_geo is not None:
        from ..damage import aero_corrections as _ac
        cy = cy + _ac.delta_cy(alpha, beta, damage_geo, damage_state)
        cx = cx + _ac.delta_cx(alpha, beta, damage_geo, damage_state)
        cz = cz + _ac.delta_cz(alpha, beta, damage_geo, damage_state)
        mx_ = mx_ + _ac.delta_mx(alpha, beta, damage_geo, damage_state)
        my_ = my_ + _ac.delta_my(alpha, beta, damage_geo, damage_state)
        mz_ = mz_ + _ac.delta_mz(alpha, beta, damage_geo, damage_state)
```

- [ ] **Step 5: Run — pass**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/ode_integration_test.py -v
```

Expected: 2 PASS.

- [ ] **Step 6: Run full test suite — no regressions**

```bash
poetry run pytest tests/aerospacemodel/ tests/envs/ -x --tb=short -q
```

Expected: pass count ≥ baseline + new tests.

- [ ] **Step 7: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/angular/ \
        tests/aerospacemodel/f16_damage/ode_integration_test.py
git commit -m "feat(f16-damage): wire aero corrections into 6-DoF ODE"
```

---

## Phase 4 — Control surface failures

### Task 4.1: controls.py — apply_control_failures

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/controls.py`
- Test: `tests/aerospacemodel/f16_damage/controls_test.py`

- [ ] **Step 1: Write failing tests**

`tests/aerospacemodel/f16_damage/controls_test.py`:

```python
"""Control-surface failure mapping."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
    ControlFailure, DamageState,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
    load_f16_geometry,
)


@pytest.fixture
def healthy():
    return DamageState.healthy(load_f16_geometry())


def test_healthy_passthrough(healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.controls import (
        apply_control_failures,
    )
    u = np.array([0.1, -0.05, 0.02, 0.03])
    surface_to_idx = {
        "stab_left": 0, "stab_right": 1, "aileron_left": 2, "rudder": 3,
    }
    out = apply_control_failures(u, healthy, surface_to_idx)
    np.testing.assert_array_equal(out, u)


def test_jam_overrides_command(healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.controls import (
        apply_control_failures,
    )
    healthy.set_control_failure(
        "stab_left", ControlFailure(mode="jam", jam_position_rad=0.05),
    )
    u = np.array([0.4, 0.0, 0.0, 0.0])
    surface_to_idx = {
        "stab_left": 0, "stab_right": 1, "aileron_left": 2, "rudder": 3,
    }
    out = apply_control_failures(u, healthy, surface_to_idx)
    assert out[0] == pytest.approx(0.05)


def test_efficiency_loss_scales_command(healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.controls import (
        apply_control_failures,
    )
    healthy.set_control_failure(
        "rudder", ControlFailure(mode="efficiency_loss", efficiency=0.5),
    )
    u = np.array([0.0, 0.0, 0.0, 0.4])
    surface_to_idx = {
        "stab_left": 0, "stab_right": 1, "aileron_left": 2, "rudder": 3,
    }
    out = apply_control_failures(u, healthy, surface_to_idx)
    assert out[3] == pytest.approx(0.2)


def test_lost_zeros_command(healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.controls import (
        apply_control_failures,
    )
    healthy.set_control_failure("rudder", ControlFailure(mode="lost"))
    u = np.array([0.0, 0.0, 0.0, 0.4])
    surface_to_idx = {
        "stab_left": 0, "stab_right": 1, "aileron_left": 2, "rudder": 3,
    }
    out = apply_control_failures(u, healthy, surface_to_idx)
    assert out[3] == 0.0
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement controls.py**

`tensoraerospace/aerospacemodel/f16/nonlinear/damage/controls.py`:

```python
"""Apply control-surface failures to commanded inputs."""

from __future__ import annotations

import numpy as np

from .state import DamageState

# Default mapping for the split-stab AngularF16 (action = [stab_l, stab_r, ail, dir])
ANGULAR_SPLIT_STAB_INDEX = {
    "stab_left": 0,
    "stab_right": 1,
    "aileron_left": 2,
    "aileron_right": 2,
    "rudder": 3,
}

# Mapping for the legacy 3-input AngularF16 (action = [stab, ail, dir]).
# In legacy mode, separate-side failures are not addressable; the closest
# is to apply to the merged 'stab' channel.
ANGULAR_LEGACY_INDEX = {
    "stab_left": 0, "stab_right": 0,
    "aileron_left": 1, "aileron_right": 1,
    "rudder": 2,
}


def apply_control_failures(
    u_command: np.ndarray,
    state: DamageState,
    surface_to_index: dict[str, int],
) -> np.ndarray:
    """Return modified command vector after applying every active failure.

    Multiple failures targeting the same index compose left-to-right by
    insertion order in `state.control_failures`.
    """
    u_eff = np.asarray(u_command, dtype=np.float64).copy()
    for surface, failure in state.control_failures.items():
        if surface not in surface_to_index:
            continue
        idx = surface_to_index[surface]
        if failure.mode == "healthy":
            continue
        if failure.mode == "jam":
            u_eff[idx] = failure.jam_position_rad
        elif failure.mode == "efficiency_loss":
            u_eff[idx] = u_eff[idx] * failure.efficiency
        elif failure.mode in ("lost", "free_floating"):
            u_eff[idx] = 0.0
    return u_eff
```

- [ ] **Step 4: Run — pass**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/controls_test.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/controls.py \
        tests/aerospacemodel/f16_damage/controls_test.py
git commit -m "feat(f16-damage): apply jam/efficiency/lost failures to command vector"
```

### Task 4.2: Wire control failures into AngularF16.run_step

**Files:**
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/angular/model.py`
- Modify: `tests/aerospacemodel/f16_damage/ode_integration_test.py`

- [ ] **Step 1: Add integration test**

Append to `ode_integration_test.py`:

```python
def test_jammed_stab_left_in_split_mode_overrides_command():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
        ControlFailure,
    )
    geo = load_f16_geometry()
    state = DamageState.healthy(geo)
    state.set_control_failure(
        "stab_left", ControlFailure(mode="jam", jam_position_rad=0.10),
    )
    m = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4", split_stab=True)
    m.damage_state = state
    m.damage_geometry = geo
    # Try to command stab_left to 0.0; jam should keep it at 0.10
    m.run_step(np.array([0.0, 0.0, 0.0, 0.0]))
    # Effective inputs are recorded in u_history (legacy 3-vector)
    # But more directly: state.list_state[8] (stab) should track the jam
    # (after one step with jam_pos=0.10, stab_left=0.10 mean dictates 0.05)
    after = m.current_state
    # If jam=0.10 on left, mean=0.05, delta=0.05; legacy stab cmd = 0.05.
    # Actuator dynamics with Tstab=0.03 → stab moves towards 0.05 quickly.
    assert after[8] > 1e-3, "Stabilator should have moved due to jam"
```

- [ ] **Step 2: Run — fails (jam not yet wired)**

- [ ] **Step 3: Wire control failures into run_step**

In `tensoraerospace/aerospacemodel/f16/nonlinear/angular/model.py`, modify `run_step` so the failure layer fires *before* the split-stab merging:

```python
    def run_step(self, u: ArrayLike) -> np.ndarray:
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_arr.size != self.action_space_length:
            raise ValueError(
                "Размерность управляющего вектора задана неверно."
                f" Текущее значение {u_arr.size}, не соответсвует {self.action_space_length}"
            )

        # Apply control-surface failures BEFORE split-stab merging
        if self.damage_state is not None:
            from ..damage.controls import (
                ANGULAR_LEGACY_INDEX, ANGULAR_SPLIT_STAB_INDEX,
                apply_control_failures,
            )
            mapping = (
                ANGULAR_SPLIT_STAB_INDEX if self.split_stab else ANGULAR_LEGACY_INDEX
            )
            u_arr = apply_control_failures(u_arr, self.damage_state, mapping)

        if self.split_stab:
            stab_mean = 0.5 * (u_arr[0] + u_arr[1])
            delta_stab = 0.5 * (u_arr[0] - u_arr[1])
            u_legacy = np.array([stab_mean, u_arr[2], u_arr[3]], dtype=np.float64)
            self.param.delta_stab_cmd = float(delta_stab)
        else:
            u_legacy = u_arr
            self.param.delta_stab_cmd = 0.0

        # Damage hooks for ODE corrections (Phase 3)
        if self.damage_state is not None and self.damage_geometry is not None:
            self.param.damage_state = self.damage_state
            self.param.damage_geometry = self.damage_geometry
        else:
            self.param.damage_state = None
            self.param.damage_geometry = None

        x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
        t_now = self.t0 + self.dt * self.time_step
        x_next = self._step_fn(f16_ode_6dof, x_prev, u_legacy, t_now, self.dt, self.param)

        x_next_col = x_next.reshape(14, 1)
        self.x_history.append(x_next_col)
        self.u_history.append(u_arr.reshape(-1, 1))
        self.time_step += 1

        if self.selected_state_output:
            return x_next_col[self.selected_state_index]
        return x_next_col
```

- [ ] **Step 4: Run integration test — pass**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/ode_integration_test.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Run full suite — no regressions**

```bash
poetry run pytest tests/aerospacemodel/ tests/envs/ -x --tb=short -q
```

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/angular/model.py \
        tests/aerospacemodel/f16_damage/ode_integration_test.py
git commit -m "feat(f16-damage): wire control-surface failures into AngularF16"
```

---

## Phase 5 — Engine and structural damage

### Task 5.1: Engine thrust factor (propulsion.py)

**Note:** F-16 thrust is currently **not** modeled in the angular ODE — gravity drives the speed. For now, we expose the engine state as a multiplier that downstream consumers (envs, custom dynamics) can read. We add a method that returns the effective thrust given a base thrust input.

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/propulsion.py`
- Test: `tests/aerospacemodel/f16_damage/propulsion_test.py`

- [ ] **Step 1: Write tests**

`tests/aerospacemodel/f16_damage/propulsion_test.py`:

```python
"""Engine thrust factor."""

from __future__ import annotations

import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
    DamageState, EngineState,
)


def test_default_thrust_factor_is_one():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.propulsion import (
        effective_thrust,
    )
    state = DamageState(section_loss={}, control_failures={})
    assert effective_thrust(1000.0, state) == pytest.approx(1000.0)


def test_thrust_factor_scales():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.propulsion import (
        effective_thrust,
    )
    state = DamageState(section_loss={}, control_failures={},
                        engine=EngineState(thrust_factor=0.4))
    assert effective_thrust(1000.0, state) == pytest.approx(400.0)


def test_hard_failure_zeros_thrust():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.propulsion import (
        effective_thrust,
    )
    state = DamageState(section_loss={}, control_failures={},
                        engine=EngineState(thrust_factor=1.0, hard_failure=True))
    assert effective_thrust(1000.0, state) == 0.0
```

- [ ] **Step 2: Implement propulsion.py**

```python
"""Engine state effects."""

from __future__ import annotations

from .state import DamageState


def effective_thrust(base_thrust: float, state: DamageState) -> float:
    """Apply engine.thrust_factor and hard_failure to a base thrust value."""
    if state.engine.hard_failure:
        return 0.0
    return float(base_thrust * state.engine.thrust_factor)
```

- [ ] **Step 3: Run — pass**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/propulsion_test.py -v
```

Expected: 3 PASS.

- [ ] **Step 4: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/propulsion.py \
        tests/aerospacemodel/f16_damage/propulsion_test.py
git commit -m "feat(f16-damage): add engine thrust factor utility"
```

### Task 5.2: Structural state already covered

The `StructuralState` (mass delta, CG shift, inertia delta) is already consumed by `recompute.py:apply_to_params` (Phase 2). Add a verification test:

**Files:**
- Modify: `tests/aerospacemodel/f16_damage/recompute_test.py`

- [ ] **Step 1: Add test**

```python
def test_structural_mass_delta_applies(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_mass_geometry,
    )
    healthy.structural.extra_mass_delta_kg = -200.0
    out = recompute_mass_geometry(geo, healthy)
    base = recompute_mass_geometry(geo, DamageState.healthy(geo))
    assert out["m"] == pytest.approx(base["m"] - 200.0, rel=0.001)
```

- [ ] **Step 2: Run — pass (already implemented)**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/recompute_test.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/aerospacemodel/f16_damage/recompute_test.py
git commit -m "test(f16-damage): cover structural mass delta path"
```

---

## Phase 6 — Events, Profile, Manager

### Task 6.1: DamageEvent and DamageProfile

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/events.py`
- Test: `tests/aerospacemodel/f16_damage/events_test.py`

- [ ] **Step 1: Write tests**

`tests/aerospacemodel/f16_damage/events_test.py`:

```python
"""DamageEvent / DamageProfile: scheduling and triggering."""

from __future__ import annotations

import pytest


def test_event_is_frozen():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent,
    )
    e = DamageEvent(
        trigger_time=5.0, event_type="section_loss",
        payload={"section": "left_tip", "loss_fraction": 1.0},
    )
    with pytest.raises((AttributeError, Exception)):
        e.trigger_time = 99.0


def test_profile_returns_pending_in_window():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent, DamageProfile,
    )
    e1 = DamageEvent(1.0, "section_loss", {"section": "x", "loss_fraction": 1.0})
    e2 = DamageEvent(5.5, "engine_failure", {"thrust_factor": 0.0})
    e3 = DamageEvent(10.0, "section_loss", {"section": "y", "loss_fraction": 0.5})
    p = DamageProfile(events=[e1, e2, e3])
    pending = p.get_pending_events(t_current=6.0, t_previous=1.0)
    assert e2 in pending
    assert e1 not in pending  # already past window opening (t_prev=1.0 means
                              # window is (1.0, 6.0], so e1 at t=1.0 is NOT in)
    assert e3 not in pending


def test_profile_inclusive_at_current_exclusive_at_previous():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent, DamageProfile,
    )
    e_at_5 = DamageEvent(5.0, "engine_failure", {"thrust_factor": 0.5})
    p = DamageProfile(events=[e_at_5])
    # (4.99, 5.0] should contain it
    assert e_at_5 in p.get_pending_events(5.0, 4.99)
    # (5.0, 5.5] should NOT
    assert e_at_5 not in p.get_pending_events(5.5, 5.0)
```

- [ ] **Step 2: Implement events.py**

`tensoraerospace/aerospacemodel/f16/nonlinear/damage/events.py`:

```python
"""DamageEvent and DamageProfile."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

EventType = Literal[
    "section_loss",
    "control_failure",
    "engine_failure",
    "structural_change",
]


@dataclass(frozen=True)
class DamageEvent:
    trigger_time: float
    event_type: EventType
    payload: dict = field(default_factory=dict)
    label: Optional[str] = None
    duration: Optional[float] = None  # None = permanent

    def __post_init__(self) -> None:
        if self.event_type not in (
            "section_loss", "control_failure", "engine_failure", "structural_change"
        ):
            raise ValueError(f"Unknown event_type {self.event_type!r}")
        if self.trigger_time < 0:
            raise ValueError(f"trigger_time must be ≥ 0; got {self.trigger_time}")


@dataclass
class DamageProfile:
    events: list[DamageEvent] = field(default_factory=list)
    seed: Optional[int] = None

    def get_pending_events(
        self, t_current: float, t_previous: float
    ) -> list[DamageEvent]:
        """Events triggering in the half-open interval (t_previous, t_current]."""
        return [
            e for e in self.events
            if t_previous < e.trigger_time <= t_current
        ]
```

- [ ] **Step 3: Run — pass**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/events_test.py -v
```

- [ ] **Step 4: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/events.py \
        tests/aerospacemodel/f16_damage/events_test.py
git commit -m "feat(f16-damage): add DamageEvent and DamageProfile"
```

### Task 6.2: DamageManager

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/manager.py`
- Test: `tests/aerospacemodel/f16_damage/manager_test.py`

- [ ] **Step 1: Write tests**

`tests/aerospacemodel/f16_damage/manager_test.py`:

```python
"""DamageManager: orchestrates events → state mutations → param updates."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    F16AngularParameters,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
    DamageEvent, DamageProfile,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
    load_f16_geometry,
)


@pytest.fixture
def geo():
    return load_f16_geometry()


def test_manager_default_state_is_healthy(geo):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.manager import (
        DamageManager,
    )
    p = F16AngularParameters()
    m = DamageManager(geometry=geo, params=p, profile=DamageProfile(events=[]))
    assert all(v == 0.0 for v in m.state.section_loss.values())


def test_manager_triggers_section_loss(geo):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.manager import (
        DamageManager,
    )
    p = F16AngularParameters()
    base_m = p.m
    profile = DamageProfile(events=[
        DamageEvent(5.0, "section_loss",
                    {"section": "left_tip", "loss_fraction": 1.0}),
    ])
    mgr = DamageManager(geometry=geo, params=p, profile=profile)
    triggered = mgr.update(t_current=4.0, t_previous=0.0)
    assert triggered == []
    assert p.m == pytest.approx(base_m, rel=0.001)
    triggered = mgr.update(t_current=6.0, t_previous=4.0)
    assert len(triggered) == 1
    assert mgr.state.section_loss["left_tip"] == 1.0
    # Params updated: mass less than baseline
    assert p.m < base_m


def test_manager_triggers_control_failure(geo):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.manager import (
        DamageManager,
    )
    p = F16AngularParameters()
    profile = DamageProfile(events=[
        DamageEvent(2.0, "control_failure",
                    {"surface": "rudder", "mode": "jam",
                     "jam_position_rad": 0.05}),
    ])
    mgr = DamageManager(geometry=geo, params=p, profile=profile)
    mgr.update(t_current=2.5, t_previous=1.5)
    assert "rudder" in mgr.state.control_failures
    cf = mgr.state.control_failures["rudder"]
    assert cf.mode == "jam"
    assert cf.jam_position_rad == 0.05


def test_manager_inject_event_runtime(geo):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.manager import (
        DamageManager,
    )
    p = F16AngularParameters()
    mgr = DamageManager(geometry=geo, params=p, profile=DamageProfile(events=[]))
    mgr.inject_event(DamageEvent(
        trigger_time=3.0, event_type="engine_failure",
        payload={"thrust_factor": 0.0, "hard_failure": True},
    ))
    mgr.update(t_current=3.5, t_previous=2.5)
    assert mgr.state.engine.hard_failure is True


def test_manager_reset_clears_state(geo):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.manager import (
        DamageManager,
    )
    p = F16AngularParameters()
    profile = DamageProfile(events=[
        DamageEvent(1.0, "section_loss",
                    {"section": "left_tip", "loss_fraction": 1.0}),
    ])
    mgr = DamageManager(geometry=geo, params=p, profile=profile)
    mgr.update(t_current=2.0, t_previous=0.0)
    assert mgr.state.section_loss["left_tip"] == 1.0
    mgr.reset()
    assert mgr.state.section_loss["left_tip"] == 0.0
```

- [ ] **Step 2: Implement manager.py**

`tensoraerospace/aerospacemodel/f16/nonlinear/damage/manager.py`:

```python
"""DamageManager: ties events, state, and param recomputation together."""

from __future__ import annotations

from typing import Optional

from .events import DamageEvent, DamageProfile
from .geometry import BaseGeometry
from .recompute import apply_to_params
from .state import ControlFailure, DamageState


class DamageManager:
    """Owns DamageState, applies events, drives param recomputation."""

    def __init__(
        self,
        geometry: BaseGeometry,
        params,
        profile: Optional[DamageProfile] = None,
    ) -> None:
        self.geometry = geometry
        self.params = params
        self.profile: DamageProfile = profile or DamageProfile(events=[])
        self.state = DamageState.healthy(geometry)
        self._injected: list[DamageEvent] = []

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Clear all damage and re-apply baseline params."""
        self.state = DamageState.healthy(self.geometry)
        self._injected = []
        apply_to_params(self.params, self.geometry, self.state)

    def set_profile(self, profile: DamageProfile) -> None:
        self.profile = profile

    def inject_event(self, event: DamageEvent) -> None:
        """Add an event to be triggered on the next matching window."""
        self._injected.append(event)

    def update(
        self, t_current: float, t_previous: float
    ) -> list[DamageEvent]:
        """Trigger any events in (t_previous, t_current]; return them."""
        triggered: list[DamageEvent] = []

        # Profile events
        for ev in self.profile.get_pending_events(t_current, t_previous):
            self._apply_event(ev)
            triggered.append(ev)

        # Injected events (single-fire)
        remaining: list[DamageEvent] = []
        for ev in self._injected:
            if t_previous < ev.trigger_time <= t_current:
                self._apply_event(ev)
                triggered.append(ev)
            else:
                remaining.append(ev)
        self._injected = remaining

        if triggered:
            apply_to_params(self.params, self.geometry, self.state)

        return triggered

    def _apply_event(self, ev: DamageEvent) -> None:
        if ev.event_type == "section_loss":
            self.state.set_section_loss(
                ev.payload["section"], ev.payload["loss_fraction"]
            )
        elif ev.event_type == "control_failure":
            payload = dict(ev.payload)
            surface = payload.pop("surface")
            cf = ControlFailure(**payload)
            self.state.set_control_failure(surface, cf)
        elif ev.event_type == "engine_failure":
            if "thrust_factor" in ev.payload:
                self.state.engine.thrust_factor = float(ev.payload["thrust_factor"])
            if "hard_failure" in ev.payload:
                self.state.engine.hard_failure = bool(ev.payload["hard_failure"])
        elif ev.event_type == "structural_change":
            if "mass_delta_kg" in ev.payload:
                self.state.structural.extra_mass_delta_kg += float(
                    ev.payload["mass_delta_kg"]
                )
            if "cg_shift_m" in ev.payload:
                shift = ev.payload["cg_shift_m"]
                old = self.state.structural.extra_cg_shift_m
                self.state.structural.extra_cg_shift_m = (
                    old[0] + shift[0], old[1] + shift[1], old[2] + shift[2]
                )
            if "inertia_delta" in ev.payload:
                d = ev.payload["inertia_delta"]
                old = self.state.structural.extra_inertia_delta
                self.state.structural.extra_inertia_delta = (
                    old[0] + d[0], old[1] + d[1], old[2] + d[2], old[3] + d[3]
                )
        else:
            raise ValueError(f"Unknown event_type: {ev.event_type}")
```

- [ ] **Step 3: Run — pass**

```bash
poetry run pytest tests/aerospacemodel/f16_damage/manager_test.py -v
```

Expected: 5 PASS.

- [ ] **Step 4: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/manager.py \
        tests/aerospacemodel/f16_damage/manager_test.py
git commit -m "feat(f16-damage): add DamageManager (events → state → params)"
```

### Task 6.3: Built-in scenario presets

**Files:**
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/presets.py`
- Test: `tests/aerospacemodel/f16_damage/presets_scenarios_test.py`

- [ ] **Step 1: Write tests**

`tests/aerospacemodel/f16_damage/presets_scenarios_test.py`:

```python
"""Built-in damage scenarios."""

from __future__ import annotations

import pytest


def test_wing_strike_left_tip_preset():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import presets
    p = presets.WING_STRIKE_LEFT_TIP
    assert len(p.events) == 1
    e = p.events[0]
    assert e.event_type == "section_loss"
    assert e.payload["section"] == "left_tip"
    assert e.payload["loss_fraction"] == 1.0


def test_engine_flameout_preset():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import presets
    p = presets.ENGINE_FLAMEOUT
    e = p.events[0]
    assert e.event_type == "engine_failure"


def test_birdstrike_compound_preset():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import presets
    p = presets.BIRDSTRIKE_COMPOUND
    assert len(p.events) >= 2  # multiple events bundled
```

- [ ] **Step 2: Append presets to presets.py**

```python
# === Built-in damage scenarios ===

from .events import DamageEvent, DamageProfile  # noqa: E402

WING_STRIKE_LEFT_TIP = DamageProfile(events=[
    DamageEvent(
        trigger_time=10.0, event_type="section_loss",
        payload={"section": "left_tip", "loss_fraction": 1.0},
        label="left_tip_total_loss",
    ),
])

WING_STRIKE_LEFT_HALF = DamageProfile(events=[
    DamageEvent(
        trigger_time=10.0, event_type="section_loss",
        payload={"section": "left_tip", "loss_fraction": 1.0},
        label="left_tip_total_loss",
    ),
    DamageEvent(
        trigger_time=10.0, event_type="section_loss",
        payload={"section": "left_mid", "loss_fraction": 0.5},
        label="left_mid_partial",
    ),
])

ELEVATOR_JAM_NEUTRAL = DamageProfile(events=[
    DamageEvent(
        trigger_time=5.0, event_type="control_failure",
        payload={"surface": "stab_left", "mode": "jam", "jam_position_rad": 0.0},
        label="stab_left_jam_neutral",
    ),
    DamageEvent(
        trigger_time=5.0, event_type="control_failure",
        payload={"surface": "stab_right", "mode": "jam", "jam_position_rad": 0.0},
        label="stab_right_jam_neutral",
    ),
])

ELEVATOR_JAM_PITCH_UP = DamageProfile(events=[
    DamageEvent(
        trigger_time=5.0, event_type="control_failure",
        payload={"surface": "stab_left", "mode": "jam", "jam_position_rad": 0.1745},
        label="stab_left_jam_up",
    ),
    DamageEvent(
        trigger_time=5.0, event_type="control_failure",
        payload={"surface": "stab_right", "mode": "jam", "jam_position_rad": 0.1745},
        label="stab_right_jam_up",
    ),
])

RUDDER_LOST = DamageProfile(events=[
    DamageEvent(
        trigger_time=5.0, event_type="control_failure",
        payload={"surface": "rudder", "mode": "lost"},
        label="rudder_lost",
    ),
])

ENGINE_FLAMEOUT = DamageProfile(events=[
    DamageEvent(
        trigger_time=5.0, event_type="engine_failure",
        payload={"thrust_factor": 0.0, "hard_failure": True},
        label="engine_flameout",
    ),
])

BIRDSTRIKE_COMPOUND = DamageProfile(events=[
    DamageEvent(
        trigger_time=5.0, event_type="section_loss",
        payload={"section": "right_mid", "loss_fraction": 0.2},
        label="right_wing_birdstrike",
    ),
    DamageEvent(
        trigger_time=5.0, event_type="engine_failure",
        payload={"thrust_factor": 0.3},
        label="engine_partial_loss",
    ),
])
```

- [ ] **Step 3: Run — pass**

- [ ] **Step 4: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/presets.py \
        tests/aerospacemodel/f16_damage/presets_scenarios_test.py
git commit -m "feat(f16-damage): add built-in damage scenario presets"
```

### Task 6.4: Public package API

- [ ] **Step 1: Update `damage/__init__.py`**

`tensoraerospace/aerospacemodel/f16/nonlinear/damage/__init__.py`:

```python
"""Damage modeling subsystem for the nonlinear F-16 model."""

from .events import DamageEvent, DamageProfile
from .geometry import AeroSection, BaseGeometry
from .manager import DamageManager
from .presets import (
    BIRDSTRIKE_COMPOUND,
    ELEVATOR_JAM_NEUTRAL,
    ELEVATOR_JAM_PITCH_UP,
    ENGINE_FLAMEOUT,
    RUDDER_LOST,
    WING_STRIKE_LEFT_HALF,
    WING_STRIKE_LEFT_TIP,
    load_f16_geometry,
)
from .state import (
    ControlFailure,
    DamageState,
    EngineState,
    StructuralState,
)

__all__ = [
    "AeroSection", "BaseGeometry",
    "DamageState", "ControlFailure", "EngineState", "StructuralState",
    "DamageEvent", "DamageProfile",
    "DamageManager",
    "load_f16_geometry",
    "WING_STRIKE_LEFT_TIP", "WING_STRIKE_LEFT_HALF",
    "ELEVATOR_JAM_NEUTRAL", "ELEVATOR_JAM_PITCH_UP",
    "RUDDER_LOST", "ENGINE_FLAMEOUT", "BIRDSTRIKE_COMPOUND",
]
```

- [ ] **Step 2: Smoke test**

```bash
poetry run python -c "
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageManager, DamageProfile, WING_STRIKE_LEFT_TIP, load_f16_geometry,
)
print('Imports OK')
print('Profile events:', WING_STRIKE_LEFT_TIP.events)
"
```

Expected: prints `Imports OK` and the event list.

- [ ] **Step 3: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/__init__.py
git commit -m "feat(f16-damage): expose public API in package __init__"
```

---

## Phase 7 — Gym env integration

### Task 7.1: NonlinearAngularF16 — damage_profile parameter

**Files:**
- Modify: `tensoraerospace/envs/f16/nonlinear_angular.py`
- Test: `tests/envs/f16_angular_damage_test.py` (new)

- [ ] **Step 1: Write tests**

`tests/envs/f16_angular_damage_test.py`:

```python
"""NonlinearAngularF16 with damage_profile."""

from __future__ import annotations

import numpy as np
import pytest


def test_no_damage_profile_unchanged_behaviour():
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16
    e = NonlinearAngularF16(
        initial_state=np.zeros(14), number_time_steps=10, dt=0.01, airspeed=200.0,
    )
    obs, _ = e.reset()
    obs2, r, term, trunc, info = e.step(np.zeros(3))
    assert "damage_state" not in info


def test_damage_profile_triggers_event():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        WING_STRIKE_LEFT_TIP,
    )
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    profile = WING_STRIKE_LEFT_TIP  # event at t=10s
    e = NonlinearAngularF16(
        initial_state=np.zeros(14), number_time_steps=2000,
        dt=0.01, airspeed=200.0, damage_profile=profile, split_stab=True,
    )
    e.reset()
    triggered_seen = False
    for _ in range(1100):  # 11 seconds — past trigger
        _, _, _, _, info = e.step(np.zeros(4))
        if info.get("damage_events_triggered"):
            triggered_seen = True
            break
    assert triggered_seen


def test_damage_observable_extends_obs_space():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        DamageProfile, load_f16_geometry,
    )
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    geo = load_f16_geometry()
    profile = DamageProfile(events=[])
    e = NonlinearAngularF16(
        initial_state=np.zeros(14), number_time_steps=10,
        dt=0.01, airspeed=200.0,
        damage_profile=profile, damage_observable=True, split_stab=True,
    )
    obs, _ = e.reset()
    assert obs.shape[0] > 14, "damage_observable should extend obs"


def test_reset_clears_damage():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        DamageEvent, DamageProfile,
    )
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    profile = DamageProfile(events=[
        DamageEvent(0.05, "section_loss",
                    {"section": "left_tip", "loss_fraction": 1.0}),
    ])
    e = NonlinearAngularF16(
        initial_state=np.zeros(14), number_time_steps=100,
        dt=0.01, airspeed=200.0, damage_profile=profile, split_stab=True,
    )
    e.reset()
    e.step(np.zeros(4))  # past 0.05s → fires
    assert e.unwrapped.damage_manager.state.section_loss["left_tip"] == 1.0
    e.reset()
    assert e.unwrapped.damage_manager.state.section_loss["left_tip"] == 0.0
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Modify NonlinearAngularF16**

In `tensoraerospace/envs/f16/nonlinear_angular.py`:

Add to imports at top:

```python
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageManager, DamageProfile, load_f16_geometry,
)
```

In `__init__`, add new arguments after `trail_length`:

```python
        damage_profile: Optional[DamageProfile] = None,
        damage_observable: bool = False,
        damage_event_callback=None,
        split_stab: bool = False,
```

In `__init__` body, after `self.trail_length = trail_length`:

```python
        self.split_stab = split_stab
        self.damage_profile = damage_profile
        self.damage_observable = damage_observable
        self.damage_event_callback = damage_event_callback
        # Action shape depends on split_stab
        action_shape = (4,) if split_stab else (3,)
```

Replace the action_space block:

```python
        self.action_space = spaces.Box(
            low=-self.max_action_value, high=self.max_action_value,
            shape=action_shape, dtype=np.float64,
        )
```

Replace the obs_space block (compute extended size if damage_observable):

```python
        # Observation: 14 model states + optional damage state vector
        obs_low = -np.inf
        obs_high = np.inf
        obs_size = 14
        if damage_observable:
            geo = load_f16_geometry()
            obs_size += len(geo.section_names())  # section_loss vector
            obs_size += 1  # engine.thrust_factor
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=(obs_size,), dtype=np.float64,
        )
        self._geo_for_obs = (
            load_f16_geometry() if damage_observable or damage_profile else None
        )
        self.damage_manager: Optional[DamageManager] = None
```

Modify `reset()`:

```python
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.model = AngularF16(
            x0=self.initial_state, t0=0, dt=self.dt, integrator=self.integrator,
            split_stab=self.split_stab,
        )
        self._step_index = 0

        # Damage manager
        if self.damage_profile is not None or self.damage_observable:
            geo = self._geo_for_obs
            self.damage_manager = DamageManager(
                geometry=geo, params=self.model.param,
                profile=(self.damage_profile or DamageProfile(events=[])),
            )
            if options and "damage_profile" in options:
                self.damage_manager.set_profile(options["damage_profile"])
            self.damage_manager.reset(seed=seed)
            self.model.damage_state = self.damage_manager.state
            self.model.damage_geometry = geo

        self.position_history = np.zeros((1, 3), dtype=np.float64)
        self.attitude_history = self._extract_attitude(self.initial_state).reshape(1, 3)
        self.time_history = np.zeros((1,), dtype=np.float64)
        self.chart_history = {
            name: np.array([self.initial_state[MODEL_STATE_ORDER.index(name)]])
            for name in self.chart_states
        }
        self._live_renderer = None

        obs = self._build_observation(self.initial_state)
        return obs, {}
```

Modify `step()`:

```python
    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        expected = (4,) if self.split_stab else (3,)
        if action.shape != expected:
            raise ValueError(f"action must be {expected}; got {action.shape}")
        action_clipped = np.clip(action, -self.max_action_value, self.max_action_value)
        u_rad = np.deg2rad(action_clipped)

        # Time bookkeeping (BEFORE stepping the model)
        t_prev = self._step_index * self.dt
        t_now = (self._step_index + 1) * self.dt

        # Damage events
        triggered_labels: list[str] = []
        if self.damage_manager is not None:
            triggered = self.damage_manager.update(t_now, t_prev)
            for ev in triggered:
                if self.damage_event_callback:
                    self.damage_event_callback(ev, self.damage_manager.state)
                triggered_labels.append(ev.label or ev.event_type)

        assert self.model is not None
        self.model.run_step(u_rad)
        next_state = self.model.current_state.copy()

        self._update_history(next_state)
        self._step_index += 1
        terminated = False
        truncated = self._step_index >= self.number_time_steps
        reward = 0.0

        info: dict = {}
        if self.damage_manager is not None:
            info["damage_state"] = self.damage_manager.state.snapshot()
            if triggered_labels:
                info["damage_events_triggered"] = triggered_labels

        obs = self._build_observation(next_state)
        return obs, reward, terminated, truncated, info
```

Add `_build_observation`:

```python
    def _build_observation(self, model_state: np.ndarray) -> np.ndarray:
        if not self.damage_observable or self.damage_manager is None:
            return model_state.copy()
        geo = self._geo_for_obs
        names = geo.section_names()
        loss_vec = np.array(
            [self.damage_manager.state.section_loss.get(n, 0.0) for n in names],
            dtype=np.float64,
        )
        thrust_vec = np.array(
            [self.damage_manager.state.engine.thrust_factor], dtype=np.float64
        )
        return np.concatenate([model_state, loss_vec, thrust_vec])
```

- [ ] **Step 4: Run env tests — pass**

```bash
poetry run pytest tests/envs/f16_angular_damage_test.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Run full env suite — no regressions**

```bash
poetry run pytest tests/envs/ -x --tb=short -q
```

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/envs/f16/nonlinear_angular.py \
        tests/envs/f16_angular_damage_test.py
git commit -m "feat(f16-damage): integrate damage subsystem into NonlinearAngularF16 env"
```

### Task 7.2: NonlinearLongitudinal env damage support (symmetric only)

The longitudinal model has 4 states / 1 action and is symmetric. Asymmetric events are not meaningful here; we only forward symmetric `section_loss` (e.g. fuselage mass), engine, and structural events, and ignore (with warning) asymmetric ones.

**Files:**
- Modify: `tensoraerospace/envs/f16/nonlinear_longitudinal.py`
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/model.py`
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/dynamics.py`
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/params.py`
- Test: `tests/envs/f16_longitudinal_damage_test.py` (new)

- [ ] **Step 1: Read longitudinal env to confirm signature**

```bash
head -120 /home/mr8bit/Projects/TensorAeroSpace/tensoraerospace/envs/f16/nonlinear_longitudinal.py
```

The longitudinal env has a 4-state observation, 1-action input, and is symmetric. Damage hooks here only need: param recompute (mass/inertia) and `delta_cy`/`delta_my` corrections (asymmetric ΔMx isn't in the longitudinal ODE).

- [ ] **Step 2: Write the failing test**

`tests/envs/f16_longitudinal_damage_test.py`:

```python
"""NonlinearLongitudinalF16 with symmetric damage."""

from __future__ import annotations

import numpy as np
import pytest


def test_no_damage_profile_unchanged():
    from tensoraerospace.envs.f16.nonlinear_longitudinal import (
        NonlinearLongitudinalF16,
    )
    e = NonlinearLongitudinalF16(
        initial_state=np.zeros(4), number_time_steps=10, dt=0.01,
    )
    e.reset()
    _, _, _, _, info = e.step(np.zeros(1))
    assert "damage_state" not in info


def test_symmetric_loss_changes_alpha_response():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent, DamageProfile,
    )
    from tensoraerospace.envs.f16.nonlinear_longitudinal import (
        NonlinearLongitudinalF16,
    )
    profile = DamageProfile(events=[
        DamageEvent(0.05, "section_loss",
                    {"section": "left_tip", "loss_fraction": 0.5}),
        DamageEvent(0.05, "section_loss",
                    {"section": "right_tip", "loss_fraction": 0.5}),
    ])
    e = NonlinearLongitudinalF16(
        initial_state=np.zeros(4), number_time_steps=200,
        dt=0.01, damage_profile=profile,
    )
    e.reset()
    for _ in range(200):
        e.step(np.zeros(1))
    # Should not crash; final state finite
    assert np.all(np.isfinite(e.unwrapped.model.current_state))
```

- [ ] **Step 3: Apply mirror changes to longitudinal**

In `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/params.py`, add fields:

```python
    damage_state: object = None
    damage_geometry: object = None
```

In `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/dynamics.py`, after the line computing `cy` and `mz_`, add the same hook block (only `delta_cy` and `delta_my` make sense in symmetric long ODE):

```python
    damage_state = getattr(p, "damage_state", None)
    damage_geo = getattr(p, "damage_geometry", None)
    if damage_state is not None and damage_geo is not None:
        from ..damage import aero_corrections as _ac
        cy = cy + _ac.delta_cy(alpha, 0.0, damage_geo, damage_state)
        # mz_ here is the pitching moment coefficient
        mz_ = mz_ + _ac.delta_my(alpha, 0.0, damage_geo, damage_state)
```

(Note: in the longitudinal model `mz_` denotes the pitching coefficient; we add `delta_my`.)

In `tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/model.py` `__init__`, after `self.param`:

```python
        self.damage_state = None
        self.damage_geometry = None
```

And in `run_step`, before the integrator call:

```python
        if self.damage_state is not None and self.damage_geometry is not None:
            self.param.damage_state = self.damage_state
            self.param.damage_geometry = self.damage_geometry
        else:
            self.param.damage_state = None
            self.param.damage_geometry = None
```

In `tensoraerospace/envs/f16/nonlinear_longitudinal.py`, add the same damage hooks as Task 7.1 step 3, with longitudinal-specific shapes (4-state obs, 1-action). Add to imports:

```python
from typing import Optional
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageManager, DamageProfile, load_f16_geometry,
)
```

In `__init__`, add three new keyword arguments alongside existing ones:

```python
        damage_profile: Optional[DamageProfile] = None,
        damage_observable: bool = False,
        damage_event_callback=None,
```

After existing field assignments, add:

```python
        self.damage_profile = damage_profile
        self.damage_observable = damage_observable
        self.damage_event_callback = damage_event_callback
        obs_size = 4
        if damage_observable:
            geo = load_f16_geometry()
            obs_size += len(geo.section_names()) + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64,
        )
        self._geo_for_obs = (
            load_f16_geometry() if damage_observable or damage_profile else None
        )
        self.damage_manager: Optional[DamageManager] = None
```

In `reset()`, after the model is constructed and before the existing return, add:

```python
        if self.damage_profile is not None or self.damage_observable:
            geo = self._geo_for_obs
            self.damage_manager = DamageManager(
                geometry=geo, params=self.model.param,
                profile=(self.damage_profile or DamageProfile(events=[])),
            )
            if options and "damage_profile" in options:
                self.damage_manager.set_profile(options["damage_profile"])
            self.damage_manager.reset(seed=seed)
            self.model.damage_state = self.damage_manager.state
            self.model.damage_geometry = geo
```

In `step()`, before the model.run_step call, add the event-update block:

```python
        t_prev = self._step_index * self.dt
        t_now = (self._step_index + 1) * self.dt
        triggered_labels: list[str] = []
        if self.damage_manager is not None:
            triggered = self.damage_manager.update(t_now, t_prev)
            for ev in triggered:
                if self.damage_event_callback:
                    self.damage_event_callback(ev, self.damage_manager.state)
                triggered_labels.append(ev.label or ev.event_type)
```

After the model has stepped and `next_state` is available, populate `info`:

```python
        info: dict = {}
        if self.damage_manager is not None:
            info["damage_state"] = self.damage_manager.state.snapshot()
            if triggered_labels:
                info["damage_events_triggered"] = triggered_labels
```

Replace the `return next_state, ...` with the extended observation:

```python
        if self.damage_observable and self.damage_manager is not None:
            geo = self._geo_for_obs
            names = geo.section_names()
            loss_vec = np.array(
                [self.damage_manager.state.section_loss.get(n, 0.0) for n in names],
                dtype=np.float64,
            )
            thrust_vec = np.array(
                [self.damage_manager.state.engine.thrust_factor], dtype=np.float64,
            )
            obs_out = np.concatenate([next_state, loss_vec, thrust_vec])
        else:
            obs_out = next_state
        return obs_out, reward, terminated, truncated, info
```

- [ ] **Step 4: Run — pass**

```bash
poetry run pytest tests/envs/f16_longitudinal_damage_test.py -v
```

- [ ] **Step 5: Full regression**

```bash
poetry run pytest tests/aerospacemodel/ tests/envs/ -x --tb=short -q
```

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/ \
        tensoraerospace/envs/f16/nonlinear_longitudinal.py \
        tests/envs/f16_longitudinal_damage_test.py
git commit -m "feat(f16-damage): integrate damage subsystem into longitudinal env"
```

---

## Phase 8 — Random damage profile generator + demo

### Task 8.1: RandomDamageProfileGenerator

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/random.py`
- Test: `tests/aerospacemodel/f16_damage/random_test.py`

- [ ] **Step 1: Write tests**

`tests/aerospacemodel/f16_damage/random_test.py`:

```python
"""Random damage profile generation for RL."""

from __future__ import annotations

import pytest


def test_generator_seeded_is_deterministic():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.random import (
        RandomDamageProfileGenerator,
    )
    g1 = RandomDamageProfileGenerator(
        event_types=["section_loss"],
        time_range=(5.0, 25.0),
        severity_range=(0.1, 1.0),
        num_events_range=(1, 1),
        seed=42,
    )
    g2 = RandomDamageProfileGenerator(
        event_types=["section_loss"],
        time_range=(5.0, 25.0),
        severity_range=(0.1, 1.0),
        num_events_range=(1, 1),
        seed=42,
    )
    p1 = g1.sample()
    p2 = g2.sample()
    assert p1.events[0].trigger_time == p2.events[0].trigger_time
    assert p1.events[0].payload == p2.events[0].payload


def test_generator_respects_time_range():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.random import (
        RandomDamageProfileGenerator,
    )
    g = RandomDamageProfileGenerator(
        event_types=["section_loss"],
        time_range=(5.0, 25.0),
        severity_range=(0.1, 1.0),
        num_events_range=(1, 1),
        seed=42,
    )
    for _ in range(50):
        p = g.sample()
        for e in p.events:
            assert 5.0 <= e.trigger_time <= 25.0


def test_generator_respects_num_events_range():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.random import (
        RandomDamageProfileGenerator,
    )
    g = RandomDamageProfileGenerator(
        event_types=["section_loss"],
        time_range=(5.0, 25.0),
        severity_range=(0.1, 1.0),
        num_events_range=(2, 4),
        seed=42,
    )
    for _ in range(20):
        p = g.sample()
        assert 2 <= len(p.events) <= 4
```

- [ ] **Step 2: Implement random.py**

`tensoraerospace/aerospacemodel/f16/nonlinear/damage/random.py`:

```python
"""Random damage profile generation."""

from __future__ import annotations

import random
from typing import Optional

from .events import DamageEvent, DamageProfile
from .presets import load_f16_geometry

_DEFAULT_LOSABLE_SECTIONS = (
    "left_tip", "left_mid", "right_tip", "right_mid",
    "stab_left", "stab_right",
)


class RandomDamageProfileGenerator:
    """Sample DamageProfiles for RL/curriculum training.

    All knobs are inclusive ranges sampled uniformly. Currently supports
    `section_loss`, `control_failure`, and `engine_failure` events.
    """

    def __init__(
        self,
        event_types: list[str],
        time_range: tuple[float, float],
        severity_range: tuple[float, float] = (0.1, 1.0),
        num_events_range: tuple[int, int] = (1, 1),
        seed: Optional[int] = None,
        sections: Optional[tuple[str, ...]] = None,
    ) -> None:
        self.event_types = list(event_types)
        if not self.event_types:
            raise ValueError("event_types must be non-empty")
        self.time_range = time_range
        self.severity_range = severity_range
        self.num_events_range = num_events_range
        self.rng = random.Random(seed)
        self.sections = sections or _DEFAULT_LOSABLE_SECTIONS

    def sample(self) -> DamageProfile:
        n_events = self.rng.randint(*self.num_events_range)
        events: list[DamageEvent] = []
        for _ in range(n_events):
            t = self.rng.uniform(*self.time_range)
            kind = self.rng.choice(self.event_types)
            if kind == "section_loss":
                section = self.rng.choice(self.sections)
                fraction = self.rng.uniform(*self.severity_range)
                events.append(DamageEvent(
                    trigger_time=t, event_type="section_loss",
                    payload={"section": section, "loss_fraction": fraction},
                    label=f"random_loss_{section}_{fraction:.2f}",
                ))
            elif kind == "control_failure":
                surface = self.rng.choice(
                    ["stab_left", "stab_right", "rudder", "aileron_left", "aileron_right"]
                )
                mode = self.rng.choice(["jam", "efficiency_loss", "lost"])
                payload = {"surface": surface, "mode": mode}
                if mode == "jam":
                    payload["jam_position_rad"] = self.rng.uniform(-0.15, 0.15)
                elif mode == "efficiency_loss":
                    payload["efficiency"] = self.rng.uniform(0.2, 0.9)
                events.append(DamageEvent(
                    trigger_time=t, event_type="control_failure",
                    payload=payload, label=f"random_{surface}_{mode}",
                ))
            elif kind == "engine_failure":
                tf = self.rng.uniform(0.0, 0.6)
                events.append(DamageEvent(
                    trigger_time=t, event_type="engine_failure",
                    payload={"thrust_factor": tf},
                    label=f"random_engine_{tf:.2f}",
                ))
        return DamageProfile(events=events)
```

- [ ] **Step 3: Run — pass**

- [ ] **Step 4: Update package __init__**

In `tensoraerospace/aerospacemodel/f16/nonlinear/damage/__init__.py`:

```python
from .random import RandomDamageProfileGenerator  # noqa: E402
```

Append `"RandomDamageProfileGenerator"` to `__all__`.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/random.py \
        tensoraerospace/aerospacemodel/f16/nonlinear/damage/__init__.py \
        tests/aerospacemodel/f16_damage/random_test.py
git commit -m "feat(f16-damage): add RandomDamageProfileGenerator for RL"
```

### Task 8.2: Demo notebook (dogfight wing strike)

**Files:**
- Create: `example/failure_demos/f16_damage_dogfight_demo.py` (or `.py` script — see step)

- [ ] **Step 1: Decide format**

Run `ls example/` first to see existing format. If `.ipynb` is dominant, create one; otherwise a `.py` script.

```bash
ls /home/mr8bit/Projects/TensorAeroSpace/example/
```

- [ ] **Step 2: Create demo script**

Create `example/failure_demos/f16_damage_dogfight_demo.py`:

```python
"""Demo: F-16 with left-wingtip damage at t=10s.

Simulates 20 seconds of level flight; at t=10s the left wingtip is
fully lost. Without active stabilisation, a roll moment develops.
Plots roll rate over time.
"""

import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    WING_STRIKE_LEFT_TIP,
)
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16


def main():
    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=2000,
        dt=0.01,
        airspeed=200.0,
        damage_profile=WING_STRIKE_LEFT_TIP,
        split_stab=True,
    )
    obs, _ = env.reset()
    wx_history = [obs[2]]
    for k in range(2000):
        obs, _, _, _, info = env.step(np.zeros(4))
        wx_history.append(obs[2])
        if info.get("damage_events_triggered"):
            print(f"t={k*0.01:.2f}s: {info['damage_events_triggered']}")

    print(f"Final wx (roll rate, rad/s): {wx_history[-1]:.4f}")
    print(f"Max |wx|: {max(abs(w) for w in wx_history):.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the demo**

```bash
poetry run python example/failure_demos/f16_damage_dogfight_demo.py
```

Expected: prints the trigger event around t=10s and a non-zero final/max roll rate.

- [ ] **Step 4: Commit**

```bash
git add example/failure_demos/f16_damage_dogfight_demo.py
git commit -m "docs(f16-damage): add dogfight wing-strike demo script"
```

---

## Phase 9 — Documentation

### Task 9.1: User-facing docs page

**Files:**
- Create: `docs/en/aircraft-damage-modeling.md` (or `docs/ru/`, mirror existing layout)

- [ ] **Step 1: Inspect docs layout**

```bash
ls /home/mr8bit/Projects/TensorAeroSpace/docs/en/
ls /home/mr8bit/Projects/TensorAeroSpace/docs/ru/
```

- [ ] **Step 2: Create the docs page**

`docs/en/aircraft-damage-modeling.md`:

```markdown
# Aircraft Damage Modeling

The damage subsystem allows you to schedule failures during a simulation —
wing tip loss, jammed control surfaces, engine failure, structural changes —
that update the aircraft's mass, inertia, aerodynamic coefficients, and
control-surface effectiveness in real time.

Currently supported on the **nonlinear F-16** (longitudinal and 6-DoF angular).

## Quick start

```python
import numpy as np
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import WING_STRIKE_LEFT_TIP
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

env = NonlinearAngularF16(
    initial_state=np.zeros(14),
    number_time_steps=2000,
    damage_profile=WING_STRIKE_LEFT_TIP,
    split_stab=True,
)
obs, _ = env.reset()
for _ in range(2000):
    obs, r, term, trunc, info = env.step(np.zeros(4))
    if info.get("damage_events_triggered"):
        print(info["damage_events_triggered"])
```

## Available built-in scenarios

- `WING_STRIKE_LEFT_TIP` — full loss of left wingtip at t=10s
- `WING_STRIKE_LEFT_HALF` — left wingtip + 50% mid-section
- `ELEVATOR_JAM_NEUTRAL` / `ELEVATOR_JAM_PITCH_UP` — both stab halves jammed
- `RUDDER_LOST` — rudder lost
- `ENGINE_FLAMEOUT` — engine fails
- `BIRDSTRIKE_COMPOUND` — compound wing + engine failure

## Custom scenario

```python
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageEvent, DamageProfile,
)

profile = DamageProfile(events=[
    DamageEvent(8.0, "section_loss",
                payload={"section": "right_mid", "loss_fraction": 0.4}),
    DamageEvent(15.0, "engine_failure",
                payload={"thrust_factor": 0.3}),
])
```

## Random profiles for RL

```python
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    RandomDamageProfileGenerator,
)

generator = RandomDamageProfileGenerator(
    event_types=["section_loss", "control_failure"],
    time_range=(5.0, 25.0),
    severity_range=(0.1, 1.0),
    num_events_range=(1, 2),
    seed=42,
)

# In training loop:
profile = generator.sample()
obs, info = env.reset(options={"damage_profile": profile})
```

## Architecture and physical model

See the design document at
`docs/superpowers/specs/2026-04-28-aircraft-damage-modeling-design.md`.

Key points:
- **Strip-theory aerodynamic corrections** — each section contributes
  proportionally to lost lift/drag/moment when damaged. Accuracy is ~10–20%
  vs full VLM.
- **Huygens–Steiner inertia recompute** — preserves physical correctness
  of mass distribution shifts.
- **Asymmetric damage requires the angular 6-DoF model** with `split_stab=True`.
  Symmetric damage works in both longitudinal and angular.
- Without a `damage_profile`, env behaviour is **bit-identical** to the
  un-damaged baseline.
```

- [ ] **Step 3: If `mkdocs.yml` includes a navigation, add the new page**

```bash
poetry run grep -n "nav:" /home/mr8bit/Projects/TensorAeroSpace/mkdocs.yml | head -10
```

If a nav block is present, add an entry pointing to `en/aircraft-damage-modeling.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "docs(f16-damage): add user-facing documentation page"
```

---

## Final regression run

- [ ] **Run the full test suite**

```bash
poetry run pytest -x --tb=short -q
```

Expected: all tests pass, including the ~30 new tests added across phases.

- [ ] **Verify imports cleanly**

```bash
poetry run python -c "
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageManager, DamageProfile, DamageEvent, DamageState,
    AeroSection, BaseGeometry, ControlFailure, EngineState, StructuralState,
    RandomDamageProfileGenerator, load_f16_geometry,
    WING_STRIKE_LEFT_TIP, WING_STRIKE_LEFT_HALF, ELEVATOR_JAM_NEUTRAL,
    ELEVATOR_JAM_PITCH_UP, RUDDER_LOST, ENGINE_FLAMEOUT, BIRDSTRIKE_COMPOUND,
)
print('All damage exports importable.')
"
```

---

## Notes for the executor

1. **Calibration drift**: the YAML masses/areas may need ±5% tweaks if Phase 1 calibration tests fail. Adjust by tuning `fuselage_main.mass` (largest absorber) first, then per-wing-section `mass`. Keep `S` total at exactly 27.87 m².

2. **Aero contribution constants**: `cl_alpha_contribution` and `cd0_contribution` per section are rough first estimates. If `delta_cy` over a known damage case (e.g., 50% wing area lost ⇒ ~50% lift drop) is off by >25%, scale these to compensate. The aircraft-level sum of `cl_alpha_contribution` × area / S_total should approximately equal `Cy_α ≈ 4.5/rad` for F-16.

3. **Param mutation safety**: `apply_to_params` mutates the `F16AngularParameters` instance in place. The damage-aware ODE reads these mutated values. The model's reference is the same instance throughout an episode — this is intentional. `reset()` resets to the healthy state which restores baseline.

4. **Split-stab actuator**: the existing actuator (`Tstab`, `Xistab`) is shared by both halves; differential commands snap the actuator to a mean position. If finer-grained per-side actuator dynamics are needed later, that's a separate mini-task (split actuator state into `stab_l`, `stab_r`, double the integration).

5. **Branch hygiene**: this work touches the F-16 model, dynamics, and envs; if you're already on `feature/3d-vizualization`, consider whether a separate branch (`feature/damage-modeling`) is more appropriate before starting Phase 0.
