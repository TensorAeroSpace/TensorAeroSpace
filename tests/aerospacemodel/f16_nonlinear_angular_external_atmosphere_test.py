"""External atmosphere and air-relative state for nonlinear AngularF16."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics import (
    f16_ode_6dof,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    F16AngularParameters,
)


def _model() -> AngularF16:
    return AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4", track_altitude=True)


def test_none_density_preserves_baseline_step():
    model = _model()

    model.run_step(np.array([0.02, -0.01, 0.03]))

    expected = np.array(
        [
            7.5757992629117774e-04,
            -1.8622616923876683e-07,
            4.2557862196987477e-05,
            -1.0210440202220448e-05,
            -3.5022232717329778e-03,
            1.3361491957313435e-07,
            -2.9246282010108312e-08,
            -1.7494384645511601e-05,
            9.4682506172839528e-04,
            1.7396488547873803e-01,
            -9.8144260416666669e-04,
            -1.7203648705208333e-01,
            1.4202375925925926e-03,
            2.6094732821810701e-01,
            2.9999995346885385e03,
            1.2000121804051957e02,
        ]
    )
    np.testing.assert_allclose(model.current_state, expected, rtol=0, atol=1e-12)


def test_isa_equivalent_density_override_matches_baseline_rhs():
    x = np.zeros(16)
    x[[0, 1, 7, 14, 15]] = [0.04, 0.01, 0.04, 3000.0, 120.0]
    u = np.array([0.02, -0.01, 0.03])
    baseline = F16AngularParameters()
    temperature = 288.15 - 0.0065 * x[14]
    density = 1.225 * (temperature / 288.15) ** (baseline.g / (0.0065 * 287.0531) - 1.0)
    override = F16AngularParameters(air_density_kg_m3=density)

    np.testing.assert_allclose(
        f16_ode_6dof(x, u, 0.0, override),
        f16_ode_6dof(x, u, 0.0, baseline),
        rtol=1e-14,
        atol=1e-12,
    )


def test_density_scales_only_aerodynamic_contributions():
    x = np.array(
        [
            0.05,
            0.02,
            0.01,
            -0.02,
            0.03,
            0.1,
            0.2,
            0.04,
            0.01,
            0.0,
            -0.02,
            0.0,
            0.03,
            0.0,
            2500.0,
            160.0,
        ]
    )
    u = np.array([0.01, -0.02, 0.03])

    def derivative(density):
        params = F16AngularParameters(air_density_kg_m3=density)
        return f16_ode_6dof(x, u, 0.0, params)

    without_aero = derivative(0.0)
    reference = derivative(0.8)
    doubled_density = derivative(1.6)

    assert not np.allclose(reference, without_aero)
    np.testing.assert_allclose(
        doubled_density,
        without_aero + 2.0 * (reference - without_aero),
        rtol=1e-12,
        atol=1e-12,
    )


def test_rk4_uses_one_external_density_for_all_rhs_evaluations(monkeypatch):
    import tensoraerospace.aerospacemodel.f16.nonlinear.angular.model as angular_model

    model = _model()
    model.param.air_density_kg_m3 = 0.73
    densities = []
    original_rhs = angular_model.f16_ode_6dof

    def recording_rhs(x, u, t, params):
        densities.append(params.air_density_kg_m3)
        return original_rhs(x, u, t, params)

    monkeypatch.setattr(angular_model, "f16_ode_6dof", recording_rhs)
    model.run_step(np.zeros(3))

    assert densities == [0.73] * 4


def test_set_air_relative_state_changes_only_four_states_without_history():
    x0 = np.arange(16, dtype=float) * 0.01
    x0[14:] = [2500.0, 180.0]
    model = AngularF16(x0=x0, dt=0.01, integrator="rk4", track_altitude=True)
    before = model.current_state
    history_length = len(model.x_history)
    time_step = model.time_step
    t0 = model.t0

    model.set_air_relative_state(
        altitude_m=3200.0,
        airspeed_mps=145.0,
        alpha_rad=0.06,
        beta_rad=-0.02,
    )

    expected = before.copy()
    expected[[14, 15, 0, 1]] = [3200.0, 145.0, 0.06, -0.02]
    np.testing.assert_array_equal(model.current_state, expected)
    assert len(model.x_history) == history_length
    assert model.time_step == time_step
    assert model.t0 == t0
    assert model.u_history == []

    installed_state = model.current_state.copy()
    model.run_step(np.zeros(3))
    np.testing.assert_array_equal(model.x_history[-2].reshape(-1), installed_state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("altitude_m", np.nan),
        ("altitude_m", np.inf),
        ("airspeed_mps", 0.0),
        ("airspeed_mps", -1.0),
        ("airspeed_mps", np.nan),
        ("airspeed_mps", np.inf),
        ("alpha_rad", np.nan),
        ("alpha_rad", -np.inf),
        ("beta_rad", np.nan),
        ("beta_rad", np.inf),
    ],
)
def test_set_air_relative_state_rejects_invalid_values(field, value):
    model = _model()
    before = model.current_state
    values = {
        "altitude_m": 3000.0,
        "airspeed_mps": 120.0,
        "alpha_rad": 0.0,
        "beta_rad": 0.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match="finite|positive"):
        model.set_air_relative_state(**values)
    np.testing.assert_array_equal(model.current_state, before)


def test_set_air_relative_state_requires_altitude_tracking():
    model = AngularF16(x0=np.zeros(14), track_altitude=False)

    with pytest.raises(RuntimeError, match="track_altitude=True"):
        model.set_air_relative_state(
            altitude_m=3000.0,
            airspeed_mps=120.0,
            alpha_rad=0.0,
            beta_rad=0.0,
        )


def test_damage_event_preserves_external_density():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        DamageEvent,
        DamageProfile,
    )
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    profile = DamageProfile(
        events=[
            DamageEvent(
                0.005,
                "section_loss",
                {"section": "left_tip", "loss_fraction": 0.5},
            )
        ]
    )
    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=2,
        dt=0.01,
        damage_profile=profile,
        track_altitude=True,
    )
    env.reset()
    env.model.param.air_density_kg_m3 = 0.65

    _, _, _, _, info = env.step(np.zeros(3))

    assert info["damage_events_triggered"]
    assert env.damage_manager.state.section_loss["left_tip"] == 0.5
    assert env.model.param.air_density_kg_m3 == 0.65


def test_external_atmosphere_sequence_is_deterministic():
    first = _model()
    second = _model()
    controls = [np.zeros(3), np.array([0.01, 0.0, -0.02]), np.zeros(3)]
    densities = [0.9, 0.7, 1.1]

    for model in (first, second):
        model.set_air_relative_state(
            altitude_m=2800.0,
            airspeed_mps=150.0,
            alpha_rad=0.03,
            beta_rad=0.01,
        )
        for control, density in zip(controls, densities):
            model.param.air_density_kg_m3 = density
            model.run_step(control)

    np.testing.assert_array_equal(
        np.asarray(first.x_history), np.asarray(second.x_history)
    )


@pytest.mark.parametrize("track_altitude", [False, True])
@pytest.mark.parametrize("density", [np.nan, np.inf, -np.inf, -1.0, "1.0", True, False])
def test_invalid_density_is_rejected_before_integration(track_altitude, density):
    model = AngularF16(
        x0=np.zeros(14),
        dt=0.01,
        integrator="rk4",
        track_altitude=track_altitude,
    )
    model.param.air_density_kg_m3 = density
    before = model.current_state

    with pytest.raises((TypeError, ValueError), match="air_density_kg_m3"):
        model.run_step(np.zeros(3))

    np.testing.assert_array_equal(model.current_state, before)
    assert len(model.x_history) == 1
    assert model.u_history == []


def test_external_density_applies_without_altitude_tracking():
    baseline = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    override = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    override.param.air_density_kg_m3 = 0.5

    baseline.run_step(np.zeros(3))
    override.run_step(np.zeros(3))

    assert not np.allclose(override.current_state, baseline.current_state)
    assert override.param.q == pytest.approx(3600.0)


def test_dynamic_pressure_tracks_density_altitude_and_speed_parameters():
    params = F16AngularParameters(air_density_kg_m3=0.5)
    assert params.q == pytest.approx(3600.0)

    params.V = 200.0
    assert params.q == pytest.approx(10000.0)

    params.air_density_kg_m3 = None
    params.Oy = 0.0
    params.V = 100.0
    assert params.q == pytest.approx(6125.0)


def test_isa_uses_isothermal_stratosphere_and_remains_finite_at_60km():
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
        _isa_dynamic_pressure,
    )

    g = 9.80665
    rho11 = 1.225 * (216.65 / 288.15) ** (g / (0.0065 * 287.0531) - 1.0)
    expected15 = rho11 * np.exp(-g * 4000.0 / (287.0531 * 216.65))
    density15 = 2.0 * _isa_dynamic_pressure(15000.0, 120.0, g) / 120.0**2
    density60 = 2.0 * _isa_dynamic_pressure(60000.0, 120.0, g) / 120.0**2

    assert density15 == pytest.approx(expected15, rel=1e-14)
    assert density15 == pytest.approx(0.195, abs=0.002)
    assert np.isfinite(density60)
    assert density60 >= 0.0


def test_high_altitude_isa_step_remains_finite():
    model = _model()
    model.set_air_relative_state(
        altitude_m=60000.0,
        airspeed_mps=120.0,
        alpha_rad=0.0,
        beta_rad=0.0,
    )

    model.run_step(np.zeros(3))

    assert np.all(np.isfinite(model.current_state))


@pytest.mark.parametrize("beta", [np.pi / 2, -np.pi / 2, np.pi, -np.pi])
def test_set_air_relative_state_rejects_singular_sideslip(beta):
    model = _model()
    before = model.current_state

    with pytest.raises(ValueError, match="beta_rad"):
        model.set_air_relative_state(
            altitude_m=3000.0,
            airspeed_mps=120.0,
            alpha_rad=0.0,
            beta_rad=beta,
        )
    np.testing.assert_array_equal(model.current_state, before)


def test_env_reset_restores_configured_external_density():
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=2,
        track_altitude=True,
        air_density_kg_m3=0.65,
    )

    env.reset()
    assert env.model.param.air_density_kg_m3 == 0.65
    env.step(np.zeros(3))
    env.reset()

    assert env.model.param.air_density_kg_m3 == 0.65
    assert env.model.param.q == pytest.approx(0.5 * 0.65 * 120.0**2)
    assert env.get_init_args()["air_density_kg_m3"] == 0.65


def test_ode_rejects_invalid_density_for_direct_caller():
    params = F16AngularParameters(air_density_kg_m3=np.nan)
    x = np.zeros(16)
    x[14:] = [3000.0, 120.0]

    with pytest.raises(ValueError, match="air_density_kg_m3"):
        f16_ode_6dof(x, np.zeros(3), 0.0, params)


def test_trim_rejects_invalid_density_for_direct_caller():
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.trim import find_trim

    params = F16AngularParameters(air_density_kg_m3=np.nan)

    with pytest.raises(ValueError, match="air_density_kg_m3"):
        find_trim(params=params)
