"""Independent checks of the published-algorithm entry points."""

import numpy as np
import pytest

from tensoraerospace.agent.aa_indi.hosm import HOSMDifferentiator
from tensoraerospace.agent.aa_indi.otse import OptimalTwoStageEKF
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig


def test_optimal_two_stage_matches_augmented_kalman_with_random_bias():
    rng = np.random.default_rng(109)
    n, b, d = 3, 2, 2
    x = rng.normal(size=n)
    filter_ = OptimalTwoStageEKF(x, np.eye(n), 2 * np.eye(b))
    mean = np.r_[x, np.zeros(b)]
    covariance = np.diag([1.0] * n + [2.0] * b)
    F = np.eye(n) + rng.normal(scale=0.02, size=(n, n))
    B = rng.normal(scale=0.1, size=(n, b))
    H = rng.normal(size=(d, n))
    Ha = np.hstack([H, np.zeros((d, b))])
    Fa = np.block([[F, B], [np.zeros((b, n)), np.eye(b)]])
    noise = rng.normal(scale=0.02, size=(n + b, n + b))
    Q = noise @ noise.T  # Includes nonzero state/bias process-noise correlation.
    R = 0.1 * np.eye(d)
    for _ in range(100):
        y = rng.normal(size=d)
        mean = Fa @ mean
        covariance = Fa @ covariance @ Fa.T + Q
        K = np.linalg.solve(Ha @ covariance @ Ha.T + R, Ha @ covariance).T
        mean = mean + K @ (y - Ha @ mean)
        residual = np.eye(n + b) - K @ Ha
        covariance = residual @ covariance @ residual.T + K @ R @ K.T
        filter_.predict(F @ filter_.state, F, B, Q[:n, :n], Q[n:, n:], Q[:n, n:])
        filter_.correct(y, H @ filter_.state, H, R)
        np.testing.assert_allclose(filter_.state, mean[:n], atol=2e-12)
        np.testing.assert_allclose(filter_.bias, mean[n:], atol=2e-12)
        np.testing.assert_allclose(
            filter_.state_covariance, covariance[:n, :n], atol=2e-12
        )
        np.testing.assert_allclose(
            filter_.V @ filter_.P_bias, covariance[:n, n:], atol=2e-12
        )


def test_hosm_uses_simultaneous_published_nonrecursive_update():
    h = HOSMDifferentiator(1, 0.01, [2.0, 3.0, 4.0, 5.0])
    h.step(np.array([0.0]))
    h.z[:] = np.array([[16.0], [3.0], [2.0], [1.0]])
    old = h.z.copy()
    h.step(np.array([0.0]))
    np.testing.assert_allclose(
        h.z[:, 0],
        old[:, 0]
        + 0.01
        * np.array([3 - 2 * 16**0.75, 2 - 3 * 16 ** (2 / 3), 1 - 4 * 16**0.5, -5]),
    )


def test_hosm_tracks_a_ramp_without_same_sensor_reintegration():
    h = HOSMDifferentiator(1, 0.001, [10.0, 100.0, 5.0, 0.1])
    rates = [h.step(np.array([0.2 * k * 0.001]))[0] for k in range(10000)]
    assert np.sqrt(np.mean((np.array(rates[-1000:]) - 0.2) ** 2)) < 0.01


def paper_config(mode):
    dt = 0.01
    signal = np.sin(np.arange(20)[:, None]) * 0.1
    return IADPConfig.paper(
        excitation_signal=signal,
        dt=dt,
        learning_mode=mode,
        model_learning_seconds=0.2,
        controller_training_seconds=0.4,
        critic_window_seconds=0.1,
        critic_update_hz=20,
        F_init=np.diag([0.9, 1.0]),
        G_init=np.array([[0.1], [0.0]]),
    )


def test_iadp_paper_profile_disables_modified_critic_objectives():
    cfg = paper_config("continuous")
    assert not hasattr(cfg, "policy_eval_regularization")
    assert not hasattr(cfg, "policy_eval_blend")
    assert not hasattr(cfg, "enforce_psd")
    agent = IADPAgent(1, 1, cfg)
    assert agent.P[0, 1] < 0
    assert np.linalg.eigvalsh(agent.P).min() > 0


@pytest.mark.parametrize("mode", ["continuous", "sequential"])
def test_iadp_published_learning_phases(mode, monkeypatch):
    a = IADPAgent(1, 1, paper_config(mode))
    fits = []
    monkeypatch.setattr(a, "_policy_evaluation", lambda: fits.append(a._step))
    x = np.zeros(1)
    for k in range(80):
        u = a.predict(x, np.ones(1), k)
        x = 0.9 * x + 0.1 * u
        a.learn(x, np.ones(1), k, applied_action=u)
        if k == 19:
            assert not a._window
    assert min(fits) >= 29  # Fresh post-identification window.
    if mode == "sequential":
        assert a.rls.num_updates == 19
        assert max(fits) < 60
        assert a.phase == "assessment"
    else:
        assert a.rls.num_updates == 79
        assert max(fits) >= 60
        assert a.phase == "controller_training"


def test_iadp_independent_reference_state_and_output_map_roundtrip(tmp_path):
    cfg = paper_config("continuous")
    cfg.n_reference = 2
    cfg.policy_eval_window = 30
    cfg.policy_eval_min_samples = 30
    cfg.output_matrix = np.array([[0.0, 1.0, 0.0]])
    cfg.reference_output_matrix = np.array([[1.0, 0.0]])
    cfg.F_init = np.eye(5)
    cfg.G_init = np.array([[0.0], [0.1], [0.0], [0.0], [0.0]])
    a = IADPAgent(3, 1, cfg)
    x, ref = np.array([7.0, 0.2, -3.0]), np.array([0.4, 2.0])
    u = a.predict(x, ref)
    result = a.learn(x, ref, applied_action=u)
    assert result["cost"] == pytest.approx(0.2**2 + u[0] ** 2)
    restored = IADPAgent.from_pretrained(a.save(tmp_path))
    assert restored.n_aug == 5
    np.testing.assert_array_equal(restored.Cr, a.Cr)
    np.testing.assert_array_equal(restored.predict(x, ref), a.predict(x, ref))


def test_reconstructed_moments_obey_rotational_energy_balance():
    from tensoraerospace.agent.aa_indi import AircraftGeometry

    inertia = np.array([[8.0, 0.3, -0.4], [0.3, 10.0, 0.2], [-0.4, 0.2, 13.0]])
    geometry = AircraftGeometry(inertia, 4.0, 3.0, 1.2)
    rate = np.array([0.2, -0.4, 0.3])
    acceleration = np.array([0.7, 0.1, -0.2])
    coefficients = geometry.coefficients(rate, acceleration, 1.1, 70.0)
    moments = coefficients * geometry.moment_scale(1.1, 70.0)
    # Gyroscopic terms redistribute momentum but do no work.
    assert rate @ moments == pytest.approx(rate @ inertia @ acceleration, abs=1e-13)
    derivatives = np.eye(3) * 0.1
    np.testing.assert_allclose(
        geometry.effectiveness(derivatives, 1.1, 140.0),
        4 * geometry.effectiveness(derivatives, 1.1, 70.0),
    )


def test_moment_identifier_recovers_three_independent_surface_derivatives():
    from tensoraerospace.agent.aa_indi import MomentIdentifier

    rng = np.random.default_rng(774)
    true = np.array([[0.1, -0.02, 0.03], [0.04, -0.2, 0.05], [-0.03, 0.01, 0.1]])
    identifier = MomentIdentifier(np.zeros((3, 3)), covariance_init=1e6)
    for surfaces in rng.normal(scale=0.1, size=(300, 3)):
        identifier.update(surfaces, true @ surfaces)
    np.testing.assert_allclose(identifier.derivatives, true, rtol=1e-5, atol=1e-8)
    assert all(e.lam_max == 1 for e in identifier.estimators)


def navigation_sample(t, fault=None):
    from tensoraerospace.agent.aa_indi import FlightMeasurement

    imu = np.array([0.0, 0.0, -9.80665, 0.0, 0.0, 0.0])
    if fault is not None:
        imu = imu + fault
    return FlightMeasurement(
        t,
        imu[3:],
        imu[:3],
        np.array([40.0, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
        40.0,
        1.2,
    )


@pytest.mark.parametrize("axis,magnitude,tolerance", [(0, 0.5, 0.02), (4, 0.02, 0.001)])
def test_otsekf_hosm_observes_new_sensor_bias_from_independent_navigation(
    axis, magnitude, tolerance
):
    from tensoraerospace.agent.aa_indi import ObserverConfig, OTSEKFHOSMObserver

    observer = OTSEKFHOSMObserver(ObserverConfig())
    estimated = []
    for k in range(1200):
        fault = np.zeros(6)
        if k >= 200:
            fault[axis] = magnitude
        observer.update(navigation_sample(k * 0.01, fault))
        estimated.append(observer.faults[axis])
    assert np.sqrt(np.mean((np.array(estimated[-200:]) - magnitude) ** 2)) < tolerance
    assert np.linalg.eigvalsh(observer.filter.state_covariance).min() >= -1e-12


def test_physical_aaindi_factory_and_pending_transition_roundtrip(tmp_path):
    from tensoraerospace.agent.aa_indi import (
        AAINDIAgent,
        AAINDIConfig,
        AircraftGeometry,
    )

    geometry = AircraftGeometry(np.diag([100.0, 150.0, 200.0]), 10.0, 8.0, 1.5)
    agent = AAINDIAgent(AAINDIConfig(geometry, np.diag([0.01, 0.02, 0.01])))
    assert isinstance(agent, AAINDIAgent)
    for k in range(20):
        agent.predict(navigation_sample(k * 0.01), np.zeros(3))
        agent.learn(navigation_sample((k + 1) * 0.01))
    agent.predict(navigation_sample(0.2), np.zeros(3))
    restored = AAINDIAgent.from_pretrained(agent.save(tmp_path))
    for k in range(21, 50):
        sample = navigation_sample(k * 0.01, np.array([0.0, 0.0, 0.0, 0.0, 0.01, 0.0]))
        assert agent.learn(sample) == restored.learn(sample)
        np.testing.assert_array_equal(
            agent.predict(sample, np.zeros(3)), restored.predict(sample, np.zeros(3))
        )
        np.testing.assert_array_equal(agent.observer.faults, restored.observer.faults)
        np.testing.assert_array_equal(
            agent.identifier.derivatives, restored.identifier.derivatives
        )


def test_moment_filters_start_from_same_nonzero_actuator_interval(monkeypatch):
    from tensoraerospace.agent.aa_indi import (
        AAINDIAgent,
        AAINDIConfig,
        AircraftGeometry,
        FlightMeasurement,
    )

    geometry = AircraftGeometry(np.eye(3), 2.0, 2.0, 2.0)
    agent = AAINDIAgent(AAINDIConfig(geometry, np.eye(3) * 0.5))
    monkeypatch.setattr(agent.observer, "update", lambda sample: sample.imu)

    def sample(t, rate):
        return FlightMeasurement(
            t,
            np.array([0.0, rate, 0.0]),
            np.array([0.0, 0.0, -9.80665]),
            np.array([1.0, 0.0, 0.0]),
            np.zeros(3),
            np.array([0.0, 0.2, 0.0]),
            1.0,
            1.0,
        )

    agent._observe(sample(0.0, 0.0))
    agent._observe(sample(0.01, 0.002))
    np.testing.assert_allclose(
        agent.identifier.derivatives, np.eye(3) * 0.5, atol=1e-14
    )
