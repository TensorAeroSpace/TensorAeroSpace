import numpy as np
import pytest
import torch

from tensoraerospace.envs.b747_vec_torch import (
    ImprovedB747VecEnvTorch,
    SignalRandomization,
    _make_signal_randomization,
)


def test_init_validations():
    with pytest.raises(ValueError):
        ImprovedB747VecEnvTorch(num_envs=0)
    with pytest.raises(ValueError):
        ImprovedB747VecEnvTorch(num_envs=1, reward_mode="bad")
    with pytest.raises(ValueError):
        ImprovedB747VecEnvTorch(num_envs=1, survival_bonus=-1.0)
    with pytest.raises(ValueError):
        ImprovedB747VecEnvTorch(num_envs=1, initial_state=np.zeros(3))


def test_reset_shapes_with_reference():
    env = ImprovedB747VecEnvTorch(
        num_envs=2,
        dt=0.1,
        tn=1.0,
        device="cpu",
        include_reference_in_obs=True,
        auto_reset=False,
        seed=0,
    )
    obs, info = env.reset()
    assert obs.shape == (2, 6)
    assert env.reference_signal.shape == (2, env.number_time_steps)
    assert torch.all(env.step_count == 0)
    assert torch.all(env.prev_action == 0)
    assert info == {}


def test_step_shapes_and_action_limits():
    env = ImprovedB747VecEnvTorch(
        num_envs=2, dt=0.1, tn=1.0, device="cpu", auto_reset=False
    )
    action = torch.full((2, 1), 10.0)
    obs, reward, terminated, truncated, _ = env.step(action)

    assert obs.shape == (2, 4)
    assert reward.shape == (2,)
    assert terminated.dtype == torch.bool
    assert truncated.dtype == torch.bool
    assert torch.all(torch.abs(env.prev_u_rad) <= env.input_magnitude_limit_rad + 1e-6)
    assert torch.all(env.prev_action.abs() <= 1.0)


def test_auto_reset_on_truncation():
    env = ImprovedB747VecEnvTorch(
        num_envs=3, dt=0.1, tn=0.1, device="cpu", auto_reset=True
    )
    env.reset()
    _, _, _, truncated, _ = env.step(torch.zeros((3, 1)))
    assert torch.any(truncated)
    assert torch.all(env.step_count == 0)


def test_reference_generation_mixed_and_sine():
    env = ImprovedB747VecEnvTorch(
        num_envs=2,
        dt=0.1,
        tn=1.0,
        device="cpu",
        auto_reset=False,
        step_randomization={
            "signal_type": "mixed",
            "amplitude_deg_range": (5.0, 5.0),
            "min_abs_amplitude_deg": 5.0,
            "p_step": 0.0,
            "p_sine": 1.0,
            "frequency_hz_range": (0.05, 0.05),
        },
        seed=123,
    )
    env.reset()
    assert torch.any(env.reference_signal != 0)
    assert env.reference_signal.shape[1] == env.number_time_steps

    env_sine = ImprovedB747VecEnvTorch(
        num_envs=1,
        dt=0.1,
        tn=1.0,
        device="cpu",
        auto_reset=False,
        step_randomization={"signal_type": "sine", "amplitude_deg_range": (2.0, 2.0)},
        seed=321,
    )
    env_sine.reset()
    ref = env_sine.reference_signal[0]
    assert torch.all(ref <= torch.max(ref))  # sanity access
    assert torch.any(ref != 0)


def test_ramp_reference_monotonic():
    env = ImprovedB747VecEnvTorch(
        num_envs=1,
        dt=0.1,
        tn=1.0,
        device="cpu",
        auto_reset=False,
        step_randomization={"signal_type": "ramp", "amplitude_deg_range": (3.0, 3.0)},
        seed=42,
    )
    env.reset()
    ref = env.reference_signal[0]
    assert ref[-1] > ref[0]
    assert torch.all(ref <= torch.max(ref))


def test_tracking_reward_branch_and_init_args():
    env = ImprovedB747VecEnvTorch(
        num_envs=1,
        dt=0.2,
        tn=1.0,
        device="cpu",
        auto_reset=False,
        reward_mode="tracking",
        survival_bonus=0.5,
        completion_bonus=0.25,
        step_randomization={"signal_type": "step"},
        seed=7,
    )
    env.reset()
    _, reward, _, _, _ = env.step(torch.zeros((1, 1)))
    assert reward.shape == (1,)
    args = env.get_init_args()
    assert args["reward_mode"] == "tracking"
    assert args["num_envs"] == 1
    assert args["step_randomization"]["signal_type"] == "step"
    assert args["survival_bonus"] == 0.5
    assert args["completion_bonus"] == 0.25


def test_step_response_penalties_and_oscillation_tracking():
    env = ImprovedB747VecEnvTorch(
        num_envs=2,
        dt=0.1,
        tn=2.0,
        device="cpu",
        auto_reset=False,
        reward_mode="step_response",
        step_randomization={
            "signal_type": "step",
            "amplitude_deg_range": (10.0, 10.0),
            "min_abs_amplitude_deg": 10.0,
            "step_time_sec_range": (0.0, 0.0),
        },
        seed=0,
    )
    env.reset()

    # Env0: create overshoot (theta above target), Env1: crossing to trigger oscillation flag.
    target_rad = env.reference_signal[0, 0].item()
    env.state[0, env._idx_theta] = target_rad * 1.5  # overshoot
    env.state[1, env._idx_theta] = -target_rad * 0.5  # opposite sign to prev_error_sign
    env._seg_amp = torch.tensor([target_rad, target_rad], device=env.device)
    env._seg_sign = torch.tensor([1.0, 1.0], device=env.device)
    env._prev_error_sign = torch.tensor([0, 1], device=env.device)  # env1 crossing
    env._sign_changes = torch.tensor([0, 1], device=env.device)

    obs, reward, terminated, truncated, _ = env.step(torch.zeros((2, 1)))
    assert obs.shape == (2, 4)
    assert reward.shape == (2,)
    assert torch.isfinite(reward).all()
    # Overshoot should be tracked for env0, oscillation count should increase for env1.
    assert env._seg_max_err_dir[0] > 0
    assert env._sign_changes[1] > 1
    assert not terminated.any()
    assert not truncated.any()


def test_early_termination_penalty_per_step_and_completion_bonus():
    env = ImprovedB747VecEnvTorch(
        num_envs=2,
        dt=0.1,
        tn=1.0,
        device="cpu",
        auto_reset=False,
        reward_mode="step_response",
        early_termination_penalty_per_step=2.0,
        completion_bonus=1.5,
        survival_bonus=0.0,
        step_randomization={"signal_type": "step", "amplitude_deg_range": (5.0, 5.0)},
        seed=1,
    )
    env.reset()
    # Env0 will terminate (theta beyond limit) early in the horizon to accrue per-step penalty.
    env.state[0, env._idx_theta] = env.max_pitch_rad * 1.5
    env.step_count[0] = 0
    # Env1 will truncate by reaching horizon without terminating.
    env.step_count[1] = env.number_time_steps - 3  # after increment => truncated True
    env.state[1, env._idx_theta] = 0.0

    _, reward, terminated, truncated, _ = env.step(torch.zeros((2, 1)))
    assert bool(terminated[0]) is True
    assert float(reward[0]) <= -100.0  # early termination penalty applied
    assert bool(truncated[1]) is True
    assert float(reward[1]) >= 1.0  # completion bonus dominates near-zero cost


def test_include_reference_in_obs_on_step_and_action_vector():
    env = ImprovedB747VecEnvTorch(
        num_envs=2,
        dt=0.2,
        tn=1.0,
        device="cpu",
        auto_reset=False,
        include_reference_in_obs=True,
        step_randomization={"signal_type": "step", "amplitude_deg_range": (2.0, 2.0)},
        seed=11,
    )
    env.reset()
    obs, reward, terminated, truncated, _ = env.step(torch.tensor([0.5, -0.5]))
    assert obs.shape == (2, 6)
    assert reward.shape == (2,)
    assert terminated.shape == (2,)
    assert truncated.shape == (2,)


def test_reset_done_resets_only_masked_envs():
    env = ImprovedB747VecEnvTorch(
        num_envs=3, dt=0.1, tn=1.0, device="cpu", auto_reset=False
    )
    env.reset()
    env.step_count = torch.tensor([5, 6, 7])
    env.prev_action = torch.tensor([0.1, 0.2, 0.3])
    done_mask = torch.tensor([True, False, True])
    env._reset_done(done_mask)
    assert torch.all(env.step_count == torch.tensor([0, 6, 0]))
    assert env.prev_action[0] == 0.0
    assert env.prev_action[2] == 0.0
    assert env.prev_action[1] == 0.2


def test_sample_amplitude_with_deadzone():
    env = ImprovedB747VecEnvTorch(
        num_envs=1,
        dt=0.1,
        tn=1.0,
        device="cpu",
        auto_reset=False,
        step_randomization={
            "signal_type": "step",
            "amplitude_deg_range": (-10.0, 10.0),
            "min_abs_amplitude_deg": 5.0,
        },
        seed=5,
    )
    amps = env._sample_amplitude(16)
    assert torch.all(torch.abs(amps) >= 5.0 - 1e-5)


def test_rate_and_magnitude_limits():
    env = ImprovedB747VecEnvTorch(
        num_envs=1,
        dt=0.1,
        tn=1.0,
        device="cpu",
        auto_reset=False,
        step_randomization={"signal_type": "step", "amplitude_deg_range": (5.0, 5.0)},
        seed=2,
    )
    env.reset()
    env.prev_u_rad[:] = env.input_magnitude_limit_rad
    action = torch.tensor([[10.0]])  # tries to push beyond magnitude and rate limits
    _, _, _, _, _ = env.step(action)
    # Rate limit and magnitude limit should both hold.
    assert torch.all(env.prev_u_rad <= env.input_magnitude_limit_rad + 1e-6)
    assert torch.all(env.prev_u_rad >= -env.input_magnitude_limit_rad - 1e-6)


def test_make_signal_randomization_filters_unknown_fields():
    params = {
        "signal_type": "sine",
        "amplitude_deg_range": (1.0, 2.0),
        "unknown_field": 123,
    }
    rand = _make_signal_randomization(params)
    assert isinstance(rand, SignalRandomization)
    assert rand.signal_type == "sine"
    assert rand.amplitude_deg_range == (1.0, 2.0)


def test_init_validations_early_term_penalty():
    """Cover lines 117-121: early_termination_penalty validations."""
    with pytest.raises(ValueError):
        ImprovedB747VecEnvTorch(num_envs=1, early_termination_penalty=-1.0)
    with pytest.raises(ValueError):
        ImprovedB747VecEnvTorch(num_envs=1, early_termination_penalty_per_step=-1.0)
    with pytest.raises(ValueError):
        ImprovedB747VecEnvTorch(num_envs=1, completion_bonus=-1.0)


def test_step_randomization_as_object():
    """Cover line 212: step_randomization passed as SignalRandomization object."""
    rand_obj = SignalRandomization(signal_type="ramp", amplitude_deg_range=(2.0, 4.0))
    env = ImprovedB747VecEnvTorch(
        num_envs=1, dt=0.1, tn=1.0, device="cpu", step_randomization=rand_obj
    )
    assert env.step_rand is rand_obj


def test_unwrapped_property():
    """Cover line 265: .unwrapped returns self."""
    env = ImprovedB747VecEnvTorch(num_envs=1, dt=0.1, tn=1.0, device="cpu")
    assert env.unwrapped is env


def test_sample_reference_empty_indices():
    """Cover line 391: _sample_reference_for_indices with empty tensor."""
    env = ImprovedB747VecEnvTorch(num_envs=2, dt=0.1, tn=1.0, device="cpu")
    env.reset()
    ref_before = env.reference_signal.clone()
    env._sample_reference_for_indices(torch.tensor([], dtype=torch.int64, device="cpu"))
    assert torch.all(env.reference_signal == ref_before)


def test_mixed_signal_step_and_ramp_branches():
    """Cover lines 414-416 (step mask) and 428-430 (ramp mask) in mixed signal."""
    env = ImprovedB747VecEnvTorch(
        num_envs=100,
        dt=0.1,
        tn=1.0,
        device="cpu",
        auto_reset=False,
        step_randomization={
            "signal_type": "mixed",
            "amplitude_deg_range": (5.0, 5.0),
            "p_step": 0.5,
            "p_sine": 0.0,  # no sine, rest is ramp
        },
        seed=999,
    )
    env.reset()
    # With 100 envs, p_step=0.5, p_sine=0, we'll have ~50 step and ~50 ramp
    assert env.reference_signal.shape == (100, env.number_time_steps)


def test_reset_with_seed_parameter():
    """Cover line 461: reset(seed=...) re-seeds generator."""
    env = ImprovedB747VecEnvTorch(num_envs=2, dt=0.1, tn=1.0, device="cpu", seed=0)
    env.reset(seed=123)
    ref1 = env.reference_signal.clone()
    env.reset(seed=123)
    ref2 = env.reference_signal.clone()
    assert torch.all(ref1 == ref2)


def test_reset_done_all_false():
    """Cover line 525: _reset_done with all False returns early."""
    env = ImprovedB747VecEnvTorch(num_envs=2, dt=0.1, tn=1.0, device="cpu")
    env.reset()
    env.step_count[:] = 5
    done_mask = torch.tensor([False, False])
    env._reset_done(done_mask)
    assert torch.all(env.step_count == 5)
