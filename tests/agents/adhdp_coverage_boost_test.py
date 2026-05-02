"""Tests to boost coverage for tensoraerospace/agent/adhdp/model.py.

Target: model.py 6% -> ~44%
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from tensoraerospace.agent.adhdp.model import ADHDP, _as_flat_np

# ---------------------------------------------------------------------------
# Helpers: lightweight fake envs
# ---------------------------------------------------------------------------


class _TinySpace:
    def __init__(self, shape, low=-1.0, high=1.0):
        self.shape = tuple(shape)
        self.low = np.full(self.shape, low, dtype=np.float32)
        self.high = np.full(self.shape, high, dtype=np.float32)


class _TinyEnv:
    """Minimal env with 6-dim obs (ImprovedB747-like: pitch_err, q, theta, prev_action, theta_ref, q_ref)."""

    def __init__(self, obs_dim=6, act_dim=1, max_steps=5):
        self.observation_space = _TinySpace((obs_dim,))
        self.action_space = _TinySpace((act_dim,))
        self._step_count = 0
        self.max_steps = int(max_steps)
        self.dt = 0.1
        self.reward_scale = 1.0
        self.max_pitch_rad = np.deg2rad(20.0)

    def reset(self):
        self._step_count = 0
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        # Provide some nonzero obs to exercise baseline branches
        obs[0] = 0.05  # pitch error
        obs[1] = 0.01  # q
        obs[2] = 0.03  # theta
        obs[3] = 0.0  # prev_action
        if obs.shape[0] >= 6:
            obs[4] = 0.04  # theta_ref
            obs[5] = 0.005  # q_ref
        return obs, {}

    def step(self, action):
        _ = np.asarray(action, dtype=np.float32).reshape(-1)
        self._step_count += 1
        obs = np.random.randn(self.observation_space.shape[0]).astype(np.float32) * 0.05
        reward = -0.05
        terminated = self._step_count >= self.max_steps
        truncated = False
        info = {"cost_total": 0.03}
        return obs, reward, terminated, truncated, info

    @property
    def unwrapped(self):
        return self


class _SmallEnv(_TinyEnv):
    """4-dim obs env (no reference dims)."""

    def __init__(self, max_steps=5):
        super().__init__(obs_dim=4, act_dim=1, max_steps=max_steps)


class _EnvWithRef(_TinyEnv):
    """Env with initial_state and reference_signal attributes (for randomization tests)."""

    def __init__(self, max_steps=5):
        super().__init__(obs_dim=6, act_dim=1, max_steps=max_steps)
        self.initial_state = np.zeros(4, dtype=np.float32)
        self.reference_signal = np.zeros((1, 10), dtype=np.float32)
        self.reference_signal[0, :] = 0.01


# ===========================================================================
# _as_flat_np
# ===========================================================================


class TestAsFlatNp:
    def test_list(self):
        r = _as_flat_np([1.0, 2.0])
        assert r.shape == (2,)
        assert r.dtype == np.float32

    def test_2d(self):
        r = _as_flat_np(np.array([[1, 2], [3, 4]]))
        assert r.shape == (4,)


# ===========================================================================
# ADHDP construction
# ===========================================================================


class TestADHDPConstruction:
    def test_default(self):
        env = _TinyEnv()
        agent = ADHDP(env=env, device="cpu")
        assert hasattr(agent, "actor")
        assert hasattr(agent, "critic")
        assert agent.policy_mode == "direct"

    def test_paper_strict(self):
        env = _TinyEnv()
        agent = ADHDP(env=env, device="cpu", paper_strict=True)
        assert agent.paper_strict is True
        assert agent.policy_mode == "direct"
        assert agent.actor_bc_l2 == 0.0

    def test_residual_mode(self):
        env = _TinyEnv()
        agent = ADHDP(env=env, device="cpu", policy_mode="residual", residual_scale=0.3)
        assert agent.policy_mode == "residual"
        assert agent.residual_scale == 0.3

    def test_invalid_obs_dim(self):
        class _Bad:
            observation_space = _TinySpace((0,))
            action_space = _TinySpace((1,))
            unwrapped = property(lambda self: self)

        with pytest.raises(ValueError, match="ADHDP expects Box-like spaces"):
            ADHDP(env=_Bad(), device="cpu")

    def test_action_momentum_invalid(self):
        env = _TinyEnv()
        agent = ADHDP(env=env, device="cpu", action_momentum=1.5)
        assert agent.action_momentum == 0.0

    def test_action_max_abs_invalid(self):
        env = _TinyEnv()
        agent = ADHDP(env=env, device="cpu", action_max_abs=0.0)
        assert agent.action_max_abs == 1.0

    def test_negative_lr_clamp(self):
        env = _TinyEnv()
        agent = ADHDP(
            env=env,
            device="cpu",
            action_grad_lr=-1.0,
            action_grad_step_clip=-1.0,
            action_grad_u_l2=-1.0,
            action_grad_du_l2=-1.0,
            actor_distill_coef=-1.0,
            actor_distill_lr=-1.0,
            teacher_rollout_noise_std=-1.0,
        )
        assert agent.action_grad_lr == 0.0
        assert agent.action_grad_step_clip == 0.0
        assert agent.action_grad_u_l2 == 0.0
        assert agent.action_grad_du_l2 == 0.0
        assert agent.actor_distill_coef == 0.0
        assert agent.actor_distill_lr == 0.0
        assert agent.teacher_rollout_noise_std == 0.0

    def test_initial_state_capture(self):
        env = _EnvWithRef()
        agent = ADHDP(env=env, device="cpu")
        assert agent._initial_state_base is not None
        assert agent._reference_signal_base is not None

    def test_get_env(self):
        env = _TinyEnv()
        agent = ADHDP(env=env, device="cpu")
        assert agent.get_env() is env


# ===========================================================================
# ADHDP select_action / predict
# ===========================================================================


class TestADHDPSelectAction:
    def test_select_action_evaluate(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(env=env, device="cpu", exploration_std=0.0)
        obs, _ = env.reset()
        act = agent.select_action(obs, evaluate=True)
        assert act.shape == (1,)
        assert np.all(act >= -1.0) and np.all(act <= 1.0)

    def test_select_action_with_exploration(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu", exploration_std=0.5)
        obs, _ = env.reset()
        acts = [agent.select_action(obs, evaluate=False) for _ in range(50)]
        assert np.std(np.stack(acts)) > 0.0

    def test_predict_positional(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu", exploration_std=0.0)
        obs, _ = env.reset()
        act1 = agent.select_action(obs, evaluate=True)
        act2 = agent.predict(obs)
        np.testing.assert_allclose(act1, act2)

    def test_predict_deterministic_false(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu", exploration_std=0.5)
        obs, _ = env.reset()
        act = agent.predict(obs, deterministic=False)
        assert act.shape == (1,)

    def test_predict_two_positional_args(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu", exploration_std=0.0)
        obs, _ = env.reset()
        act = agent.predict(obs, True)
        assert act.shape == (1,)

    def test_predict_no_state_raises(self):
        env = _TinyEnv()
        agent = ADHDP(env=env, device="cpu")
        with pytest.raises(ValueError, match="expects `state`"):
            agent.predict()


# ===========================================================================
# ADHDP reset
# ===========================================================================


class TestADHDPReset:
    def test_reset(self):
        env = _TinyEnv()
        agent = ADHDP(env=env, device="cpu")
        agent._pid_i = 0.5
        agent._pid_prev_meas = 0.1
        agent.reset()
        assert agent._pid_i == 0.0
        assert agent._pid_prev_meas == 0.0


# ===========================================================================
# ADHDP baseline action
# ===========================================================================


class TestADHDPBaseline:
    def test_pd_baseline_4dim(self):
        env = _SmallEnv()
        agent = ADHDP(env=env, device="cpu", baseline_type="pd")
        obs_t = torch.randn(1, 4)
        u = agent._baseline_action(obs_t)
        assert u.shape == (1, 1)

    def test_pid_baseline_6dim(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(env=env, device="cpu", baseline_type="pid")
        obs_t = torch.randn(1, 6)
        u = agent._baseline_action(obs_t)
        assert u.shape == (1, 1)

    def test_pid_baseline_4dim(self):
        env = _SmallEnv()
        agent = ADHDP(env=env, device="cpu", baseline_type="pid")
        obs_t = torch.randn(1, 4)
        u = agent._baseline_action(obs_t)
        assert u.shape == (1, 1)

    def test_pid_baseline_batch(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(env=env, device="cpu", baseline_type="pid")
        obs_t = torch.randn(5, 6)
        u = agent._baseline_action(obs_t)
        assert u.shape == (5, 1)

    def test_pid_baseline_no_max_pitch_rad(self):
        env = _SmallEnv()
        del env.max_pitch_rad  # remove attribute
        agent = ADHDP(env=env, device="cpu", baseline_type="pid")
        obs_t = torch.randn(1, 4)
        u = agent._baseline_action(obs_t)
        assert u.shape == (1, 1)

    def test_baseline_invalid_ndim(self):
        env = _TinyEnv()
        agent = ADHDP(env=env, device="cpu")
        with pytest.raises(ValueError, match="Expected obs shape"):
            agent._baseline_action(torch.randn(4))

    def test_baseline_too_few_dims(self):
        env = _TinyEnv()
        agent = ADHDP(env=env, device="cpu")
        with pytest.raises(ValueError, match="at least 2 dims"):
            agent._baseline_action(torch.randn(1, 1))

    def test_pid_antiwindup(self):
        """Exercise PID anti-windup clipping."""
        env = _SmallEnv()
        agent = ADHDP(env=env, device="cpu", baseline_type="pid", pid_i_clip=0.01)
        # Large error to force saturation
        obs_t = torch.tensor([[10.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        u1 = agent._baseline_action(obs_t)
        u2 = agent._baseline_action(obs_t)
        assert u1.shape == (1, 1)
        assert u2.shape == (1, 1)


# ===========================================================================
# ADHDP policy action
# ===========================================================================


class TestADHDPPolicyAction:
    def test_direct_mode(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(env=env, device="cpu", policy_mode="direct", exploration_std=0.0)
        obs_t = torch.randn(1, 6)
        act = agent._policy_action_t(obs_t, evaluate=True)
        assert act.shape == (1, 1)

    def test_residual_mode(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(
            env=env,
            device="cpu",
            policy_mode="residual",
            residual_scale=0.2,
            exploration_std=0.0,
        )
        obs_t = torch.randn(1, 6)
        act = agent._policy_action_t(obs_t, evaluate=True)
        assert act.shape == (1, 1)

    def test_force_baseline(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(env=env, device="cpu")
        obs_t = torch.randn(1, 6)
        act = agent._policy_action_t(obs_t, evaluate=True, force_baseline=True)
        assert act.shape == (1, 1)

    def test_action_momentum(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(env=env, device="cpu", action_momentum=0.5, exploration_std=0.0)
        obs_t = torch.randn(1, 6)
        act = agent._policy_action_t(obs_t, evaluate=True)
        assert act.shape == (1, 1)

    def test_action_max_abs_clipping(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu", action_max_abs=0.5, exploration_std=0.0)
        obs_t = torch.randn(1, 4) * 10
        act = agent._policy_action_t(obs_t, evaluate=True)
        assert torch.all(act <= 0.5 + 1e-6)
        assert torch.all(act >= -0.5 - 1e-6)

    def test_critic_gradient_action_selection(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(
            env=env,
            device="cpu",
            action_selection="critic_gradient",
            action_grad_steps=3,
            action_grad_lr=0.1,
            exploration_std=0.05,
        )
        obs_t = torch.randn(1, 6)
        act = agent._policy_action_t(obs_t, evaluate=False)
        assert act.shape == (1, 1)


# ===========================================================================
# ADHDP optimize action by critic
# ===========================================================================


class TestOptimizeAction:
    def test_basic(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(
            env=env,
            device="cpu",
            action_grad_steps=5,
            action_grad_lr=0.1,
        )
        obs_t = torch.randn(1, 6)
        a_init = torch.zeros(1, 1)
        a = agent._optimize_action_by_critic(obs_t, a_init=a_init, n_steps=5)
        assert a.shape == (1, 1)

    def test_with_regularizers(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(
            env=env,
            device="cpu",
            action_grad_steps=3,
            action_grad_lr=0.1,
            action_grad_u_l2=0.1,
            action_grad_du_l2=0.1,
            action_grad_step_clip=0.05,
            action_max_abs=0.8,
        )
        obs_t = torch.randn(1, 6)
        a_init = torch.zeros(1, 1)
        a = agent._optimize_action_by_critic(obs_t, a_init=a_init, n_steps=3)
        assert a.shape == (1, 1)

    def test_zero_lr(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu", action_grad_lr=0.0)
        obs_t = torch.randn(1, 4)
        a_init = torch.tensor([[0.5]])
        a = agent._optimize_action_by_critic(obs_t, a_init=a_init, n_steps=5)
        # Should return clipped a_init without modification
        assert a.shape == (1, 1)


# ===========================================================================
# ADHDP teacher action
# ===========================================================================


class TestTeacherAction:
    def test_teacher_action(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(
            env=env,
            device="cpu",
            action_selection="critic_gradient",
            action_grad_steps=2,
            action_grad_lr=0.1,
            teacher_rollout_noise_std=0.1,
        )
        obs_t = torch.randn(1, 6)
        a = agent._teacher_action_t(obs_t, evaluate=False)
        assert a.shape == (1, 1)

    def test_teacher_action_4dim(self):
        env = _SmallEnv()
        agent = ADHDP(
            env=env,
            device="cpu",
            action_selection="critic_gradient",
            action_grad_steps=2,
            action_grad_lr=0.1,
        )
        obs_t = torch.randn(1, 4)
        a = agent._teacher_action_t(obs_t, evaluate=True)
        assert a.shape == (1, 1)

    def test_teacher_action_small_obs(self):
        """Test with obs dim < 4 (no prev action in obs)."""
        env = _TinyEnv(obs_dim=3, act_dim=1)
        agent = ADHDP(
            env=env,
            device="cpu",
            action_selection="critic_gradient",
            action_grad_steps=2,
            action_grad_lr=0.1,
        )
        obs_t = torch.randn(1, 3)
        a = agent._teacher_action_t(obs_t, evaluate=True)
        assert a.shape == (1, 1)


# ===========================================================================
# ADHDP env randomization
# ===========================================================================


class TestMaybeRandomize:
    def test_randomize_initial_state(self):
        env = _EnvWithRef()
        agent = ADHDP(env=env, device="cpu", initial_state_noise_std=0.1)
        agent._maybe_randomize_env_for_episode()
        # initial_state should have been perturbed
        # (We can't check exact value but should not raise)

    def test_randomize_reference(self):
        env = _EnvWithRef()
        agent = ADHDP(
            env=env,
            device="cpu",
            reference_roll_steps=2,
            reference_noise_std=0.001,
        )
        orig_ref = env.reference_signal.copy()
        agent._maybe_randomize_env_for_episode()
        # reference may have been modified
        assert env.reference_signal is not None

    def test_no_randomize_when_zero(self):
        env = _EnvWithRef()
        agent = ADHDP(
            env=env, device="cpu", initial_state_noise_std=0.0, reference_roll_steps=0
        )
        orig_init = env.initial_state.copy()
        agent._maybe_randomize_env_for_episode()
        # Should remain unchanged
        np.testing.assert_array_equal(env.initial_state, orig_init)


# ===========================================================================
# ADHDP _td_update
# ===========================================================================


class TestTdUpdate:
    def test_td_update_both(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu")
        obs = np.random.randn(4).astype(np.float32)
        act = np.random.randn(1).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)

        c_loss, a_loss = agent._td_update(
            obs=obs,
            act=act,
            reward_for_update=-0.1,
            next_obs=next_obs,
            done_bootstrap=0.0,
            do_critic_update=True,
            do_actor_update=True,
        )
        assert isinstance(c_loss, float)
        assert isinstance(a_loss, float)

    def test_td_update_critic_only(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu")
        obs = np.random.randn(4).astype(np.float32)
        act = np.random.randn(1).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)

        c_loss, a_loss = agent._td_update(
            obs=obs,
            act=act,
            reward_for_update=-0.1,
            next_obs=next_obs,
            done_bootstrap=0.0,
            do_critic_update=True,
            do_actor_update=False,
        )
        assert isinstance(c_loss, float)
        assert a_loss == 0.0

    def test_td_update_with_bc(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu", actor_bc_l2=0.5)
        obs = np.random.randn(4).astype(np.float32)
        act = np.random.randn(1).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)

        c_loss, a_loss = agent._td_update(
            obs=obs,
            act=act,
            reward_for_update=-0.1,
            next_obs=next_obs,
            done_bootstrap=0.0,
            do_critic_update=True,
            do_actor_update=True,
        )
        assert isinstance(a_loss, float)

    def test_td_update_residual_bc(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(
            env=env,
            device="cpu",
            policy_mode="residual",
            residual_scale=0.2,
            actor_bc_l2=0.5,
        )
        obs = np.random.randn(6).astype(np.float32)
        act = np.random.randn(1).astype(np.float32)
        next_obs = np.random.randn(6).astype(np.float32)

        c_loss, a_loss = agent._td_update(
            obs=obs,
            act=act,
            reward_for_update=-0.1,
            next_obs=next_obs,
            done_bootstrap=0.0,
            do_critic_update=True,
            do_actor_update=True,
        )
        assert isinstance(a_loss, float)

    def test_td_update_force_baseline_policy(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(env=env, device="cpu")
        obs = np.random.randn(6).astype(np.float32)
        act = np.random.randn(1).astype(np.float32)
        next_obs = np.random.randn(6).astype(np.float32)

        c_loss, a_loss = agent._td_update(
            obs=obs,
            act=act,
            reward_for_update=-0.1,
            next_obs=next_obs,
            done_bootstrap=0.0,
            do_critic_update=True,
            do_actor_update=True,
            force_baseline_policy=True,
        )
        assert isinstance(c_loss, float)


# ===========================================================================
# ADHDP _critic_update_with_target / _actor_update
# ===========================================================================


class TestCriticActorUpdate:
    def test_critic_update_with_target(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu")
        obs_t = torch.randn(1, 4)
        act_t = torch.randn(1, 1)
        target_q = torch.tensor([[0.5]])
        loss = agent._critic_update_with_target(
            obs_t=obs_t,
            act_t=act_t,
            target_q=target_q,
            n_steps=3,
        )
        assert isinstance(loss, float)

    def test_actor_update_minimize_critic(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu", actor_update_mode="minimize_critic")
        obs_t = torch.randn(1, 4)
        loss = agent._actor_update(obs_t=obs_t, n_steps=2)
        assert isinstance(loss, float)

    def test_actor_update_with_bc(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu", actor_bc_l2=0.5)
        obs_t = torch.randn(1, 4)
        loss = agent._actor_update(obs_t=obs_t, n_steps=2)
        assert isinstance(loss, float)

    def test_actor_update_residual_bc(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(
            env=env,
            device="cpu",
            policy_mode="residual",
            actor_bc_l2=0.5,
        )
        obs_t = torch.randn(1, 6)
        loss = agent._actor_update(obs_t=obs_t, n_steps=2)
        assert isinstance(loss, float)

    def test_actor_update_distill_mode(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADHDP(
            env=env,
            device="cpu",
            actor_update_mode="distill_critic_gradient",
            actor_distill_coef=1.0,
            actor_distill_steps=2,
            actor_distill_lr=0.01,
            action_grad_steps=3,
            action_grad_lr=0.1,
        )
        obs_t = torch.randn(1, 6)
        loss = agent._actor_update(obs_t=obs_t, n_steps=1)
        assert isinstance(loss, float)


# ===========================================================================
# ADHDP warmstart actor
# ===========================================================================


class TestWarmstart:
    def test_warmstart_direct(self):
        env = _TinyEnv(obs_dim=6, act_dim=1, max_steps=3)
        agent = ADHDP(
            env=env,
            device="cpu",
            warmstart_actor_episodes=1,
            warmstart_actor_epochs=1,
        )
        agent._warmstart_actor(episodes=1, max_steps=3)

    def test_warmstart_residual(self):
        env = _TinyEnv(obs_dim=6, act_dim=1, max_steps=3)
        agent = ADHDP(
            env=env,
            device="cpu",
            policy_mode="residual",
            warmstart_actor_episodes=1,
            warmstart_actor_epochs=1,
        )
        agent._warmstart_actor(episodes=1, max_steps=3)

    def test_warmstart_zero_episodes(self):
        env = _TinyEnv()
        agent = ADHDP(env=env, device="cpu")
        agent._warmstart_actor(episodes=0, max_steps=5)  # should be no-op


# ===========================================================================
# ADHDP train (integration)
# ===========================================================================


class TestADHDPTrain:
    def test_train_basic(self):
        env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=5)
        agent = ADHDP(
            env=env,
            device="cpu",
            exploration_std=0.0,
            log_every_updates=2,
        )
        agent.train(num_episodes=2, max_steps=5, show_progress=False)
        assert agent._updates > 0

    def test_train_with_warmstart(self):
        env = _TinyEnv(obs_dim=6, act_dim=1, max_steps=5)
        agent = ADHDP(
            env=env,
            device="cpu",
            warmstart_actor_episodes=1,
            warmstart_actor_epochs=1,
            exploration_std=0.0,
            log_every_updates=2,
        )
        agent.train(num_episodes=1, max_steps=5, show_progress=False)

    def test_train_with_critic_warmup(self):
        env = _TinyEnv(obs_dim=6, act_dim=1, max_steps=5)
        agent = ADHDP(
            env=env,
            device="cpu",
            critic_warmup_episodes=1,
            exploration_std=0.0,
            log_every_updates=2,
        )
        agent.train(num_episodes=2, max_steps=5, show_progress=False)

    def test_train_alternating_cycles(self):
        env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=5)
        agent = ADHDP(
            env=env,
            device="cpu",
            critic_cycle_episodes=1,
            action_cycle_episodes=1,
            log_every_updates=2,
        )
        agent.train(num_episodes=4, max_steps=5, show_progress=False)

    def test_train_baseline_warmup(self):
        env = _TinyEnv(obs_dim=6, act_dim=1, max_steps=5)
        agent = ADHDP(
            env=env,
            device="cpu",
            baseline_warmup_episodes=1,
            log_every_updates=2,
        )
        agent.train(num_episodes=2, max_steps=5, show_progress=False)

    def test_train_bc_decay(self):
        env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=5)
        agent = ADHDP(
            env=env,
            device="cpu",
            actor_bc_l2=0.1,
            actor_bc_decay=0.5,
            log_every_updates=2,
        )
        agent.train(num_episodes=3, max_steps=5, show_progress=False)
        assert agent._actor_bc_coef < 0.1

    def test_train_env_cost(self):
        env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=5)
        agent = ADHDP(
            env=env,
            device="cpu",
            use_env_cost=True,
            log_every_updates=2,
        )
        agent.train(num_episodes=2, max_steps=5, show_progress=False)

    def test_train_critic_gradient_selection(self):
        env = _TinyEnv(obs_dim=6, act_dim=1, max_steps=5)
        agent = ADHDP(
            env=env,
            device="cpu",
            action_selection="critic_gradient",
            action_grad_steps=2,
            action_grad_lr=0.1,
            log_every_updates=2,
        )
        agent.train(num_episodes=1, max_steps=5, show_progress=False)

    def test_train_with_env_randomization(self):
        env = _EnvWithRef(max_steps=5)
        agent = ADHDP(
            env=env,
            device="cpu",
            initial_state_noise_std=0.01,
            reference_roll_steps=1,
            reference_noise_std=0.001,
            log_every_updates=2,
        )
        agent.train(num_episodes=2, max_steps=5, show_progress=False)

    def test_train_multi_step_critic_actor(self):
        env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=5)
        agent = ADHDP(
            env=env,
            device="cpu",
            critic_updates_per_step=2,
            actor_updates_per_step=2,
            log_every_updates=2,
        )
        agent.train(num_episodes=1, max_steps=5, show_progress=False)

    def test_train_with_distill_execute_teacher(self):
        env = _TinyEnv(obs_dim=6, act_dim=1, max_steps=5)
        agent = ADHDP(
            env=env,
            device="cpu",
            actor_update_mode="distill_critic_gradient",
            distill_execute_teacher=True,
            actor_distill_coef=1.0,
            actor_distill_steps=2,
            actor_distill_lr=0.01,
            action_grad_steps=2,
            action_grad_lr=0.1,
            log_every_updates=2,
        )
        agent.train(num_episodes=1, max_steps=5, show_progress=False)


# ===========================================================================
# ADHDP save / load
# ===========================================================================


class TestADHDPSaveLoad:
    def test_save_basic(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = agent.save(path=tmpdir)
            assert os.path.isdir(run_dir)
            assert os.path.isfile(os.path.join(run_dir, "config.json"))
            assert os.path.isfile(os.path.join(run_dir, "actor.pth"))
            assert os.path.isfile(os.path.join(run_dir, "critic.pth"))
            actor_state = torch.load(
                os.path.join(run_dir, "actor.pth"), weights_only=True
            )
            assert isinstance(actor_state, dict)

    def test_save_with_gradients(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = agent.save(path=tmpdir, save_gradients=True)
            assert os.path.isfile(os.path.join(run_dir, "actor_optim.pth"))
            assert os.path.isfile(os.path.join(run_dir, "critic_optim.pth"))

    def test_save_default_path(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = agent.save(path=tmpdir)
            assert os.path.isdir(run_dir)

    def test_load_roundtrip(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu", exploration_std=0.0)
        obs, _ = env.reset()
        act_before = agent.select_action(obs, evaluate=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = agent.save(path=tmpdir)
            # Create a fresh agent and load into it
            agent2 = ADHDP(env=env, device="cpu", exploration_std=0.0)
            agent2.load(path=run_dir)
            act_after = agent2.select_action(obs, evaluate=True)
            np.testing.assert_allclose(act_before, act_after, atol=1e-5)

    def test_load_missing_files(self):
        env = _TinyEnv()
        agent = ADHDP(env=env, device="cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="Missing actor/critic"):
                agent.load(path=tmpdir)

    def test_load_no_path_raises(self):
        env = _TinyEnv()
        agent = ADHDP(env=env, device="cpu")
        with pytest.raises(ValueError, match="expects a folder path"):
            agent.load()

    def test_get_param_env(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADHDP(env=env, device="cpu")
        params = agent.get_param_env()
        assert "env" in params
        assert "policy" in params
        assert params["policy"]["params"]["gamma"] == 0.99

    def test_filter_kwargs_for_init(self):
        class _Dummy:
            def __init__(self, a=1, b=2):
                pass

        filtered = ADHDP._filter_kwargs_for_init(_Dummy, {"a": 10, "b": 20, "c": 30})
        assert "c" not in filtered
        assert filtered["a"] == 10
