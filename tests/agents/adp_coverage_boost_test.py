"""Tests to boost coverage for tensoraerospace/agent/adp/ modules.

Covers: adp.py, networks.py, replay.py
Targets: adp.py 20% -> ~60%, networks.py 50% -> ~85%, replay.py 40% -> ~85%
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from tensoraerospace.agent.adp.adp import ADP, _as_flat_np
from tensoraerospace.agent.adp.networks import (
    DeterministicActor,
    JCritic,
    JLambdaActionCritic,
    JLambdaCritic,
    LambdaCritic,
    QCritic,
    _mlp,
    _mlp_body,
    polyak_update,
)
from tensoraerospace.agent.adp.replay import ReplayBuffer, Transition

# ---------------------------------------------------------------------------
# Helpers: lightweight fake envs
# ---------------------------------------------------------------------------


class _TinySpace:
    def __init__(self, shape, low=-1.0, high=1.0):
        self.shape = tuple(shape)
        self.low = np.full(self.shape, low, dtype=np.float32)
        self.high = np.full(self.shape, high, dtype=np.float32)


class _TinyEnv:
    """Minimal env that follows Gymnasium API."""

    def __init__(self, obs_dim=4, act_dim=1, max_steps=5):
        self.observation_space = _TinySpace((obs_dim,))
        self.action_space = _TinySpace((act_dim,))
        self._step_count = 0
        self.max_steps = int(max_steps)

    def reset(self):
        self._step_count = 0
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        _ = np.asarray(action, dtype=np.float32).reshape(-1)
        self._step_count += 1
        obs = np.random.randn(self.observation_space.shape[0]).astype(np.float32) * 0.1
        reward = -0.1
        terminated = self._step_count >= self.max_steps
        truncated = False
        info = {"cost_total": 0.05}
        return obs, reward, terminated, truncated, info

    @property
    def unwrapped(self):
        return self


class _ModelBasedEnv:
    """Env that provides model matrices (filt_A, filt_B) + reference_signal for DHP designs."""

    def __init__(self, state_dim=4, act_dim=1, max_steps=5, n_steps=10):
        obs_dim = state_dim + 2  # state + theta_ref + q_ref in obs
        self.observation_space = _TinySpace((obs_dim,))
        self.action_space = _TinySpace((act_dim,))
        self._step_count = 0
        self.max_steps = max_steps
        self.initial_state = np.zeros(state_dim, dtype=np.float32)
        self.state = np.zeros(state_dim, dtype=np.float32)
        self.current_step = 0
        self.reference_signal = np.zeros((1, n_steps), dtype=np.float32)
        self.reference_signal[0, :] = 0.01  # small constant ref
        self.dt = 0.1
        self.max_stabilizer_angle_deg = 10.0
        self.max_pitch_rad = np.deg2rad(20.0)
        self.max_pitch_rate_rad_s = np.deg2rad(5.0)
        self.ref_theta_dot_clip_rad_s = self.max_pitch_rate_rad_s
        self.w_pitch = 5.0
        self.w_q = 0.2
        self.w_action = 0.01
        self.w_smooth = 0.02
        self.w_jerk = 0.0
        self.reward_scale = 1.0

        class _Model:
            filt_A = np.eye(state_dim, dtype=np.float32) * 0.99
            filt_B = np.ones((state_dim, act_dim), dtype=np.float32) * 0.01

        self.model = _Model()

    def reset(self):
        self._step_count = 0
        self.current_step = 0
        self.state = self.initial_state.copy()
        obs = np.concatenate([self.state, [0.01, 0.0]]).astype(np.float32)
        return obs, {}

    def step(self, action):
        _ = np.asarray(action, dtype=np.float32).reshape(-1)
        self._step_count += 1
        self.current_step = self._step_count
        self.state = (
            self.state * 0.99
            + np.random.randn(*self.state.shape).astype(np.float32) * 0.001
        )
        obs = np.concatenate([self.state, [0.01, 0.0]]).astype(np.float32)
        reward = -0.05
        terminated = self._step_count >= self.max_steps
        truncated = False
        info = {"cost_total": 0.05}
        return obs, reward, terminated, truncated, info

    @property
    def unwrapped(self):
        return self


# ===========================================================================
# replay.py tests
# ===========================================================================


class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=10, seed=0)
        assert len(buf) == 0
        obs = np.zeros(4, dtype=np.float32)
        act = np.zeros(1, dtype=np.float32)
        buf.push(obs, act, 1.0, obs, 0.0)
        assert len(buf) == 1

    def test_sample_basic(self):
        buf = ReplayBuffer(capacity=10, seed=0)
        for i in range(5):
            obs = np.full(4, float(i), dtype=np.float32)
            act = np.full(1, float(i), dtype=np.float32)
            buf.push(obs, act, float(i), obs, 0.0)
        o, a, r, no, d = buf.sample(3)
        assert o.shape == (3, 4)
        assert a.shape == (3, 1)
        assert r.shape == (3, 1)
        assert no.shape == (3, 4)
        assert d.shape == (3, 1)

    def test_overflow_circular(self):
        buf = ReplayBuffer(capacity=3, seed=0)
        for i in range(5):
            obs = np.full(2, float(i), dtype=np.float32)
            act = np.full(1, float(i), dtype=np.float32)
            buf.push(obs, act, float(i), obs, 0.0)
        assert len(buf) == 3
        # Oldest items (0, 1) should have been overwritten
        o, a, r, no, d = buf.sample(3)
        assert o.shape == (3, 2)

    def test_sample_invalid_batch_size(self):
        buf = ReplayBuffer(capacity=10, seed=0)
        buf.push(np.zeros(2), np.zeros(1), 0.0, np.zeros(2), 0.0)
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            buf.sample(0)
        with pytest.raises(ValueError, match="Cannot sample"):
            buf.sample(5)

    def test_invalid_capacity(self):
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            ReplayBuffer(capacity=0)

    def test_transition_dataclass(self):
        t = Transition(
            obs=np.zeros(2),
            act=np.zeros(1),
            reward=1.0,
            next_obs=np.zeros(2),
            done_bootstrap=0.0,
        )
        assert t.reward == 1.0


# ===========================================================================
# networks.py tests
# ===========================================================================


class TestNetworks:
    def test_mlp_basic(self):
        net = _mlp(4, 2, [32, 16])
        x = torch.randn(3, 4)
        out = net(x)
        assert out.shape == (3, 2)

    def test_mlp_with_out_activation(self):
        net = _mlp(4, 2, [32], out_activation=torch.nn.Tanh)
        x = torch.randn(3, 4)
        out = net(x)
        assert out.shape == (3, 2)
        # tanh output should be bounded
        assert torch.all(out >= -1.0) and torch.all(out <= 1.0)

    def test_mlp_body_empty_hidden(self):
        mod, last_dim = _mlp_body(8, [])
        assert last_dim == 8
        x = torch.randn(2, 8)
        out = mod(x)
        assert out.shape == (2, 8)

    def test_mlp_body_with_hidden(self):
        mod, last_dim = _mlp_body(8, [32, 16])
        assert last_dim == 16
        x = torch.randn(2, 8)
        out = mod(x)
        assert out.shape == (2, 16)

    def test_deterministic_actor_default_bounds(self):
        actor = DeterministicActor(obs_dim=4, act_dim=2)
        x = torch.randn(5, 4)
        out = actor(x)
        assert out.shape == (5, 2)
        # default bounds [-1, 1]
        assert torch.all(out >= -1.0 - 1e-6) and torch.all(out <= 1.0 + 1e-6)

    def test_deterministic_actor_custom_bounds(self):
        low = np.array([-2.0, -3.0], dtype=np.float32)
        high = np.array([2.0, 3.0], dtype=np.float32)
        actor = DeterministicActor(
            obs_dim=4, act_dim=2, action_low=low, action_high=high
        )
        x = torch.randn(5, 4)
        out = actor(x)
        assert out.shape == (5, 2)
        assert torch.all(out >= -3.0 - 1e-3) and torch.all(out <= 3.0 + 1e-3)

    def test_deterministic_actor_invalid_bounds(self):
        with pytest.raises(ValueError, match="action_low/high must have shape"):
            DeterministicActor(
                obs_dim=4,
                act_dim=2,
                action_low=np.array([-1.0]),
                action_high=np.array([1.0]),
            )

    def test_qcritic_forward(self):
        critic = QCritic(obs_dim=4, act_dim=2, hidden_sizes=(32,))
        obs = torch.randn(3, 4)
        act = torch.randn(3, 2)
        q = critic(obs, act)
        assert q.shape == (3, 1)

    def test_jcritic_forward(self):
        critic = JCritic(input_dim=6, hidden_sizes=(32,))
        x = torch.randn(3, 6)
        j = critic(x)
        assert j.shape == (3, 1)

    def test_lambda_critic_forward(self):
        critic = LambdaCritic(state_dim=4, hidden_sizes=(32,))
        x = torch.randn(3, 4)
        lam = critic(x)
        assert lam.shape == (3, 4)

    def test_lambda_critic_different_input_dim(self):
        critic = LambdaCritic(state_dim=4, input_dim=6, hidden_sizes=(32,))
        x = torch.randn(3, 6)
        lam = critic(x)
        assert lam.shape == (3, 4)

    def test_jlambda_critic_forward(self):
        critic = JLambdaCritic(input_dim=6, r_dim=6, hidden_sizes=(32,))
        x = torch.randn(3, 6)
        j, lam = critic(x)
        assert j.shape == (3, 1)
        assert lam.shape == (3, 6)

    def test_jlambda_action_critic_forward(self):
        critic = JLambdaActionCritic(r_dim=6, a_dim=1, hidden_sizes=(32,))
        r = torch.randn(3, 6)
        a = torch.randn(3, 1)
        j, jr, ja = critic(r, a)
        assert j.shape == (3, 1)
        assert jr.shape == (3, 6)
        assert ja.shape == (3, 1)

    def test_polyak_update_all_params(self):
        net_a = torch.nn.Linear(4, 2)
        net_b = torch.nn.Linear(4, 2)
        # Set net_b weights to all ones
        with torch.no_grad():
            net_b.weight.fill_(1.0)
            net_b.bias.fill_(1.0)
        old_a_weight = net_a.weight.data.clone()
        polyak_update(net_a, net_b, tau=0.5)
        # net_a should be 0.5*old + 0.5*1.0
        expected = 0.5 * old_a_weight + 0.5 * torch.ones_like(old_a_weight)
        assert torch.allclose(net_a.weight.data, expected, atol=1e-6)

    def test_polyak_update_specific_params(self):
        net_a = torch.nn.Linear(4, 2)
        net_b = torch.nn.Linear(4, 2)
        with torch.no_grad():
            net_b.weight.fill_(1.0)
            net_b.bias.fill_(1.0)
        old_bias = net_a.bias.data.clone()
        old_weight = net_a.weight.data.clone()
        polyak_update(net_a, net_b, tau=0.5, params=["bias"])
        # weight should be untouched
        assert torch.allclose(net_a.weight.data, old_weight, atol=1e-6)
        # bias should be averaged
        expected_bias = 0.5 * old_bias + 0.5 * torch.ones_like(old_bias)
        assert torch.allclose(net_a.bias.data, expected_bias, atol=1e-6)

    def test_polyak_update_invalid_tau(self):
        net_a = torch.nn.Linear(2, 2)
        net_b = torch.nn.Linear(2, 2)
        with pytest.raises(ValueError, match="tau must be in"):
            polyak_update(net_a, net_b, tau=0.0)
        with pytest.raises(ValueError, match="tau must be in"):
            polyak_update(net_a, net_b, tau=1.5)

    def test_polyak_update_missing_param_name_skipped(self):
        net_a = torch.nn.Linear(4, 2)
        net_b = torch.nn.Linear(4, 2)
        # Should not raise even if param name does not exist
        polyak_update(net_a, net_b, tau=0.5, params=["nonexistent_param"])


# ===========================================================================
# adp.py tests: _as_flat_np
# ===========================================================================


class TestAsFlat:
    def test_list_input(self):
        r = _as_flat_np([1.0, 2.0, 3.0])
        assert r.shape == (3,)
        assert r.dtype == np.float32

    def test_2d_input(self):
        r = _as_flat_np(np.array([[1, 2], [3, 4]]))
        assert r.shape == (4,)


# ===========================================================================
# adp.py tests: ADP construction
# ===========================================================================


class TestADPConstruction:
    def test_default_adhdp(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu")
        assert agent.design == "adhdp"
        assert hasattr(agent, "actor")
        assert hasattr(agent, "critic")

    def test_ddpg_design(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu", design="ddpg")
        assert agent.design == "ddpg"
        assert agent.memory is None  # no replay by default

    def test_ddpg_with_replay_and_targets(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(
            env=env,
            device="cpu",
            design="ddpg",
            use_replay=True,
            memory_capacity=100,
            use_target_networks=True,
            tau=0.01,
        )
        assert agent.memory is not None
        assert agent.actor_target is not None
        assert agent.critic_target is not None

    def test_adhdp_rejects_replay(self):
        env = _TinyEnv()
        with pytest.raises(
            ValueError, match="Canonical design='adhdp' does not support replay"
        ):
            ADP(env=env, device="cpu", design="adhdp", use_replay=True)

    def test_adhdp_rejects_baseline(self):
        env = _TinyEnv()
        with pytest.raises(
            ValueError, match="Canonical design='adhdp' does not support baseline"
        ):
            ADP(env=env, device="cpu", design="adhdp", adhdp_use_baseline=True)

    def test_invalid_design(self):
        env = _TinyEnv()
        with pytest.raises(ValueError, match="design must be one of"):
            ADP(env=env, device="cpu", design="invalid_design")

    def test_invalid_batch_size(self):
        env = _TinyEnv()
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            ADP(env=env, device="cpu", batch_size=0)

    def test_invalid_gamma(self):
        env = _TinyEnv()
        with pytest.raises(ValueError, match="gamma must be"):
            ADP(env=env, device="cpu", gamma=-0.5)

    def test_invalid_exploration_std(self):
        env = _TinyEnv()
        with pytest.raises(ValueError, match="exploration_std must be >= 0"):
            ADP(env=env, device="cpu", exploration_std=-1.0)

    def test_invalid_tau_with_targets(self):
        env = _TinyEnv()
        with pytest.raises(ValueError, match="tau must be in"):
            ADP(env=env, device="cpu", design="ddpg", use_target_networks=True, tau=0.0)

    def test_invalid_updates_per_step(self):
        env = _TinyEnv()
        with pytest.raises(ValueError, match="updates_per_step must be >= 1"):
            ADP(env=env, device="cpu", updates_per_step=0)

    def test_invalid_log_every(self):
        env = _TinyEnv()
        with pytest.raises(ValueError, match="log_every_updates must be >= 1"):
            ADP(env=env, device="cpu", log_every_updates=0)

    def test_invalid_env_dims(self):
        class _BadEnv:
            observation_space = _TinySpace((0,))
            action_space = _TinySpace((1,))
            unwrapped = property(lambda self: self)

        with pytest.raises(ValueError, match="ADP expects Box-like spaces"):
            ADP(env=_BadEnv(), device="cpu")


# ===========================================================================
# adp.py tests: select_action / predict
# ===========================================================================


class TestADPSelectAction:
    def test_select_action_evaluate(self):
        env = _TinyEnv(obs_dim=4, act_dim=2, max_steps=3)
        agent = ADP(env=env, device="cpu", exploration_std=0.1)
        obs, _ = env.reset()
        act = agent.select_action(obs, evaluate=True)
        assert act.shape == (2,)
        assert np.all(act >= -1.0) and np.all(act <= 1.0)

    def test_select_action_with_exploration(self):
        env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=3)
        agent = ADP(env=env, device="cpu", exploration_std=0.5)
        obs, _ = env.reset()
        # Collect many actions to check that exploration adds variance
        acts = [agent.select_action(obs, evaluate=False) for _ in range(50)]
        acts_np = np.stack(acts)
        # Should have some variance due to exploration
        assert np.std(acts_np) > 0.0

    def test_predict_alias(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu", exploration_std=0.0)
        obs, _ = env.reset()
        act1 = agent.select_action(obs, evaluate=True)
        act2 = agent.predict(obs, deterministic=True)
        np.testing.assert_allclose(act1, act2)

    def test_get_env(self):
        env = _TinyEnv()
        agent = ADP(env=env, device="cpu")
        assert agent.get_env() is env


# ===========================================================================
# adp.py tests: train (online ADHDP)
# ===========================================================================


class TestADPTrainOnline:
    def test_train_adhdp_online_short(self):
        env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=5)
        agent = ADP(
            env=env,
            device="cpu",
            design="adhdp",
            exploration_std=0.0,
            log_every_updates=2,
        )
        agent.train(num_episodes=2, max_steps=5)
        assert agent._updates > 0

    def test_train_ddpg_online(self):
        env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=5)
        agent = ADP(
            env=env,
            device="cpu",
            design="ddpg",
            exploration_std=0.0,
            log_every_updates=2,
        )
        agent.train(num_episodes=2, max_steps=5)
        assert agent._updates > 0

    def test_train_ddpg_with_replay_and_targets(self):
        env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=8)
        agent = ADP(
            env=env,
            device="cpu",
            design="ddpg",
            use_replay=True,
            memory_capacity=200,
            batch_size=4,
            use_target_networks=True,
            tau=0.1,
            exploration_std=0.1,
            log_every_updates=2,
        )
        agent.train(num_episodes=3, max_steps=8)
        assert agent._updates > 0
        assert len(agent.memory) > 0

    def test_train_adhdp_with_env_cost(self):
        env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=5)
        agent = ADP(
            env=env,
            device="cpu",
            design="adhdp",
            adhdp_use_env_cost=True,
            log_every_updates=2,
        )
        agent.train(num_episodes=2, max_steps=5)
        assert agent._updates > 0

    def test_train_adhdp_warmup_episodes(self):
        env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=5)
        agent = ADP(
            env=env,
            device="cpu",
            design="adhdp",
            adhdp_critic_warmup_episodes=1,
            log_every_updates=2,
        )
        agent.train(num_episodes=2, max_steps=5)
        assert agent._updates > 0

    def test_train_adhdp_alternating_cycles(self):
        env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=5)
        agent = ADP(
            env=env,
            device="cpu",
            design="adhdp",
            adhdp_critic_cycle_episodes=1,
            adhdp_action_cycle_episodes=1,
            log_every_updates=2,
        )
        agent.train(num_episodes=4, max_steps=5)
        assert agent._updates > 0

    def test_train_adhdp_bc_decay(self):
        env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=5)
        agent = ADP(
            env=env,
            device="cpu",
            design="adhdp",
            adhdp_actor_bc_l2=0.1,
            adhdp_actor_bc_decay=0.5,
            log_every_updates=2,
        )
        agent.train(num_episodes=3, max_steps=5)
        # Coefficient should have decayed
        assert agent._adhdp_actor_bc_coef < 0.1

    def test_train_adhdp_warmstart_from_pd(self):
        env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=5)
        agent = ADP(
            env=env,
            device="cpu",
            design="adhdp",
            adhdp_warmstart_actor_episodes=1,
            adhdp_warmstart_actor_epochs=1,
            log_every_updates=2,
        )
        agent.train(num_episodes=1, max_steps=5)
        assert agent._updates > 0


# ===========================================================================
# adp.py tests: _td_update_batch
# ===========================================================================


class TestTdUpdateBatch:
    def test_td_update_critic_only(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu", design="adhdp")
        obs = np.random.randn(2, 4).astype(np.float32)
        act = np.random.randn(2, 1).astype(np.float32)
        rew = np.array([[0.1], [-0.1]], dtype=np.float32)
        next_obs = np.random.randn(2, 4).astype(np.float32)
        done = np.array([[0.0], [1.0]], dtype=np.float32)

        c_loss, a_loss = agent._td_update_batch(
            obs,
            act,
            rew,
            next_obs,
            done,
            do_critic_update=True,
            do_actor_update=False,
        )
        assert isinstance(c_loss, float)
        assert a_loss == 0.0

    def test_td_update_actor_only(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu", design="adhdp")
        obs = np.random.randn(2, 4).astype(np.float32)
        act = np.random.randn(2, 1).astype(np.float32)
        rew = np.array([[0.1], [-0.1]], dtype=np.float32)
        next_obs = np.random.randn(2, 4).astype(np.float32)
        done = np.array([[0.0], [1.0]], dtype=np.float32)

        c_loss, a_loss = agent._td_update_batch(
            obs,
            act,
            rew,
            next_obs,
            done,
            do_critic_update=False,
            do_actor_update=True,
        )
        assert c_loss == 0.0
        assert isinstance(a_loss, float)

    def test_td_update_both(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu", design="adhdp")
        obs = np.random.randn(2, 4).astype(np.float32)
        act = np.random.randn(2, 1).astype(np.float32)
        rew = np.array([[0.1], [-0.1]], dtype=np.float32)
        next_obs = np.random.randn(2, 4).astype(np.float32)
        done = np.array([[0.0], [1.0]], dtype=np.float32)

        c_loss, a_loss = agent._td_update_batch(
            obs,
            act,
            rew,
            next_obs,
            done,
            do_critic_update=True,
            do_actor_update=True,
        )
        assert isinstance(c_loss, float) and c_loss > 0.0 or c_loss == 0.0
        assert isinstance(a_loss, float)

    def test_td_update_with_target_networks(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(
            env=env,
            device="cpu",
            design="ddpg",
            use_target_networks=True,
            tau=0.1,
        )
        obs = np.random.randn(2, 4).astype(np.float32)
        act = np.random.randn(2, 1).astype(np.float32)
        rew = np.array([[0.1], [-0.1]], dtype=np.float32)
        next_obs = np.random.randn(2, 4).astype(np.float32)
        done = np.array([[0.0], [1.0]], dtype=np.float32)

        c_loss, a_loss = agent._td_update_batch(
            obs,
            act,
            rew,
            next_obs,
            done,
            do_critic_update=True,
            do_actor_update=True,
        )
        assert isinstance(c_loss, float)
        assert isinstance(a_loss, float)

    def test_td_update_with_bc_regularizer(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(
            env=env,
            device="cpu",
            design="adhdp",
            adhdp_actor_bc_l2=0.5,
        )
        obs = np.random.randn(2, 4).astype(np.float32)
        act = np.random.randn(2, 1).astype(np.float32)
        rew = np.array([[0.1], [-0.1]], dtype=np.float32)
        next_obs = np.random.randn(2, 4).astype(np.float32)
        done = np.array([[0.0], [0.0]], dtype=np.float32)

        c_loss, a_loss = agent._td_update_batch(
            obs,
            act,
            rew,
            next_obs,
            done,
            do_critic_update=True,
            do_actor_update=True,
        )
        assert isinstance(a_loss, float)


# ===========================================================================
# adp.py tests: ADHDP baseline action
# ===========================================================================


class TestADHDPBaseline:
    def test_adhdp_baseline_pd(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu", design="adhdp")
        obs_t = torch.randn(1, 4)
        u = agent._adhdp_baseline_action(obs_t)
        assert u.shape == (1, 1)

    def test_adhdp_baseline_pid(self):
        env = _TinyEnv(obs_dim=6, act_dim=1)
        agent = ADP(env=env, device="cpu", design="adhdp")
        agent._dhp_baseline_type = "pid"
        agent._dhp_baseline_kp = 0.5
        agent._dhp_baseline_ki = 0.1
        agent._dhp_baseline_kd = 0.2
        agent._adhdp_pid_i = 0.0
        agent._adhdp_pid_i_clip = 1.0
        # Obs with 6 dims to exercise q_ref_n path
        obs_t = torch.randn(1, 6)
        u = agent._adhdp_baseline_action(obs_t)
        assert u.shape == (1, 1)

    def test_adhdp_baseline_batch_gt1(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu", design="adhdp")
        obs_t = torch.randn(5, 4)
        u = agent._adhdp_baseline_action(obs_t)
        assert u.shape == (5, 1)

    def test_adhdp_baseline_invalid_shape(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu", design="adhdp")
        with pytest.raises(ValueError, match="Expected obs shape"):
            agent._adhdp_baseline_action(torch.randn(4))

    def test_adhdp_baseline_too_small(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu", design="adhdp")
        with pytest.raises(ValueError, match="at least 2 dims"):
            agent._adhdp_baseline_action(torch.randn(1, 1))


# ===========================================================================
# adp.py tests: ADHDP policy action
# ===========================================================================


class TestADHDPPolicyAction:
    def test_adhdp_policy_action(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu", design="adhdp")
        obs_t = torch.randn(1, 4)
        u = agent._adhdp_policy_action(obs_t)
        assert u.shape == (1, 1)


# ===========================================================================
# adp.py tests: reset
# ===========================================================================


class TestADPReset:
    def test_reset_basic(self):
        env = _TinyEnv()
        agent = ADP(env=env, device="cpu", design="adhdp")
        agent.reset()  # should not raise

    def test_reset_model_based(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(env=env, device="cpu", design="hdp")
        agent.reset()  # should reset prev_u_norm arrays


# ===========================================================================
# adp.py tests: model-based designs (HDP, DHP, GDHP, ADDHP, ADGDHP)
# ===========================================================================


class TestADPModelBased:
    def test_hdp_construction(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(env=env, device="cpu", design="hdp")
        assert agent.design == "hdp"
        assert isinstance(agent.critic, JCritic)

    def test_dhp_construction(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(env=env, device="cpu", design="dhp")
        assert agent.design == "dhp"
        assert isinstance(agent.critic, LambdaCritic)

    def test_gdhp_construction(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(env=env, device="cpu", design="gdhp")
        assert agent.design == "gdhp"
        assert isinstance(agent.critic, JLambdaCritic)

    def test_addhp_construction(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(env=env, device="cpu", design="addhp")
        assert agent.design == "addhp"
        assert isinstance(agent.critic, JLambdaActionCritic)

    def test_adgdhp_construction(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(env=env, device="cpu", design="adgdhp")
        assert agent.design == "adgdhp"
        assert isinstance(agent.critic, JLambdaActionCritic)

    def test_model_based_rejects_replay(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        with pytest.raises(ValueError, match="only supported for design='ddpg'"):
            ADP(env=env, device="cpu", design="hdp", use_replay=True)

    def test_hdp_select_action(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(env=env, device="cpu", design="hdp", exploration_std=0.0)
        obs, _ = env.reset()
        act = agent.select_action(obs, evaluate=True)
        assert act.shape == (1,)

    def test_hdp_select_action_with_exploration(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(env=env, device="cpu", design="hdp", exploration_std=0.3)
        obs, _ = env.reset()
        act = agent.select_action(obs, evaluate=False)
        assert act.shape == (1,)

    def test_dhp_train_short(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(
            env=env,
            device="cpu",
            design="dhp",
            exploration_std=0.0,
            log_every_updates=2,
        )
        agent.train(num_episodes=1, max_steps=3)
        assert agent._updates > 0

    def test_hdp_train_short(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(
            env=env,
            device="cpu",
            design="hdp",
            exploration_std=0.0,
            log_every_updates=2,
        )
        agent.train(num_episodes=1, max_steps=3)
        assert agent._updates > 0

    def test_gdhp_train_short(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(
            env=env,
            device="cpu",
            design="gdhp",
            exploration_std=0.0,
            log_every_updates=2,
        )
        agent.train(num_episodes=1, max_steps=3)
        assert agent._updates > 0

    def test_addhp_train_short(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(
            env=env,
            device="cpu",
            design="addhp",
            exploration_std=0.0,
            log_every_updates=2,
        )
        agent.train(num_episodes=1, max_steps=3)
        assert agent._updates > 0

    def test_adgdhp_train_short(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(
            env=env,
            device="cpu",
            design="adgdhp",
            exploration_std=0.0,
            log_every_updates=2,
        )
        agent.train(num_episodes=1, max_steps=3)
        assert agent._updates > 0

    def test_dhp_train_with_alternating_cycles(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(
            env=env,
            device="cpu",
            design="dhp",
            dhp_critic_cycle_episodes=1,
            dhp_action_cycle_episodes=1,
            exploration_std=0.0,
            log_every_updates=2,
        )
        agent.train(num_episodes=4, max_steps=3)
        assert agent._updates > 0

    def test_dhp_select_action_no_baseline(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(env=env, device="cpu", design="dhp", dhp_use_baseline=False)
        obs, _ = env.reset()
        act = agent.select_action(obs, evaluate=True)
        assert act.shape == (1,)

    def test_dhp_select_action_with_pd_baseline(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(
            env=env,
            device="cpu",
            design="dhp",
            dhp_use_baseline=True,
            dhp_baseline_type="pd",
        )
        obs, _ = env.reset()
        act = agent.select_action(obs, evaluate=True)
        assert act.shape == (1,)


# ===========================================================================
# adp.py tests: save / load
# ===========================================================================


class TestADPSaveLoad:
    def test_save_basic(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu", design="adhdp")
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = agent.save(path=tmpdir)
            assert os.path.isdir(run_dir)
            assert os.path.isfile(os.path.join(run_dir, "config.json"))
            assert os.path.isfile(os.path.join(run_dir, "actor.pth"))
            assert os.path.isfile(os.path.join(run_dir, "critic.pth"))

    def test_save_with_gradients(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu", design="adhdp")
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = agent.save(path=tmpdir, save_gradients=True)
            assert os.path.isfile(os.path.join(run_dir, "actor_optim.pth"))
            assert os.path.isfile(os.path.join(run_dir, "critic_optim.pth"))

    def test_save_ddpg_with_targets(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(
            env=env,
            device="cpu",
            design="ddpg",
            use_target_networks=True,
            tau=0.1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = agent.save(path=tmpdir)
            assert os.path.isfile(os.path.join(run_dir, "actor_target.pth"))
            assert os.path.isfile(os.path.join(run_dir, "critic_target.pth"))

    def test_save_default_path(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu")
        # Save to cwd (within tmpdir)
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = agent.save(path=tmpdir)
            assert os.path.isdir(run_dir)

    def test_get_param_env_adhdp(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu", design="adhdp")
        params = agent.get_param_env()
        assert "env" in params
        assert "policy" in params
        assert params["policy"]["params"]["design"] == "adhdp"

    def test_get_param_env_model_based(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(env=env, device="cpu", design="hdp")
        params = agent.get_param_env()
        assert "dhp_w_theta" in params["policy"]["params"]

    def test_filter_kwargs_for_init(self):
        class _Dummy:
            def __init__(self, a=1, b=2):
                pass

        filtered = ADP._filter_kwargs_for_init(_Dummy, {"a": 10, "b": 20, "c": 30})
        assert "c" not in filtered
        assert filtered["a"] == 10

    def test_filter_kwargs_for_init_var_keyword(self):
        class _DummyVarKW:
            def __init__(self, a=1, **kwargs):
                pass

        filtered = ADP._filter_kwargs_for_init(_DummyVarKW, {"a": 10, "extra": 99})
        # With **kwargs, all keys should pass through
        assert "extra" in filtered


# ===========================================================================
# adp.py tests: DHP baseline helpers
# ===========================================================================


class TestDHPBaseline:
    def test_dhp_baseline_u_norm_pd(self):
        env = _ModelBasedEnv(max_steps=3, n_steps=10)
        agent = ADP(env=env, device="cpu", design="hdp", dhp_use_baseline=False)
        x = np.zeros(4, dtype=np.float32)
        u = agent._dhp_baseline_u_norm(x=x, theta_ref=0.01, q_ref=0.0)
        assert isinstance(u, float)
        assert -1.0 <= u <= 1.0


# ===========================================================================
# adp.py: from_pretrained / local folder loading
# ===========================================================================


class TestADPFromPretrained:
    def test_from_pretrained_local(self):
        env = _TinyEnv(obs_dim=4, act_dim=1)
        agent = ADP(env=env, device="cpu", design="ddpg")
        agent.train(num_episodes=1, max_steps=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = agent.save(path=tmpdir)
            loaded = ADP.from_pretrained(run_dir)
            assert loaded.design == "ddpg"
            obs, _ = env.reset()
            act = loaded.select_action(obs, evaluate=True)
            assert act.shape == (1,)

    def test_from_pretrained_nonexistent_path(self):
        with pytest.raises(FileNotFoundError):
            ADP.from_pretrained("/tmp/nonexistent_adp_dir_12345")
