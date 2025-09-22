import numpy as np
import torch

from tensoraerospace.agent.ppo.model import PPO, Actor, Critic, ppo_iter


class _DummyEnv:
    def __init__(self):
        self.observation_space = type("S", (), {"shape": (3,)})
        self.action_space = type("A", (), {"shape": (2,)})

    def reset(self):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, _action):
        return (
            np.zeros(3, dtype=np.float32),
            0.0,
            False,
            False,
            {},
        )


def test_actor_critic_forward():
    actor = Actor(input_dim=3, out_dim=2)
    critic = Critic(input_dim=3)
    x = torch.zeros((4, 3))

    a, r = actor(x, return_reward=True, continous_actions=False)
    assert a.shape == (4, 2)
    assert r.shape == (4,)

    v = critic(x)
    assert v.shape == (4, 1)


def test_ppo_iter_shapes():
    n = 8
    states = torch.zeros((n, 3))
    actions = torch.zeros((n, 1))
    log_probs = torch.zeros((n, 1))
    returns = torch.zeros((n, 1))
    advantages = torch.zeros((n, 1))
    rewards = torch.zeros((n, 1))

    it = ppo_iter(
        epoch=1,
        mini_batch_size=4,
        states=states,
        actions=actions,
        log_probs=log_probs,
        returns=returns,
        advantages=advantages,
        rewards=rewards,
    )
    s, a, lp, _, _, _ = next(it)
    assert s.shape == (4, 3)
    assert a.shape == (4, 1)
    assert lp.shape == (4, 1)


def test_ppo_act_and_actor_loss():
    env = _DummyEnv()
    agent = PPO(env=env, max_episodes=1, rollout_len=8, num_epochs=1, batch_size=4)
    state = np.zeros(3, dtype=np.float32)
    action, _mean, _log_prob = agent.act(state)
    assert action.shape[-1] == 2

    # Actor loss with dummy tensors
    probs = torch.zeros((4, 1))
    entropy = torch.tensor(0.0)
    actions = torch.zeros((4, 1))
    adv = torch.zeros((4, 1))
    old_probs = torch.zeros((4, 1))
    loss = agent.actor_loss(probs, entropy, actions, adv, old_probs)
    assert isinstance(loss.item(), float)
