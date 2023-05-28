import os
import pytest
import torch
import numpy as np
import torch.nn as nn
from pytest import approx
from tensoraerospace.agent.sac import ReplayMemory, ValueNetwork, QNetwork


class TestReplayMemory:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.replay_memory = ReplayMemory(10, 42)  # max capacity 10, seed 42
        self.save_path = "test_replay_memory.pkl"
        yield
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

    def test_push_and_len(self):
        state = np.array([1, 2, 3])
        action = 0
        reward = 1.0
        next_state = np.array([2, 3, 4])
        done = False
        self.replay_memory.push(state, action, reward, next_state, done)
        assert len(self.replay_memory) == 1

    def test_sample(self):
        for i in range(10):
            state = np.array([i, i + 1, i + 2])
            action = i
            reward = 1.0
            next_state = np.array([i + 1, i + 2, i + 3])
            done = False
            self.replay_memory.push(state, action, reward, next_state, done)

        states, actions, rewards, next_states, dones = self.replay_memory.sample(5)
        assert states.shape == (5, 3)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)
        assert next_states.shape == (5, 3)
        assert dones.shape == (5,)

    def test_save_and_load_buffer(self):
        for i in range(10):
            state = np.array([i, i + 1, i + 2])
            action = i
            reward = 1.0
            next_state = np.array([i + 1, i + 2, i + 3])
            done = False
            self.replay_memory.push(state, action, reward, next_state, done)

        self.replay_memory.save_buffer("test_env", save_path=self.save_path)
        assert os.path.exists(self.save_path)

        loaded_replay_memory = ReplayMemory(10, 42)
        loaded_replay_memory.load_buffer(self.save_path)
        assert len(loaded_replay_memory) == len(self.replay_memory)



def test_ValueNetwork():
    num_inputs = 5
    hidden_dim = 10
    value_network = ValueNetwork(num_inputs, hidden_dim)

    # Проверяем прямой проход
    state = torch.randn(1, num_inputs)
    output = value_network(state)
    assert output.size() == (1, 1)  # Проверка размера вывода

    # Проверяем, что вывод не NaN и не бесконечность
    assert torch.all(torch.isfinite(output)).item()

    # Проверяем, что сеть обрабатывает векторы и матрицы
    state = torch.randn(2, num_inputs)
    output = value_network(state)
    assert output.size() == (2, 1)  # Проверка размера вывода

    # Проверяем, что сеть обрабатывает большие входные данные
    state = torch.randn(1000, num_inputs)
    output = value_network(state)
    assert output.size() == (1000, 1)  # Проверка размера вывода

    # Проверяем, что сеть обрабатывает данные с большим количеством входных признаков
    num_inputs_large = 100
    value_network_large = ValueNetwork(num_inputs_large, hidden_dim)
    state = torch.randn(1, num_inputs_large)
    output = value_network_large(state)
    assert output.size() == (1, 1)  # Проверка размера вывода

def test_QNetwork():
    num_inputs = 5
    num_actions = 3
    hidden_dim = 10
    q_network = QNetwork(num_inputs, num_actions, hidden_dim)

    # Проверяем прямой проход
    state = torch.randn(1, num_inputs)
    action = torch.randn(1, num_actions)
    q1, q2 = q_network(state, action)

    assert q1.size() == (1, 1)  # Проверка размера Q1
    assert q2.size() == (1, 1)  # Проверка размера Q2

    # Проверяем, что вывод не NaN и не бесконечность
    assert torch.all(torch.isfinite(q1)).item()
    assert torch.all(torch.isfinite(q2)).item()

    # Проверяем, что сеть обрабатывает векторы и матрицы
    state = torch.randn(2, num_inputs)
    action = torch.randn(2, num_actions)
    q1, q2 = q_network(state, action)

    assert q1.size() == (2, 1)  # Проверка размера Q1
    assert q2.size() == (2, 1)  # Проверка размера Q2

    # Проверяем, что сеть обрабатывает большие входные данные
    state = torch.randn(1000, num_inputs)
    action = torch.randn(1000, num_actions)
    q1, q2 = q_network(state, action)

    assert q1.size() == (1000, 1)  # Проверка размера Q1
    assert q2.size() == (1000, 1)  # Проверка размера Q2
