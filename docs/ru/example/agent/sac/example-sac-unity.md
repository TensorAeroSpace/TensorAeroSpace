# SAC с Unity ML-Agents

![Unity Demo](../../../model/img/example_run.jpg){ width=800 }

## О чём пример

Этот пример демонстрирует обучение **Soft Actor-Critic (SAC)** агента из TensorAeroSpace в Unity ML-Agents окружении для непрерывного управления.

!!! info "Репозиторий Unity окружения"
    Исходный код Unity окружения доступен по ссылке: [TensorAeroSpace/UnityAirplaneEnvironment](https://github.com/TensorAeroSpace/UnityAirplaneEnvironment)

## Цели и требования

**Что вы сделаете:**

- Подключите Unity-среду к TensorAeroSpace (Editor или билд)
- Обучите SAC-агента на непрерывных действиях
- Оцените качество обученной политики
- Сохраните и загрузите модель

**Требования:**

- Unity + ML-Agents (Python package `mlagents==1.1.0`)
- Python 3.8+
- TensorAeroSpace
- GPU рекомендуется для обучения

## Установка

```bash
pip install mlagents==1.1.0
```

## Импорты

```python
import gymnasium as gym
import numpy as np
from tensoraerospace.agent import SAC

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
```

## Обёртка для Gymnasium API

Unity ML-Agents использует старый `gym` API. Нам нужна обёртка для преобразования в современный `gymnasium` API:

```python
class UnityToGymnasiumWrapper(gym.Wrapper):
    """Обёртка для преобразования Unity ML-Agents gym в Gymnasium API.
    
    Args:
        env: Unity окружение, обёрнутое в UnityToGymWrapper
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self, *, seed=None, options=None):
        """Сброс окружения."""
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        """Выполнение шага в окружении."""
        result = self.env.step(action)
        obs, reward, done, info = result
        # Gymnasium использует 5-tuple вместо 4-tuple
        return obs, reward, done, False, info

    def close(self):
        """Закрытие Unity окружения."""
        self.env.close()
```

## Подключение к Unity

=== "Unity Editor"

    ```python
    unity_env = UnityEnvironment(
        file_name=None,  # None для подключения к Editor
        log_folder="./logs/",
        additional_args=["-logfile", "unity.log"]
    )
    ```

=== "Standalone Build"

    ```python
    # Linux
    unity_env = UnityEnvironment("/path/to/build.x86_64")
    
    # Windows
    unity_env = UnityEnvironment("C:\\path\\to\\build.exe")
    ```

```python
# Обёртка для совместимости
gym_env = UnityToGymWrapper(unity_env, uint8_visual=True)
env = UnityToGymnasiumWrapper(gym_env)

print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
```

!!! note "Порт подключения"
    По умолчанию используется порт 5004. Если он занят — закройте другие процессы Unity.

## Конфигурация SAC агента

```python
agent = SAC(
    env=env,
    # Динамика обучения
    updates_per_step=1,
    batch_size=256,
    memory_capacity=1_000_000,
    
    # Скорости обучения
    lr=3e-4,          # Критик
    policy_lr=3e-4,   # Актёр
    
    # RL гиперпараметры
    gamma=0.99,       # Коэффициент дисконтирования
    tau=0.005,        # Мягкое обновление
    alpha=0.2,        # Начальный коэффициент энтропии
    
    # Конфигурация политики
    policy_type="Gaussian",
    target_update_interval=1,
    automatic_entropy_tuning=True,
    
    # Архитектура сети
    hidden_size=256,
    
    # Устройство и логирование
    device="cuda",  # Используйте "cpu" если нет GPU
    verbose_histogram=False,
    seed=42,
)
```

### Таблица гиперпараметров

| Параметр | Значение | Описание |
|----------|----------|----------|
| `batch_size` | 256 | Размер мини-батча |
| `memory_capacity` | 1,000,000 | Размер replay buffer |
| `lr` | 3e-4 | Скорость обучения критиков |
| `policy_lr` | 3e-4 | Скорость обучения актёра |
| `gamma` | 0.99 | Коэффициент дисконтирования |
| `tau` | 0.005 | Коэффициент мягкого обновления |
| `automatic_entropy_tuning` | True | Автоподстройка исследования |
| `hidden_size` | 256 | Нейроны скрытого слоя |

## Обучение

```python
NUM_EPISODES = 1000
agent.train(num_episodes=NUM_EPISODES)
```

!!! tip "TensorBoard"
    Логи обучения сохраняются для TensorBoard:
    ```bash
    tensorboard --logdir runs/
    ```

## Оценка

```python
def evaluate_agent(agent, env, num_episodes=5):
    """Оценка обученного агента."""
    rewards = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")
    
    print(f"\nСредняя награда: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    return np.mean(rewards), np.std(rewards), rewards

# Запуск оценки
mean_reward, std_reward, all_rewards = evaluate_agent(agent, env, num_episodes=5)
```

## Сохранение и загрузка модели

```python
# Сохранение
agent.save(path="./checkpoints")

# Загрузка
loaded_agent = SAC.from_pretrained("./checkpoints/Nov24_11-01-27_SAC")
```

## Закрытие окружения

```python
env.close()
```

## Решение проблем

| Проблема | Решение |
|----------|---------|
| `Port 5004 is busy` | Закройте другие Unity процессы |
| `allow_multiple_obs` предупреждение | Установите `allow_multiple_obs=True` или игнорируйте |
| Таймаут соединения | Запустите Unity сцену перед Python скриптом |
| CUDA out of memory | Уменьшите `batch_size` или `hidden_size` |

## Типичный лог подключения

```text
[INFO] Listening on port 5004...
[INFO] Connected to Unity environment with package version X.X.X
[INFO] Connected new brain: BehaviorName?team=0
```

## Ссылки

- [SAC Paper](https://arxiv.org/abs/1801.01290)
- [Unity ML-Agents Documentation](https://unity-technologies.github.io/ml-agents/)
- [TensorAeroSpace SAC API](../../../agent/sac.md)

