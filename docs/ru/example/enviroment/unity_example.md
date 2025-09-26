# Пример запуска Unity среды с DQN агентом

![Unity Demo](../../model/img/example_run.jpg){ width=800 }

## О чем урок

Короткий путь: подключить Unity‑среду (через Editor или standalone‑билд), обучить DQN‑агента и проверить взаимодействие случайным агентом. Для подготовки Unity следуйте разделу «Настройка Unity среды» — см. страницу [Unity Environment](../../guide/unity_env.md).

## Цели и требования

- Что вы сделаете:
  - Подключите Unity‑среду к TensorAeroSpace (Editor или билд).
  - Запустите DQN‑тренировку и оценку.
  - Проверите взаимодействие со случайным агентом.
- Что потребуется:
  - Установленные Unity + ML‑Agents, Python 3.8+, `gym`, `gym-unity`, `tensoraerospace`.

## Импорты модели и среды

```python
from tensoraerospace.agent.dqn.model import Model, PERAgent
from tensoraerospace.envs.unity_env import get_plane_env, unity_discrete_env
```

## Подключение среды

=== "Editor"

    ```python
    # Подключение к Unity Editor (нажмите Play после запуска скрипта)
    env = unity_discrete_env()
    ```

=== "Standalone build"

    ```python
    # Укажите путь к собранной среде Unity
    build_path = "/abs/path/to/build.x86_64"   # Linux
    # build_path = "C:\\path\\to\\build.exe"  # Windows
    env = get_plane_env(build_path, server=True)
    ```

!!! note "Порт и подключение"
    По умолчанию используется порт 5004. Если он занят — закройте другие процессы или измените порт в настройках ML‑Agents/окружения.

## Запуск DQN‑тренировки и оценки

```python
num_actions = env.action_space.n
model = Model(num_actions)
target_model = Model(num_actions)

agent = PERAgent(model, target_model, env, train_nums=100)
agent.train()

# Оценка после обучения
rewards_sum = agent.evaluation(env)
print("After Training: %d out of 200" % rewards_sum)
```

## Типичные логи подключения

```text
[INFO] Listening on port 5004. Start training by pressing the Play button in the Unity Editor.
[INFO] Connected to Unity environment with package version 2.2.1-exp.1 and communication version 1.5.0
[INFO] Connected new brain: My Behavior?team=0
[WARNING] uint8_visual was set to true, but visual observations are not in use. This setting will not have any effect.
[WARNING] The environment contains multiple observations. You must define allow_multiple_obs=True to receive them all.
```

## Экран запуска взаимодействия

![Unity Interface](../../guide/img/6.png){ width=800 }

## Случайный агент: быстрое взаимодействие

=== "Plane env"

    ```python
    env = get_plane_env()
    env.reset()

    print(env.action_space)       # Количество действий в среде
    print(env.observation_space)  # Размер состояния среды

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()
    ```

=== "Discrete env"

    ```python
    env = unity_discrete_env()
    env.reset()

    print(env.action_space)

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()
    ```

## Запуск в Docker на множестве GPU/CPU

Обучение на нескольких GPU ускоряет процесс, позволяет использовать более сложные модели и параллелить сбор опыта.

```bash
FROM tensorflow/tensorflow:2.4.0-gpu-jupyter

RUN pip install gym==0.20.0 scipy==1.5.4 gym-unity==0.28.0
RUN mkdir /tf/logs
COPY a3c_example.py /tf

ENTRYPOINT tensorboard --logdir /tf/logs --port 8889 --host 0.0.0.0 & python a3c_example.py
```

### Скрипт запуска A3C в Docker

```python
from tensoraerospace.envs.unity_env import get_plane_env
from tensoraerospace.agent.a3c import Agent, setup_global_params

def env_function(worker_id):
    # /tf/linux_build/build.x86_64 — путь к собранному Unity окружению
    return get_plane_env("/tf/linux_build/build.x86_64", server=True, worker=worker_id)

actor_lr = 0.0005
critic_lr = 0.001
gamma = 0.99
hidden_size = 128
update_interval = 1
max_episodes = 100

setup_global_params(actor_lr, critic_lr, gamma, hidden_size, update_interval, max_episodes)

agent = Agent(env_function, gamma)
agent.train()
```

### Запуск контейнера

```bash
docker run \
  -v ./tensoraerospace:/tf/tensoraerospace \
  -v ./linux_build:/tf/linux_build \
  -p 8889:8889 unity_docker
```

## Трудности и решения

- Порт 5004 занят: измените порт в конфигурации или остановите конфликтующий процесс.
- Лог `allow_multiple_obs=True`: либо включите параметр в обертке среды, либо используйте первый наблюдаемый канал.
- Несоответствие версий `gym`/`gym-unity`: убедитесь, что версии совместимы с используемым ML‑Agents.
- Не подключается к билду: проверьте `build_path` и права на запуск (Linux: `chmod +x`).

## Пример запуска обучения модели

![Training Example](../../model/img/example_run.jpg){ width=800 }
