# A3C (Asynchronous Advantage Actor‑Critic)

A3C сочетает преимущества policy‑based и value‑based подходов: несколько асинхронных рабочих агентов параллельно исследуют среду и обновляют общую (глобальную) сеть, используя функцию преимущества. Данная PyTorch реализация использует многопроцессность с общей глобальной сетью и оптимизатором SharedAdam.

![A3C схема](../agent/img/a3c/a3c.png){ width=800 }

## Компоненты

- **Глобальная сеть**: Общие параметры Actor (политики) и Critic (оценки V) в едином модуле `Net`
- **Воркеры**: Независимые процессы, каждый со своей средой и локальной копией сети
- **SharedAdam**: Оптимизатор с общим состоянием между процессами для согласованного обновления параметров
- **Advantage**: TD-ошибка используется для взвешивания градиентов политики и обновления функции ценности

## Теория (на базе реализации)

### Архитектура сети

Модуль `Net` объединяет Actor и Critic:

**Ветвь Actor:**
- Вход → Linear(s_dim, 256) → ReLU6
- → mu: Linear(256, a_dim) → Tanh → масштабирование *2 (диапазон действий: [-2, 2])
- → sigma: Linear(256, a_dim) → Softplus + 0.001 (для численной стабильности)

**Ветвь Critic:**
- Вход → Linear(s_dim, 256) → ReLU6
- → value: Linear(256, 1)

### Политика (Actor) — Гауссовское распределение

Актор выводит среднее \(\mu(s)\) и стандартное отклонение \(\sigma(s)\). Действия семплируются из:

$$
a \sim \mathcal{N}\big(\mu(s),\ \sigma^2(s)\big)
$$

Для многомерных действий базовое нормальное распределение оборачивается в `Independent` распределение.

Лог-вероятность:

$$
\log \pi_\theta(a|s) = -\tfrac{1}{2}\,\frac{(a-\mu)^2}{\sigma^2} - \tfrac{1}{2}\,\log(2\pi\sigma^2)
$$

### Функция ценности (Critic)

Критик оценивает ценность состояния \(V_\phi(s)\). Temporal difference ошибка:

$$
\text{TD} = R_t^{(n)} - V_\phi(s_t)
$$

Потеря ценности (среднеквадратичная ошибка):

$$
\mathcal{L}_\text{value} = \mathbb{E}[\text{TD}^2]
$$

### N-шаговые возвраты с бутстрэпом

Реализация использует правильные n-шаговые возвраты с бутстрэпом:

$$
R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n})
$$

Если эпизод завершается, \(V_\phi(s_{t+n}) = 0\).

### Функция потерь

**Потеря политики** (с регуляризацией энтропии):

$$
\mathcal{L}_\text{policy} = -\mathbb{E}\big[\log \pi_\theta(a_t|s_t) \cdot \text{TD} + 0.005 \cdot H[\pi]\big]
$$

где \(H[\pi]\) — энтропия политики.

**Общая потеря**:

$$
\mathcal{L}_\text{total} = \mathbb{E}[\mathcal{L}_\text{policy} + \mathcal{L}_\text{value}]
$$

Advantage (TD-ошибка) отсоединяется при вычислении потери политики для предотвращения обратного распространения через функцию ценности.

### Асинхронность и синхронизация

Реализация использует `torch.multiprocessing` для параллельного обучения:

1. **Вычисление градиентов**: Каждый воркер вычисляет градиенты на своей локальной сети
2. **Отправка градиентов**: Локальные градиенты передаются в параметры глобальной сети (`gp._grad = lp.grad`)
3. **Клиппинг градиентов**: Глобальные градиенты обрезаются (max_norm=40.0) для стабильности
4. **Шаг оптимизатора**: SharedAdam обновляет параметры глобальной сети
5. **Получение параметров**: Локальная сеть загружает обновленные глобальные параметры (`load_state_dict`)

Эта процедура push-and-pull выполняется каждые `update_global_iter` шагов или при завершении эпизода.

### Гиперпараметры

- `lr`: Скорость обучения для SharedAdam (по умолчанию: 1e-4)
- `gamma`: Коэффициент дисконтирования (по умолчанию: 0.99)
- `n_workers`: Количество параллельных воркеров (по умолчанию: количество CPU)
- `max_episodes`: Общее количество эпизодов (по умолчанию: 10)
- `max_ep_step`: Максимум шагов в эпизоде (по умолчанию: 200)
- `update_global_iter`: Частота глобальных обновлений (по умолчанию: 10)
- Коэффициент энтропии: 0.005 (жестко задан в функции потерь)
- Размер скрытого слоя: 256 (жестко задан в архитектуре Net)

## Алгоритм обучения (псевдокод)

```text
# Глобальная настройка
global_net = Net(s_dim, a_dim).share_memory()
optimizer = SharedAdam(global_net.parameters(), lr)

# Каждый воркер выполняется параллельно:
def worker_process(worker_id):
    local_net = Net(s_dim, a_dim)
    local_net.load_state_dict(global_net.state_dict())  # Начальная синхронизация
    env = env_function(worker_id)
    
    while global_episodes < max_episodes:
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        episode_reward = 0
        
        for t in range(max_ep_step):
            # Выбор действия
            a = local_net.choose_action(s)
            s', r, done = env.step(clip(a, action_space))
            
            # Сохранение перехода
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)
            episode_reward += r
            
            # Условие обновления
            if t % update_global_iter == 0 or done:
                # Вычисление n-шаговых возвратов с бутстрэпом
                if done:
                    v_s_ = 0
                else:
                    v_s_ = local_net.forward(s')[2]  # оценка ценности
                
                # Обратное накопление
                returns = []
                for r in reversed(buffer_r):
                    v_s_ = r + gamma * v_s_
                    returns.insert(0, v_s_)
                
                # Вычисление потерь
                loss = local_net.loss_func(buffer_s, buffer_a, returns)
                
                # Отправка градиентов в глобальную сеть, получение обновленных параметров
                optimizer.zero_grad()
                loss.backward()
                transfer_gradients(local_net, global_net)
                clip_grad_norm(global_net.parameters(), max_norm=40.0)
                optimizer.step()
                local_net.load_state_dict(global_net.state_dict())
                
                # Очистка буферов
                buffer_s, buffer_a, buffer_r = [], [], []
                
                if done:
                    record_episode(episode_reward)
                    break
            
            s = s'
```

## Быстрый старт

Полный пример обучения A3C на окружении B747 для отслеживания синусоидального угла тангажа:

```python
import numpy as np
import torch
from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standard import sinusoid_vertical_shift
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period
from tensoraerospace.agent.a3c import Agent, setup_global_params

# Установка random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Создание временной базы и опорного сигнала
dt = 0.1
_tp = generate_time_period(tn=20, dt=dt)
tps = convert_tp_to_sec_tp(_tp, dt=dt)
number_time_steps = len(_tp)

reference_signals = np.reshape(
    sinusoid_vertical_shift(
        tp=np.asarray(tps),
        frequency=0.05,
        amplitude=np.deg2rad(1.0),
        vertical_shift=0.0,
    ),
    [1, -1],
)

# Начальное состояние: [u, w, q, theta]
init_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Настройка гиперпараметров
setup_global_params(
    max_episodes=3000,
    max_ep_step=number_time_steps,
    gamma=0.99,
    update_global_iter=10,
    lr=1e-4,
)

# Функция-фабрика окружения
def make_env(worker_id: int):
    return ImprovedB747Env(
        initial_state=init_state,
        reference_signal=reference_signals,
        number_time_steps=number_time_steps,
        dt=dt,
        initial_elevator_deg=0.0,
    )

# Создание и обучение агента
agent = Agent(
    env_function=make_env,
    gamma=0.99,
    n_workers=4,
    lr=1e-4,
    max_episodes=3000,
    max_ep_step=number_time_steps,
    update_global_iter=10,
    render=False,
    run_in_main=True,  # Установите False для многопроцессности
    log_dir="runs/a3c_b747",
)

# Обучение
agent.train()

# Оценка
eval_env = make_env(0)
obs, _ = eval_env.reset()
agent.gnet.eval()
episode_reward = 0.0

with torch.no_grad():
    terminated = truncated = False
    while not (terminated or truncated):
        obs_tensor = torch.from_numpy(np.array(obs).reshape(1, -1).astype(np.float32))
        mu, _, _ = agent.gnet.forward(obs_tensor)
        action = mu.cpu().numpy().reshape(-1)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        episode_reward += reward

print(f"Оценка награды: {episode_reward:.4f}")
eval_env.close()
agent.close()
```

### Мониторинг с помощью TensorBoard

```bash
tensorboard --logdir=runs/a3c_b747
```

Метрики включают:
- **Loss/w*/total**: Общая потеря для каждого воркера
- **Loss/w*/value**: Потеря ценности (TD ошибка)
- **Loss/w*/policy**: Потеря политики
- **Loss/w*/entropy**: Энтропия политики
- **Performance/w*/episode_reward**: Награды за эпизод
- **Performance/w*/moving_avg_reward**: Скользящее среднее

!!! tip "Лучшие практики"
    - Используйте `run_in_main=True` для notebook/отладки
    - Установите `run_in_main=False` и `n_workers=8` для продакшн обучения
    - Действия автоматически обрезаются по `env.action_space.low/high`
    - Sigma имеет минимальное значение 0.001 для численной стабильности
    - Следите в TensorBoard за коллапсом энтропии или расхождением потери ценности

---

## Продвинутый пример: Обучение на окружении B747

Полный пример демонстрирует обучение агента A3C на окружении `ImprovedB747Env` для отслеживания синусоидального опорного сигнала угла тангажа.

### Настройка и создание окружения

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from queue import Empty

from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standard import sinusoid_vertical_shift
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period
from tensoraerospace.agent.a3c import Agent, setup_global_params

# Установка random seed для воспроизводимости
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Создание временной базы
dt = 0.1  # секунды
_tp = generate_time_period(tn=20, dt=dt)
tps = convert_tp_to_sec_tp(_tp, dt=dt)
number_time_steps = len(_tp)

print(f"Длина эпизода: {number_time_steps} шагов ({number_time_steps * dt:.1f} секунд)")

# Генерация синусоидального опорного сигнала для угла тангажа (theta)
reference_signals = np.reshape(
    sinusoid_vertical_shift(
        tp=np.asarray(tps),
        frequency=0.05,             # Гц
        amplitude=np.deg2rad(1.0),  # амплитуда 1 градус
        vertical_shift=0.0,
    ),
    [1, -1],
)

# Определение начального состояния: [u, w, q, theta]
init_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Создание окружения
env = ImprovedB747Env(
    initial_state=init_state,
    reference_signal=reference_signals,
    number_time_steps=number_time_steps,
    dt=dt,
    initial_elevator_deg=0.0,
)

print(f"Пространство наблюдений: {env.observation_space}")
print(f"Пространство действий: {env.action_space}")
```

### Настройка и создание агента

```python
# Настройка гиперпараметров
setup_global_params(
    max_episodes=3000,
    max_ep_step=number_time_steps,
    gamma=0.99,
    update_global_iter=10,
    lr=1e-4,
)

# Функция-фабрика окружения
def make_env(worker_id: int):
    """Создает окружение для каждого воркера."""
    return ImprovedB747Env(
        initial_state=init_state,
        reference_signal=reference_signals,
        number_time_steps=number_time_steps,
        dt=dt,
        initial_elevator_deg=0.0,
    )

# Создание агента A3C
agent = Agent(
    env_function=make_env,
    gamma=0.99,
    n_workers=4,              # Используем 4 параллельных воркера
    lr=1e-4,
    max_episodes=3000,
    max_ep_step=number_time_steps,
    update_global_iter=10,
    render=False,
    run_in_main=True,         # Установите False для настоящей многопроцессности
    log_dir="runs/a3c_b747",
)

print("Агент A3C успешно создан!")
```

### Обучение агента

```python
import time

print("Запуск обучения A3C...\n")

episode_rewards = []
start_time = time.time()

# Запуск обучения (синхронный если run_in_main=True)
agent.train()

# Сбор наград из очереди
while True:
    try:
        r = agent.res_queue.get_nowait()
    except Empty:
        break
    if r is None:
        break
    episode_rewards.append(float(r))

training_time = time.time() - start_time
print(f"\nОбучение завершено за {training_time:.2f} секунд")
print(f"Всего эпизодов: {len(episode_rewards)}")
print(f"Финальная награда (скользящее среднее): {episode_rewards[-1]:.4f}")
```

### График прогресса обучения

```python
plt.figure(figsize=(12, 5))
plt.plot(episode_rewards, label='Скользящее среднее награды', alpha=0.7)

# Добавление сглаженного тренда
window = 50
if len(episode_rewards) >= window:
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(episode_rewards)), smoothed, 
             'r-', linewidth=2, label=f'Сглаженное (MA{window})')

plt.grid(True, alpha=0.3)
plt.xlabel('Эпизод')
plt.ylabel('Награда (скользящее среднее)')
plt.title('Прогресс обучения A3C на окружении B747')
plt.legend()
plt.tight_layout()
plt.show()
```

### Оценка обученной политики

```python
# Детерминированная оценка с использованием среднего политики
eval_env = make_env(0)
obs, info = eval_env.reset()

agent.gnet.eval()
episode_reward = 0.0
terminated = False
truncated = False

with torch.no_grad():
    while not (terminated or truncated):
        obs_tensor = torch.from_numpy(np.array(obs).reshape(1, -1).astype(np.float32))
        mu, sigma, value = agent.gnet.forward(obs_tensor)
        
        # Используем среднее для детерминированной политики
        action = mu.cpu().numpy().reshape(-1)
        
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += float(reward)

print(f"Детерминированная оценка награды: {episode_reward:.4f}")

# График отслеживания угла тангажа
eval_env.unwrapped.model.plot_transient_process(
    'theta',
    tps,
    reference_signals[0],
    to_deg=True,
    figsize=(15, 4)
)

eval_env.close()
agent.close()
```

### Мониторинг с помощью TensorBoard

```bash
tensorboard --logdir=runs/a3c_b747
```

Доступные метрики:
- **Loss/w*/total**: Общая потеря для каждого воркера
- **Loss/w*/value**: Потеря функции ценности (квадрат TD ошибки)
- **Loss/w*/policy**: Потеря политики (отрицательное ожидаемое преимущество)
- **Loss/w*/entropy**: Энтропия политики (мера исследования)
- **Performance/w*/episode_reward**: Сырые награды за эпизод
- **Performance/w*/moving_avg_reward**: Экспоненциально взвешенное скользящее среднее

### Ожидаемые результаты

После 3000 эпизодов обучения:
- Агент учится отслеживать синусоидальный опорный сигнал тангажа с амплитудой ~1°
- Финальное скользящее среднее награды: примерно от -1.6 до -2.0
- Ошибка отслеживания тангажа: < 0.5° среднеквадратичное

### Советы для улучшения производительности

1. **Увеличьте длительность обучения**: 10000+ эпизодов для лучшей сходимости
2. **Настройте гиперпараметры**:
   - Уменьшите `lr` (5e-5) для более стабильного обучения
   - Увеличьте `update_global_iter` (20-30) для более плавных градиентов
3. **Используйте больше воркеров**: Установите `run_in_main=False` и `n_workers=8` для более быстрого обучения
4. **Настройте опорный сигнал**: Попробуйте разные частоты и амплитуды
5. **Следите за TensorBoard**: Отслеживайте коллапс энтропии или расхождение потери ценности

---

## Документация API

### Agent

::: tensoraerospace.agent.a3c.pytorch.Agent

### Worker

::: tensoraerospace.agent.a3c.pytorch.Worker

### Network

::: tensoraerospace.agent.a3c.pytorch.Net

### Optimizer

::: tensoraerospace.agent.a3c.shared_optim.SharedAdam

### Утилиты

::: tensoraerospace.agent.a3c.pytorch.setup_global_params

## Детали реализации

### Ключевые особенности

1. **Единая сеть**: Один модуль `Net` с общими слоями, снижающий потребление памяти
2. **Активация ReLU6**: Более стабильные градиенты по сравнению со стандартным ReLU
3. **Клиппинг градиентов**: Максимальная норма 40.0 предотвращает взрывающиеся градиенты
4. **Регуляризация энтропии**: Коэффициент 0.005 поощряет исследование
5. **SharedAdam**: Состояние оптимизатора разделяется между процессами для согласованных обновлений
6. **Правильный бутстрэп**: N-шаговые возвраты включают ценность терминального состояния, когда эпизод продолжается

### Преимущества перед синхронными методами

- **Параллельный сбор опыта**: Несколько воркеров исследуют одновременно
- **Некоррелированные сэмплы**: Разные воркеры в разных состояниях снижают корреляцию
- **Без буфера повторов**: Онлайн обучение снижает требования к памяти
- **Естественное исследование**: Асинхронность обеспечивает разнообразие без ε-greedy

### Советы по отладке

- Используйте `run_in_main=True` для запуска одного воркера без многопроцессности
- Проверяйте TensorBoard на расхождение потерь или коллапс энтропии
- Уменьшайте `lr` если обучение нестабильно
- Увеличивайте `update_global_iter` для более стабильных градиентов
- Убедитесь, что окружение правильно инициализировано для воспроизводимости

## Источники

- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) (Mnih et al., 2016)
- [PyTorch Multiprocessing Best Practices](https://pytorch.org/docs/stable/notes/multiprocessing.html)

## Протестированные окружения

- Unity ML-Agents окружения
- Gymnasium задачи непрерывного управления
- TensorAeroSpace LinearLongitudinal* окружения
- Кастомные аэрокосмические окружения управления
