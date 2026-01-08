# MPC + NARX динамика для B747 — Отслеживание ступенчатого сигнала

Этот пример демонстрирует полный пайплайн **Model Predictive Control (MPC)** для модели продольной динамики Boeing 747 с использованием обученной модели **NARX (нелинейная авторегрессия с экзогенными входами)**.

## Постановка задачи

Мы управляем **углом тангажа (θ)** самолёта Boeing 747 для отслеживания ступенчатого референсного сигнала с использованием нейронной сети NARX в качестве модели динамики.

### Что такое NARX?

**NARX** (Nonlinear AutoRegressive with eXogenous inputs) — это рекуррентная архитектура, которая предсказывает будущие выходы на основе прошлых входов и выходов:

$$x_{t+1} = f(x_t, x_{t-1}, ..., x_{t-n_x}, u_t, u_{t-1}, ..., u_{t-n_u})$$

Где:

- \(n_x\) = `state_lags` — количество прошлых состояний
- \(n_u\) = `control_lags` — количество прошлых управлений

!!! info "One-Step vs Multi-Lag NARX"
    В этом примере мы используем NARX в **режиме одного шага** (`state_lags=1`, `control_lags=1`), что делает его эквивалентным стандартному MLP, получающему `concat([x_t, u_t])`. Для систем со значительными эффектами памяти используйте бóльшие лаги и подавайте расширенный вектор истории в MPC.

### Вектор состояния

| Индекс | Переменная | Описание | Единицы |
| --- | --- | --- | --- |
| 0 | u | Возмущение скорости по продольной оси | м/с |
| 1 | w | Возмущение скорости по вертикальной оси | м/с |
| 2 | q | Угловая скорость тангажа | рад/с |
| 3 | θ | Угол тангажа | рад |

### Управляющий вход

| Индекс | Переменная | Описание | Единицы |
| --- | --- | --- | --- |
| 0 | δe | Отклонение руля высоты | град (среда) / рад (внутри) |

## Обзор метода

1. **Настройка среды**: Создание `LinearLongitudinalB747-v0` со ступенчатым референсом для тангажа (θ)
2. **Сбор данных**: Сбор переходов состояний с использованием разнообразных исследовательских сигналов
3. **Обучение NARX**: Обучение `NARXDynamicsModel` предсказывать дельты состояния
4. **MPC управление**: Градиентная оптимизация с предсказаниями NARX
5. **Оценка качества**: Анализ качества управления через `ControlBenchmark`

## Конфигурация

### Параметры симуляции

```python
DT = 0.1            # Шаг по времени [с]
TN = 20.0           # Длительность симуляции [с]
N_STEPS = 201       # Общее количество шагов

REF_STEP_DEG = 1.0      # Целевой тангаж [град] — меньшая ступенька для демо NARX
REF_STEP_TIME_S = 5.0   # Ступенька в момент t=5с
```

### Сбор данных

```python
COLLECT_EPISODES = 1500     # Количество эпизодов исследования
ACTION_RANGE_DEG = 25.0     # Максимальная амплитуда управления [град]
```

### Архитектура NARX

```python
NARX_HIDDEN = 256     # Размер скрытого слоя
NARX_LAYERS = 3       # Количество скрытых слоёв
STATE_LAGS = 1        # История состояний (1 = только текущее)
CONTROL_LAGS = 1      # История управлений (1 = только текущее)
```

### Обучение

```python
EPOCHS = 120          # Эпохи обучения
BATCH_SIZE = 512      # Размер мини-батча
LR = 1e-4             # Скорость обучения
```

### Параметры MPC

```python
HORIZON = 20          # Горизонт предсказания [шаги]
MPC_ITERS = 60        # Итерации оптимизации на шаг
MPC_LR = 0.02         # Скорость обучения оптимизатора
DU_MAX_DEG = 3.0      # Предел скорости управления [град/шаг]
```

## Импорты

```python
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from tensoraerospace.signals.standart import unit_step
from tensoraerospace.agent.mpc import (
    MPCAgent,
    MPCConstraints,
    MPCStepResponseExtraCostConfig,
    MPCTrackingExtraCostConfig,
    MPCWeights,
    NARXDynamicsModel,
)
from tensoraerospace.benchmark import ControlBenchmark
```

## Архитектура модели NARX

`NARXDynamicsModel` внутренне строит:

```
Вход: concat([x_{t}, ..., x_{t-n_x+1}, u_{t}, ..., u_{t-n_u+1}])
       ↓
   Linear(input_dim → hidden_size)
       ↓
   [LayerNorm → ReLU → Linear(hidden_size → hidden_size)] × (num_layers - 1)
       ↓
   Linear(hidden_size → output_dim)
       ↓
Выход: Δx (дельта состояния)
```

```python
narx_model = NARXDynamicsModel(
    state_dim=4,
    action_dim=1,
    hidden_size=NARX_HIDDEN,      # 256 нейронов на слой
    num_layers=NARX_LAYERS,       # 3 скрытых слоя
    state_lags=STATE_LAGS,        # 1 (только текущее состояние)
    control_lags=CONTROL_LAGS,    # 1 (только текущее управление)
)
```

## Конфигурация MPCAgent

### Веса и ограничения

```python
weights = MPCWeights(
    Q_diag=np.array([0.0, 0.0, 0.2, 2000.0], dtype=np.float32),
    R_diag=np.array([0.01], dtype=np.float32),
    S_diag=np.array([5.0], dtype=np.float32),
    terminal_weight=10.0,
)

u_lim = float(np.deg2rad(25.0))
du_max = float(np.deg2rad(DU_MAX_DEG))

constraints = MPCConstraints(
    u_min=np.array([-u_lim], dtype=np.float32),
    u_max=np.array([u_lim], dtype=np.float32),
    du_min=np.array([-du_max], dtype=np.float32),
    du_max=np.array([du_max], dtype=np.float32),
)
```

### Конфигурация переходного процесса

```python
step_cfg = MPCStepResponseExtraCostConfig.from_degrees(
    tracked_idx=3,
    rate_idx=2,
    dt=float(DT),
    overshoot_limit_deg=0.05,
    settle_band_deg=0.10,
    settle_time_target_s=1.0,
    w_overshoot=8000.0,
    w_settle=8000.0,
    w_sse_steady=40000.0,
    w_osc=500.0,
)
```

### Агент с пользовательской моделью NARX

```python
agent = MPCAgent(
    env,
    state_dim=4,
    action_dim=1,
    horizon=HORIZON,
    weights=weights,
    constraints=constraints,
    tracking_type="step_response",
    step_response_config=step_cfg,
    # Пользовательская модель NARX
    model=narx_model,
    model_predict_delta=True,
    normalize=True,
    dynamics_lr=LR,
    # Настройки MPC
    iters=MPC_ITERS,
    mpc_lr=MPC_LR,
    warm_start=True,
    mpc_track_best=True,
    # Адаптеры
    obs_to_state=obs_to_state,
    action_to_env=action_to_env,
    action_from_env=action_from_env,
    device="cuda",
)
```

## Сбор данных

```python
agent.collect_data(
    num_episodes=COLLECT_EPISODES,
    exploration="signals",
    signal_kinds=[
        "random_steps",
        "unit_step",
        "multi_step",
        "ramp",
        "sinusoid",
        "multisine",
        "chirp",
        "square_wave",
        "triangular_wave",
        "sawtooth",
        "doublet",
        "pulse",
        "gaussian_pulse",
        "damped_sinusoid",
    ],
)
print(f"Собрано {len(agent.memory)} переходов")
```

## Обучение динамики NARX

```python
metrics = agent.train_dynamics(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    loss="mse",
)
print(f"Финальная ошибка: {metrics['loss']:.2e}")
```

**Ожидаемый вывод:**
```
Train dynamics: 100%|██████████| 69600/69600 [08:54<00:00, 130.33step/s, loss=4.36e-6]
```

!!! note "Время обучения"
    NARX с 3 скрытыми слоями обучается дольше, чем простой 2-слойный MLP. Ожидайте ~9 минут на GPU для 1500 эпизодов.

## MPC Rollout

Процедура rollout идентична примеру с MLP — `MPCAgent` внутренне обрабатывает модель динамики:

```python
_ = env.reset()
agent.reset()

hist_theta_deg, hist_ref_deg, hist_u_deg = [], [], []
ref_theta_rad = np.asarray(env.unwrapped.reference_signal).reshape(-1)

for step in tqdm(range(env.unwrapped.number_time_steps - 2)):
    k = int(env.unwrapped.current_step)
    x0 = np.asarray(env.unwrapped.model.xt, dtype=np.float32).reshape(-1)
    
    target = float(ref_theta_rad[min(k, len(ref_theta_rad)-1)])
    x_ref = np.zeros((HORIZON + 1, 4), dtype=np.float32)
    x_ref[:, 3] = target
    
    action = agent.select_action(x0, x_ref=x_ref)
    obs, reward, terminated, truncated, info = env.step(action)
    
    theta_deg = float(np.rad2deg(env.unwrapped.model.xt[3]))
    hist_theta_deg.append(theta_deg)
    hist_ref_deg.append(float(np.rad2deg(target)))
    hist_u_deg.append(float(action[0]))
    
    if terminated or truncated:
        break
```

## Результаты

### Визуализация переходного процесса

<!-- TODO: ВСТАВИТЬ ГРАФИК ПЕРЕХОДНОГО ПРОЦЕССА СЮДА -->
<!-- ![Переходный процесс NARX](../../../../../assets/mpc/narx_step_response.png) -->

### Метрики качества

| Метрика | Значение | Описание |
| --- | --- | --- |
| **Перерегулирование** | ~-1.87% | Небольшой недоход (отрицательное перерегулирование) |
| **Время установления** | ~3.0 с | Медленнее MLP из-за консервативного управления |
| **Время нарастания** | ~1.0 с | Сравнимо с MLP |
| **Время пика** | ~1.7 с | Время до первого пика |
| **Статическая ошибка** | ~0.026 | Небольшое, но заметное смещение |

### Анализ

Модель NARX даёт другое поведение управления по сравнению с MLP:

1. **Недоход вместо перерегулирования**: Система подходит к уставке снизу, что указывает на консервативное управление. Это проявляется как отрицательное "перерегулирование" (~-1.87%).

2. **Более медленное установление**: 3.0с против 1.7с у MLP — NARX занимает почти вдвое больше времени. Причины:
   - Более глубокая сеть (3 слоя vs 2) может иметь немного другие градиенты
   - Более консервативные предсказания в начале переходного процесса

3. **Бóльшая статическая ошибка**: 0.026 против 0.001 у MLP. Можно улучшить через:
   - Увеличение веса `w_sse_steady`
   - Использование интегральной составляющей (не реализовано в базовом MPC)

4. **Хорошая генерализация**: Несмотря на более медленное установление, NARX надёжно отслеживает референс и избегает нестабильности.

### Когда использовать NARX

NARX особенно полезен когда:

- **Система имеет память**: Бóльшие лаги (`state_lags > 1`) могут захватить задержанные эффекты
- **Нелинейная динамика**: Более глубокая архитектура лучше аппроксимирует сложные функции
- **Зашумлённые наблюдения**: LayerNorm в NARX помогает с нормализацией входов

Для линейной модели B747 MLP работает лучше, но NARX демонстрирует модульную архитектуру, где любая совместимая модель может быть использована.

## Сравнение с MLP

| Метрика | MLP | NARX |
| --- | --- | --- |
| Перерегулирование | +0.30% | -1.87% |
| Время установления | 1.7 с | 3.0 с |
| Время нарастания | 1.1 с | 1.0 с |
| Статическая ошибка | 0.001 | 0.026 |
| Время обучения | ~2.5 мин | ~9 мин |

## Исходный код

Полный ноутбук: [`example/mpc_controllers/example-mpc-b747-torch-mpc-narx.ipynb`](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/main/example/mpc_controllers/example-mpc-b747-torch-mpc-narx.ipynb)
