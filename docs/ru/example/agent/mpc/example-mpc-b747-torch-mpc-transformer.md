# MPC + Transformer динамика для B747 — Отслеживание ступенчатого сигнала

Этот пример демонстрирует полный пайплайн **Model Predictive Control (MPC)** для модели продольной динамики Boeing 747 с использованием обученной модели динамики на основе **Transformer**.

## Постановка задачи

Мы управляем **углом тангажа (θ)** самолёта Boeing 747 для отслеживания ступенчатого референсного сигнала с использованием архитектуры Transformer encoder в качестве модели динамики.

### Что такое Transformer Dynamics?

**TransformerDynamicsModel** применяет архитектуру Transformer encoder к обучению динамики:

$$\hat{x}_{t+1} = \text{MLP}(\text{TransformerEncoder}(\text{Embed}([x_t, u_t])))$$

Ключевые компоненты:

- **Входной эмбеддинг**: Проецирует `concat([x_t, u_t])` в `d_model` измерений
- **Позиционное кодирование**: Опционально (отключено для `seq_len=1`)
- **Слои self-attention**: Захватывают связи внутри входа
- **Feed-forward слои**: Преобразуют выход attention
- **Выходная проекция**: Отображает обратно в размерность состояния

!!! info "Упрощённая конфигурация"
    В этом примере мы используем `seq_len=1` (без явной истории), поэтому Transformer действует как продвинутый нелинейный аппроксиматор функций. Для последовательных данных увеличьте `seq_len` и подавайте прошлые пары состояние-действие.

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

1. **Настройка среды**: Создание `LinearLongitudinalB747-v0` со ступенчатым референсом
2. **Сбор данных**: Сбор переходов состояний с использованием разнообразных исследовательских сигналов
3. **Обучение Transformer**: Обучение `TransformerDynamicsModel` предсказывать дельты состояния
4. **MPC управление**: Градиентная оптимизация с предсказаниями Transformer
5. **Оценка качества**: Анализ качества управления через `ControlBenchmark`

## Конфигурация

### Параметры симуляции

```python
DT = 0.1            # Шаг по времени [с]
TN = 20.0           # Длительность симуляции [с]
N_STEPS = 201       # Общее количество шагов

REF_STEP_DEG = 5.0      # Целевой тангаж [град]
REF_STEP_TIME_S = 5.0   # Ступенька в момент t=5с
```

### Архитектура Transformer

```python
D_MODEL = 64          # Размерность эмбеддинга
N_HEAD = 4            # Количество голов внимания
N_LAYERS = 2          # Количество слоёв encoder
FF_DIM = 256          # Скрытая размерность feed-forward
DROPOUT = 0.1         # Dropout (регуляризация)
```

!!! note "Выбор архитектуры"
    - **d_model=64**: Маленький эмбеддинг для быстрого инференса. Увеличьте для сложных систем.
    - **nhead=4**: Позволяет 4 параллельных паттерна внимания.
    - **num_layers=2**: Минимальная глубина; более глубокие модели могут захватить более сложную динамику.
    - **dim_feedforward=256**: 4× `d_model` — распространённый выбор.

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

from tensoraerospace.signals.standard import unit_step
from tensoraerospace.agent.mpc import (
    MPCAgent,
    MPCConstraints,
    MPCStepResponseExtraCostConfig,
    MPCTrackingExtraCostConfig,
    MPCWeights,
    TransformerDynamicsModel,
)
from tensoraerospace.benchmark import ControlBenchmark
```

## Архитектура модели Transformer

```
Вход: [x_t, u_t] ∈ ℝ^5
    ↓
Входной эмбеддинг: Linear(5 → 64)
    ↓
Позиционное кодирование (опционально)
    ↓
┌─────────────────────────────────────────┐
│ TransformerEncoderLayer × 2:            │
│   • Multi-Head Self-Attention (4 голов) │
│   • Add & Norm                          │
│   • Feed-Forward (64 → 256 → 64)        │
│   • Add & Norm                          │
│   • Dropout (0.1)                       │
└─────────────────────────────────────────┘
    ↓
Выходная проекция: Linear(64 → 4)
    ↓
Выход: Δx ∈ ℝ^4
```

```python
transformer_model = TransformerDynamicsModel(
    input_dim=4 + 1,              # state_dim + action_dim
    output_dim=4,                 # state_dim
    d_model=D_MODEL,              # 64
    nhead=N_HEAD,                 # 4
    num_encoder_layers=N_LAYERS,  # 2
    dim_feedforward=FF_DIM,       # 256
    dropout=DROPOUT,              # 0.1
    seq_len=1,                    # Один временной шаг
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

### Агент с пользовательской моделью Transformer

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
    # Пользовательская модель Transformer
    model=transformer_model,
    model_predict_delta=True,
    normalize=True,
    dynamics_lr=LR,
    # Настройки MPC
    iters=MPC_ITERS,
    mpc_lr=MPC_LR,
    warm_start=True,
    mpc_track_best=True,
    mpc_compile_dynamics=True,  # JIT компиляция для скорости (только CUDA)
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

## Обучение динамики Transformer

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
Train dynamics: 100%|██████████| 69600/69600 [18:18<00:00, 63.37step/s, loss=1.57e-5]
```

!!! warning "Время обучения"
    Модели Transformer обучаются значительно дольше, чем MLP из-за механизма внимания. Ожидайте ~18 минут на GPU для 1500 эпизодов (против ~2.5 мин для MLP).

## MPC Rollout

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

### Метрики качества

| Метрика | Значение | Описание |
| --- | --- | --- |
| **Перерегулирование** | ~-0.10% | Минимальный недоход |
| **Время установления** | ~1.5 с | **Лучший** результат среди всех моделей |
| **Время нарастания** | ~0.8 с | **Лучший** результат среди всех моделей |
| **Время пика** | ~2.5 с | Время до первого пика |
| **Статическая ошибка** | ~0.009 | Низкая установившаяся ошибка |
| **Количество колебаний** | 6 | Больше колебаний, чем у MLP |

### Анализ

Модель Transformer достигает **лучших общих показателей**:

1. **Самое быстрое время нарастания (0.8с)**: Механизм внимания обеспечивает точные краткосрочные предсказания, позволяя агрессивное управление без нестабильности.

2. **Самое быстрое время установления (1.5с)**: Несмотря на бо́льшее количество колебаний, Transformer быстро сходится к уставке.

3. **Минимальное перерегулирование (-0.10%)**: Небольшой недоход пренебрежимо мал и хорошо в пределах типичных спецификаций.

4. **Больше колебаний (6)**: Способность Transformer захватывать мелкие детали приводит к небольшим колебаниям вокруг уставки. Это можно настроить через:
   - Увеличение `w_osc` в конфигурации step response
   - Уменьшение горизонта MPC
   - Добавление бо́льшего сглаживания в весах

5. **Хорошая статическая ошибка (0.009)**: Лучше, чем NARX, но немного хуже MLP.

### Почему Transformer работает хорошо

1. **Self-attention**: Может моделировать сложные связи вход-выход за один прямой проход
2. **Residual connections**: Улучшают поток градиентов при обучении и инференсе
3. **Layer normalization**: Стабилизирует активации, важно для градиентной оптимизации MPC
4. **Гибкая ёмкость**: 2-слойный encoder обеспечивает достаточную ёмкость для динамики B747

### Когда использовать Transformer

Transformer dynamics рекомендуется когда:

- **Сложные нелинейные системы**: Механизм внимания может захватить сложную динамику
- **Длинные последовательности** (`seq_len > 1`): Естественно подходит для последовательных данных
- **Достаточно вычислений**: Обучение медленнее, но инференс параллелизуется на GPU
- **Исследования/прототипирование**: Современная архитектура для изучения новых методов управления

## Сравнение с другими моделями

| Метрика | MLP | NARX | **Transformer** |
| --- | --- | --- | --- |
| **Перерегулирование** | +0.30% | -1.87% | **-0.10%** |
| **Время установления** | 1.7 с | 3.0 с | **1.5 с** |
| **Время нарастания** | 1.1 с | 1.0 с | **0.8 с** |
| **Статическая ошибка** | **0.001** | 0.026 | 0.009 |
| **Колебания** | Мало | Мало | Средне |
| **Время обучения** | **~2.5 мин** | ~9 мин | ~18 мин |

### Рекомендации

- **Для продакшена**: Начните с **MLP** (самое быстрое обучение, хорошие результаты)
- **Для систем с памятью**: Используйте **NARX** с подходящими лагами
- **Для лучшего отслеживания**: Используйте **Transformer**, если время обучения не критично
- **Для исследований**: **Transformer** предлагает наибольшую гибкость для архитектурных изменений

## Исходный код

Полный ноутбук: [`example/mpc_controllers/example-mpc-b747-torch-mpc-transformer.ipynb`](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/main/example/mpc_controllers/example-mpc-b747-torch-mpc-transformer.ipynb)
