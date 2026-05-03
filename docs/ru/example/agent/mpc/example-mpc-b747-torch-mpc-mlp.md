# MPC + MLP динамика для B747 — Отслеживание ступенчатого сигнала

Этот пример демонстрирует полный пайплайн **Model Predictive Control (MPC)** для модели продольной динамики Boeing 747 с использованием обученного **многослойного персептрона (MLP)** в качестве модели динамики.

## Постановка задачи

Мы управляем **углом тангажа (θ)** самолёта Boeing 747 для отслеживания ступенчатого референсного сигнала. Модель самолёта — это линейное пространство состояний 4-го порядка, описывающее продольную динамику на крейсерском режиме.

### Вектор состояния

Модель продольной динамики B747 имеет 4 переменные состояния:

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

!!! warning "Соглашение о единицах измерения"
    Среда B747 ожидает **действия в градусах**, в то время как внутренняя линейная модель и MPC работают в **радианах**. `MPCAgent` обрабатывает это преобразование через адаптеры `action_to_env` и `action_from_env`.

## Обзор метода

1. **Настройка среды**: Создание `LinearLongitudinalB747-v0` со ступенчатым референсом для тангажа (θ)
2. **Сбор данных**: Сбор переходов состояний \((x_t, u_t) \to x_{t+1}\) с использованием разнообразных исследовательских сигналов
3. **Обучение динамики**: Обучение `OneStepMLP` предсказывать \(\Delta x = x_{t+1} - x_t\)
4. **MPC управление**: Градиентная оптимизация для поиска оптимальной последовательности управления
5. **Оценка качества**: Анализ качества управления через `ControlBenchmark`

## Конфигурация

### Параметры симуляции

```python
# Дискретизация по времени
DT = 0.1          # Шаг по времени [с]
TN = 20.0         # Длительность симуляции [с]
N_STEPS = 201     # Общее количество шагов

# Ступенчатый референсный сигнал
REF_STEP_DEG = 5.0      # Целевой тангаж [град]
REF_STEP_TIME_S = 5.0   # Ступенька в момент t=5с
```

### Сбор данных

```python
COLLECT_EPISODES = 1500     # Количество эпизодов исследования
ACTION_RANGE_DEG = 25.0     # Максимальная амплитуда управления [град]
```

### Нейронная сеть (OneStepMLP)

```python
HIDDEN = 256        # Размер скрытого слоя
LR = 1e-4           # Скорость обучения
EPOCHS = 120        # Количество эпох обучения
BATCH_SIZE = 1024   # Размер мини-батча
```

### Параметры MPC

```python
HORIZON = 20        # Горизонт предсказания [шаги]
MPC_ITERS = 60      # Итерации оптимизации на каждом шаге
MPC_LR = 0.02       # Скорость обучения оптимизатора (Adam)
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
)
from tensoraerospace.benchmark import ControlBenchmark
```

## Настройка среды

```python
# Временной вектор
T = np.arange(N_STEPS, dtype=np.float32) * DT

# Генерация ступенчатого референсного сигнала (внутри конвертируется в радианы)
reference_signal = unit_step(
    tp=T,
    degree=float(REF_STEP_DEG),
    time_step=float(REF_STEP_TIME_S),
    output_rad=True,
).reshape(1, -1)

# Создание среды
env = gym.make(
    "LinearLongitudinalB747-v0",
    number_time_steps=N_STEPS,
    initial_state=np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float32),
    reference_signal=reference_signal,
    dt=float(DT),
)
```

## Конфигурация MPCAgent

### Адаптеры состояния/действия

Эти функции обрабатывают преобразование между наблюдениями среды и внутренними состояниями MPC:

```python
def obs_to_state(env, obs) -> np.ndarray:
    """Извлечение истинного состояния из среды (в радианах)."""
    return np.asarray(env.unwrapped.model.xt, dtype=np.float32).reshape(-1)

def action_from_env(a_env_deg: np.ndarray) -> np.ndarray:
    """Преобразование действия из единиц среды (град) во внутренние (рад)."""
    return np.deg2rad(a_env_deg.astype(np.float32)).astype(np.float32)

def action_to_env(u_rad: np.ndarray) -> np.ndarray:
    """Преобразование действия из внутренних (рад) в единицы среды (град)."""
    return np.rad2deg(u_rad.astype(np.float32)).astype(np.float32)
```

### Веса MPC

Функция стоимости:
$$J = \sum_{k=0}^{H} \left[ (x_k - x_{ref})^T Q (x_k - x_{ref}) + u_k^T R\, u_k + \Delta u_k^T S\, \Delta u_k \right] + J_{extra}$$

```python
weights = MPCWeights(
    Q_diag=np.array([0.0, 0.0, 0.2, 2000.0], dtype=np.float32),  # [u, w, q, θ]
    R_diag=np.array([0.01], dtype=np.float32),                   # затраты на управление
    S_diag=np.array([5.0], dtype=np.float32),                    # скорость изменения управления
    terminal_weight=10.0,                                         # терминальный множитель
)
```

- **Q_diag**: Веса отслеживания состояния. Высокий вес на θ (2000.0) для точного отслеживания тангажа.
- **R_diag**: Штраф за величину управления. Маленький (0.01) для разрешения агрессивного управления при необходимости.
- **S_diag**: Штраф за скорость изменения управления. Умеренный (5.0) для плавных команд руля.

### Ограничения

```python
u_lim = float(np.deg2rad(25.0))    # ±25° предел руля высоты
du_max = float(np.deg2rad(10.0))   # ±10°/шаг предел скорости

constraints = MPCConstraints(
    u_min=np.array([-u_lim], dtype=np.float32),
    u_max=np.array([u_lim], dtype=np.float32),
    du_min=np.array([-du_max], dtype=np.float32),
    du_max=np.array([du_max], dtype=np.float32),
)
```

### Дополнительные штрафы для переходного процесса

Конфигурация step response добавляет штрафы за:

- **Перерегулирование**: Штраф за превышение уставки
- **Время установления**: Поощрение быстрой сходимости
- **Колебания**: Снижение "охоты" вокруг уставки
- **Рывок**: Плавные переходы управления

```python
step_cfg = MPCStepResponseExtraCostConfig.from_degrees(
    tracked_idx=3,              # Индекс θ в векторе состояния
    rate_idx=2,                 # Индекс q (скорость тангажа)
    dt=float(DT),
    overshoot_limit_deg=0.05,   # Допустимое перерегулирование [град]
    settle_band_deg=0.10,       # Полоса установления [град]
    settle_time_target_s=1.0,   # Целевое время установления [с]
    w_overshoot=8000.0,         # Вес штрафа за перерегулирование
    w_settle=8000.0,            # Вес штрафа за установление
)
```

### Инициализация агента

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
    # Конфигурация MLP
    hidden_layers=(HIDDEN, HIDDEN),  # Два скрытых слоя
    model_predict_delta=True,         # Предсказываем Δx, не x_next
    normalize=True,                   # Нормализация входов/выходов
    dynamics_lr=LR,
    # MPC оптимизация
    iters=MPC_ITERS,
    mpc_lr=MPC_LR,
    mpc_optimizer="adam",
    warm_start=True,                  # Инициализация из предыдущего решения
    # Адаптеры
    obs_to_state=obs_to_state,
    action_to_env=action_to_env,
    action_from_env=action_from_env,
    device="cuda",
)
```

## Сбор данных

Агент собирает переходы используя разнообразные исследовательские сигналы для обеспечения хорошего покрытия пространства состояний-действий:

```python
agent.collect_data(
    num_episodes=COLLECT_EPISODES,
    exploration="signals",
    signal_kinds=[
        "random_steps",      # Случайные ступеньки
        "unit_step",         # Простые ступеньки
        "multi_step",        # Несколько ступенек за эпизод
        "ramp",              # Линейные рампы
        "sinusoid",          # Синусоидальное возбуждение
        "multisine",         # Сумма синусоид
        "chirp",             # Развёртка по частоте
        "square_wave",       # Меандр
        "triangular_wave",   # Треугольная волна
        "sawtooth",          # Пилообразная волна
        "doublet",           # Дублет (лётные испытания)
        "pulse",             # Прямоугольный импульс
        "gaussian_pulse",    # Гауссов импульс
        "damped_sinusoid",   # Затухающая синусоида
    ],
    action_amplitude_frac=1.0,
)
print(f"Собрано {len(agent.memory)} переходов")
```

## Обучение динамики

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
Train dynamics: 100%|██████████| 23520/23520 [02:31<00:00, 155.23step/s, loss=1.28e-6]
```

## MPC Rollout

```python
_ = env.reset()
agent.reset()

hist_theta_deg, hist_ref_deg, hist_u_deg = [], [], []
ref_theta_rad = np.asarray(env.unwrapped.reference_signal).reshape(-1)

for step in tqdm(range(env.unwrapped.number_time_steps - 2)):
    k = int(env.unwrapped.current_step)
    x0 = np.asarray(env.unwrapped.model.xt, dtype=np.float32).reshape(-1)
    
    # Построение референсной траектории на горизонте
    target = float(ref_theta_rad[min(k, len(ref_theta_rad)-1)])
    x_ref = np.zeros((HORIZON + 1, 4), dtype=np.float32)
    x_ref[:, 3] = target  # Только референс по θ имеет значение
    
    # Получение оптимального действия от MPC
    action = agent.select_action(x0, x_ref=x_ref)
    
    # Шаг среды
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Логирование результатов
    theta_deg = float(np.rad2deg(env.unwrapped.model.xt[3]))
    hist_theta_deg.append(theta_deg)
    hist_ref_deg.append(float(np.rad2deg(target)))
    hist_u_deg.append(float(action[0]))
    
    if terminated or truncated:
        break
```

## Результаты

### Визуализация переходного процесса

```python
t_plot = np.arange(len(hist_theta_deg)) * DT

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Отслеживание угла тангажа
axes[0].plot(t_plot, hist_ref_deg, 'r--', linewidth=2, label="θ_ref")
axes[0].plot(t_plot, hist_theta_deg, 'b-', linewidth=1.5, label="θ")
axes[0].axhline(y=REF_STEP_DEG * 0.9, color='g', linestyle=':', alpha=0.5, label="90% нарастание")
axes[0].axhline(y=REF_STEP_DEG * 1.02, color='orange', linestyle=':', alpha=0.5, label="2% полоса")
axes[0].axhline(y=REF_STEP_DEG * 0.98, color='orange', linestyle=':', alpha=0.5)
axes[0].set_ylabel("Угол тангажа θ [град]")
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].set_title("MPC + MLP динамика: Отслеживание ступенчатого сигнала")

# Сигнал управления
axes[1].plot(t_plot, hist_u_deg, 'g-', linewidth=1.5, label="δe")
axes[1].axhline(y=25, color='r', linestyle='--', alpha=0.5, label="пределы")
axes[1].axhline(y=-25, color='r', linestyle='--', alpha=0.5)
axes[1].set_xlabel("Время [с]")
axes[1].set_ylabel("Руль высоты δe [град]")
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Метрики качества

| Метрика | Значение | Описание |
| --- | --- | --- |
| **Перерегулирование** | ~0.30% | Отлично — хорошо в пределах типичного требования 2% |
| **Время установления** | ~1.7 с | Быстрая сходимость к ±2% полосе |
| **Время нарастания** | ~1.1 с | Время достижения 90% уставки |
| **Статическая ошибка** | ~0.001 | Пренебрежимо малая установившаяся ошибка |

```python
bench = ControlBenchmark()
metrics = bench.becnchmarking_one_step(
    control_signal=np.array(hist_ref_deg),
    system_signal=np.array(hist_theta_deg),
    signal_val=0.0,
    dt=DT,
)
print(bench.generate_report(metrics))
```

### Анализ

MPC на основе MLP достигает отличных характеристик переходного процесса:

1. **Минимальное перерегулирование** (~0.3%): Штрафы step response эффективно подавляют перерегулирование.
2. **Быстрое установление** (~1.7с): Высокий вес на отслеживание θ (Q[3]=2000) обеспечивает быструю сходимость.
3. **Плавное управление**: Штраф за скорость (S=5.0) предотвращает резкие движения руля.
4. **Почти нулевая статическая ошибка**: Обученная динамика точно отражает поведение системы.

## Ключевые выводы

- **OneStepMLP** — самая простая модель динамики, подходит для систем с гладкой, приблизительно линейной динамикой
- **Предсказание дельты** (`model_predict_delta=True`) улучшает стабильность обучения
- **Разнообразные исследовательские сигналы** критически важны для хорошей генерализации
- **Step response extra cost** помогает достичь строгих требований по перерегулированию/установлению

## Исходный код

Полный ноутбук: [`example/mpc_controllers/example-mpc-b747-torch-mpc-mlp.ipynb`](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/main/example/mpc_controllers/example-mpc-b747-torch-mpc-mlp.ipynb)
