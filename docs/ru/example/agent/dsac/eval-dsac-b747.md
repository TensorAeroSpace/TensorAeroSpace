# Оценка DSAC на Boeing 747

Этот ноутбук демонстрирует, как **загрузить обученный агент DSAC** и оценить его производительность на задаче продольного управления тангажом Boeing 747. Он строит графики переходного процесса и вычисляет стандартные метрики качества управления.

## Обзор

| Аспект | Описание |
|--------|----------|
| **Задача** | Отработка ступенчатого воздействия (θ = 0° → 1°) |
| **Среда** | `ImprovedB747Env` (нормализованное пространство состояний) |
| **Агент** | Обученный чекпоинт DSAC |
| **Результат** | Графики переходного процесса, таблица метрик |

## Что делает этот ноутбук

1. **Загружает обученный чекпоинт DSAC** с помощью `DSAC.from_pretrained()` — поддерживает локальные пути и репозитории HuggingFace Hub.
2. **Настраивает среду оценки** с единичным ступенчатым воздействием (1° по тангажу при t=5с, dt=0.1с, T=20с).
3. **Запускает один эпизод оценки** без шума исследования (`evaluate=True`).
4. **Визуализирует переходный процесс** встроенными методами `plot_transient_process()` и `plot_control()`.
5. **Вычисляет метрики качества** с помощью `ControlBenchmark` — перерегулирование, время установления, время нарастания, IAE, ISE, ITAE и др.

## Запуск

Откройте ноутбук в Jupyter или VS Code:

```bash
jupyter notebook example/reinforcement_learning/deep_rl/eval_dsac_b747.ipynb
```

Или запустите напрямую с `nbconvert`:

```bash
jupyter nbconvert --execute --to notebook example/reinforcement_learning/deep_rl/eval_dsac_b747.ipynb
```

## Основной код

```python
from tensoraerospace.agent import DSAC
from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standard import unit_step
from tensoraerospace.benchmark.bench import ControlBenchmark
import numpy as np

# 1. Настройка среды оценки
dt, tn, step_deg, step_time = 0.1, 20.0, 1.0, 5.0
t = np.arange(0.0, tn + dt, dt, dtype=np.float32)
ref = unit_step(t, degree=step_deg, time_step=step_time, output_rad=True).reshape(1, -1)

env = ImprovedB747Env(
    initial_state=np.zeros(4, dtype=float),
    reference_signal=ref,
    number_time_steps=ref.shape[1],
    dt=dt,
    include_reference_in_obs=True,
    reward_mode="step_response",
)

# 2. Загрузка обученного агента
# Можно загрузить из Hugging Face Hub:
agent = DSAC.from_pretrained("TensorAeroSpace/dsac-b747-step-response")
agent.env = env
agent.to_device("cpu")  # или "cuda" / "mps"
agent.eval()

# 3. Запуск эпизода оценки
obs, _ = env.reset()
done = False
while not done:
    action = agent.select_action(obs, evaluate=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# 4. Построение графика переходного процесса
env.unwrapped.model.plot_transient_process(
    "theta", t, ref[0], to_deg=True, figsize=(15, 4)
)

# 5. Вычисление метрик бенчмарка
system_signal = env.unwrapped.model.get_state("theta", to_deg=True)
benchmark = ControlBenchmark()
benchmark.plot(ref[0], system_signal, tps=t, signal_val=0.0, dt=dt)
print(benchmark.generate_report(ref[0], system_signal, signal_val=0.0, dt=dt))
```

## Пример результатов

Ноутбук выдаёт:

1. **График переходного процесса** — θ(t) относительно задающего сигнала с выделенной зоной установления
2. **График управляющего сигнала** — отклонение руля высоты δe(t)
3. **Таблица метрик**:
   - Перерегулирование (%)
   - Время установления (с) — время входа и удержания в ±5% полосе
   - Время нарастания (с) — время достижения 90% установившегося значения
   - Время пика (с)
   - Степень затухания
   - Статическая ошибка
   - Интегралы IAE, ISE, ITAE
   - Индекс качества (взвешенная комбинация)

## Советы

!!! tip "Выбор устройства"
    Ноутбук автоматически определяет CUDA/MPS/CPU. На Apple Silicon MPS даёт значительное ускорение по сравнению с CPU.

!!! tip "Пути к чекпоинтам"
    Можно загружать из:
    
    - **Локального пути**: `/path/to/checkpoint/folder`
    - **HuggingFace Hub**: `TensorAeroSpace/dsac-b747-step-response`

!!! tip "Настройка задающего сигнала"
    Измените `step_deg`, `step_time`, `tn` и `dt` для тестирования разных сценариев (больший шаг, более быстрая динамика).

## См. также

- [Алгоритм DSAC](../../../agent/dsac.md) — архитектура и гиперпараметры
- [Обучение DSAC (ступенчатое воздействие)](train-dsac-b747-step-response.md) — как обучить с нуля
- [Обучение DSAC (слежение за синусоидой)](train-dsac-b747-tracking.md) — обучение для непрерывного слежения
- [Сравнение DSAC vs PID](../../../comparison/dsac_vs_pid_b747.md) — сравнительный бенчмарк

