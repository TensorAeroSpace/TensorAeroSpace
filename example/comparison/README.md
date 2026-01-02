# 🔬 Сравнение методов управления: ML vs PID

Эта папка содержит примеры и эксперименты для сравнения методов машинного обучения (SAC, PPO, MPC) с классическим ПИД-регулятором.

## 📋 Содержание

### Базовые ноутбуки (baseline)

| Метод | Объект управления | Файл | Описание |
|-------|------------------|------|----------|
| **PID** | F-16 | `pid_f16_baseline.ipynb` | ПИД-регулятор для продольного управления F-16 |
| **SAC** | F-16 | `sac_f16_baseline.ipynb` | Soft Actor-Critic для F-16 |
| **PPO** | B747 | `ppo_b747_baseline.ipynb` | Proximal Policy Optimization для Boeing 747 |
| **MPC** | B747 | `mpc_b747_baseline.ipynb` | Model Predictive Control для B747 |

### Сравнительные эксперименты

| Эксперимент | Файл | Описание |
|-------------|------|----------|
| **ML vs PID (F-16)** | `comparison_sac_vs_pid_f16.ipynb` | Сравнение SAC и PID на модели F-16 |
| **ML vs PID (B747)** | `comparison_ppo_vs_pid_b747.ipynb` | Сравнение PPO и PID на модели B747 |
| **ML vs PID (MPC)** | `comparison_mpc_vs_pid_b747.ipynb` | Сравнение MPC и PID на модели B747 |

## 📊 Метрики сравнения

Для каждого эксперимента измеряются следующие метрики качества переходного процесса:

- **Время регулирования (Settling Time)** — время достижения установившегося режима
- **Перерегулирование (Overshoot)** — максимальное превышение заданного значения в %
- **Статическая ошибка (Static Error)** — разность между заданным и установившимся значением

## 🎯 Цель экспериментов

Продемонстрировать, что методы машинного обучения обеспечивают **на 30% и более быстрый переходный процесс** по сравнению с классическим ПИД-регулятором при сопоставимом или лучшем качестве управления.

## 🚀 Запуск

```bash
# Запуск Jupyter
cd example/comparison
jupyter lab
```

## 📚 Использованные функции

```python
from tensoraerospace.benchmark.function import overshoot, settling_time, static_error
from tensoraerospace.agent.pid import PID
from tensoraerospace.agent.sac import SAC
from tensoraerospace.agent.ppo.model import PPO
```

## 📈 Результаты

Сводная таблица результатов сравнения представлена в ноутбуке `comparison_summary.ipynb` и в разделе НТО.





