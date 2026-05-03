# DSAC для Boeing 747 (ступенчатый сигнал)

Минимальный запуск DSAC для продольного управления тангажом в нормализованной среде `ImprovedB747Env`. Используются квантильные (IQN) сдвоенные критики и регуляризация CAPS для слежения за ступенчатым отклонением руля высоты на 5°.

## Как запустить

```bash
python example/reinforcement_learning/example_dsac_b747.py
```

Что делает скрипт:

- Формирует 800 шагов эталона (переход 0 → 5° на 20% эпизода)
- Создаёт `ImprovedB747Env` с нормализацией наблюдений и действий
- Инициализирует `DSAC` с 32 квантилями, CAPS и автонастройкой энтропии
- Обучает 5 эпизодов и пишет логи TensorBoard

!!! note
    Устройство выбирается автоматически: `device = "cuda" if torch.cuda.is_available() else "cpu"`.

## Основной фрагмент

```python
import numpy as np
import torch
from tensoraerospace.agent import DSAC
from tensoraerospace.envs.b747 import ImprovedB747Env

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_reference(steps: int, step_deg: float = 5.0) -> np.ndarray:
    ref = np.zeros((1, steps), dtype=np.float32)
    ref[:, steps // 5 :] = np.deg2rad(step_deg)
    return ref

env = ImprovedB747Env(
    initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
    reference_signal=make_reference(800, step_deg=5.0),
    number_time_steps=800,
    dt=0.02,
    reward_mode="step_response",
)

agent = DSAC(
    env,
    batch_size=128,
    memory_capacity=200_000,
    learning_starts=1_000,
    updates_per_step=1,
    num_quantiles=32,
    embedding_dim=32,
    hidden_layers=[64, 64],
    huber_threshold=1.0,
    lr=3e-4,
    policy_lr=3e-4,
    device=device,
    log_every_updates=50,
    automatic_entropy_tuning=True,
)

agent.train(num_episodes=5, save_best=False)
agent.close()
```

## Советы

- Увеличьте `num_episodes` для реального качества; 5 эпизодов — лишь быстрый smoke-test.
- Следите за логами в TensorBoard (`tensorboard --logdir runs`) для контроля потерь и энтропии.
- Если действия выглядят слишком сглаженными, уменьшите `updates_per_step` или коэффициенты сглаживания в `DSAC.update_parameters`.

