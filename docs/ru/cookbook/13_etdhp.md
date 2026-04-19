# Рецепт 13 — ET-DHP на нелинейной F-16

ET-DHP (Event-Triggered Dual Heuristic Programming) добавляет поверх IM-GDHP **триггер на основе Липшица**: регулятор выдаёт новое действие только когда состояние достаточно ушло от точки последнего триггера, чтобы оправдать пересчёт. На встраиваемом контроллере это снижает вычисления на тик на порядок.

**Документация.** [ET-DHP](../agent/et_dhp.md) · **Полный ноутбук.** [`example_etdhp_nonlinear_f16.ipynb`](../example/agent/et_dhp/example_etdhp_nonlinear.md) · **Связано.** [Рецепт 12 — IM-GDHP](12_imgdhp.md).

## Когда использовать ET-DHP

- **Маломощные / встраиваемые системы**, где нельзя позволить полный forward NN каждую миллисекунду.
- **Каналы с ограниченной пропускной способностью**, где каждое обновление управления имеет ненулевую стоимость передачи (БПЛА, спутники).

Компромисс: между триггерами привод **держит последнее значение**. На объекте с быстрой динамикой это даёт лёгкий пилообразный эффект на трассе состояния. Настраивайте порог под полосу объекта.

## Шаг 1 — Минимальная конфигурация

```python
import numpy as np
from tensoraerospace.agent.et_dhp import ETDHPAgent, ETDHPConfig

cfg = ETDHPConfig(
    dt=0.01,
    # те же актор / критик, что у IM-GDHP
    actor_hidden=(32, 32),
    critic_hidden=(64, 64),
    actor_lr=1e-3,
    critic_lr=5e-4,
    gamma=0.95,
    lambda_weight=0.5,
    # RLS-идентификатор
    rls_forgetting=0.995,
    rls_cov_init=1e2,
    G_init=np.array([[-0.5]]),
    # event-trigger: срабатывает, когда ||x_t - x_last_trigger|| > trigger_threshold * Lipschitz_constant
    trigger_threshold=0.02,
    lipschitz_estimate=10.0,
    u_magnitude_limit=15.0,
    u_rate_limit=60.0,
    seed=0,
)
```

**Порог триггера** — главная ручка. Меньше → чаще обновления → тугое слежение ценой компьюта. Типичный старт: `0.01–0.05`, масштабированный величиной типового состояния.

## Шаг 2 — Цикл

Тот же трёхшаговый паттерн, что у остальных адаптивных агентов:

```python
agent = ETDHPAgent(n_state=1, n_control=1, config=cfg)
env.reset()

for k in range(n_steps):
    obs = env.get_state()
    u = agent.predict(obs[controlled_channels], ref[:, k], k)
    obs, *_ = env.step(u)
    agent.learn(obs[controlled_channels], ref[:, k], k)
```

Внутри `predict()` либо прогоняет полный актор+критик (на триггере), либо возвращает кэшированное действие. `learn()` всегда делает шаг RLS, чтобы модель оставалась свежей между триггерами.

## Шаг 3 — Ожидаемое поведение на нелинейной F-16

Слежение за ступенькой α с активным триггером:

![ET-DHP на нелинейной F-16](img/13_etdhp_nonlinear.png)

Что вы должны увидеть:

- Слежение, похожее на IM-GDHP (тот же актор / критик).
- Редкие **event-срабатывания** на одной из диагностических трасс — обычно < 10 % всех тиков приводят к пересчёту НС.
- Слегка «блочную» команду руля, потому что привод держится между триггерами.

См. [`example_etdhp_nonlinear_f16.ipynb`](../example/agent/et_dhp/example_etdhp_nonlinear.md) — там есть лог счётчика триггеров и пошаговый разбор.

## Шаг 4 — Save / load / публикация в HuggingFace

```python
run_dir = agent.save('./checkpoints')
restored = ETDHPAgent.from_pretrained(run_dir)
agent.publish_to_hub('me/my-etdhp', folder_path=run_dir, access_token='hf_…')
```

## Подводные камни

- **Триггер никогда не срабатывает.** `trigger_threshold` слишком велик для ваших состояний — половините, пока не проявится ошибка слежения.
- **Триггер срабатывает каждый тик.** Порог слишком мал; тогда нет смысла в ET-DHP — возьмите IM-GDHP.
- **Слежение ухудшается на быстрых эталонах.** Поднимите `lipschitz_estimate`, чтобы порог был жёстче при высокой скорости изменения состояния.

## Куда дальше

- **[Рецепт 14 — AA-INDI](14_aaindi.md)** — не-НС отказоустойчивая альтернатива.
- **[Рецепт 09 — Отказоустойчивость](09_fault_tolerance.md)** — сравнение в лоб с iADP и AA-INDI.
- **[Документация ET-DHP](../agent/et_dhp.md)** — теория + API.
