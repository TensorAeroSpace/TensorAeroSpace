# Рецепт 06 — Онлайн-адаптивные агенты

Развернуть IHDP, IM-GDHP, ET-DHP, AA-INDI или iADP. У всех пяти один и тот же трёхшаговый жизненный цикл: **warm-start → цикл predict/learn → save/reload**. Этот рецепт даёт скелет и указывает на готовый ноутбук для каждого агента.

**Связано.** [Рецепт 04](04_choosing_agent.md) (дерево решений) · [iADP](../agent/iadp.md) · [AA-INDI](../agent/aa_indi.md) · [IHDP](../agent/ihdp.md) · [IM-GDHP](../agent/imgdhp.md) · [ET-DHP](../agent/et_dhp.md).

## Трёхшаговый жизненный цикл

1. **Warm-start** инкрементной модели (и ядерной матрицы для iADP) из короткой офлайн-идентификации.
2. **Пошаговый онлайн-цикл** — `agent.predict(x, ref, k)` → `env.step(u)` → `agent.learn(x_next, ref, k)`.
3. **Save / reload** в любой момент; сохранённое состояние восстанавливается бит-в-бит ([Рецепт 08](08_huggingface.md)).

**Нет фазы обучения в смысле deep-RL.** «Обучение» — внутри `learn()`, каждый тик.

## Шаг 1 — Warm-start F̃, G̃ через короткое PE-возбуждение

Общая ловушка: все пять агентов требуют нетривиальной оценки `G̃`. При `G̃ ≈ 0`:

- INDI-стиль (AA-INDI): `G_pinv = pinv(0)` взрывается — привод насыщается.
- LQR-стиль (iADP): `γ·G^T P X` обнуляется — управления нет.

**Решение.** 300 тиков мультисинуса, скалярный `G` через МНК:

```python
env_pe = make_env(300); obs, _ = env_pe.reset()
wz_hist, u_hist = [float(obs[1])], [0.0]
for t in range(300):
    u = 2.0*math.sin(2*math.pi*0.7*t*dt) + 1.0*math.sin(2*math.pi*1.5*t*dt)
    obs, *_ = env_pe.step(np.array([u]))
    wz_hist.append(float(obs[1])); u_hist.append(float(u))

dwz, du = np.diff(wz_hist), np.diff(u_hist)
A_pe = np.column_stack([dwz[:-1], du[:-1]])
F_wz, G_wz = np.linalg.lstsq(A_pe, dwz[1:], rcond=None)[0]

F_init = np.array([[F_wz, 0.0], [0.0, 1.0]])   # строка референса стационарна
G_init = np.array([[G_wz], [0.0]])             # руль влияет только на строку wz
```

На нелинейной F-16 должно получиться `F_wz ≈ 1.00, G_wz ≈ -0.0014` (per ° руля, дискретное время).

## Шаг 2 — (только iADP) DARE warm-start для P̃

Самый крупный выигрыш точности после настройки `(Q, R, γ)`:

```python
from scipy.linalg import solve_discrete_are

Q, R, gamma = 30_000.0, 0.1, 0.9
Q_aug = Q * np.array([[1.0, -1.0], [-1.0, 1.0]])
P_init = solve_discrete_are(
    np.sqrt(gamma) * F_init, np.sqrt(gamma) * G_init,
    Q_aug, np.array([[R]]),
)
```

Полный разбор (почему дисконт вносится через подстановку `√γ`, почему это лучше ad-hoc блочной инициализации) — в [примере iADP на F-16](../example/agent/iadp/example_iadp_nonlinear.md).

## Шаг 3 — Цикл

Идентичен для всех пяти агентов:

```python
agent = IADPAgent(n_state=1, n_control=1, config=cfg)   # или AAINDIAgent, IHDPAgent, …
env.reset()

for k in range(n_steps):
    obs = env.get_state()
    u = agent.predict(obs[controlled_channels], ref[:, k], k)
    obs, *_ = env.step(u)
    agent.learn(obs[controlled_channels], ref[:, k], k)
```

`predict(obs, ref, k)` возвращает команду и кэширует состояние для `learn()`. `learn(next_obs, ref, k)` делает шаг RLS и, по страйду, обновление политики.

**Ожидаемое поведение (iADP на нелинейной F-16, 0.12 Гц синусоида):**

![iADP базовое слежение](img/06_iadp_baseline.png)

Измеренная ω_z отслеживает команду в пределах ~12 % от амплитуды (`0.09 °/с` RMSE на `0.8 °/с` синусоиде). Команда руля не выходит за ±0.5°. `G̃` уходит от PE-затравки к локально валидному коэффициенту.

## Шаг 4 — Save / reload

```python
run_dir = agent.save('./checkpoints')
restored = IADPAgent.from_pretrained(run_dir)
agent.publish_to_hub('me/my-iadp', folder_path=run_dir, access_token='hf_…')
```

Детали — в [Рецепте 08](08_huggingface.md).

## Когда что выбирать

| Сценарий | Выбор |
|---|---|
| Резкое повреждение привода в полёте | **AA-INDI** — VFF-RLS сжимает фактор забывания на больших невязках, самое быстрое восстановление. |
| Интерпретируемая LQT-стоимость, мало гиперпараметров | **iADP** — ручки `(Q, R, γ)` + DARE warm-start + soft-update. |
| Нейросетевой актор с двойным критиком `(J, λ)` | **IM-GDHP** — богаче критик, больше вычислений. |
| Событийные обновления для эмбеддед | **ET-DHP** — правило триггера на основе Липшица. |
| Классический ADP | **IHDP** — исторический baseline. |

## Подсказки по тюнингу

- **`gamma_rls`** — ближе к 1 — длиннее память. Старт с `0.995` для чистого sim; `0.9999` для шумного реального мира.
- **`phi_init`** — начальная ковариация RLS. Дефолт `1.0` — безопасно. Меньше → медленная адаптация; больше → шумные первые тики.
- **(iADP) `policy_eval_regularization`** — масштабируется с квадратом величины состояния. Дефолт `1e-4` для O(1); для рад/с — `1e-10`.
- **(iADP) `policy_eval_blend`** — `0.1` убирает пилообразный эффект на трассе руля в каждом такте LS.

## Отказоустойчивость на практике

Онлайн-адаптивные агенты восстанавливаются после изменений объекта **только если вход их возбуждает**. При постоянной уставке нет Δx/Δu для RLS. Смягчите небольшим dither'ом, либо замораживайте RLS (`gamma_rls = 1.0`), когда невязка спокойна.

См. [Рецепт 09](09_fault_tolerance.md) — сравнение iADP и AA-INDI в лоб с настоящей потерей 50 % эффективности руля.

## Куда дальше

- **[Рецепт 07 — Гиперпоиск через Optuna](07_optuna_search.md)** — автоматизация тюнинга.
- **[Рецепт 08 — Save/load/публикация в HuggingFace](08_huggingface.md)** — контракт персистентности в деталях.
- **[Рецепт 09 — Отказоустойчивость](09_fault_tolerance.md)** — сравнение адаптивных агентов на общем профиле отказа.
