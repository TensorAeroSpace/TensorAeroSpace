# Incremental Approximate Dynamic Programming (iADP)

iADP — **онлайн-адаптивный закон управления полётом** на основе обучения с подкреплением. Алгоритм объединяет три блока:

- **онлайн-идентификация инкрементальной модели** методом рекурсивного МНК (RLS),
- **квадратичная аппроксимация функции ценности** \(V_\pi(X_t) = X_t^T \tilde{P} X_t\), подбираемая пакетным МНК по невязкам уравнения Беллмана,
- **аналитический шаг улучшения политики**, вытекающий из принципа оптимальности Беллмана.

Поскольку модель идентифицируется онлайн, регулятор при работе **не требует априорной модели** объекта управления и автоматически подстраивается под смену конфигурации воздушного судна, деградацию приводов и немоделируемую аэродинамику. Проверен в лётных испытаниях на Cessna Citation II (PH-LAB) в TU Delft / DLR. См. также нелинейную модель F-16: [NonlinearLongitudinalF16](../model/f16_nonlinear_longitudinal.md).

**Источник**: Konatala, Milz, Weiser, Looye, van Kampen, *"Flight Testing Reinforcement Learning based Online Adaptive Flight Control Laws on CS-25 Class Aircraft"*, AIAA SCITECH 2024, [DOI 10.2514/6.2024-2402](https://doi.org/10.2514/6.2024-2402).

## Ключевые идеи

- **Инкрементальная модель.** Линеаризация нелинейного объекта вокруг последней рабочей точки даёт \(\Delta X_{t+1} \approx \tilde{F}_t \Delta X_t + \tilde{G}_t \Delta\delta_t\). Стек параметров \(\tilde{\Theta}_t = [\tilde{F}_t; \tilde{G}_t]^T\) и регрессоров \(W_t = [\Delta X_t; \Delta \delta_t]\) даёт стандартную рекурсию RLS с постоянным забыванием.
- **Квадратичная функция ценности.** В духе линейно-квадратичного трекинга (LQT) полагаем \(V_\pi(X) = X^T \tilde{P} X\) на дополненном состоянии \(X_t = [x_t; x_t^r]\) (система + референсные состояния). Это обобщает оценку ценности с наблюдённых траекторий на всё достижимое множество.
- **Пакетный МНК для политики оценки.** На скользящем окне каждое наблюдение даёт одно скалярное уравнение Беллмана \((X_t \otimes X_t)^T \mathrm{vec}(\tilde{P}^{j+1}) = c_t + \gamma X_{t+1}^T \tilde{P}^{j} X_{t+1}\); стекуя их и решая МНК, получаем новую матрицу \(\tilde{P}\). После симметризации опционально проектируется в конус положительно определённых матриц.
- **Аналитическое улучшение политики.** Минимизация правой части уравнения Беллмана по \(\Delta \delta_t\) даёт уравнение (11) из статьи — без нейросети, без градиентного спуска: \(\Delta \delta_t = -(R + \gamma \tilde{G}^T \tilde{P} \tilde{G})^{-1} [R \delta_{t-1} + \gamma \tilde{G}^T \tilde{P} X_t + \gamma \tilde{G}^T \tilde{P} \tilde{F} \Delta X_t]\).

## Отличия от близких методов

| Аспект | IHDP | IM-GDHP | ET-DHP | **iADP** |
| --- | --- | --- | --- | --- |
| Модель | Онлайн-МНК по градиентам актёра/критика | Онлайн-RLS (dual) | Онлайн-RLS (событийно) | **Онлайн-RLS** (eq. (9)) |
| Критик | Нейросеть | Две головы \(J, \lambda\) | \(\lambda\)-критик | **Параметрическая квадратичная** \(X^T P X\) |
| Актёр | НС + градиентные апдейты | НС + GDHP-градиенты | НС + событийные апдейты | **Аналитический** (eq. (11)) |
| Расписание апдейтов | Каждый шаг | Каждый шаг | Событийно | Модель 1 кГц / политика 20 Гц |
| Число гиперпараметров | Много (слои, lr, …) | Много | Много | Мало: \(Q\), \(R\), \(\gamma\), \(\gamma_{RLS}\) |

Главные плюсы iADP — **интерпретируемость**: LQT-функция стоимости, положительно определённая матрица \(\tilde{P}\) и аналитическая политика. Нет тренировочного датасета, replay-буфера и стохастического градиентного спуска.

## Состав iADP

| Компонент | Роль | Реализация |
| --- | --- | --- |
| IncrementalRLS | Онлайн-идентификация \(\tilde{\Theta} = [\tilde{F}; \tilde{G}]^T\) с постоянным забыванием | `tensoraerospace.agent.iadp.IncrementalRLS` |
| Policy-evaluation LS | Пакетный МНК по скользящему окну для \(\tilde{P}\) | `IADPAgent._policy_evaluation` |
| Policy improvement | Аналитическое уравнение (11) | `IADPAgent._compute_policy_increment` |
| IADPAgent | Оркестрирует идентификацию, оценку, улучшение | `tensoraerospace.agent.iadp.IADPAgent` |

## Алгоритм

На каждом шаге управления \(k\), при измерении \(x_k\) и референсе \(r_k\):

1. **Дополненное состояние.**
\[
X_k = \begin{pmatrix} x_k \\ r_k \end{pmatrix}, \qquad \Delta X_k = X_k - X_{k-1}.
\]

2. **Улучшение политики (eq. (11)).** По текущим оценкам \(\tilde{F}, \tilde{G}, \tilde{P}\):
\[
\Delta \delta_k = -\bigl(R + \gamma \tilde{G}^T \tilde{P} \tilde{G}\bigr)^{-1}
\bigl[R \delta_{k-1} + \gamma \tilde{G}^T \tilde{P} X_k + \gamma \tilde{G}^T \tilde{P} \tilde{F} \Delta X_k\bigr].
\]
Ограничение по скорости \(\pm \dot{u}_{\max} \cdot dt\), затем \(\delta_k = \mathrm{clip}(\delta_{k-1} + \Delta \delta_k, \pm u_{\max})\).

3. **Применить к объекту и наблюдать** \(x_{k+1}\).

4. **Обновление модели (RLS).** Регрессор \(W = [\Delta X_{k-1}; \Delta \delta_{k-1}]\), цель \(\Delta X_k\). Шаг:
\[
\varepsilon = \Delta X_k - \tilde{\Theta}^T W, \quad
K = \frac{\Phi W}{\gamma_{RLS} + W^T \Phi W}, \quad
\tilde{\Theta} \leftarrow \tilde{\Theta} + K \varepsilon^T, \quad
\Phi \leftarrow \tfrac{1}{\gamma_{RLS}}(\Phi - K W^T \Phi).
\]

5. **Накопление стоимости.** Добавить \((X_k, X_{k+1}, c_k)\) в окно,
\(c_k = (x_k - r_k)^T Q (x_k - r_k) + \delta_k^T R \delta_k\).

6. **Оценка политики (каждые `policy_eval_every` шагов).** С текущей \(\tilde{P}^j\) решить ридж-МНК
\[
A \mathrm{vec}(\tilde{P}^{j+1}) = b, \quad
A_i = \mathrm{vec}(X_i X_i^T)^T, \quad
b_i = c_i + \gamma X_{i+1}^T \tilde{P}^j X_{i+1}.
\]
Симметризовать и опционально спроектировать на множество ПО-матриц.

## Быстрый старт

```python
import numpy as np
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig

# Warm-start из бортовой линеаризации (в точке трима).
F_init = np.eye(2)
G_init = np.array([[-0.5], [0.0]])

cfg = IADPConfig(
    dt=0.01,
    Q=np.array([[10.0]]),
    R=np.array([[0.1]]),
    gamma=0.8,
    gamma_rls=0.995,
    policy_eval_window=200,
    policy_eval_every=50,                       # ≈ 20 Гц при dt = 0.01 с
    policy_eval_warmup_updates=30,
    F_init=F_init, G_init=G_init,
    u_magnitude_limit=15.0,
    u_rate_limit=60.0,
    seed=0,
)
agent = IADPAgent(n_state=1, n_control=1, config=cfg)

x = np.zeros(1)
for k in range(2000):
    ref = np.array([0.1 if k > 100 else 0.0])
    u = agent.predict(x, ref, k)
    # Подставьте шаг вашей среды здесь.
    x = x + cfg.dt * (-0.5 * u - 2.0 * x)
    agent.learn(x, ref, k)
```

!!! tip "Warm-start `G_init` критичен"
    Уравнение (11) требует разумной оценки \(\tilde{G}\) на первых шагах, иначе \((R + \gamma \tilde{G}^T \tilde{P} \tilde{G}) \approx R\) даёт лишь отклик по регуляризации управления. Возьмите `G_init` из линеаризации бортовой модели — онлайн-RLS уточнит его.

!!! tip "Warm-start `P_init` по DARE для быстрой сходимости"
    Онлайн-МНК подбирает \(\tilde{P}\) на окнах переходов, но смещение конечного окна оставляет политику на несколько процентов ниже оптимума. Засейте `P_init` аналитическим LQT-DARE, вычисленным по warm-start модели:

    ```python
    from scipy.linalg import solve_discrete_are
    Q_aug = Q_val * np.block([[np.eye(n_state), -np.eye(n_state)],
                              [-np.eye(n_state), np.eye(n_state)]])
    P_init = solve_discrete_are(
        np.sqrt(gamma) * F_init, np.sqrt(gamma) * G_init,
        Q_aug, R,
    )
    ```

    Дисконт заходит через \(\bar{F} = \sqrt{\gamma}F,\, \bar{G} = \sqrt{\gamma}G\), поэтому `solve_discrete_are` применяется напрямую. См. [пример на нелинейной F-16](../example/agent/iadp/example_iadp_nonlinear.md).

!!! note "Sequential vs Continuous learning"
    Установите `model_learning_only_steps > 0` вместе с `excitation_signal`, чтобы воспроизвести **Sequential Learning Approach** из статьи: первые *N* шагов запускаются в открытом контуре с пользовательским мультисинусоидом, пока RLS не сойдётся. По умолчанию обе фазы идут одновременно (**Continuous Learning Approach**).

## Гиперпараметры

### Стоимость и дисконт

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `Q` | `I` | Весовая матрица ошибки слежения, форма `(n_state, n_state)` |
| `R` | `I` | Весовая матрица управления, форма `(n_control, n_control)` |
| `gamma` | 0.8 | Коэффициент дисконтирования Беллмана \(\gamma \in (0, 1)\) |

### RLS-идентификатор

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `gamma_rls` | 0.995 | Постоянный фактор забывания, ближе к 1 ⇒ больше память |
| `phi_init` | 1e2 | Начальный масштаб ковариационной матрицы |
| `F_init` | None | Warm-start \(\tilde{F}\), форма `(n_aug, n_aug)` |
| `G_init` | None | Warm-start \(\tilde{G}\), форма `(n_aug, n_control)` |

### Оценка политики

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `policy_eval_window` | 200 | Размер скользящего окна для пакетного МНК |
| `policy_eval_every` | 50 | Шаг между МНК-апдейтами (в вызовах `learn()`) |
| `policy_eval_iterations` | 1 | Внутренние итерации фиксированной точки на апдейт |
| `policy_eval_regularization` | 1e-4 | Ридж-регуляризация нормальных уравнений |
| `policy_eval_warmup_updates` | 20 | Число RLS-обновлений до первого МНК |
| `enforce_psd` | True | Обрезать собственные числа \(\tilde{P}\) для сохранения ПО-свойства |
| `psd_floor` | 1e-6 | Нижняя граница собственного числа |
| `policy_eval_blend` | 1.0 | EMA-коэффициент для обновлений \(\tilde{P}\). `1.0` — полная замена; малые значения (0.05–0.2) — плавное обновление \(\tilde{P}\) как у target-сети; устраняет пилообразный эффект на трассе управления, возникающий иначе на каждом тике МНК. |
| `P_init` | `I` | Warm-start ядерной матрицы |

### Ограничения привода

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `dt` | 0.01 | Шаг управления (с) |
| `u_magnitude_limit` | 25.0 | Жёсткое ограничение по амплитуде на канал |
| `u_rate_limit` | 60.0 | Макс. \(|\Delta\delta|\) в секунду на канал |
| `pinv_rcond` | 1e-8 | Порог для `np.linalg.pinv` при обращении матрицы политики |

### Фаза sequential-learning

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `model_learning_only_steps` | 0 | Длина окна открытого контура для идентификации |
| `excitation_signal` | None | Опциональное расписание \((T, n_\text{control})\) абсолютных значений управления |

## Поддерживаемые окружения

- Любые Gymnasium-среды, в которых наблюдаемый вектор содержит управляемые состояния \(x_t\), а пространство действий — непрерывное \(\delta_t\). Дополненное состояние \(X_t = [x_t; x_t^r]\) строится внутри агента — референс передаётся рядом с наблюдением в `predict` / `learn`.
- Типовые цели: `NonlinearLongitudinalF16-v0` (трекинг тангажной угловой скорости или угла тангажа через руль высоты), внутренние контуры 6-DoF по угловым скоростям.

## Сохранение/загрузка

```python
run_dir = agent.save("./checkpoints")           # создаёт <date>_IADPAgent/
restored = IADPAgent.from_pretrained(run_dir)
agent.publish_to_hub("me/my-iadp", folder_path=run_dir, access_token="hf_...")
```

Сохраняемые артефакты:

- `config.json` — полный `IADPConfig` + `n_state` / `n_control`, все массивы — как списки.
- `rls.npz` — параметрическая матрица RLS `theta`, ковариация `Phi`, счётчик обновлений, последняя невязка.
- `value.npz` — текущая ядерная матрица \(\tilde{P}\).
- `weights.npz` — активные `Q` и `R` (включая значения, подставленные по умолчанию).
- `loop_state.npz` — \(X_{t-1}\), последнее управление, последнее приращение, кэшированное \(\Delta X\), счётчик шагов.
- `window.npz` — буфер переходов для policy-evaluator, сохранённый стеком массивов, — перезагрузка бит-в-бит.

## Документация API

::: tensoraerospace.agent.iadp.model.IADPAgent

::: tensoraerospace.agent.iadp.model.IADPConfig

::: tensoraerospace.agent.iadp.rls.IncrementalRLS

## Источники

- Konatala, Milz, Weiser, Looye, van Kampen. *"Flight Testing Reinforcement Learning based Online Adaptive Flight Control Laws on CS-25 Class Aircraft"*, AIAA SCITECH 2024, [DOI 10.2514/6.2024-2402](https://doi.org/10.2514/6.2024-2402).
- Sieberling, Chu, Mulder. *"Robust Flight Control Using Incremental Nonlinear Dynamic Inversion and Angular Acceleration Prediction"*, J. Guid. Control Dyn., 2010.
- Lewis, Vrabie, Syrmos. *"Optimal Control"*, Wiley, 2012 — теория LQT, лежащая в основе квадратичной функции ценности.
