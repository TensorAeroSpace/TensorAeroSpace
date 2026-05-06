---
hide:
  - navigation
  - toc
---

<div class="hero">
  <h1>TensorAeroSpace</h1>
  <p class="tagline">Open-source аэрокосмический симулятор + библиотека адаптивного управления. Trim, fly, fail, recover — всё в чистом NumPy.</p>
  <p>
    <a href="guide/installation.md" class="md-button md-button--primary">Установить</a>
    <a href="cookbook/01_hello.md" class="md-button">Quickstart</a>
    <a href="model/b747_nonlinear.md" class="md-button">Модели</a>
    <a href="agent/ihdp.md" class="md-button">Алгоритмы</a>
  </p>
  <p>
    <a href="https://github.com/TensorAeroSpace/TensorAeroSpace"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-TensorAeroSpace-000?logo=github"></a>
    <a href="https://pypi.org/project/tensoraerospace/"><img alt="PyPI" src="https://img.shields.io/pypi/v/tensoraerospace?color=3775A9&logo=pypi&label=PyPI"></a>
    <a href="https://huggingface.co/TensorAeroSpace"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TensorAeroSpace-FFD21E"></a>
    <a href="https://pypi.org/project/tensoraerospace/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/tensoraerospace?logo=python&label=Python"></a>
    <a href="https://pypi.org/project/tensoraerospace/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/tensoraerospace?label=Downloads"></a>
    <a href="https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
    <a href="https://deepwiki.com/TensorAeroSpace/TensorAeroSpace"><img alt="Ask DeepWiki" src="https://deepwiki.com/badge.svg"></a>
  </p>
</div>

<style>
.hero {
  text-align: center;
  margin: 2rem 0 2.5rem 0;
  padding: 2.2rem 1rem;
  background: linear-gradient(120deg, rgba(59,130,246,.18), rgba(59,130,246,0) 50%),
              radial-gradient(60rem 60rem at 10% -20%, rgba(59,130,246,.25), transparent 40%),
              radial-gradient(50rem 50rem at 90% 120%, rgba(59,130,246,.18), transparent 40%),
              linear-gradient(135deg, rgba(59,130,246,.08), rgba(59,130,246,0));
  background-size: 200% 200%, auto, auto, auto;
  animation: gradientShift 12s ease-in-out infinite alternate;
  border-radius: 16px;
}
@keyframes gradientShift {
  0% { background-position: 0% 50%, 0 0, 0 0, 0 0; }
  100% { background-position: 100% 50%, 0 0, 0 0, 0 0; }
}
.hero .tagline {
  font-size: 1.08rem;
  color: var(--md-default-fg-color--light);
  margin-top: .3rem;
}
.hero .md-button { margin: .25rem .25rem; }
.hero a img { vertical-align: middle; margin: 0 .22rem; }
.cards .card-icon { font-size: 1.6rem; }
.stats { text-align: center; margin: 1.5rem 0 0; color: var(--md-default-fg-color--light); }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin: 1.5rem 0; text-align: center; }
.stat-grid .stat { padding: 1rem; border-radius: 12px; background: var(--md-code-bg-color); }
.stat-grid .stat-num { font-size: 2.1rem; font-weight: 600; color: var(--md-primary-fg-color); line-height: 1; }
.stat-grid .stat-lbl { font-size: 0.85rem; margin-top: .35rem; color: var(--md-default-fg-color--light); }
.logos { display: flex; gap: 1.2rem; align-items: center; justify-content: center; flex-wrap: wrap; margin: 1rem 0; }
.logos img { height: 42px; opacity: .9; filter: saturate(0) contrast(1.1); }
</style>

## Зачем TensorAeroSpace?

<div class="grid cards" markdown>

-   :material-airplane-takeoff: **Реальные самолёты, не игрушки**

    Каждая модель транскрибирована из рецензированных источников: NASA CR-2144 (B-747), NASA TM X-1669 (X-15), CEAS Aeronautical Journal 2025 (Skywalker X8), JSBSim (B-737), Roskam Vol VI + class-II UAV литература (RQ-7 Shadow). Trim-точки достигают машинной точности; cross-validated против опубликованных производных.

-   :material-flash: **Ядро — чистый NumPy**

    Никаких проприетарных симуляторов, MATLAB-лицензий и скомпилированных бинарников. 6-DoF Newton-Euler ОДУ на порядок быстрее JSBSim для свипов синтеза управления, и тривиально дифференцируема для adjoint-методов.

-   :material-brain: **Уникальный каталог адаптивного управления**

    Стандартный RL-стек (PPO, SAC, DDPG, DQN, A2C, A3C, GAIL) **плюс** полное семейство incremental-ADP (IHDP, IM-GDHP, ET-DHP, iADP, AA-INDI, AIDI) — редко встречаются в одном open-source пакете. Всё через единый Gymnasium API.

-   :material-shield-check: **Подсистема повреждений из коробки**

    Per-surface потеря эффективности, hard-overs, jam-события, асимметричная тяга при отказе двигателя, override конфигурации flap-jam — всё компонуется в `DamageProfile`. Работает на B-747 и F-16 без дополнительных настроек; хуки паритетны на остальных самолётах.

-   :material-puzzle-outline: **Native Gymnasium**

    Каждая среда реализует стандартный контракт `reset()` / `step()` / `action_space` / `observation_space`. Drop-in в любой Gymnasium / Stable Baselines3 / CleanRL pipeline.

-   :material-test-tube: **Постоянно проверяется**

    Сходимость trim, конвенции знаков отклонения поверхностей, время выгорания топлива — всё зафиксировано 894 unit-тестами с регрессионным покрытием на каждый push.

</div>

---

## 30-секундный старт

=== ":fontawesome-solid-plane: Запустить полётную симуляцию"

    ```python
    import gymnasium as gym
    import tensoraerospace  # регистрирует все envs
    import numpy as np

    env = gym.make("NonlinearB747-v0", trim_at=(20_000.0, 674.0),
                   number_time_steps=2000, dt=0.01)
    obs, _ = env.reset()
    trim_action = np.array([-0.0126, 0.0, 0.0, 0.555])  # rad / [0, 1]
    for _ in range(2000):
        obs, _, _, trunc, _ = env.step(trim_action)
        if trunc: break

    print(f"V={float(np.linalg.norm(obs[:3])):.1f} ft/s, "
          f"alt={-float(obs[11]):.0f} ft")
    # → V=674.0 ft/s, alt=20000 ft  (идеальное удержание trim)
    ```

=== ":material-cog: Классический PID"

    ```python
    import gymnasium as gym
    import numpy as np
    from tensoraerospace.agent.pid import PID
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standard import unit_step

    dt = 0.01
    tp = generate_time_period(tn=10, dt=dt)
    ref = unit_step(degree=5, tp=tp, time_step=2.0,
                    output_rad=True).reshape(1, -1)

    env = gym.make('LinearLongitudinalF16-v0',
                   number_time_steps=len(tp),
                   initial_state=[[0], [0]],
                   reference_signal=ref, use_reward=False)
    pid = PID(env, kp=-14.29, ki=-8.24, kd=-1.30, dt=dt)

    obs, _ = env.reset()
    for t in range(len(tp) - 1):
        u = pid.select_action(ref[0, t], float(obs[0]))
        obs, _, term, trunc, _ = env.step(np.array([[float(u)]],
                                                    dtype=np.float32))
        if term or trunc: break
    ```

=== ":material-brain: Online ADP (IHDP)"

    ```python
    from tensoraerospace.aerospacemodel.b747.nonlinear import trim
    from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env
    from tensoraerospace.agent.ihdp.model import IHDPAgent
    import numpy as np, math

    # Trim на FL200 cruise
    r = trim(altitude_ft=20_000.0, V_ft_s=674.0)
    env = NonlinearB747Env(trim_at=(20_000.0, 674.0),
                            number_time_steps=3000, dt=0.02)

    agent = IHDPAgent(actor_settings={...}, critic_settings={...},
                      incremental_settings={...},
                      tracking_states=["d_theta"],
                      selected_states=["d_theta", "q"],
                      selected_input=["d_elev"],
                      number_time_steps=3000,
                      indices_tracking_states=[0])

    # Single-pass online learning — без offline pre-training
    obs, _ = env.reset()
    # ... запуск rollout ...
    # → Late-half MAE θ = 0.043° на 1° ступеньке (4.3% амплитуды)
    ```

=== ":material-rocket: Pretrained SAC"

    ```bash
    python example/reinforcement_learning/deep_rl/sac-b747-render.py \
        --render --dt 0.1 --tn 200 \
        --repo TensorAeroSpace/sac-b747 --device cuda
    ```

    Или через Python:

    ```python
    from tensoraerospace.agent.sac import SAC
    from tensoraerospace.envs.b747 import ImprovedB747Env

    agent = SAC.from_pretrained("TensorAeroSpace/sac-b747")
    env = ImprovedB747Env(dt=0.1, number_time_steps=200)
    obs, _ = env.reset()
    while True:
        action = agent.select_action(obs, evaluate=True)
        obs, _, term, trunc, _ = env.step(action)
        env.render(mode="human")
        if term or trunc: break
    ```

---

## Библиотека самолётов

У каждой модели есть Gymnasium-среда, trim solver, полная 6-DoF динамика и рецензированные исходные данные.

| Самолёт | Класс | Конфигурации | Источник аэродинамики | Особенность |
|---|---|---|---|---|
| **F-16 Fighting Falcon** | Истребитель | продольный · 6-DoF угловой · damage | NASA / Stevens-Lewis tables | Cubic-spline aero, полная damage-подсистема |
| **Boeing 747-100** | Тяжёлый транспорт | NOMINAL · POWER_APPROACH · LANDING | NASA CR-2144 (Heffley & Jewell) | Помоторная асимметричная тяга + flap jam |
| **Boeing 737-100/800** | Средний транспорт | 737-100 · 737-800 (NG) | JSBSim + Roskam Vol VI | Бенчмарки координированного поворота |
| **X-15** | Гиперзвуковой research | BASIC · A2 record | NASA TM X-1669 + Thompson 2000 | Mach-table 0.4–6.7, XLR99 ракета, переменная масса |
| **Skywalker X8** | Малый UAV (3.4 кг) | flying-wing | CEAS Aeronautical Journal 2025 | Рецензированная flight-test ID |
| **AAI RQ-7 Shadow** | UAV класса II (170 кг) | RQ-7B | Beard & McLain + NASA TM-2014-218686 | V-tail mixed control |
| **Quadrotor** | Мультиротор | nonlinear 6-DoF + damage | Standard quad-X derivation | Per-rotor отказы, X-config allocator |
| **F-4C, ELV, ComSat, GeoSat, LSU, Ultrastick, UAV, Missile** | Linear / improved | разные | Roskam, AIAA conf | Классический state-space + RL-friendly wrapper |

[Смотреть полную галерею моделей →](model/f16.md)

---

## Каталог управления

20 алгоритмов, разбитых по семьям:

=== "Классическое"

    | Алгоритм | Описание |
    |---|---|
    | **PID** | Классический PID с несколькими методами тюнинга |
    | **MPC** | Model-Predictive Control с MLP / NARX / Transformer моделями объекта |

=== "Adaptive Dynamic Programming (offline)"

    Классическая batch-trained ADP-семья:

    | Алгоритм | Notebook |
    |---|---|
    | **HDP** (Heuristic Dynamic Programming) | `acd_hdp_b747.ipynb` |
    | **DHP** (Dual Heuristic Programming) | `acd_dhp_b747.ipynb` |
    | **GDHP** (Globalized Dual HP) | `acd_gdhp_b747.ipynb` |
    | **AD-HDP** | `acd_adhdp_b747.ipynb` |
    | **AD-GDHP** | `acd_adgdhp_b747.ipynb` |
    | **AD-DHP** | `acd_addhp_b747.ipynb` |

=== "Incremental ADP (online)"

    Онлайн single-pass adaptive critic — уникальная часть каталога:

    | Алгоритм | Слежение за командой / крейс | Сценарии повреждения |
    |---|---|---|
    | **IHDP** | F-16 sin-α, B-747 θ-step, B-737 90° turn, quadrotor | failure recovery |
    | **IM-GDHP** | F-16 нелинейный | — |
    | **ET-DHP** | F-16 синусоида | B-747 engine-out (0.28° ψ-error), F-16 damage |
    | **iADP** | F-16 нелинейный | F-16 damage |
    | **AA-INDI** | F-16 нелинейный | — |
    | **AIDI** | — | F-16 damage |

=== "Deep RL"

    Стандартный model-free RL-стек:

    | Алгоритм | Тип | Pretrained checkpoint |
    |---|---|---|
    | **SAC** | Off-policy actor-critic | [HF: TensorAeroSpace/sac-b747](https://huggingface.co/TensorAeroSpace/sac-b747) |
    | **DSAC** | Distributional SAC | step-response & tracking варианты |
    | **PPO** | On-policy clipped objective | 8 самолётов |
    | **DDPG** | Deterministic policy gradient | B-747 |
    | **DQN** | Discrete value iteration | B-747, Unity |
    | **A2C** | Synchronous actor-critic | B-747 + NARX critic |
    | **A3C** | Asynchronous A-C | B-747 |
    | **GAIL** | Generative imitation | F-16 dataset |

---

## Архитектура

```
┌────────────────────────────────────────────────────────────────────────┐
│                        пакет tensoraerospace                           │
├──────────────────┬──────────────────────┬───────────────────────────────┤
│  aerospacemodel  │         envs         │            agent              │
├──────────────────┼──────────────────────┼───────────────────────────────┤
│  pure-NumPy      │   Gymnasium-spec     │   Classical · ADP · Deep RL   │
│  6-DoF dynamics  │   обёртки            │   PID, IHDP, ET-DHP, SAC, ... │
│  Trim solvers    │   "virtual" /        │   Все потребляют Gym env API  │
│  Damage subsys.  │   "normalized"       │                               │
│                  │   action modes       │                               │
└──────────────────┴──────────────────────┴───────────────────────────────┘
                                │
                                ▼
                         Gymnasium / SB3 / CleanRL  ← любой RL pipeline подключается
```

Три пакета слабо связаны. Можно использовать dynamics автономно (без Gym), env без агента (просто обёртка), или построить свой регулятор против любой модели. Единый Gymnasium API на linear / nonlinear / damaged моделях означает, что код регулятора портируется без изменений.

---

## Сценарии использования

=== ":material-account-cog: Инженер по управлению"

    «Нужна точная 6-DoF модель для синтеза регулятора.»

    1. Выберите планер из [галереи моделей](model/f16.md).
    2. Используйте **trim solver** (`trim(altitude, V)`) для нахождения операционной точки.
    3. Линеаризуйте вокруг trim или запустите нелинейную симуляцию напрямую.
    4. Валидируйте против опубликованных cruise / loiter / hypersonic условий в docs.

    Пример: [Boeing 747 нелинейный](model/b747_nonlinear.md), [B-737 координированный поворот](example/agent/ihdp/example_ihdp_nonlinear_b737_turn.md).

=== ":material-school-outline: RL-исследователь"

    «Хочу benchmark нового агента на аэрокосмических задачах.»

    1. Drop-in любая из 20+ envs зарегистрированных через `gym.make("...-v0")`.
    2. Использовать встроенные PID / IHDP / SAC baselines для честного сравнения.
    3. Сравнить метрики с опубликованными [comparison-исследованиями](comparison/all_vs_pid_b747.md).
    4. Запушить обученную модель в Hugging Face через тот же wrapper.

    Пример: [SAC на B-747](example/agent/sac/example-sac-b747.md), [IHDP vs PID на F-16](comparison/ihdp_imgdhp_vs_pid_f16_nonlinear.md).

=== ":material-shield-alert: FTC / отказоустойчивое управление"

    «Исследую реконфигурацию управления после повреждения.»

    1. Используйте [подсистему повреждений](model/aircraft-damage-modeling.md) — surface jam, потеря эффективности, engine flameout, flap jam, отказ RCS.
    2. Декларативно компонуйте `DamageProfile`.
    3. Запустите online ADP-агент (ET-DHP, iADP, AIDI), который адаптируется в реальном времени.

    Пример: [ET-DHP B-747 engine-out удержание курса (0.28°)](example/agent/et_dhp/example_etdhp_b747_engine_failure.md).

=== ":material-book-open: Студент / преподаватель"

    «Изучаю динамику полёта или адаптивное управление.»

    1. Начните с [cookbook](cookbook/01_hello.md) — 16 step-by-step рецептов от "hello world" до FTC под повреждением.
    2. Прочитайте [11 уроков](lesson/base/tutor_1.md) по state-space, controllability, RL fundamentals и hands-on practical (XFLR5 → Simulink → Python).
    3. Откройте любой notebook из [галереи примеров](https://github.com/TensorAeroSpace/TensorAeroSpace/tree/main/example) и запустите локально.

    Пример: [Урок 1 — Введение в State-Space](lesson/base/tutor_1.md), [Cookbook — Online-адаптивные агенты](cookbook/06_online_adaptive.md).

---

## Что нового

<div class="grid cards" markdown>

-   :material-airplane: **AAI RQ-7 Shadow нелинейный**

    Тактический UAV класса II, 170 кг, V-tail mixed convention, 4-канальное управление. Синтезирован из Beard & McLain + NASA TM-2014-218686 + Roskam Vol VI.

    [:octicons-arrow-right-24: model/aai_shadow_nonlinear](model/aai_shadow_nonlinear.md)

-   :material-airplane-takeoff: **Boeing 737 нелинейный**

    737-100 / 737-800 с JSBSim аэродинамикой, JT8D / CFM56-7B engine models, машинно-точный trim, MIMO IHDP пример координированного поворота.

    [:octicons-arrow-right-24: model/b737_nonlinear](model/b737_nonlinear.md)

-   :material-rocket-launch: **X-15 hypersonic**

    M = 0.4 → 6.7, ракетный двигатель XLR99, переменная масса, 13-state vector с propellant-каналом. Время выгорания 79.8 с — соответствует Thompson 2000 с точностью 0.2 %.

    [:octicons-arrow-right-24: model/x15_nonlinear](model/x15_nonlinear.md)

-   :material-quadcopter: **Skywalker X8 малый UAV**

    Рецензированная flight-test идентификация (CEAS Aeronautical Journal 2025). 3.4 кг flying-wing, 3-канальное управление, propeller-airframe drag coupling.

    [:octicons-arrow-right-24: model/skywalker_x8_nonlinear](model/skywalker_x8_nonlinear.md)

-   :material-target: **B-747 damage subsystem v2**

    Помоторная асимметричная тяга + flap-jam configuration override + 5 готовых пресетов включая `LEFT_OUTER_ENGINE_FAILURE`.

    [:octicons-arrow-right-24: model/b747_nonlinear](model/b747_nonlinear.md#подсистема-повреждений)

-   :material-format-list-bulleted-square: **Реструктуризация `example/`**

    Top-level сгруппирован по классу регулятора; `reinforcement_learning/` разбит на `incremental_adp/` и `deep_rl/`. 101 notebook, все пути обновлены в docs.

    [:octicons-arrow-right-24: example README](https://github.com/TensorAeroSpace/TensorAeroSpace/tree/main/example)

</div>

---

## Featured-метрики

Реальные числа из исполняемых example-notebook:

| Сценарий | Результат | Источник |
|---|---|---|
| **B-747 ET-DHP heading hold под отказом двигателя** | ψ-error **0.28°** vs open-loop −85.5° | [notebook](example/agent/et_dhp/example_etdhp_b747_engine_failure.md) |
| **B-737 MIMO IHDP координированный поворот 90°** | финальная ψ-error **0.98°**, max сайдслип **0.11°** | [notebook](example/agent/ihdp/example_ihdp_nonlinear_b737_turn.md) |
| **B-747 IHDP ступенька θ (1°)** | late-half MAE **0.043°** | [notebook](example/agent/ihdp/example_ihdp_nonlinear_b747.md) |
| **X-15 boost-burnout (full throttle)** | выгорание **79.8 с** vs Thompson 2000: 80 с | [model](model/x15_nonlinear.md) |
| **Skywalker X8 cruise trim** | машинная точность (residual 1e-15) | [model](model/skywalker_x8_nonlinear.md) |

---

## Установка

=== "pip"

    ```bash
    pip install tensoraerospace
    ```

=== "poetry"

    ```bash
    poetry add tensoraerospace
    ```

=== "из исходников"

    ```bash
    git clone https://github.com/TensorAeroSpace/TensorAeroSpace.git
    cd TensorAeroSpace
    poetry install
    ```

=== "Docker"

    ```bash
    docker run --rm -p 8888:8888 ghcr.io/tensoraerospace/tas-jupyter
    ```

Python 3.10–3.12, MATLAB не требуется, нет проприетарных code paths.

---

## Ресурсы

| | |
|---|---|
| 📦 **Пакет** | [PyPI](https://pypi.org/project/tensoraerospace/) · [GitHub](https://github.com/TensorAeroSpace/TensorAeroSpace) · [Hugging Face](https://huggingface.co/TensorAeroSpace) |
| 📚 **Документация** | [Модели](model/f16.md) · [Алгоритмы](agent/sac.md) · [Cookbook](cookbook/01_hello.md) · [Уроки](lesson/base/tutor_1.md) |
| 🧪 **Примеры** | [101 notebook на GitHub](https://github.com/TensorAeroSpace/TensorAeroSpace/tree/main/example) |
| 📊 **Бенчмарки** | [Comparison-исследования](comparison/all_vs_pid_b747.md) · [Метрики](benchmark/metrics.md) |
| 💬 **Сообщество** | [Issues](https://github.com/TensorAeroSpace/TensorAeroSpace/issues) · [DeepWiki Q&A](https://deepwiki.com/TensorAeroSpace/TensorAeroSpace) |

---

## Цитирование

Если вы используете TensorAeroSpace в исследовании, пожалуйста, сошлитесь:

```bibtex
@software{tensoraerospace,
  title  = {TensorAeroSpace: An Open-Source Aerospace Simulation and Adaptive Control Toolkit},
  author = {Mazaev, A. and contributors},
  year   = {2026},
  url    = {https://github.com/TensorAeroSpace/TensorAeroSpace},
  note   = {Pure-NumPy 6-DoF aerospace dynamics + Gymnasium envs + classical/adaptive/deep RL agents}
}
```

Для базовых аэродинамических источников — NASA CR-2144 (B-747), NASA TM X-1669 (X-15), CEAS Aeronautical Journal 2025 (Skywalker X8), JSBSim (B-737), Beard & McLain (Aerosonde / Shadow), Roskam Vol VI — пожалуйста, также сошлитесь на оригинальные references, перечисленные на странице каждой модели.

---

<div style="text-align:center; margin: 1.6rem 0 0.5rem;">
  <a href="guide/installation.md" class="md-button md-button--primary">Начать</a>
  <a href="cookbook/01_hello.md" class="md-button">Открыть cookbook</a>
  <a href="https://github.com/TensorAeroSpace/TensorAeroSpace" class="md-button">Star на GitHub</a>
</div>

<p style="text-align:center; color: var(--md-default-fg-color--light); margin-top: 1rem;">
MIT licensed · построено на NumPy, PyTorch, Gymnasium · powered by aerospace research community.
</p>
