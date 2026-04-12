# Урок 10 -- Обучение первого RL-агента для управления аэрокосмическими системами

## 1. Цель

В этом уроке мы пройдем полный рабочий процесс от начала до конца:

1. Настроим среду Boeing 747 для задачи отслеживания тангажа.
2. Создадим базовый **ПИД-регулятор** с автоматической настройкой.
3. Обучим агент **PPO** (Proximal Policy Optimization).
4. Обучим агент **SAC** (Soft Actor-Critic).
5. Оценим все три регулятора и сравним их по стандартным метрикам качества.
6. Сохраним, загрузим и визуализируем результаты.

По окончании урока у вас будет рабочий конвейер обучения RL-агентов, который
можно адаптировать к любой среде TensorAeroSpace.

---

## 2. Требования

| Требование | Версия |
|---|---|
| Python | >= 3.10 |
| PyTorch | >= 2.0 |
| TensorAeroSpace | последняя |
| matplotlib | любая |
| numpy | любая |

Установка:

```bash
pip install tensoraerospace matplotlib
```

---

## 3. Шаг 1 -- Настройка среды

Мы используем `ImprovedB747Env` -- нормализованную среду продольного канала
Boeing 747 с композитной функцией вознаграждения, разработанной для обучения
с подкреплением.

Основные свойства:

* **Пространство наблюдений**: нормализованный вектор `[ошибка тангажа, угловая скорость, тангаж, предыдущее действие]`, все в `[-1, 1]`.
* **Пространство действий**: скалярное отклонение руля высоты в `[-1, 1]`.
* **Вектор состояния**: `[u, w, q, theta]` -- продольная скорость, вертикальная скорость, угловая скорость тангажа, угол тангажа.
* **Вознаграждение**: композитный сигнал, штрафующий за ошибку слежения, угловую скорость, управляющее воздействие, дрожание и (в режиме `step_response`) перерегулирование и колебания.

```python
import numpy as np
import torch
import matplotlib.pyplot as plt

from tensoraerospace.envs import ImprovedB747Env
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp

# --- Временная база ---
dt = 0.1                                         # частота дискретизации 10 Гц
tp = generate_time_period(tn=40, dt=dt)           # 40 секунд моделирования
tps = convert_tp_to_sec_tp(tp, dt=dt)             # ось времени в секундах
number_time_steps = len(tp)

# --- Задающий сигнал: ступенька 1 градус в момент t = 5 с ---
reference = np.reshape(
    unit_step(degree=1, tp=tp, time_step=50, output_rad=True),
    (1, -1),
)

# --- Начальное состояние [u, w, q, theta] (все нули) ---
initial_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# --- Создание среды ---
env = ImprovedB747Env(
    initial_state=initial_state,
    reference_signal=reference,
    number_time_steps=number_time_steps,
    dt=dt,
)

obs, info = env.reset()
print(f"Размер наблюдения  : {np.array(obs).shape}")
print(f"Пространство действий   : {env.action_space}")
print(f"Пространство наблюдений : {env.observation_space}")
```

### Что означают параметры

| Параметр | Описание |
|---|---|
| `degree=1` | Амплитуда ступеньки в градусах (преобразуется в радианы при `output_rad=True`). |
| `time_step=50` | Индекс начала ступеньки (50 * dt = 5 с). |
| `initial_state` | Самолет начинает в балансировочном режиме: нулевые отклонения. |
| `dt=0.1` | Шаг дискретизации; 0.1 с -- хороший баланс между разрешением и скоростью. |

---

## 4. Шаг 2 -- Базовый ПИД-регулятор

Перед обучением нейросетевых регуляторов необходимо установить базовый уровень.
`tensoraerospace.agent.pid.PID` предоставляет классический ПИД-регулятор с
автоматической настройкой в стиле MATLAB.

```python
from tensoraerospace.agent.pid import PID

# Создание ПИД-регулятора
pid = PID(env=env, kp=1.0, ki=1.0, kd=0.5, dt=dt)

# Автоматическая настройка через оптимизацию по модели пространства состояний
# track_state_idx=3 означает отслеживание theta (индекс 3 в [u, w, q, theta])
tune_result = pid.tune_matlab_style(
    track_state_idx=3,
    target_settling_time=5.0,   # целевое время переходного процесса 5 с
    target_overshoot=5.0,       # целевое перерегулирование 5%
    n_iterations=100,
    verbose=True,
)
print(tune_result)
```

### Запуск эпизода с ПИД

```python
pid.reset()
obs_pid, info_pid = env.reset()
done = False
pid_states = []
pid_rewards = []
step_idx = 0

while not done:
    # Задающее значение на текущем шаге (в радианах)
    ref_val = float(reference[0, min(step_idx, reference.shape[1] - 1)])

    # Наблюдение среды содержит нормализованный тангаж;
    # для ПИД нам нужен необработанный theta из модели.
    raw_theta = float(env.unwrapped.model.get_output()[3])

    # Вычисление управляющего воздействия ПИД (возвращает float)
    action_pid = pid.select_action(setpoint=ref_val, measurement=raw_theta)

    # Нормализация действия в [-1, 1] для ImprovedB747Env
    max_ele_deg = env.unwrapped.max_stabilizer_angle_deg
    action_norm = np.clip(
        np.array([action_pid / np.deg2rad(max_ele_deg)], dtype=np.float32),
        -1.0, 1.0,
    )

    obs_pid, reward, terminated, truncated, info_pid = env.step(action_norm)
    done = terminated or truncated
    pid_states.append(raw_theta)
    pid_rewards.append(reward)
    step_idx += 1

pid_response = np.array(pid_states)
print(f"Суммарное вознаграждение ПИД: {sum(pid_rewards):.2f}")
```

---

## 5. Шаг 3 -- Обучение PPO

PPO -- это алгоритм обучения с подкреплением по политике (on-policy), который
собирает пакет траекторий, вычисляет преимущества с помощью GAE и обновляет
политику с обрезанной суррогатной функцией потерь. Это надежный первый выбор
для непрерывного управления.

### 5.1 Создание агента

```python
from tensoraerospace.agent.ppo.model import PPO

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

ppo_agent = PPO(
    env=env,
    gamma=0.99,              # коэффициент дисконтирования -- высокое значение для длинного горизонта
    max_episodes=50,         # общее количество эпизодов обучения (увеличьте до 200+ для лучших результатов)
    rollout_len=2048,        # шаги, собираемые перед каждым обновлением политики
    clip_pram=0.2,           # параметр обрезки PPO (epsilon)
    num_epochs=10,           # проходы SGD по каждому роллауту
    batch_size=64,           # размер мини-пакета
    entropy_coef=0.01,       # бонус энтропии -- поддерживает исследование
    actor_lr=3e-4,           # скорость обучения актора
    critic_lr=1e-3,          # скорость обучения критика
    gae_lambda=0.95,         # GAE lambda для снижения дисперсии
    actor_hidden_dim=256,    # размер скрытого слоя сети актора
    critic_hidden_dim=256,   # размер скрытого слоя сети критика
    normalize_obs=True,      # нормализация наблюдений по скользящему среднему
    seed=42,
    device=device,
)
```

### 5.2 Описание ключевых гиперпараметров

| Параметр | Назначение | Типичный диапазон |
|---|---|---|
| `gamma` | Коэффициент дисконтирования. Более высокие значения заставляют агента больше заботиться о будущих наградах. | 0.95 -- 0.999 |
| `clip_pram` | Ограничивает величину обновления политики за один шаг. | 0.1 -- 0.3 |
| `rollout_len` | Количество шагов среды перед обновлением. Более длинные роллауты дают более разнообразные данные, но замедляют обучение. | 512 -- 4096 |
| `num_epochs` | Сколько раз алгоритм итерирует по собранному роллауту. | 3 -- 30 |
| `entropy_coef` | Бонус за энтропию политики. Предотвращает преждевременную сходимость, но слишком высокое значение ведет к случайному поведению. | 0.001 -- 0.05 |
| `actor_lr` / `critic_lr` | Скорости обучения. Критик часто обучается с более высокой скоростью. | 1e-4 -- 3e-3 |
| `gae_lambda` | Компромисс смещение-дисперсия в оценке преимущества. 1.0 = Монте-Карло, 0.0 = 1-шаговый TD. | 0.9 -- 0.99 |

### 5.3 Запуск обучения

```python
print("Начинаем обучение PPO ...")
ppo_agent.train()
print("Обучение PPO завершено.")
```

Прогресс обучения автоматически записывается в TensorBoard. Запустите
панель мониторинга:

```bash
tensorboard --logdir runs
```

---

## 6. Шаг 4 -- Обучение SAC

SAC -- это алгоритм обучения вне политики (off-policy), который максимизирует
компромисс между ожидаемой отдачей и энтропией. Он использует буфер воспроизведения,
двойные Q-сети и мягкие обновления целевой сети. Для многих задач непрерывного
управления SAC более эффективен по числу взаимодействий, чем PPO.

### 6.1 Создание агента

```python
from tensoraerospace.agent.sac import SAC

# Пересоздание среды для SAC (те же параметры)
env_sac = ImprovedB747Env(
    initial_state=initial_state,
    reference_signal=reference,
    number_time_steps=number_time_steps,
    dt=dt,
)

sac_agent = SAC(
    env=env_sac,
    hidden_size=256,                   # ширина сетей критика и политики
    lr=3e-4,                           # скорость обучения критика
    policy_lr=3e-4,                    # скорость обучения политики
    gamma=0.99,                        # коэффициент дисконтирования
    tau=0.005,                         # коэффициент мягкого обновления целевой сети
    alpha=0.2,                         # начальный коэффициент энтропии
    automatic_entropy_tuning=True,     # автоматическая настройка alpha
    batch_size=64,                     # размер мини-пакета для выборки из буфера
    memory_capacity=100000,            # емкость буфера воспроизведения
    updates_per_step=1,                # шаги градиента на каждый шаг среды
    policy_type="Gaussian",            # стохастическая гауссова политика
    seed=42,
    device=device,
)
```

### 6.2 Ключевые концепции

* **Буфер воспроизведения** (`memory_capacity`): Хранит прошлые переходы `(s, a, r, s', done)`.
  Обучение вне политики повторно использует старые данные, повышая эффективность.
* **Мягкие обновления** (`tau`): Целевая Q-сеть медленно подтягивается к
  онлайн-сети: `theta_target = tau * theta + (1 - tau) * theta_target`.
* **Автоматическая настройка энтропии**: SAC подстраивает коэффициент энтропии `alpha`
  так, чтобы политика поддерживала целевой уровень энтропии. Это убирает один
  гиперпараметр.
* **Гауссова политика**: Актор выдает среднее и логарифм стандартного отклонения
  сжатого гауссова распределения, гарантируя, что действия остаются в допустимых
  пределах.

### 6.3 Запуск обучения

```python
print("Начинаем обучение SAC ...")
sac_agent.train(num_episodes=300)
print("Обучение SAC завершено.")
```

---

## 7. Шаг 5 -- Оценка

После обучения мы оцениваем каждого агента в детерминированном режиме
(без шума исследования) и собираем траектории для сравнения.

### 7.1 Вспомогательные функции

```python
def evaluate_agent_sac(agent, env, reference_signal, label="Агент"):
    """Запуск одного оценочного эпизода с агентом SAC.

    Возвращает:
        response (np.ndarray): Записанный угол тангажа на каждом шаге.
        total_reward (float): Суммарное вознаграждение за эпизод.
    """
    obs, _ = env.reset()
    done = False
    states = []
    total_reward = 0.0
    step_idx = 0

    while not done:
        action = agent.select_action(obs, evaluate=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        raw_theta = float(env.unwrapped.model.get_output()[3])
        states.append(raw_theta)
        total_reward += float(reward)
        step_idx += 1

    print(f"{label} -- вознаграждение: {total_reward:.2f}, шаги: {step_idx}")
    return np.array(states), total_reward


def evaluate_agent_ppo(agent, env, reference_signal, label="Агент"):
    """Запуск одного оценочного эпизода с агентом PPO.

    Возвращает:
        response (np.ndarray): Записанный угол тангажа на каждом шаге.
        total_reward (float): Суммарное вознаграждение за эпизод.
    """
    obs, _ = env.reset()
    done = False
    states = []
    total_reward = 0.0
    step_idx = 0

    while not done:
        # PPO использует act(), возвращающий (action_tensor, mean_action, log_prob)
        _, mean_action, _ = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(mean_action[0])
        done = terminated or truncated
        raw_theta = float(env.unwrapped.model.get_output()[3])
        states.append(raw_theta)
        total_reward += float(reward)
        step_idx += 1

    print(f"{label} -- вознаграждение: {total_reward:.2f}, шаги: {step_idx}")
    return np.array(states), total_reward
```

### 7.2 Запуск оценки

```python
# Создание чистой среды для каждого оценочного прогона
def make_env():
    return ImprovedB747Env(
        initial_state=initial_state,
        reference_signal=reference,
        number_time_steps=number_time_steps,
        dt=dt,
    )

# Оценка PPO
env_eval_ppo = make_env()
ppo_response, ppo_reward = evaluate_agent_ppo(
    ppo_agent, env_eval_ppo, reference, label="PPO"
)

# Оценка SAC
env_eval_sac = make_env()
sac_response, sac_reward = evaluate_agent_sac(
    sac_agent, env_eval_sac, reference, label="SAC"
)
```

---

## 8. Шаг 6 -- Бенчмаркинг и сравнение

TensorAeroSpace предоставляет специализированные функции для оценки качества
управления.

### 8.1 Вычисление метрик

```python
from tensoraerospace.benchmark.function import (
    overshoot,
    settling_time,
    static_error,
)

# Формируем массив задающего сигнала в градусах (той же длины, что и отклик)
n_pid = len(pid_response)
n_ppo = len(ppo_response)
n_sac = len(sac_response)

ref_deg = np.rad2deg(reference[0])  # полный задающий сигнал в градусах

ref_pid = ref_deg[:n_pid]
ref_ppo = ref_deg[:n_ppo]
ref_sac = ref_deg[:n_sac]

pid_deg = np.rad2deg(pid_response)
ppo_deg = np.rad2deg(ppo_response)
sac_deg = np.rad2deg(sac_response)

# Перерегулирование (%)
os_pid = overshoot(ref_pid, pid_deg)
os_ppo = overshoot(ref_ppo, ppo_deg)
os_sac = overshoot(ref_sac, sac_deg)

# Время переходного процесса (индекс -> умножаем на dt для получения секунд)
st_pid = settling_time(ref_pid, pid_deg)
st_ppo = settling_time(ref_ppo, ppo_deg)
st_sac = settling_time(ref_sac, sac_deg)

# Статическая ошибка (градусы)
se_pid = static_error(ref_pid, pid_deg)
se_ppo = static_error(ref_ppo, ppo_deg)
se_sac = static_error(ref_sac, sac_deg)
```

### 8.2 Таблица результатов

```python
def fmt_settling(st_index, dt):
    """Форматирование времени переходного процесса: индекс * dt или 'Н/Д'."""
    if st_index is None:
        return "Н/Д"
    return f"{st_index * dt:.2f} с"


print(f"{'Метрика':<25} {'ПИД':>12} {'PPO':>12} {'SAC':>12}")
print("-" * 63)
print(f"{'Перерегулирование (%)':<25} {os_pid:>12.2f} {os_ppo:>12.2f} "
      f"{os_sac:>12.2f}")
print(f"{'Время перех. процесса':<25} {fmt_settling(st_pid, dt):>12} "
      f"{fmt_settling(st_ppo, dt):>12} {fmt_settling(st_sac, dt):>12}")
print(f"{'Стат. ошибка (град)':<25} {se_pid:>12.4f} {se_ppo:>12.4f} "
      f"{se_sac:>12.4f}")
print(f"{'Суммарное вознагр.':<25} {sum(pid_rewards):>12.2f} "
      f"{ppo_reward:>12.2f} {sac_reward:>12.2f}")
```

### 8.3 Использование ControlBenchmark

Для более полного отчета можно использовать класс `ControlBenchmark`:

```python
from tensoraerospace.benchmark import ControlBenchmark

bench = ControlBenchmark()

# signal_val=0 указывает, с какого значения начинается ступенька
# (до ступеньки задающий сигнал равен нулю).
pid_metrics = bench.benchmarking_one_step(
    control_signal=ref_pid,
    system_signal=pid_deg,
    signal_val=0,
    dt=dt,
)
print("Метрики ПИД:", pid_metrics)
```

---

## 9. Шаг 7 -- Сохранение и загрузка моделей

### 9.1 Сохранение

```python
# PPO
ppo_agent.save("./checkpoints/ppo_b747")

# SAC
sac_agent.save("./checkpoints/sac_b747")
```

### 9.2 Загрузка из локального чекпоинта

```python
# SAC -- используем from_pretrained с локальным путем к директории
loaded_sac = SAC.from_pretrained("./checkpoints/sac_b747/<папка_с_временной_меткой>")

# PPO -- аналогично
loaded_ppo = PPO.from_pretrained("./checkpoints/ppo_b747/<папка_с_временной_меткой>")
```

### 9.3 Загрузка из Hugging Face Hub

Если предобученная модель доступна в Hub, загрузка выполняется одной строкой:

```python
agent = SAC.from_pretrained("TensorAeroSpace/sac-b747")
```

---

## 10. Шаг 8 -- Визуализация

### 10.1 Сравнение отслеживания тангажа

```python
time_axis = np.array(tps[:max(n_pid, n_ppo, n_sac)])

fig, ax = plt.subplots(figsize=(14, 5))

# Задающий сигнал
ax.plot(tps[:reference.shape[1]], np.rad2deg(reference[0]),
        "k--", linewidth=2, label="Задание")

# ПИД
ax.plot(tps[:n_pid], pid_deg,
        linewidth=1.5, label=f"ПИД (Перерег.={os_pid:.1f}%)")

# PPO
ax.plot(tps[:n_ppo], ppo_deg,
        linewidth=1.5, label=f"PPO (Перерег.={os_ppo:.1f}%)")

# SAC
ax.plot(tps[:n_sac], sac_deg,
        linewidth=1.5, label=f"SAC (Перерег.={os_sac:.1f}%)")

ax.set_xlabel("Время (с)")
ax.set_ylabel("Угол тангажа (град)")
ax.set_title("Отслеживание тангажа: ПИД vs PPO vs SAC")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("comparison_pitch_tracking.png", dpi=150)
plt.show()
```

### 10.2 Кривые вознаграждения

Во время обучения история вознаграждений записывается в TensorBoard.
Вы можете просмотреть их интерактивно:

```bash
tensorboard --logdir runs
```

Альтернативно, можно записывать вознаграждения вручную во время обучения и
строить графики с помощью matplotlib.

---

## 11. Рекомендации по гиперпараметрам для аэрокосмических задач

### 11.1 Рекомендуемые начальные значения

| Гиперпараметр | PPO | SAC |
|---|---|---|
| `gamma` | 0.99 | 0.99 |
| `learning_rate` (актор) | 3e-4 | 3e-4 |
| `learning_rate` (критик) | 1e-3 | 3e-4 |
| `hidden_size` | 256 | 256 |
| `batch_size` | 64 | 64 |
| `entropy_coef` / `alpha` | 0.01 | 0.2 (авто) |
| `episodes` | 50 -- 200 | 200 -- 500 |

### 11.2 Типичные ошибки

1. **Слишком высокая скорость обучения** -- обновления политики становятся
   нестабильными, и агент расходится (вознаграждения падают до больших
   отрицательных значений). Начинайте с 3e-4 и уменьшайте при необходимости.

2. **Недостаточное количество эпизодов обучения** -- аэрокосмические среды имеют
   длинные горизонты. При `rollout_len=2048` и 50 эпизодах агент может не
   увидеть достаточного разнообразия. Увеличьте до 200+ эпизодов для надежной
   сходимости.

3. **Разреженное вознаграждение** -- если вознаграждение выдается только в конце
   эпизода, агент с трудом обучается. `ImprovedB747Env` уже предоставляет
   плотное формообразующее вознаграждение, но при создании собственной среды
   убедитесь, что вы даете обратную связь на каждом шаге.

4. **Игнорирование ПИД-базы** -- если ПИД с автоматической настройкой уже
   достигает хорошего слежения, RL потребуется намного больше эпизодов, чтобы
   его достичь. Используйте метрики ПИД как целевой ориентир.

5. **Забыли про нормализацию** -- PPO сильно выигрывает от нормализации
   наблюдений (`normalize_obs=True`). SAC менее чувствителен, поскольку буфер
   воспроизведения содержит разнообразные данные, но нормализованные наблюдения
   помогают обучению сетей в целом.

6. **Слишком низкий терминальный штраф** -- если агент научится завершать эпизод
   раньше времени, чтобы избежать накопления отрицательных вознаграждений,
   увеличьте `early_termination_penalty` и `early_termination_penalty_per_step`
   в конструкторе среды.

---

## 12. Что дальше

* **DSAC (Distributional SAC)**: Управление с учетом рисков, где критик
  предсказывает полное распределение отдачи. См. `tensoraerospace.agent.dsac`
  и пример `example/reinforcement_learning/train_dsac_b747_step_response.py`.
* **MPC (Model Predictive Control)**: Использование обученной или аналитической
  модели динамики с оптимизацией на скользящем горизонте. См.
  `tensoraerospace.agent.mpc`.
* **ADP / ADHDP**: Методы адаптивного динамического программирования, сочетающие
  нейронные сети с принципом Беллмана. См. `tensoraerospace.agent.adp`.
* **Сравнительные исследования**: Каталог `example/comparison/` содержит
  ноутбуки, сравнивающие несколько агентов на одной среде.
* **Собственные среды**: Адаптируйте этот рабочий процесс к другим моделям
  летательных аппаратов -- `ImprovedX15Env`, `F4CPitchEnvNormalized`,
  `ImprovedComSatEnv` -- заменив класс среды и скорректировав начальное
  состояние и задающий сигнал.

---

## 13. Полный скрипт

Ниже приведен полный скрипт, собранный из шагов выше. Скопируйте его в файл
`.py` и запустите.

```python
"""Урок 10 -- Обучение первого RL-агента для управления аэрокосмическими системами.

Полный конвейер: ПИД-база -> обучение PPO -> обучение SAC ->
оценка -> сравнение по метрикам -> визуализация.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from tensoraerospace.envs import ImprovedB747Env
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.agent.pid import PID
from tensoraerospace.agent.ppo.model import PPO
from tensoraerospace.agent.sac import SAC
from tensoraerospace.benchmark.function import overshoot, settling_time, static_error
from tensoraerospace.benchmark import ControlBenchmark

# =====================================================================
# 1. Настройка среды
# =====================================================================
dt = 0.1
tp = generate_time_period(tn=40, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)

reference = np.reshape(
    unit_step(degree=1, tp=tp, time_step=50, output_rad=True),
    (1, -1),
)
initial_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def make_env():
    return ImprovedB747Env(
        initial_state=initial_state,
        reference_signal=reference,
        number_time_steps=number_time_steps,
        dt=dt,
    )


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"Устройство: {device}")

# =====================================================================
# 2. ПИД-база
# =====================================================================
env_pid = make_env()
pid = PID(env=env_pid, dt=dt)
pid.tune_matlab_style(
    track_state_idx=3,
    target_settling_time=5.0,
    target_overshoot=5.0,
    n_iterations=100,
    verbose=True,
)

pid.reset()
obs_pid, _ = env_pid.reset()
done = False
pid_states, pid_rewards = [], []
step_idx = 0

while not done:
    ref_val = float(reference[0, min(step_idx, reference.shape[1] - 1)])
    raw_theta = float(env_pid.unwrapped.model.get_output()[3])
    action_pid = pid.select_action(setpoint=ref_val, measurement=raw_theta)
    max_ele_deg = env_pid.unwrapped.max_stabilizer_angle_deg
    action_norm = np.clip(
        np.array([action_pid / np.deg2rad(max_ele_deg)], dtype=np.float32),
        -1.0, 1.0,
    )
    obs_pid, reward, terminated, truncated, _ = env_pid.step(action_norm)
    done = terminated or truncated
    pid_states.append(raw_theta)
    pid_rewards.append(reward)
    step_idx += 1

pid_response = np.array(pid_states)
print(f"Суммарное вознаграждение ПИД: {sum(pid_rewards):.2f}")

# =====================================================================
# 3. Обучение PPO
# =====================================================================
env_ppo = make_env()
ppo_agent = PPO(
    env=env_ppo,
    gamma=0.99,
    max_episodes=50,
    rollout_len=2048,
    clip_pram=0.2,
    num_epochs=10,
    batch_size=64,
    entropy_coef=0.01,
    actor_lr=3e-4,
    critic_lr=1e-3,
    gae_lambda=0.95,
    actor_hidden_dim=256,
    critic_hidden_dim=256,
    normalize_obs=True,
    seed=42,
    device=device,
)

print("Начинаем обучение PPO ...")
ppo_agent.train()
print("Обучение PPO завершено.")

# =====================================================================
# 4. Обучение SAC
# =====================================================================
env_sac = make_env()
sac_agent = SAC(
    env=env_sac,
    hidden_size=256,
    lr=3e-4,
    policy_lr=3e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    automatic_entropy_tuning=True,
    batch_size=64,
    memory_capacity=100000,
    updates_per_step=1,
    policy_type="Gaussian",
    seed=42,
    device=device,
)

print("Начинаем обучение SAC ...")
sac_agent.train(num_episodes=300)
print("Обучение SAC завершено.")

# =====================================================================
# 5. Оценка
# =====================================================================

def evaluate_ppo(agent, env):
    obs, _ = env.reset()
    done, states, total = False, [], 0.0
    while not done:
        _, mean_action, _ = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(mean_action[0])
        done = terminated or truncated
        states.append(float(env.unwrapped.model.get_output()[3]))
        total += float(reward)
    return np.array(states), total


def evaluate_sac(agent, env):
    obs, _ = env.reset()
    done, states, total = False, [], 0.0
    while not done:
        action = agent.select_action(obs, evaluate=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(float(env.unwrapped.model.get_output()[3]))
        total += float(reward)
    return np.array(states), total


ppo_response, ppo_reward = evaluate_ppo(ppo_agent, make_env())
sac_response, sac_reward = evaluate_sac(sac_agent, make_env())

# =====================================================================
# 6. Бенчмарк
# =====================================================================
ref_deg = np.rad2deg(reference[0])

n_pid, n_ppo, n_sac = len(pid_response), len(ppo_response), len(sac_response)
pid_deg = np.rad2deg(pid_response)
ppo_deg = np.rad2deg(ppo_response)
sac_deg = np.rad2deg(sac_response)

os_pid = overshoot(ref_deg[:n_pid], pid_deg)
os_ppo = overshoot(ref_deg[:n_ppo], ppo_deg)
os_sac = overshoot(ref_deg[:n_sac], sac_deg)

st_pid = settling_time(ref_deg[:n_pid], pid_deg)
st_ppo = settling_time(ref_deg[:n_ppo], ppo_deg)
st_sac = settling_time(ref_deg[:n_sac], sac_deg)

se_pid = static_error(ref_deg[:n_pid], pid_deg)
se_ppo = static_error(ref_deg[:n_ppo], ppo_deg)
se_sac = static_error(ref_deg[:n_sac], sac_deg)


def fmt_st(idx):
    return f"{idx * dt:.2f} с" if idx is not None else "Н/Д"


print(f"\n{'Метрика':<25} {'ПИД':>12} {'PPO':>12} {'SAC':>12}")
print("-" * 63)
print(f"{'Перерегулирование (%)':<25} {os_pid:>12.2f} {os_ppo:>12.2f} "
      f"{os_sac:>12.2f}")
print(f"{'Время перех. процесса':<25} {fmt_st(st_pid):>12} "
      f"{fmt_st(st_ppo):>12} {fmt_st(st_sac):>12}")
print(f"{'Стат. ошибка (град)':<25} {se_pid:>12.4f} {se_ppo:>12.4f} "
      f"{se_sac:>12.4f}")
print(f"{'Суммарное вознагр.':<25} {sum(pid_rewards):>12.2f} "
      f"{ppo_reward:>12.2f} {sac_reward:>12.2f}")

# =====================================================================
# 7. Визуализация
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(tps[:reference.shape[1]], np.rad2deg(reference[0]),
        "k--", linewidth=2, label="Задание")
ax.plot(tps[:n_pid], pid_deg, linewidth=1.5,
        label=f"ПИД (Перерег.={os_pid:.1f}%)")
ax.plot(tps[:n_ppo], ppo_deg, linewidth=1.5,
        label=f"PPO (Перерег.={os_ppo:.1f}%)")
ax.plot(tps[:n_sac], sac_deg, linewidth=1.5,
        label=f"SAC (Перерег.={os_sac:.1f}%)")
ax.set_xlabel("Время (с)")
ax.set_ylabel("Угол тангажа (град)")
ax.set_title("Отслеживание тангажа: ПИД vs PPO vs SAC на Boeing 747")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("comparison_pitch_tracking.png", dpi=150)
plt.show()

# =====================================================================
# 8. Сохранение моделей
# =====================================================================
ppo_agent.save("./checkpoints/ppo_b747")
sac_agent.save("./checkpoints/sac_b747")
print("Модели сохранены.")
```

---

## 14. Литература

1. Schulman, J. et al. *Proximal Policy Optimization Algorithms*, 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
2. Haarnoja, T. et al. *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor*, 2018. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)
3. Документация и примеры TensorAeroSpace: [GitHub](https://github.com/TensorAeroSpace/TensorAeroSpace)
