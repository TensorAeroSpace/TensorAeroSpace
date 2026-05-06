# 🚀 TensorAeroSpace

<div align="center">

[![en](https://img.shields.io/badge/lang-en-red.svg)](./readme.md)
[![ru](https://img.shields.io/badge/lang-ru-green.svg)](./README.ru-ru.md)
[![Documentation Status](https://readthedocs.org/projects/tensoraerospace/badge/?version=latest)](https://tensoraerospace.readthedocs.io/en/latest/?badge=latest)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TensorAeroSpace/TensorAeroSpace)
[![Python](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/tensoraerospace/tensoraerospace.svg)](https://github.com/tensoraerospace/tensoraerospace/stargazers)

![Логотип TensorAeroSpace](./img/logo-no-background.png)

**Продвинутая платформа для аэрокосмических систем управления и обучения с подкреплением**

*Комплексная Python библиотека для аэрокосмического моделирования, алгоритмов управления и реализации обучения с подкреплением*

[📖 Документация](https://tensoraerospace.readthedocs.io/) • [🚀 Быстрый старт](#-быстрый-старт) • [💡 Примеры](./example/) • [🤝 Участие в разработке](CONTRIBUTING.md)

</div>

---

## 🌟 Обзор

**TensorAeroSpace** — это передовая Python платформа, которая объединяет аэрокосмическую инженерию с современным машинным обучением. Она предоставляет:

- 🎯 **Системы управления**: Продвинутые алгоритмы управления, включая PID, MPC и современные подходы RL
- ✈️ **Аэрокосмические модели**: Высокоточные модели симуляции самолётов и космических аппаратов — включая **полную нелинейную F-16** (продольная + 6-DoF угловая)
- 💥 **Моделирование повреждений в полёте**: запланированные отказы — потеря законцовки крыла, заклинивание рулевых поверхностей, остановка двигателя — среда пересчитывает массу, инерцию и аэродинамику в реальном времени
- 🎮 **Интеграция с OpenAI Gym**: Готовые к использованию среды для обучения с подкреплением
- 🧠 **RL алгоритмы**: Современные реализации обучения с подкреплением, включая онлайн-адаптивные критики (iADP, IM-GDHP, ET-DHP, AIDI, AA-INDI) для отказоустойчивого управления
- 🔧 **Расширяемая архитектура**: Легко расширяется и настраивается под ваши конкретные потребности

## 🧭 Направления прикладного использования

Перечень сценариев, подтверждённых Заключением НТО:

1. **Автоматическое управление летательными аппаратами** — стабилизация, следование траектории, управление угловым положением для самолётов, БПЛА и экспериментальных аппаратов.
2. **Управление ракетно-космическими системами** — моделирование и управление ракетами-носителями, спутниками различных классов орбит, оптимизация траекторий выведения.
3. **Гибридные системы управления** — проектирование и настройка контуров, сочетающих классические и интеллектуальные методы управления.
4. **Оптимизация и сравнительный анализ алгоритмов** — автоматизированный подбор гиперпараметров, бенчмаркинг алгоритмов управления, визуализация метрик качества.
5. **Интеграция с симуляционными платформами** — работа с игровыми движками, CAD/CAE-системами, экспорт и импорт моделей между средами.
6. **Анализ надёжности и диагностика** — исследование отказных режимов, оценка устойчивости систем управления, подготовка данных для обучения.

Каждый сценарий имеет рабочие примеры и документацию (см. раздел [📚 Примеры и руководства](#-примеры-и-руководства)).

## 🚀 Быстрый старт

> 💡 **Интерактивный walkthrough**: откройте блокнот [quickstart.ipynb](./example/quickstart.ipynb), чтобы запустить SAC‑бенчмарк для B747 от начала до конца прямо в Jupyter/VS Code.

### ✅ Минимальные технические требования

| Компонент | Минимум | Рекомендовано |
| --- | --- | --- |
| **ОС** | Linux x86_64, Windows 10, macOS 13 | Ubuntu 22.04 LTS / Windows 11 |
| **CPU** | 4 ядра, AVX | 8+ ядер, AVX2/FMA |
| **RAM** | 8 ГБ | 16–32 ГБ для RL/Simulink |
| **GPU** | Необязательно | NVIDIA RTX с ≥8 ГБ VRAM для SAC/DSAC/PPO, поддержка CUDA 12.2 |
| **Python** | 3.10–3.13 | 3.11/3.12 |
| **Доп. ПО** | Git, Poetry или pip, Docker (опционально) | MATLAB/Simulink R2022b+ (для simulink-example), Unity 2021.3.5f1/2023.2.20f1 |

### 📦 Установка

#### Установка Poetry (1 раз)

```bash
curl -sSL https://install.python-poetry.org | python3 -
# Windows (PowerShell):
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
poetry --version   # проверка установки
poetry self update --preview  # при необходимости обновления
```

После установки добавьте `$HOME/.local/bin` (Linux/macOS) или `%APPDATA%\Python\Scripts` (Windows) в `PATH`.

#### Использование Poetry (Рекомендуется)
```bash
git clone https://github.com/tensoraerospace/tensoraerospace.git
cd tensoraerospace
poetry install
poetry shell        # активация виртуального окружения
poetry run pytest   # быстрая проверка
```

#### Использование pip
```bash
pip install tensoraerospace
```

#### 🐳 Docker
Образ **по умолчанию запускает JupyterLab** (см. `Dockerfile`). Runtime-образ собирается из исходников репозитория, устанавливает TensorAeroSpace как wheel и содержит примеры в `/workspace/examples`.

**Ubuntu / Linux (bash):**

```bash
docker pull ghcr.io/tensoraerospace/tensoraerospace:latest
docker run --rm -it -p 8888:8888 \
  -v "$(pwd)/projects:/workspace/projects" \
  ghcr.io/tensoraerospace/tensoraerospace:latest

# Или соберите такой же образ локально из исходников
docker build -t tensoraerospace:local . --platform=linux/amd64
docker run --rm -it -p 8888:8888 \
  -v "$(pwd)/projects:/workspace/projects" \
  tensoraerospace:local

# Опционально: включить GPU (NVIDIA) внутри контейнера
docker run --rm -it --gpus all -p 8888:8888 \
  -v "$(pwd)/projects:/workspace/projects" \
  ghcr.io/tensoraerospace/tensoraerospace:latest
```

**Windows (PowerShell):**

```powershell
docker pull ghcr.io/tensoraerospace/tensoraerospace:latest
docker run --rm -it -p 8888:8888 `
  -v "${PWD}\projects:/workspace/projects" `
  ghcr.io/tensoraerospace/tensoraerospace:latest

# Или соберите такой же образ локально из исходников
docker build -t tensoraerospace:local . --platform=linux/amd64
docker run --rm -it -p 8888:8888 `
  -v "${PWD}\projects:/workspace/projects" `
  tensoraerospace:local

# Опционально: включить GPU (NVIDIA) внутри контейнера
docker run --rm -it --gpus all -p 8888:8888 `
  -v "${PWD}\projects:/workspace/projects" `
  ghcr.io/tensoraerospace/tensoraerospace:latest
```
> Откройте выданную ссылку (обычно `http://127.0.0.1:8888`) и перейдите к `examples/quickstart.ipynb`, чтобы выполнить SAC walkthrough внутри контейнера.

#### Рекомендуемые версии CUDA и cuDNN

- Docker-образ использует Python runtime и PyTorch wheels из зависимостей проекта; для GPU запускайте контейнер через NVIDIA Container Toolkit с `--gpus all`.
- Для ручной CUDA/cuDNN установки вне Docker сверяйтесь с CUDA-версией установленного `torch`.
- Для Apple Silicon используйте `torch` с backend `mps` (CUDA не требуется).

### 🏃‍♂️ Быстрый пример

```python
import gymnasium as gym
import numpy as np

from tensoraerospace.agent.pid import PID
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standard import unit_step

# Параметры симуляции
dt = 0.01
tp = generate_time_period(tn=10, dt=dt)  # 10 секунд
N = len(tp)

# Опорный сигнал (ступенька 5° в радианах)
reference = unit_step(degree=5, tp=tp, time_step=100, output_rad=True).reshape(1, -1)

# Создание среды F-16 (порядок состояний: [alpha, q])
env = gym.make(
    'LinearLongitudinalF16-v0',
    number_time_steps=N,
    initial_state=[[0], [0]],
    reference_signal=reference,
    use_reward=False,
)

# ПИД-контроллер (коэффициенты из примера PID)
pid = PID(env, kp=-14.290139135229715, ki=-8.240470780203491, kd=-1.2991634935096958, dt=dt)

obs, info = env.reset()
for t in range(N - 1):
    setpoint = reference[0, t]
    alpha = float(obs[0])  # env возвращает [alpha, q]
    u = pid.select_action(setpoint, alpha)
    action = np.array([[float(u)]], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

#### 💥 Моделирование повреждений в полёте (нелинейная F-16)

Запланируйте отказы в ходе симуляции — потерю законцовки крыла, заклинивание рулевых поверхностей, остановку двигателя, структурные изменения — и среда пересчитает массу, тензор инерции, аэродинамические коэффициенты и эффективность рулей в реальном времени. Управляющий агент с момента срабатывания события сталкивается уже с другим объектом управления.

```python
import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    WING_STRIKE_LEFT_TIP,  # готовый пресет: полная потеря левой законцовки на t=10 с
)
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

env = NonlinearAngularF16(
    initial_state=np.zeros(14),
    number_time_steps=2000,
    damage_profile=WING_STRIKE_LEFT_TIP,
    split_stab=True,
)
obs, _ = env.reset()
for _ in range(2000):
    obs, r, term, trunc, info = env.step(np.zeros(4))
    if info.get("damage_events_triggered"):
        print(info["damage_events_triggered"])  # → ['left_tip_full_loss']
```

Что моделируется:

- **Потеря секции** (крыло/стабилизатор/киль): масса *m*, площадь крыла *S*, размах *b*, MAC, ЦМ, тензор инерции **J**, аэродинамические коэффициенты — всё пересчитывается из посекционных вкладов через теорему Гюйгенса-Штейнера.
- **Отказ рулевой поверхности** (`jam` / `efficiency_loss` / `lost`): команда **u**<sub>cmd</sub> → **u**<sub>eff</sub> перед интегратором.
- **Отказ двигателя** (частичный/полный): эффективная тяга масштабируется или зануляется.
- **Структурные изменения** (сброс груза, обледенение): Δ массы / ЦМ / инерции.

7 готовых пресетов (`WING_STRIKE_LEFT_TIP`, `ELEVATOR_JAM_NEUTRAL`, `RUDDER_LOST`, `ENGINE_FLAMEOUT`, `BIRDSTRIKE_COMPOUND` и др.) плюс `RandomDamageProfileGenerator` для RL-курикулумов. Без `damage_profile` среда бит-в-бит идентична неповреждённому baseline.

> 📖 **Полный референс**: [Документация по моделированию повреждений ЛА](https://tensoraerospace.readthedocs.io/ru/latest/model/aircraft-damage-modeling/) — обзор, реализация и примеры, математика.

## 🤖 Поддерживаемые алгоритмы

### Классическое управление

| Алгоритм | Описание |
|----------|----------|
| **PID** | Пропорционально-интегрально-дифференциальный регулятор с anti-windup и автонастройкой в стиле MATLAB. Сильный baseline для state-space задач; автотюнер извлекает матрицы `(A, B, C, D)` среды и оптимизирует PID-коэффициенты дифференциальной эволюцией по критерию переходной характеристики. |
| **MPC** | Модельно-предиктивное управление с тремя сменными вариантами модели объекта — MLP, NARX и Transformer — поверх QP/численной оптимизации с уходящим горизонтом. Применять там, где ОУ известен или может быть выучен из данных. |

### Глубокое RL — on-policy

| Алгоритм | Описание |
|----------|----------|
| **PPO** | Proximal Policy Optimization — on-policy актор-критик с clip-сурогатной целевой функцией. Стабилен и легко настраивается; стандартный «just works» старт для непрерывных и дискретных задач. |
| **A2C** | Advantage Actor-Critic — синхронный on-policy актор-критик; проще PPO, удобен как чистая референсная реализация. |
| **A2C-NARX** | A2C с NARX-критиком вместо MLP — лучше захватывает временную структуру; полезен в задачах, где состояние не показывает фазу/задержку. |
| **A3C** | Asynchronous Advantage Actor-Critic — несколько воркеров параллельно обновляют общую глобальную сеть. Лучший выбор для CPU-параллельного распределённого обучения (например, Unity-окружения). |

### Глубокое RL — off-policy

| Алгоритм | Описание |
|----------|----------|
| **SAC** | Soft Actor-Critic — off-policy стохастический актор-критик с максимизацией энтропии. Эффективный по данным дефолт для непрерывного управления; поставляется с `from_pretrained` / `publish_to_hub` интеграцией HuggingFace. |
| **DSAC** | Распределительный (distributional) SAC — SAC с квантильными (IQN-стиль) сдвоенными критиками + регуляризацией CAPS. Лучшая динамика слежения по сравнению с обычным SAC, особенно при шуме сенсоров или мультимодальной функции стоимости. |
| **DDPG** | Deep Deterministic Policy Gradient — off-policy детерминированный актор-критик. Основополагающий алгоритм; SAC превосходит его в большинстве случаев, но DDPG остаётся полезен для тихих низкочастотных задач. |
| **DQN** | Deep Q-Learning — off-policy value-based обучение для дискретных пространств действий; используется здесь для Unity-окружений с дискретным управлением. |

### Имитационное обучение

| Алгоритм | Описание |
|----------|----------|
| **GAIL** | Generative Adversarial Imitation Learning — обучение по экспертным демонстрациям без явного reward'а. Удобен для клонирования заранее построенной PID/MPC-траектории перед дообучением через RL-критик. |

### Adaptive Dynamic Programming (модельные критики)

| Алгоритм | Описание |
|----------|----------|
| **HDP** | Heuristic Dynamic Programming — актор-критик с предобученной оффлайн plant-сетью, которая даёт градиент политики через `∂f/∂u`. |
| **ADHDP** | Action-Dependent HDP — функция ценности зависит от пары `(состояние, действие)`; не требует явного градиента модели за счёт расширения входа критика. |
| **ADP** | Базовое Adaptive Dynamic Programming — value-iteration-стиль адаптивного управления без явной модели ОУ. |
| **IHDP** | Incremental HDP — актор-критик с **онлайн инкрементальной линеаризацией** ОУ; адаптивен без необходимости предобученной plant-сети. Сильный baseline для онлайн-управления полётом. |
| **NARX** | Нелинейная авторегрессионная сеть — используется и как plant-модель для MPC, и как критик в `A2C-NARX`. |

### Онлайн-адаптивные критики для отказоустойчивого полёта (новое)

| Алгоритм | Описание |
|----------|----------|
| **iADP** | Incremental Approximate Dynamic Programming — онлайн RLS-идентификация локальной инкрементальной модели `(F̃, G̃)` плюс замкнутая квадратичная политика. Восстанавливается после изменения ОУ за десятки миллисекунд; не требует детектора отказов. |
| **IM-GDHP** | Incremental-Model GDHP — онлайн RLS-идентификатор объекта в паре с GDHP-критиком. Лёгкий (без нейросети ОУ), интерпретируемый, с явными матрицами `(F, G)`. |
| **ET-DHP** | Event-Triggered Dual HDP — Липшицев event-trigger запускает обновления actor/critic только когда ошибка слежения превышает порог. Bandwidth-aware исполнение полезно для embedded-развёртываний. |
| **AIDI** | Adaptive Incremental Dynamic Inversion — INDI с поканальным VFF-RLS, который адаптирует мультипликативное масштабирование `Θ` известной онбордной матрицы эффективности управления. Отказоустойчив и model-agnostic. |
| **AA-INDI** | Adaptive Augmented INDI — инкрементальная нелинейная динамическая инверсия с онлайн-RLS-адаптацией; рассчитан на асимметричные отказы приводов и схемы flying-wing с связанными управляющими поверхностями. |

## ✈️ Модели самолетов и космических аппаратов

<details>
<summary><b>🛩️ Самолеты с неподвижным крылом</b></summary>

- **General Dynamics F-16 Fighting Falcon** — высокоточная модель истребителя в **трёх вариантах**:
  - линейная продольная (state-space, быстрая),
  - **нелинейная продольная** (NumPy ОДУ, полные таблицы аэродинамических коэффициентов),
  - **нелинейная 6-DoF угловая** (полная угловая динамика в связанной СК, опционально **split-stab** для асимметричного управления).
- **Boeing 747** — динамика коммерческого авиалайнера (линейная + нормализованная `ImprovedB747Env`)
- **McDonnell Douglas F-4C Phantom II** — модель военного самолёта
- **North American X-15** — гиперзвуковой исследовательский самолёт

</details>

<details>
<summary><b>🚁 БПЛА и дроны</b></summary>

- **LAPAN Surveillance Aircraft (LSU)-05** - Индонезийский разведывательный БПЛА
- **Ultrastick-25e** - Модель радиоуправляемого самолета
- **Универсальный БПЛА в пространстве состояний** - Настраиваемая динамика БПЛА

</details>

<details>
<summary><b>🚀 Ракеты и спутники</b></summary>

- **ELV (Expendable Launch Vehicle)** - Динамика ракеты-носителя
- **Универсальная модель ракеты** - Настраиваемая симуляция ракеты
- **Геостационарный спутник** - Симуляция орбитальной механики
- **Спутник связи** - Динамика и управление спутником связи

</details>

## 🎮 Среды моделирования

### 🎯 Интеграция с Unity ML-Agents

<div align="center">

![Демо Unity](./docs/ru/example/enviroment/img/img_demo_unity.gif)

</div>

TensorAeroSpace легко интегрируется с Unity ML-Agents для захватывающих 3D симуляций:

- 🎮 **3D визуализация**: Симуляция самолетов в реальном времени
- 🔄 **Обучение в реальном времени**: Обучение агентов в реалистичных средах
- 📊 **Богатые сенсоры**: Камера, LiDAR и физические сенсоры
- 🌍 **Пользовательские среды**: Создавайте свои собственные аэрокосмические сценарии

> 📁 **Пример среды**: [UnityAirplaneEnvironment](https://github.com/TensorAeroSpace/UnityAirplaneEnvironment)

### 🔧 Поддержка MATLAB Simulink

![Модель Simulink](./docs/ru/example/simulink/img/model.png)

- 📐 **Импорт моделей**: Конвертация моделей Simulink в Python
- ⚡ **Высокая производительность**: Интеграция скомпилированного C++
- 🔄 **Двунаправленный**: Рабочий процесс MATLAB ↔ Python
- 📊 **Валидация**: Кроссплатформенная валидация моделей

### 📊 Матрицы пространства состояний

Математическая основа для проектирования систем управления:

- 🧮 **Линейные модели**: Представление в пространстве состояний
- 🎛️ **Проектирование управления**: Реализация современной теории управления
- 📈 **Инструменты анализа**: Устойчивость, управляемость, наблюдаемость
- 🔄 **Линеаризация**: Линеаризация нелинейных моделей

## 📚 Примеры и руководства

Изучите нашу обширную коллекцию примеров в директории [`./example`](./example/) и на ReadTheDocs:

| Категория | Описание | Блокноты |
|-----------|----------|----------|
| 🚀 **Быстрый старт** | Базовое использование, экспорт моделей в HuggingFace | [`quickstart.ipynb`](./example/quickstart.ipynb) |
| 🤖 **Обучение с подкреплением** | SAC/DDPG/PPO/GAIL скрипты и ноутбуки | [`reinforcement_learning/`](./example/reinforcement_learning/) |
| 🎛️ **Системы управления** | PID/MPC, Transformers для MPC | [`pid_controllers/`](./example/pid_controllers/), [`mpc_controllers/`](./example/mpc_controllers/) |
| ✈️ **Модели самолётов и ракет** | Линеаризованные среды, Unity, Simulink | [`environments/`](./example/environments/) |
| 🔧 **Оптимизация** | Optuna/Benchmark, гиперпараметры IHDP/NARX | [`optimization/`](./example/optimization/), [`utilities/hyperparam_optimization.ipynb`](./example/utilities/hyperparam_optimization.ipynb) |

> Быстрые команды запуска:
>
> ```bash
> poetry run python example/reinforcement_learning/sac-b747-render.py --render --dt 0.1
> poetry run python example/reinforcement_learning/ddpg-b747-render.py --repo TensorAeroSpace/ddpg-b747
> poetry run python example/reinforcement_learning/gail_pendulum_generate_expert.py
> poetry run python example/general_examples/example.py --env LinearLongitudinalF16-v0
> ```
>
> Подробные walkthrough доступны в документации:  
> - [Example SAC F-16](https://tensoraerospace.readthedocs.io/en/latest/example/agent/sac/example-sac-f16.html)  
> - [Optuna optimization](https://tensoraerospace.readthedocs.io/en/latest/example/optimization/example_optimization.html)  
> - [Unity Guide](https://tensoraerospace.readthedocs.io/en/latest/guide/unity_env.html)

## 🛠️ Разработка и участие

Мы приветствуем вклад в развитие! Пожалуйста, ознакомьтесь с нашим [Руководством по участию](CONTRIBUTING.md) для получения подробностей.

### 🏗️ Настройка для разработки

```bash
git clone https://github.com/tensoraerospace/tensoraerospace.git
cd tensoraerospace
poetry install --with dev
poetry run pytest  # Запуск тестов
```

### 🧪 Тестирование

```bash
# Запуск всех тестов
poetry run pytest

# Запуск конкретной категории тестов
poetry run pytest tests/envs/
poetry run pytest tests/agents/
```

## 📖 Документация

- 📚 **Полная документация**: [tensoraerospace.readthedocs.io](https://tensoraerospace.readthedocs.io/)
- 🚀 **Справочник API**: Подробная документация API
- 📝 **Руководства**: Пошаговые инструкции
- 💡 **Примеры**: Практические случаи использования

## 🤝 Сообщество и поддержка

- 💬 **Обсуждения**: [GitHub Discussions](https://github.com/tensoraerospace/tensoraerospace/discussions)
- 🐛 **Проблемы**: [Отчеты об ошибках](https://github.com/tensoraerospace/tensoraerospace/issues)
- 📧 **Контакты**: [Поддержка по email](mailto:support@tensoraerospace.org)

## 📄 Лицензия

Этот проект лицензирован под лицензией MIT - см. файл [LICENSE](LICENSE) для подробностей.

## 🙏 Благодарности

- Команде OpenAI Gym за отличную RL платформу
- Команде Unity ML-Agents за возможности 3D симуляции
- Сообществу аэрокосмической инженерии за экспертизу в предметной области
- Всем участникам, которые делают этот проект возможным

---

<div align="center">

**⭐ Поставьте нам звезду на GitHub, если TensorAeroSpace полезен для вас! ⭐**

Сделано с ❤️ командой TensorAeroSpace

</div>
