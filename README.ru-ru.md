# 🚀 TensorAeroSpace

<div align="center">

[![en](https://img.shields.io/badge/lang-en-red.svg)](./README.md)
[![ru](https://img.shields.io/badge/lang-ru-green.svg)](./README.ru-ru.md)
[![Documentation Status](https://readthedocs.org/projects/tensoraerospace/badge/?version=latest)](https://tensoraerospace.readthedocs.io/en/latest/?badge=latest)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TensorAeroSpace/TensorAeroSpace)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
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
- ✈️ **Аэрокосмические модели**: Высокоточные модели симуляции самолетов и космических аппаратов
- 🎮 **Интеграция с OpenAI Gym**: Готовые к использованию среды для обучения с подкреплением
- 🧠 **RL алгоритмы**: Современные реализации обучения с подкреплением
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
| **Python** | 3.10–3.12 | 3.11/3.12 |
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
Образ **по умолчанию запускает JupyterLab** (см. `Dockerfile`).

**Ubuntu / Linux (bash):**

```bash
docker build -t tensoraerospace . --platform=linux/amd64
docker run --rm -it -p 8888:8888 \
  -v "$(pwd)/example:/app/example" \
  tensoraerospace

# Опционально: включить GPU (NVIDIA) внутри контейнера
docker run --rm -it --gpus all -p 8888:8888 \
  -v "$(pwd)/example:/app/example" \
  tensoraerospace
```

**Windows (PowerShell):**

```powershell
docker build -t tensoraerospace . --platform=linux/amd64
docker run --rm -it -p 8888:8888 `
  -v "${PWD}\example:/app/example" `
  tensoraerospace

# Опционально: включить GPU (NVIDIA) внутри контейнера
docker run --rm -it --gpus all -p 8888:8888 `
  -v "${PWD}\example:/app/example" `
  tensoraerospace
```
> Откройте выданную ссылку (обычно `http://127.0.0.1:8888`) и перейдите к `example/quickstart.ipynb`, чтобы выполнить SAC walkthrough внутри контейнера.

#### Рекомендуемые версии CUDA и cuDNN

- **CUDA Toolkit**: 12.2.2 (совместим с базовым Docker-образом).
- **cuDNN**: 8.9.x для CUDA 12 → [официальная документация](https://docs.nvidia.com/deeplearning/cudnn/latest).
- Для Apple Silicon используйте `torch` с backend `mps` (CUDA не требуется).

### 🏃‍♂️ Быстрый пример

```python
import gymnasium as gym
import numpy as np

from tensoraerospace.agent.pid import PID
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standart import unit_step

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

## 🤖 Поддерживаемые алгоритмы

| Алгоритм | Тип | Save/Load | HuggingFace Hub | Статус |
|----------|-----|:---------:|:---------------:|:------:|
| **SAC** | Мягкий актор-критик | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |
| **PPO** | Проксимальная оптимизация политики | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |
| **DDPG** | Глубокий детерминированный градиент политики | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |
| **DSAC** | Дистрибутивный мягкий актор-критик | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |
| **DQN** | Глубокое Q-обучение | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |
| **A2C** | Актор-критик с преимуществом | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |
| **A2C-NARX** | A2C с NARX критиком | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |
| **A3C** | Асинхронный актор-критик | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |
| **GAIL** | Имитационное обучение (состязательное) | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |
| **MPC** | Модельно-предиктивное управление | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |
| **ADP** | Адаптивное динамическое программирование | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |
| **ADHDP** | Действие-зависимое HDP | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |
| **HDP** | Эвристическое динамическое программирование | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |
| **PID** | ПИД-регулятор | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |
| **IHDP** | Инкрементальное эвристическое ДП | ❌ | ❌ (требует TensorFlow) | ✅ |
| **NARX** | Нелинейная авторегрессия | ✅ | ✅ `from_pretrained` / `publish_to_hub` | ✅ |

## ✈️ Модели самолетов и космических аппаратов

<details>
<summary><b>🛩️ Самолеты с неподвижным крылом</b></summary>

- **General Dynamics F-16 Fighting Falcon** - Высокоточная модель истребителя
- **Boeing 747** - Динамика коммерческого авиалайнера
- **McDonnell Douglas F-4C Phantom II** - Модель военного самолета
- **North American X-15** - Гиперзвуковой исследовательский самолет

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

![Демо Unity](./docs/example/env/img/img_demo_unity.gif)

</div>

TensorAeroSpace легко интегрируется с Unity ML-Agents для захватывающих 3D симуляций:

- 🎮 **3D визуализация**: Симуляция самолетов в реальном времени
- 🔄 **Обучение в реальном времени**: Обучение агентов в реалистичных средах
- 📊 **Богатые сенсоры**: Камера, LiDAR и физические сенсоры
- 🌍 **Пользовательские среды**: Создавайте свои собственные аэрокосмические сценарии

> 📁 **Пример среды**: [UnityAirplaneEnvironment](https://github.com/TensorAeroSpace/UnityAirplaneEnvironment)

### 🔧 Поддержка MATLAB Simulink

![Модель Simulink](docs/example/simulink/img/model.png)

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
