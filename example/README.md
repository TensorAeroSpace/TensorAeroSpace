# 📚 Примеры TensorAeroSpace

<div align="center">

![TensorAeroSpace Logo](../img/logo-no-background.png)

**Комплексная коллекция примеров и руководств**

*Изучите возможности TensorAeroSpace через практические примеры*

[🏠 Главная](../) • [📖 Документация](https://tensoraerospace.readthedocs.io/) • [🚀 Быстрый старт](../README.ru-ru.md#-быстрый-старт)

</div>

---

## 🌟 Обзор

Эта папка содержит обширную коллекцию примеров использования библиотеки TensorAeroSpace, организованных по категориям для удобства навигации и изучения. Каждый пример включает подробные объяснения и готовый к запуску код.

## 📁 Структура примеров

### ✈️ Аэрокосмические среды
> **Папка:** [`environments/`](./environments/)

Примеры различных аэрокосмических окружений с высокоточными моделями:

| Модель | Описание | Блокнот |
|--------|----------|---------|
| **🛩️ Boeing 747** | Продольное управление коммерческим авиалайнером | [`example-env-LinearLongitudinalB747.ipynb`](./environments/example-env-LinearLongitudinalB747.ipynb) |
| **⚡ F-16 Fighting Falcon** | Высокоманевренный истребитель | [`example-env-LinearLongitudinalF16.ipynb`](./environments/example-env-LinearLongitudinalF16.ipynb) |
| **🚀 F-4C Phantom II** | Военный истребитель-бомбардировщик | [`example-env-f4c.ipynb`](./environments/example-env-f4c.ipynb) |
| **🚀 ELV Rocket** | Ракета-носитель | [`example-env-LinearLongitudinalELVRocket.ipynb`](./environments/example-env-LinearLongitudinalELVRocket.ipynb) |
| **🛸 UAV** | Беспилотные летательные аппараты | [`example-env-LinearLongitudinalUAV.ipynb`](./environments/example-env-LinearLongitudinalUAV.ipynb) |
| **🎯 Missile Model** | Модель управляемой ракеты | [`example-env-LinearLongitudinalMissileModel.ipynb`](./environments/example-env-LinearLongitudinalMissileModel.ipynb) |
| **🛰️ ComSat** | Спутник связи | [`example-env-comsat.ipynb`](./environments/example-env-comsat.ipynb) |
| **🌍 GeoSat** | Геостационарный спутник | [`example-env-geosat.ipynb`](./environments/example-env-geosat.ipynb) |
| **🎮 Unity** | Интеграция с Unity ML-Agents | [`example-env-unity.ipynb`](./environments/example-env-unity.ipynb) |
| **🚁 X-15** | Экспериментальный гиперзвуковой самолет | [`example-env-x15.ipynb`](./environments/example-env-x15.ipynb) |

### 🤖 Обучение с подкреплением
> **Папка:** [`reinforcement_learning/`](./reinforcement_learning/)

Современные алгоритмы RL для аэрокосмических задач:

| Алгоритм | Описание | Примеры |
|----------|----------|---------|
| **🎯 A3C** | Асинхронный актор-критик | [`example-a3c.ipynb`](./reinforcement_learning/example-a3c.ipynb), [`a3c-classic.ipynb`](./reinforcement_learning/a3c-classic.ipynb) |
| **🎭 SAC** | Мягкий актор-критик | [`example-sac.ipynb`](./reinforcement_learning/example-sac.ipynb), [`example-sac-f16.ipynb`](./reinforcement_learning/example-sac-f16.ipynb) |
| **🎪 A2C** | Актор-критик с преимуществом | [`example_a2c.ipynb`](./reinforcement_learning/example_a2c.ipynb) |
| **🧠 DQN** | Глубокое Q-обучение | [`example_dqn.ipynb`](./reinforcement_learning/example_dqn.ipynb), [`torch_dqn.ipynb`](./reinforcement_learning/torch_dqn.ipynb) |
| **🚀 PPO** | Проксимальная оптимизация политики | [`example_ppo.ipynb`](./reinforcement_learning/example_ppo.ipynb), [`example_ppo_torch.ipynb`](./reinforcement_learning/example_ppo_torch.ipynb) |
| **🎨 GAIL** | Генеративное состязательное обучение имитации | [`create_dataset_for_gail.ipynb`](./reinforcement_learning/create_dataset_for_gail.ipynb) |

### 🎛️ Системы управления
> **Папки:** [`mpc_controllers/`](./mpc_controllers/), [`pid_controllers/`](./pid_controllers/)

#### 🔮 Модельно-предиктивное управление (MPC)
- **📊 Пространство состояний**: [`example-mpc-state-space.ipynb`](./mpc_controllers/example-mpc-state-space.ipynb)
- **🤖 MPC с трансформерами**: [`example-mpc-state-space-transformers.ipynb`](./mpc_controllers/example-mpc-state-space-transformers.ipynb)

#### ⚙️ ПИД-регуляторы
- **🎯 Настройка ПИД**: [`tune_pid.ipynb`](./pid_controllers/tune_pid.ipynb)
- **🔧 Оптимизация ПИД**: [`pid_optimization.ipynb`](./pid_controllers/pid_optimization.ipynb)
- **🌀 Алгоритм Twiddle**: [`pid_twiddle.ipynb`](./pid_controllers/pid_twiddle.ipynb)
- **💼 Практическое использование**: [`pid_use.ipynb`](./pid_controllers/pid_use.ipynb)

### 🛠️ Утилиты и инструменты
> **Папка:** [`utilities/`](./utilities/)

Вспомогательные инструменты для анализа и разработки:

| Инструмент | Описание | Блокнот |
|------------|----------|---------|
| **📡 Генерация сигналов** | Создание тестовых сигналов | [`signals.ipynb`](./utilities/signals.ipynb) |
| **🔄 Конвертация Simulink** | Преобразование моделей в Python | [`example_sim_model_to_python.ipynb`](./utilities/example_sim_model_to_python.ipynb) |
| **🔍 Исследование** | Анализ и визуализация данных | [`example_explarotaion.ipynb`](./utilities/example_explarotaion.ipynb) |
| **⚡ Оптимизация гиперпараметров** | Настройка параметров алгоритмов | [`hyperparam_optimization.ipynb`](./utilities/hyperparam_optimization.ipynb) |

### 📚 Общие примеры
> **Папка:** [`general_examples/`](./general_examples/)

Базовые концепции и классические задачи:

| Пример | Описание | Блокнот |
|--------|----------|---------|
| **🎯 Классический пример** | Основы использования библиотеки | [`classic_example.ipynb`](./general_examples/classic_example.ipynb) |
| **🧮 IHDP** | Бесконечно-горизонтное динамическое программирование | [`example_ihdp.ipynb`](./general_examples/example_ihdp.ipynb), [`example_ihdp_beautiful.ipynb`](./general_examples/example_ihdp_beautiful.ipynb) |
| **📈 NARX** | Нелинейная авторегрессия с внешними входами | [`example-narx.ipynb`](./general_examples/example-narx.ipynb) |
| **⚠️ Обработка отказов** | Работа с отказами систем | [`example-ihdp-failure.ipynb`](./general_examples/example-ihdp-failure.ipynb) |

### 🔧 Оптимизация
> **Папка:** [`optimization/`](./optimization/)

Алгоритмы и методы оптимизации:
- **📊 Общая оптимизация**: [`example_optimization.ipynb`](./optimization/example_optimization.ipynb)

## 🚀 Быстрый старт

### 1. 📋 Предварительные требования

Убедитесь, что у вас установлены все необходимые зависимости:

```bash
# Установка основной библиотеки
pip install tensoraerospace

# Или с использованием Poetry
poetry install
```

### 2. 🔀 Выберите формат примера

- **CLI-сценарии** — подойдут, если нужно быстро воспроизвести готовый результат или интегрировать пример в собственный пайплайн. Скрипты находятся в `example/**` и запускаются напрямую через `python` или `poetry run python`.
- **Jupyter Notebook** — используйте для интерактивного изучения, пошаговых пояснений и экспериментов. Каждый раздел выше содержит ссылку на соответствующий ноутбук.

### 3. 🏃‍♂️ Команды CLI для типовых сценариев

Запускайте скрипты из корня репозитория (после установки зависимостей):

- **F-16 baseline (general_examples/example.py)** — простейшая проверка среды `LinearLongitudinalF16-v0`.
  ```bash
  poetry run python example/general_examples/example.py
  # или, если использовали pip/venv
  python example/general_examples/example.py
  ```

- **DDPG Boeing 747 (reinforcement_learning/ddpg-b747-render.py)** — воспроизводит готового агента и при необходимости визуализирует траекторию.
  ```bash
  poetry run python example/reinforcement_learning/ddpg-b747-render.py \
    --repo TensorAeroSpace/ddpg-b747 --dt 0.1 --tn 200 --render
  # через pip/venv:
  python example/reinforcement_learning/ddpg-b747-render.py \
    --repo TensorAeroSpace/ddpg-b747 --dt 0.1 --tn 200 --render
  ```
  Ключи `--repo`, `--dt`, `--tn`, `--render/--no-render` настраивают источник весов, дискретизацию и визуализацию.

- **GAIL Pendulum dataset (reinforcement_learning/gail_pendulum_generate_expert.py)** — собирает демонстрации маятника для дальнейшего обучения.
  ```bash
  poetry run python example/reinforcement_learning/gail_pendulum_generate_expert.py
  # через pip/venv:
  python example/reinforcement_learning/gail_pendulum_generate_expert.py
  ```
  Используйте флаг `--help`, чтобы уточнить параметры генерации (количество эпизодов, путь сохранения и т.д.).

### 4. 📓 Запуск ноутбуков

```bash
# Запуск Jupyter Lab
jupyter lab

# Или Jupyter Notebook
jupyter notebook
```

## 📖 Рекомендуемый порядок изучения

### 🌱 Для начинающих:
1. 📚 [`quickstart.ipynb`](./quickstart.ipynb) - Основы работы с библиотекой
2. 🎯 [`classic_example.ipynb`](./general_examples/classic_example.ipynb) - Классические задачи управления
3. ✈️ [`example-env-LinearLongitudinalF16.ipynb`](./environments/example-env-LinearLongitudinalF16.ipynb) - Простая модель самолета

### 🚀 Для продвинутых:
1. 🤖 [`example_ppo.ipynb`](./reinforcement_learning/example_ppo.ipynb) - Обучение с подкреплением
2. 🔮 [`example-mpc-state-space.ipynb`](./mpc_controllers/example-mpc-state-space.ipynb) - Модельно-предиктивное управление
3. 🎮 [`example-env-unity.ipynb`](./environments/example-env-unity.ipynb) - Интеграция с Unity

## 🔧 Дополнительные зависимости

Некоторые примеры могут требовать дополнительных библиотек:

```bash
# Для работы с Unity
pip install mlagents

# Для продвинутой визуализации
pip install plotly seaborn

# Для оптимизации
pip install optuna ray[tune]

# Для работы с PyTorch
pip install torch torchvision
```

## 🤝 Участие в разработке

Хотите добавить свой пример? Мы будем рады вашему вкладу!

1. 🍴 **Fork** репозитория
2. 🌿 **Создайте ветку** для вашего примера
3. 📝 **Добавьте документацию** и комментарии
4. 🧪 **Протестируйте** ваш код
5. 📤 **Создайте Pull Request**

## 📞 Поддержка

Нужна помощь с примерами?

- 💬 **GitHub Discussions**: [Обсуждения](https://github.com/tensoraerospace/tensoraerospace/discussions)
- 🐛 **Issues**: [Сообщить о проблеме](https://github.com/tensoraerospace/tensoraerospace/issues)
- 📧 **Email**: support@tensoraerospace.org

---

<div align="center">

**🌟 Изучайте, экспериментируйте и создавайте удивительные аэрокосмические системы! 🌟**

[⬆️ Наверх](#-примеры-tensoraerospace) • [🏠 Главная](../) • [📖 Документация](https://tensoraerospace.readthedocs.io/)

</div>