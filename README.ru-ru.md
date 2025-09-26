# 🚀 TensorAeroSpace

<div align="center">

[![en](https://img.shields.io/badge/lang-en-red.svg)](./README.md)
[![ru](https://img.shields.io/badge/lang-ru-green.svg)](./README.ru-ru.md)
[![Documentation Status](https://readthedocs.org/projects/tensoraerospace/badge/?version=latest)](https://tensoraerospace.readthedocs.io/en/latest/?badge=latest)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TensorAeroSpace/TensorAeroSpace)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
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

## 🚀 Быстрый старт

### 📦 Установка

#### Использование Poetry (Рекомендуется)
```bash
git clone https://github.com/tensoraerospace/tensoraerospace.git
cd tensoraerospace
poetry install
```

#### Использование pip
```bash
pip install tensoraerospace
```

#### 🐳 Docker
```bash
docker build -t tensoraerospace . --platform=linux/amd64
docker run -v $(pwd)/example:/app/example -p 8888:8888 -it tensoraerospace
```

### 🏃‍♂️ Быстрый пример

```python
import tensoraerospace as tas

# Создаем среду F-16
env = tas.envs.F16Env()

# Инициализируем PPO агента
agent = tas.agent.PPO(env.observation_space, env.action_space)

# Обучаем агента
for episode in range(1000):
    obs = env.reset()
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
```

## 🤖 Поддерживаемые алгоритмы

| Алгоритм | Тип | Экспорт в HuggingFace | Статус |
|----------|-----|:---------------------:|:------:|
| **IHDP** | Инкрементальное эвристическое динамическое программирование | ❌ | ✅ |
| **DQN** | Глубокое Q-обучение | ❌ | ✅ |
| **SAC** | Мягкий актор-критик | ✅ | ✅ |
| **A3C** | Асинхронный актор-критик с преимуществом | ❌ | ✅ |
| **PPO** | Проксимальная оптимизация политики | ✅ | ✅ |
| **MPC** | Модельно-предиктивное управление | ✅ | ✅ |
| **A2C** | Актор-критик с преимуществом | ✅ | ✅ |
| **A2C-NARX** | A2C с NARX критиком | ❌ | ✅ |
| **PID** | Пропорционально-интегрально-дифференциальный регулятор | ✅ | ✅ |

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

Изучите нашу обширную коллекцию примеров в директории [`./example`](./example/):

| Категория | Описание | Блокноты |
|-----------|----------|----------|
| 🚀 **Быстрый старт** | Базовое использование и концепции | [`quickstart.ipynb`](./example/quickstart.ipynb) |
| 🤖 **Обучение с подкреплением** | Реализации RL алгоритмов | [`reinforcement_learning/`](./example/reinforcement_learning/) |
| 🎛️ **Системы управления** | PID, MPC контроллеры | [`pid_controllers/`](./example/pid_controllers/), [`mpc_controllers/`](./example/mpc_controllers/) |
| ✈️ **Модели самолетов** | Примеры сред | [`environments/`](./example/environments/) |
| 🔧 **Оптимизация** | Настройка гиперпараметров | [`optimization/`](./example/optimization/) |

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
