# 🚀 Руководство по настройке среды разработки TensorAeroSpace

Это руководство поможет вам настроить среду разработки для проекта TensorAeroSpace.

## 📋 Предварительные требования

- **Python 3.9+** (рекомендуется 3.10 или 3.11)
- **Poetry** для управления зависимостями
- **Git** для контроля версий
- **Make** для выполнения команд разработки

### Установка Poetry

```bash
# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Или через pip
pip install poetry
```

## 🛠️ Быстрая настройка

1. **Клонируйте репозиторий:**
```bash
git clone https://github.com/your-username/TensorAeroSpace.git
cd TensorAeroSpace
```

2. **Настройте среду разработки одной командой:**
```bash
make dev-setup
```

Эта команда автоматически:
- Установит все зависимости
- Настроит pre-commit hooks
- Покажет доступные команды

## 📝 Ручная настройка

Если вы предпочитаете ручную настройку:

1. **Установите зависимости:**
```bash
poetry install --with dev,test
```

2. **Активируйте виртуальную среду:**
```bash
poetry shell
```

3. **Установите pre-commit hooks:**
```bash
pre-commit install
```

## 🧪 Проверка установки

Запустите быстрые тесты для проверки:
```bash
make test-quick
```

Или полный набор проверок:
```bash
make check-all
```

## 📚 Основные команды разработки

### Тестирование
```bash
make test           # Все тесты с покрытием
make test-quick     # Быстрые тесты
make test-agents    # Тесты агентов
make test-envs      # Тесты окружений
```

### Качество кода
```bash
make format         # Форматирование кода
make lint           # Проверка линтерами
make security       # Проверка безопасности
make pre-commit     # Запуск pre-commit hooks
```

### Документация
```bash
make docs           # Генерация документации
make docs-serve     # Запуск сервера документации
```

### Сборка и публикация
```bash
make build          # Сборка пакета
make clean          # Очистка временных файлов
```

### Версионирование
```bash
make version        # Показать текущую версию
make bump-patch     # Увеличить patch версию
make bump-minor     # Увеличить minor версию
make bump-major     # Увеличить major версию
```

## 🔧 Конфигурация IDE

### VS Code

Рекомендуемые расширения (создайте `.vscode/extensions.json`):
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.isort",
    "ms-python.flake8",
    "ms-python.mypy-type-checker",
    "charliermarsh.ruff",
    "ms-python.pylint"
  ]
}
```

Настройки (создайте `.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

### PyCharm

1. Откройте проект в PyCharm
2. Настройте интерпретатор Python: Settings → Project → Python Interpreter
3. Выберите Poetry Environment
4. Настройте форматирование: Settings → Tools → External Tools

## 🐳 Docker разработка

Для разработки в Docker:

```bash
# Сборка образа
make docker_build

# Запуск в debug режиме
make docker_debug
```

## 🔄 Workflow разработки

1. **Создайте новую ветку:**
```bash
git checkout -b feature/your-feature-name
```

2. **Внесите изменения и проверьте качество:**
```bash
make format
make lint
make test
```

3. **Зафиксируйте изменения:**
```bash
git add .
git commit -m "feat: add your feature description"
```

Pre-commit hooks автоматически проверят ваш код.

4. **Отправьте изменения:**
```bash
git push origin feature/your-feature-name
```

5. **Создайте Pull Request** через GitHub интерфейс.

## 🚨 Решение проблем

### Проблемы с Poetry

```bash
# Очистка кэша Poetry
poetry cache clear pypi --all

# Переустановка зависимостей
rm poetry.lock
poetry install
```

### Проблемы с pre-commit

```bash
# Переустановка pre-commit hooks
pre-commit uninstall
pre-commit install

# Запуск на всех файлах
pre-commit run --all-files
```

### Проблемы с тестами

```bash
# Очистка кэша pytest
rm -rf .pytest_cache

# Запуск с подробным выводом
poetry run pytest -v --tb=long
```

## 📞 Получение помощи

- 📖 [Документация проекта](docs/)
- 🐛 [Сообщить об ошибке](https://github.com/your-username/TensorAeroSpace/issues/new?template=bug_report.md)
- ✨ [Запросить функцию](https://github.com/your-username/TensorAeroSpace/issues/new?template=feature_request.md)
- ❓ [Задать вопрос](https://github.com/your-username/TensorAeroSpace/issues/new?template=question.md)

## 🎯 Следующие шаги

После настройки среды:

1. Изучите [примеры](example/) для понимания API
2. Прочитайте [документацию](docs/) для глубокого понимания
3. Запустите [тесты](tests/) для проверки функциональности
4. Начните разработку своих функций!

---

**Удачной разработки! 🚀**