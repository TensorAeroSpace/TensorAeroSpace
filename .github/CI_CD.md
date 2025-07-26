# CI/CD Documentation

Этот документ описывает настройку и использование CI/CD пайплайнов для проекта TensorAeroSpace.

## Обзор GitHub Actions

Проект использует несколько GitHub Actions workflow для автоматизации различных процессов:

### 1. Quick Check (`quick-check.yml`)
**Триггеры:** Push и Pull Request на ветки `main`, `develop`, `master`, `feature/*`, `bugfix/*`, `hotfix/*`

**Задачи:**
- **quick-test**: Быстрые тесты на Python 3.10/Ubuntu
- **code-quality**: Проверка форматирования (black, isort, flake8)
- **dependency-check**: Проверка безопасности зависимостей (safety check)

### 2. Main CI/CD (`action.yml`)
**Триггеры:** Push на `main`, `develop` и Pull Request

**Задачи:**
- **test**: Полное тестирование на матрице Python версий (3.9, 3.10, 3.11) и ОС (Ubuntu, macOS, Windows)
- **docs-coverage**: Проверка покрытия документации (docstr-coverage)
- **security**: Сканирование безопасности (safety, bandit)
- **build**: Сборка пакета и загрузка артефактов

### 3. Publishing (`publish.yml`)
**Триггеры:** 
- Release (автоматическая публикация в PyPI)
- Manual dispatch (ручная публикация в TestPyPI или PyPI)

**Задачи:**
- **test-before-publish**: Предварительное тестирование
- **publish-testpypi**: Публикация в Test PyPI
- **publish-pypi**: Публикация в PyPI

### 4. Release (`release.yml`)
**Триггеры:** Push тегов версий (v*.*.*)

**Задачи:**
- Автоматическое создание GitHub Release
- Генерация changelog
- Загрузка артефактов сборки

## Локальная разработка

### Быстрая настройка
```bash
# Установка зависимостей
make install

# Настройка pre-commit hooks
make pre-commit-install

# Проверка окружения
make dev-setup
```

### Основные команды

#### Тестирование
```bash
make test           # Все тесты
make test-quick     # Быстрые тесты
make test-agents    # Тесты агентов
make test-envs      # Тесты окружений
make test-signals   # Тесты сигналов
make test-bench     # Бенчмарки
```

#### Качество кода
```bash
make lint           # Линтинг (flake8, mypy, bandit)
make format         # Форматирование (black, isort)
make format-check   # Проверка форматирования
make security       # Проверка безопасности
```

#### Документация
```bash
make docs           # Генерация документации
make docs-serve     # Запуск локального сервера документации
make docs-coverage  # Проверка покрытия документации
```

#### Сборка и публикация
```bash
make build          # Сборка пакета
make publish-test   # Публикация в Test PyPI
make publish        # Публикация в PyPI (с подтверждением)
```

#### Версионирование
```bash
make version        # Показать текущую версию
make bump-patch     # Увеличить patch версию
make bump-minor     # Увеличить minor версию
make bump-major     # Увеличить major версию
```

### Pre-commit hooks

Проект настроен с pre-commit hooks для автоматической проверки кода перед коммитом:

```bash
# Установка hooks
pre-commit install

# Ручной запуск на всех файлах
pre-commit run --all-files

# Или через Makefile
make pre-commit
```

## Workflow для разработчиков

### 1. Создание новой функции
```bash
# Создание ветки
git checkout -b feature/new-feature

# Разработка с проверками
make test-quick     # Быстрая проверка
make lint          # Проверка качества кода
make format        # Форматирование

# Коммит (pre-commit hooks запустятся автоматически)
git add .
git commit -m "feat: добавить новую функцию"

# Push (запустится quick-check workflow)
git push origin feature/new-feature
```

### 2. Pull Request
- Создайте PR в GitHub
- Автоматически запустятся все проверки
- После одобрения и merge в main запустится полный CI/CD

### 3. Релиз
```bash
# Обновление версии
make bump-minor  # или bump-patch/bump-major

# Создание тега
git tag v1.2.0
git push origin v1.2.0

# Автоматически создастся GitHub Release и публикация в PyPI
```

## Конфигурация IDE

### VS Code
Проект включает настройки VS Code:
- `.vscode/settings.json` - настройки редактора
- `.vscode/tasks.json` - задачи для выполнения команд
- `.vscode/launch.json` - конфигурации отладки
- `.vscode/extensions.json` - рекомендуемые расширения

### PyCharm
Рекомендуемые настройки:
- Интерпретатор: `.venv/bin/python`
- Форматтер: Black
- Линтер: Flake8
- Type checker: MyPy

## Troubleshooting

### Проблемы с тестами
```bash
# Очистка кэша
make clean

# Переустановка зависимостей
make install

# Проверка окружения
python -c "import tensoraerospace; print('OK')"
```

### Проблемы с форматированием
```bash
# Автоматическое исправление
make format

# Проверка без изменений
make format-check
```

### Проблемы с безопасностью
```bash
# Проверка уязвимостей
make security

# Обновление зависимостей
poetry update
```

## Мониторинг и уведомления

- **GitHub Actions**: Автоматические уведомления о статусе сборки
- **Dependabot**: Еженедельные обновления зависимостей
- **Security alerts**: Уведомления о уязвимостях

## Дополнительные ресурсы

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Black Documentation](https://black.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)