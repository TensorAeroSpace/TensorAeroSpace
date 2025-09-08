# GitHub Actions Workflows

Этот каталог содержит все GitHub Actions workflows для автоматизации CI/CD процессов проекта TensorAeroSpace.

## 📁 Структура

```
.github/
├── workflows/
│   ├── action.yml          # Основной CI/CD пайплайн
│   ├── quick-check.yml     # Быстрые проверки
│   ├── publish.yml         # Публикация пакетов
│   └── release.yml         # Создание релизов
├── ISSUE_TEMPLATE/         # Шаблоны для issues
├── dependabot.yml          # Настройки Dependabot
├── CODEOWNERS             # Владельцы кода
├── pull_request_template.md # Шаблон PR
├── settings.yml           # Настройки репозитория
└── CI_CD.md              # Документация CI/CD
```

## 🔄 Workflows

### 1. Main CI/CD (`action.yml`)
**Триггеры:** 
  - Push на **все ветки** (`**`)
  - Pull Request в **любые ветки** (`**`)
  - Ручной запуск
- ✅ Тестирование на матрице Python версий и ОС
- 📊 Покрытие кода и отчеты
- 📚 Проверка документации
- 🔒 Сканирование безопасности
- 🏗️ Сборка пакета

### 2. Quick Check (`quick-check.yml`)
**Триггеры:** 
  - Push на **все ветки** (`**`)
  - Pull Request в **любые ветки** (`**`)
- ⚡ Быстрые тесты
- 🎨 Проверка форматирования
- 🔍 Базовый линтинг
- 🛡️ Проверка зависимостей

### 3. Publishing (`publish.yml`)
**Триггеры:** Release, Manual dispatch
- 🧪 Предварительное тестирование
- 📦 Публикация в TestPyPI
- 🚀 Публикация в PyPI
- 📝 Создание GitHub Release

### 4. Release (`release.yml`)
**Триггеры:** Push тегов версий
- 📋 Автогенерация changelog
- 🏗️ Сборка артефактов
- 📦 Создание GitHub Release
- 📤 Загрузка файлов

## 🏷️ Статусы и бейджи

Добавьте эти бейджи в основной README:

```markdown
[![CI/CD](https://github.com/asmazaev/TensorAeroSpace/actions/workflows/action.yml/badge.svg)](https://github.com/asmazaev/TensorAeroSpace/actions/workflows/action.yml)
[![Quick Check](https://github.com/asmazaev/TensorAeroSpace/actions/workflows/quick-check.yml/badge.svg)](https://github.com/asmazaev/TensorAeroSpace/actions/workflows/quick-check.yml)
[![PyPI](https://img.shields.io/pypi/v/tensoraerospace)](https://pypi.org/project/tensoraerospace/)
[![Python](https://img.shields.io/pypi/pyversions/tensoraerospace)](https://pypi.org/project/tensoraerospace/)
[![License](https://img.shields.io/github/license/asmazaev/TensorAeroSpace)](LICENSE)
[![Codecov](https://codecov.io/gh/asmazaev/TensorAeroSpace/branch/main/graph/badge.svg)](https://codecov.io/gh/asmazaev/TensorAeroSpace)
```

## 🔧 Настройка секретов

Для корректной работы workflows необходимо настроить следующие секреты в GitHub:

### Обязательные секреты:
- `PYPI_API_TOKEN` - токен для публикации в PyPI
- `TEST_PYPI_API_TOKEN` - токен для публикации в TestPyPI

### Опциональные секреты:
- `CODECOV_TOKEN` - токен для Codecov (если репозиторий приватный)
- `SLACK_WEBHOOK` - webhook для уведомлений в Slack

## 📋 Переменные окружения

Workflows используют следующие переменные:
- `PYTHON_VERSION` - версия Python по умолчанию (3.10)
- `POETRY_VERSION` - версия Poetry
- `NODE_VERSION` - версия Node.js для некоторых действий

## 🚀 Использование

### Для разработчиков:
1. Создайте feature ветку: `git checkout -b feature/new-feature`
2. Внесите изменения и закоммитьте
3. Push запустит `quick-check` workflow
4. Создайте PR - запустится полный CI/CD

### Для мейнтейнеров:
1. Merge PR в main запустит полный CI/CD
2. Создание тега версии запустит release workflow
3. Manual dispatch позволяет публиковать в TestPyPI

## 🔍 Мониторинг

- **GitHub Actions**: Вкладка Actions в репозитории
- **Codecov**: Отчеты о покрытии кода
- **Dependabot**: Автоматические PR для обновления зависимостей
- **Security**: Вкладка Security для уязвимостей

## 📚 Дополнительная информация

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [PyPI Publishing Guide](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)

## 🤝 Вклад в развитие

При добавлении новых workflows:
1. Следуйте существующим соглашениям об именовании
2. Добавьте соответствующую документацию
3. Протестируйте на feature ветке
4. Обновите этот README