# Makefile для TensorAeroSpace
# Использование: make <команда>

.PHONY: help install test lint format security docs clean build publish

# Цвета для вывода
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Укажите файлы, которые нужно протестировать
NOTEBOOK_FILES = example/example-env-LinearLongitudinalB747.ipynb example/example-env-LinearLongitudinalF16.ipynb

help: ## Показать справку по командам
	@echo "$(BLUE)TensorAeroSpace - Команды разработки$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-25s$(RESET) %s\n", $$1, $$2}'

# === УСТАНОВКА И НАСТРОЙКА ===
install: ## Установить зависимости
	@echo "$(BLUE)Установка зависимостей...$(RESET)"
	poetry install --with dev,test
	poetry run pre-commit install

install_dev: ## Установить dev зависимости (legacy)
	poetry install

install-ci: ## Установить зависимости для CI
	@echo "$(BLUE)Установка зависимостей для CI...$(RESET)"
	poetry install --with dev,test --no-interaction

# === ТЕСТИРОВАНИЕ ===
test: ## Запустить все тесты
	@echo "$(BLUE)Запуск всех тестов...$(RESET)"
	poetry run pytest tests/ -v --cov=tensoraerospace --cov-report=html --cov-report=term

test-quick: ## Запустить быстрые тесты
	@echo "$(BLUE)Запуск быстрых тестов...$(RESET)"
	poetry run pytest tests/envs/ -x --maxfail=3 -q

run_env_test: ## Тестировать окружения (legacy)
	@echo "Running enviroments tests..."
	poetry run pytest -s tests/envs 

run_signal_test: ## Тестировать сигналы (legacy)
	@echo "Running signals tests..."
	poetry run pytest -s tests/signals 

run_bench_test: ## Тестировать бенчмарки (legacy)
	@echo "Running bench tests..."
	poetry run pytest -s tests/bench 

test-agents: ## Тестировать агентов
	@echo "$(BLUE)Тестирование агентов...$(RESET)"
	poetry run pytest tests/agents/ -v

test-envs: ## Тестировать окружения
	@echo "$(BLUE)Тестирование окружений...$(RESET)"
	poetry run pytest tests/envs/ -v

test-signals: ## Тестировать сигналы
	@echo "$(BLUE)Тестирование сигналов...$(RESET)"
	poetry run pytest tests/signals/ -v

test-bench: ## Тестировать бенчмарки
	@echo "$(BLUE)Тестирование бенчмарков...$(RESET)"
	poetry run pytest tests/bench/ -v

jupyter_example_test: ## Тестировать Jupyter примеры
	@echo "Starting tests for Jupyter Notebook files..."
	@for nb_file in $(NOTEBOOK_FILES); do \
		echo "Testing $$nb_file..."; \
		poetry run jupyter nbconvert --to notebook --execute --inplace $$nb_file --ExecutePreprocessor.timeout=600; \
		if [ $$? -ne 0 ]; then \
			echo "Test failed for $$nb_file"; \
			exit 1; \
		fi; \
	done
	@echo "All tests passed successfully!"

# === КАЧЕСТВО КОДА ===
lint: ## Проверить код линтерами
	@echo "$(BLUE)Проверка кода...$(RESET)"
	poetry run flake8 tensoraerospace --max-line-length=88 --extend-ignore=E203,W503
	poetry run mypy tensoraerospace --ignore-missing-imports
	poetry run bandit -r tensoraerospace -f json

format: ## Форматировать код
	@echo "$(BLUE)Форматирование кода...$(RESET)"
	poetry run black tensoraerospace tests example
	poetry run isort tensoraerospace tests example --profile=black

fmt: ## Форматировать код (legacy)
	poetry run ruff check --fix tensoraerospace --select I

clean_code: ## Очистить код (legacy)
	poetry run ruff check --fix tensoraerospace

format-check: ## Проверить форматирование без изменений
	@echo "$(BLUE)Проверка форматирования...$(RESET)"
	poetry run black --check tensoraerospace tests examples
	poetry run isort --check-only tensoraerospace tests examples --profile=black

# === БЕЗОПАСНОСТЬ ===
security: ## Проверить безопасность
	@echo "$(BLUE)Проверка безопасности...$(RESET)"
	poetry run safety check
	poetry run bandit -r tensoraerospace

# === ДОКУМЕНТАЦИЯ ===
docs: ## Сгенерировать документацию
	@echo "$(BLUE)Генерация документации...$(RESET)"
	poetry run docstr-coverage tensoraerospace --skip-magic --skip-init --fail-under=70
	cd docs && poetry run make html

build_docs: ## Собрать документацию (legacy)
	cd docs && poetry run make html

check_doc_quality: ## Проверить качество документации (legacy)
	poetry run docstr-coverage ./tensoraerospace --skip-magic --skip-init --skip-file-doc  --fail-under=90.0

docs-serve: ## Запустить сервер документации
	@echo "$(BLUE)Запуск сервера документации...$(RESET)"
	cd docs/_build/html && python -m http.server 8000

# === СБОРКА И ПУБЛИКАЦИЯ ===
clean: ## Очистить временные файлы
	@echo "$(BLUE)Очистка...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

build: ## Собрать пакет
	@echo "$(BLUE)Сборка пакета...$(RESET)"
	poetry build

publish-test: ## Опубликовать в TestPyPI
	@echo "$(BLUE)Публикация в TestPyPI...$(RESET)"
	poetry config repositories.testpypi https://test.pypi.org/legacy/
	poetry publish -r testpypi

publish: ## Опубликовать в PyPI
	@echo "$(YELLOW)Публикация в PyPI...$(RESET)"
	@echo "$(RED)Внимание! Это опубликует пакет в продакшн!$(RESET)"
	@read -p "Продолжить? [y/N] " confirm && [ "$$confirm" = "y" ]
	poetry publish

# === DOCKER ===
docker_build: ## Собрать Docker образ
	docker build -t tensor_aero_space .  --platform=linux/amd64

docker_debug: ## Запустить Docker в debug режиме
	docker run -v ${PWD}/example:/app/example -p 8888:8888 -it tensor_aero_space

# === КОМПЛЕКСНЫЕ КОМАНДЫ ===
pre_commit: fmt test ## Pre-commit проверки (legacy)

pre-commit: ## Запустить pre-commit hooks
	@echo "$(BLUE)Запуск pre-commit hooks...$(RESET)"
	poetry run pre-commit run --all-files

check-all: format-check lint security test docs ## Полная проверка кода

ci-test: install-ci test lint security docs ## CI pipeline тестирование

dev-setup: install ## Настройка среды разработки
	@echo "$(GREEN)Среда разработки настроена!$(RESET)"
	@echo "$(BLUE)Доступные команды:$(RESET)"
	@make help

# === ВЕРСИОНИРОВАНИЕ ===
version: ## Показать версию
	@poetry version

bump-patch: ## Увеличить patch версию
	@poetry version patch
	@echo "$(GREEN)Версия обновлена до: $(shell poetry version -s)$(RESET)"

bump-minor: ## Увеличить minor версию
	@poetry version minor
	@echo "$(GREEN)Версия обновлена до: $(shell poetry version -s)$(RESET)"

bump-major: ## Увеличить major версию
	@poetry version major
	@echo "$(GREEN)Версия обновлена до: $(shell poetry version -s)$(RESET)"

# === АЛИАСЫ ===
t: test ## Алиас для test
f: format ## Алиас для format
l: lint ## Алиас для lint
c: clean ## Алиас для clean
b: build ## Алиас для build