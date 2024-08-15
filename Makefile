docker_debug:
	docker run -v ${PWD}/example:/app/example -p 8888:8888 -it tensor_aero_space

docker_build:
	docker build -t tensor_aero_space .  --platform=linux/amd64

install_dev:
	poetry install

build_docs:
	cd docs && poetry run make html

clean_code:
	poetry run ruff check --fix tensoraerospace

check_doc_quality:
	poetry run docstr-coverage ./tensoraerospace --skip-magic --skip-init --skip-file-doc  --fail-under=90.0

run_env_test:
	@echo "Running enviroments tests..."
	poetry run pytest -s tests/envs 

run_signal_test:
	@echo "Running signals tests..."
	poetry run pytest -s tests/signals 

run_bench_test:
	@echo "Running bench tests..."
	poetry run pytest -s tests/bench 

fmt:
	poetry run ruff check --fix tensoraerospace --select I

test:
	poetry run pytest

pre_commit: fmt test




# Makefile

# Укажите файлы, которые нужно протестировать
NOTEBOOK_FILES = example/example-env-LinearLongitudinalB747.ipynb example/example-env-LinearLongitudinalF16.ipynb

# Команда для тестирования Jupyter Notebook файлов
jupyter_example_test:
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

.PHONY: jupyter_example_test