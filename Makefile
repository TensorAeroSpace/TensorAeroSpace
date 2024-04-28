docker_build:
	docker build -t tensor_aero_space .

docker_debug:
	docker run -v ${PWD}/example:/app/example -p 8888:8888 -it tensor_aero_space

docker_build_macos:
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

test:
	poetry run ruff check tensoraerospace