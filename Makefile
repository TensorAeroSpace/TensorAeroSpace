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


run_env_test:
	@echo "Running enviroments tests..."
	poetry run pytest -s tests/envs 
