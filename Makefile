docker_build:
	docker build -t tensor_aero_space .

docker_debug:
	docker run -v ${PWD}/example:/app/example -p 8888:8888 -it tensor_aero_space

docker_build_macos:
	docker build -t tensor_aero_space .  --platform=linux/amd64

build_docs:
	poetry install
	cd docs && poetry run make html

clean_code:
	poetry run ruff check --fix tensoraerospace