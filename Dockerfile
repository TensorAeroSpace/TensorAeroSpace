# CUDA + cuDNN base on Ubuntu 22.04 (system Python is 3.10).
# We install Python 3.11 via deadsnakes so that `poetry env use python3.11`
# matches the recommended interpreter from README and the project's
# `pyproject.toml` (python = ">=3.10,<3.13").
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install Python 3.11 from deadsnakes PPA, plus pip/poetry build prerequisites.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates wget curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils && \
    wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3.11 /tmp/get-pip.py && \
    rm /tmp/get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata first so the dependency-install layer is cached
# independently of source changes.
COPY pyproject.toml poetry.lock readme.md README.ru-ru.md ./

# Install poetry under python3.11, then create the project venv on python3.11
# explicitly. Without `poetry env use python3.11`, poetry would default to the
# system `python3` (3.10 on ubuntu22.04), which technically works but does not
# match the recommended interpreter advertised in README.
RUN python3.11 -m pip install "poetry>=1.8,<3.0" && \
    poetry env use python3.11 && \
    poetry install --with jupyter --no-root && \
    rm -rf /root/.cache/pip /root/.cache/pypoetry

# Copy source last so app changes don't bust the dep-install cache layer.
COPY tensoraerospace ./tensoraerospace
COPY example ./example

# Install the package itself now that the source is present.
RUN poetry install --with jupyter --only-root

EXPOSE 8888

# `ENTRYPOINT ["poetry", "run"]` keeps `docker run ... <cmd>` ergonomic:
# the CMD below is appended as the actual command. Previously a hard-coded
# ENTRYPOINT with full jupyter args caused duplicate `--notebook-dir` when
# users passed an extra command.
#
# NOTE: JupyterLab generates a random access token at startup and prints it
# to the container logs. Retrieve it via `docker logs <container>` (or run
# `jupyter server list` inside the container). Do not disable the token on
# network-reachable deployments.
ENTRYPOINT ["poetry", "run"]
CMD ["jupyter", "lab", "--notebook-dir=/app", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--port=8888"]
