FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv && \
    python3 -m pip install --no-cache-dir "poetry>=1.8,<3.0" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip

WORKDIR /app

COPY pyproject.toml poetry.lock readme.md README.ru-ru.md ./

RUN poetry env use python3 && \
    poetry install --with jupyter --no-root --no-ansi && \
    rm -rf /root/.cache/pypoetry

COPY tensoraerospace ./tensoraerospace
COPY example ./example

RUN poetry install --only-root --no-ansi

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app /app

EXPOSE 8888

CMD ["jupyter", "lab", "--notebook-dir=/app", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--port=8888"]
