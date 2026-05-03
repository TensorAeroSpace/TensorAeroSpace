FROM python:3.14-slim-bookworm AS wheel-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY pyproject.toml readme.md README.ru-ru.md LICENSE ./
COPY tensoraerospace ./tensoraerospace

RUN python -m pip install --upgrade pip build && \
    python -m build --wheel --outdir /tmp/dist

FROM python:3.14-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYGAME_HIDE_SUPPORT_PROMPT=1 \
    SDL_VIDEODRIVER=dummy \
    BROWSER_PATH=/usr/bin/chromium \
    TENSORAEROSPACE_EXAMPLES=/workspace/examples

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        chromium \
        curl \
        ffmpeg \
        fontconfig \
        fonts-dejavu \
        git \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgles2 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxrender1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=wheel-builder /tmp/dist/*.whl /tmp/

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install /tmp/tensoraerospace-*.whl && \
    python -m pip install \
        ipykernel \
        ipywidgets \
        jupyterlab \
        nbconvert \
        notebook \
        tqdm && \
    python -m ipykernel install --sys-prefix \
        --name tensoraerospace \
        --display-name "Python (TensorAeroSpace)" && \
    rm -f /tmp/tensoraerospace-*.whl

RUN useradd --create-home --shell /bin/bash --uid 1000 tensor && \
    mkdir -p /workspace/examples /workspace/projects && \
    chown -R tensor:tensor /workspace

WORKDIR /workspace

COPY --chown=tensor:tensor example ./examples

USER tensor

EXPOSE 8888

CMD ["jupyter", "lab", "--notebook-dir=/workspace", "--ip=0.0.0.0", "--no-browser", "--port=8888"]
