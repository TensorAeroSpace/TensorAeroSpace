FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04


RUN apt-get update && \
    apt-get install -y wget python3 python3-pip python3-dev python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install poetry
WORKDIR /app

COPY tensoraerospace /app/tensoraerospace
COPY pyproject.toml /app/pyproject.toml
COPY poetry.lock /app/poetry.lock
COPY readme.md /app/readme.md

RUN poetry install
RUN rm -rf $HOME/.cache/pip
RUN poetry add ipykernel
RUN poetry add jupyter

EXPOSE 8888
COPY start.sh start.sh
RUN chmod +x /app/start.sh
ENTRYPOINT [ "/app/start.sh" ] 