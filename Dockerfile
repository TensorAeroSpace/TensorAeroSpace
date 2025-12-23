# Combine updates, installations, and clean-up into a single layer to reduce image size
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Combine updates, installations, and clean-up into a single layer to reduce image size
RUN apt-get update && \
    apt-get install -y wget python3 python3-pip python3-dev python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set a work directory
WORKDIR /app

# Copy only required files
COPY tensoraerospace ./tensoraerospace
COPY pyproject.toml poetry.lock readme.md README.ru-ru.md ./ 

# Install dependencies using poetry and remove cache within the same layer
RUN pip3 install poetry && \
    poetry install --with jupyter && \
    rm -rf $HOME/.cache/pip

# Expose the port the app runs on
EXPOSE 8888

# Default: run JupyterLab. Keep ENTRYPOINT minimal so `docker run ... <cmd>`
# does not accidentally append a second Jupyter command/args (which previously
# caused duplicate root_dir errors).
ENTRYPOINT ["poetry", "run"]
CMD ["jupyter", "lab", "--notebook-dir=/app", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--port=8888", "--ServerApp.token=", "--ServerApp.password="]