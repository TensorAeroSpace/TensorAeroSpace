FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04


RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda 
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /app

COPY start.sh start.sh
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip

COPY tensoraerospace /app/tensoraerospace
COPY setup.py /app/setup.py

RUN pip install -e .
RUN pip install jupyter
RUN rm -rf $HOME/.cache/pip


EXPOSE 8888
RUN chmod +x /app/start.sh
ENTRYPOINT "/app/start.sh"