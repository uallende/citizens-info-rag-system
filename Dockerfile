# Nvidia latest ubuntu image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV CUDA_HOME=/usr/local/cuda

# Install CUDA and cuDNN dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    build-essential \
    cuda-cudart-12-4 \
    cuda-nvml-dev-12-4 \
    cuda-command-line-tools-12-4 \
    libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY . .

RUN python3 app/init_data.py
ENV INITIALIZED=true

CMD ["poetry", "run", "streamlit", "run", "--server.address", "0.0.0.0", "app/main.py"]