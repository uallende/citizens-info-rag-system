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

# Update alternatives to use python3 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /app
COPY ./app .

RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry config virtualenvs.in-project true \
    && poetry install --no-interaction --no-ansi --no-root

CMD ["poetry", "run", "streamlit", "run", "--server.port", "8501", "--server.address", "0.0.0.0", "main.py"]