FROM python:3.10-slim

# Install system deps for some scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./mlops_pipeline /app/mlops_pipeline
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy repository root files (dataset may be mounted in dev; in prod it will be baked into image if needed)
COPY Base_de_datos.csv /app/Base_de_datos.csv

EXPOSE 8000 8501

CMD ["/bin/bash"]
