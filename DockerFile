FROM python:3.10-slim

# Layer 1: System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libusb-1.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Core Python stack
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    openvino==2023.2.0 \
    langchain==0.1.0 \
    mlflow==2.9.2

# Layer 3: Agentic dependencies
RUN pip install --no-cache-dir \
    faiss-cpu==1.7.4 \
    sentence-transformers==2.2.2 \
    transformers==4.36.0

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 5000
CMD ["python", "src/main.py"]
