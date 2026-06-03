FROM python:3.9-slim

WORKDIR /app

# System deps needed by opencv/ffmpeg + misc runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY model/ ./model/
COPY inputs/ ./inputs/
COPY outputs/ ./outputs/

CMD ["python", "src/Simulated_Modular_Test_metric.py"]