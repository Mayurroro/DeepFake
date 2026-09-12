FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serve.txt .
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip3 install --no-cache-dir -r requirements-serve.txt

COPY . .

EXPOSE 8000 8501

# ponytail: single service image; API server runs via `docker run <img> uvicorn ...`
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]