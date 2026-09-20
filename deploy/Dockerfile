# CPU / local prototype image (mock pipeline). For real MagiCut use the GPU host + setup_ubuntu.sh.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart pillow opencv-python-headless httpx aiofiles pyyaml numpy

COPY server ./server
COPY clients/mobile_web ./clients/mobile_web
COPY configs ./configs
COPY server_data/.gitkeep ./server_data/.gitkeep

ENV MAGICUT_PIPELINE_MODE=mock \
    MAGICUT_API_HOST=0.0.0.0 \
    MAGICUT_API_PORT=8080 \
    PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
