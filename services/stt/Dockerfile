# faster-whisper STT on CUDA 12 + cuDNN 9 (CTranslate2 4.x), weights baked in, no egress at runtime.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG STT_MODEL=small
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models STT_MODEL=${STT_MODEL}

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libsndfile1 \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m pip install --upgrade pip \
 && python3 -m pip install faster-whisper soundfile scipy fastapi "uvicorn[standard]" prometheus-client

RUN python3 -c "from faster_whisper.utils import download_model; download_model('${STT_MODEL}')"

ENV HF_HUB_OFFLINE=1
WORKDIR /app
COPY app.py .
RUN useradd -r -u 10001 stt && chown -R stt /models
USER 10001
EXPOSE 8081
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]
