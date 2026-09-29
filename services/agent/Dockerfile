# Voice agent (ARI app). CPU-only: STT and TTS are separate GPU services, the LLM is an
# OpenAI/Ollama-compatible endpoint. Silero VAD needs torch, so use the small CPU wheels.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install aiohttp numpy soundfile scipy silero-vad prometheus-client requests

WORKDIR /app
COPY *.py console.html ./
RUN useradd -r -u 10001 agent
USER 10001
ENV STT_BACKEND=http CONSOLE_HOST=0.0.0.0 CONSOLE_AUDIO_DIR=/tmp/console-audio
EXPOSE 8090 9464
CMD ["python", "main.py"]
