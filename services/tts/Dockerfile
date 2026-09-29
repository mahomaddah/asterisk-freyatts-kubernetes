# FreyaTTS inference service, CUDA build for Pascal-era GPUs (GTX 10xx) and newer.
#
# Model weights are baked into the image and the hub is forced offline at runtime:
# the pod needs no internet egress, which is how it has to run inside a bank.
FROM python:3.11-slim

ARG FREYATTS_COMMIT=146d36c1cb6660646be57d31339db4eed9315de3
# PyTorch's CUDA 12.8+ wheels dropped sm_50-sm_61 (Maxwell/Pascal); cu126 still ships them.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models FREYATTS_PATH=/opt/FreyaTTS

RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.8.0 --index-url ${TORCH_INDEX} \
 && pip install numpy einops soundfile librosa huggingface_hub safetensors pydantic \
                fastapi "uvicorn[standard]" prometheus-client

# FreyaTTS only needs VoxCPM2's AudioVAE decoder. voxcpm's top-level __init__ imports its
# full TTS stack (gradio, funasr, modelscope, transformers...), so install it without deps
# and drop that import: several GB smaller, same decoder code.
RUN pip install --no-deps "voxcpm==2.0.3" \
 && echo "" > "$(python -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("voxcpm").origin))')/__init__.py"

RUN git clone https://github.com/freyavoiceai/FreyaTTS.git ${FREYATTS_PATH} \
 && git -C ${FREYATTS_PATH} checkout ${FREYATTS_COMMIT} \
 && rm -rf ${FREYATTS_PATH}/.git ${FREYATTS_PATH}/training

RUN python -c "from huggingface_hub import hf_hub_download as d; \
d('freyavoice/freya-tts','config.json'); d('freyavoice/freya-tts','model.safetensors'); \
d('openbmb/VoxCPM2','audiovae.pth')"

# numba (via librosa) must cache somewhere writable for the non-root user
ENV HF_HUB_OFFLINE=1 NUMBA_CACHE_DIR=/tmp/numba
WORKDIR /app
COPY app.py .
RUN useradd -r -u 10001 tts && chown -R tts /models
USER 10001
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
