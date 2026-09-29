"""Whisper STT service (faster-whisper / CTranslate2).

POST /v1/transcribe   body: audio/wav (any rate, mono)  ->  {"text": "...", "seconds": 0.31}
"""
import io
import os
import time

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from faster_whisper import WhisperModel
from prometheus_client import Histogram, make_asgi_app
from scipy.signal import resample_poly

MODEL = os.environ.get("STT_MODEL", "small")
DEVICE = os.environ.get("STT_DEVICE", "cuda")
# Pascal (sm_61) has no fast fp16 path; int8 runs on CTranslate2 GPUs from sm_61 up.
COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "int8")
LANGUAGE = os.environ.get("STT_LANGUAGE", "tr")

model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
model.transcribe(np.zeros(16000, dtype=np.float32), language=LANGUAGE)  # warm-up

SECONDS = Histogram("stt_transcribe_seconds", "Wall time per request", ["model"],
                    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0))

app = FastAPI(title="whisper-stt")
app.mount("/metrics", make_asgi_app())


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL, "device": DEVICE, "compute_type": COMPUTE_TYPE}


@app.post("/v1/transcribe")
async def transcribe(request: Request):
    audio, rate = sf.read(io.BytesIO(await request.body()), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:
        audio = resample_poly(audio, 16000, rate).astype(np.float32)
    t0 = time.perf_counter()
    segments, _ = model.transcribe(audio, language=LANGUAGE, beam_size=1, vad_filter=False)
    text = " ".join(s.text for s in segments).strip()
    took = time.perf_counter() - t0
    SECONDS.labels(MODEL).observe(took)
    return {"text": text, "seconds": round(took, 3), "model": MODEL}
