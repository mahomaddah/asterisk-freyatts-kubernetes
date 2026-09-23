"""FreyaTTS HTTP service.

POST /v1/synthesize  {"text": "...", "sample_rate": 8000}  ->  audio/wav (PCM16 mono)

Telephony legs are 8 kHz, so the service resamples FreyaTTS' 48 kHz output
server-side and callers never ship wideband audio they cannot use.
"""
import io
import os
import sys
import time

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Histogram, make_asgi_app
from pydantic import BaseModel, Field

# default: FreyaTTS cloned next to this repo (containers set FREYATTS_PATH)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get("FREYATTS_PATH", os.path.join(REPO_ROOT, "..", "FreyaTTS")))
from freyatts import FreyaTTS  # noqa: E402

MODEL = os.environ.get("FREYATTS_MODEL", "freyavoice/freya-tts")
SUPPORTED_RATES = {8000, 16000, 24000, 48000}


def pick_device() -> str:
    wanted = os.environ.get("TTS_DEVICE", "auto")
    if wanted != "auto":
        return wanted
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
tts = FreyaTTS.from_pretrained(MODEL, device=DEVICE)
tts.synthesize("Hazır.")  # warm up kernels before the first real call

SYNTH_SECONDS = Histogram("tts_synthesis_seconds", "Wall time per request", ["device"])
RTF = Histogram("tts_real_time_factor", "Synthesis wall time / audio duration", ["device"],
                buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0))

app = FastAPI(title="freya-tts")
app.mount("/metrics", make_asgi_app())


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    sample_rate: int = 8000


@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": DEVICE, "model": MODEL}


@app.post("/v1/synthesize")
def synthesize(req: SynthesizeRequest):
    if req.sample_rate not in SUPPORTED_RATES:
        raise HTTPException(400, f"sample_rate must be one of {sorted(SUPPORTED_RATES)}")

    t0 = time.perf_counter()
    wav = tts.synthesize(req.text)
    wall = time.perf_counter() - t0

    duration = len(wav) / tts.sample_rate
    SYNTH_SECONDS.labels(DEVICE).observe(wall)
    RTF.labels(DEVICE).observe(wall / max(duration, 1e-6))

    if req.sample_rate != tts.sample_rate:
        wav = librosa.resample(wav, orig_sr=tts.sample_rate, target_sr=req.sample_rate)
    wav = np.clip(wav, -1.0, 1.0)

    buf = io.BytesIO()
    sf.write(buf, wav, req.sample_rate, subtype="PCM_16", format="WAV")
    return Response(
        buf.getvalue(),
        media_type="audio/wav",
        headers={"X-Synthesis-Seconds": f"{wall:.3f}", "X-Audio-Seconds": f"{duration:.3f}"},
    )
