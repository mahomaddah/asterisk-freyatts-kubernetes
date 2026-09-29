"""Speech I/O backends: VAD, STT and the TTS client.

STT is pluggable: mlx-whisper on Apple Silicon (Metal) for the home lab,
faster-whisper on CUDA for the Kubernetes GPU node.
"""
import io

import aiohttp
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from silero_vad import load_silero_vad

SAMPLE_RATE = 8000   # telephony audio (G.711) end to end
STT_RATE = 16000     # Whisper's input rate
VAD_WINDOW = 256     # samples at 8 kHz (32 ms), what Silero expects


def to_stt_rate(audio: np.ndarray) -> np.ndarray:
    return resample_poly(audio, STT_RATE // SAMPLE_RATE, 1).astype(np.float32)


class VAD:
    """Silero VAD over a stream of 8 kHz float32 audio, one instance per call."""

    def __init__(self, threshold: float = 0.5):
        self.model = load_silero_vad()
        self.threshold = threshold
        self.buf = np.zeros(0, dtype=np.float32)

    def feed(self, pcm: np.ndarray) -> list[bool]:
        """Returns one speech/non-speech flag per completed 32 ms window."""
        self.buf = np.concatenate([self.buf, pcm])
        flags = []
        while len(self.buf) >= VAD_WINDOW:
            window, self.buf = self.buf[:VAD_WINDOW], self.buf[VAD_WINDOW:]
            with torch.no_grad():
                prob = self.model(torch.from_numpy(window), SAMPLE_RATE).item()
            flags.append(prob >= self.threshold)
        return flags


class MlxWhisper:
    def __init__(self, model: str):
        import mlx_whisper
        self._mlx = mlx_whisper
        self.repo = model if "/" in model else f"mlx-community/whisper-{model}-mlx"
        self.transcribe(np.zeros(STT_RATE, dtype=np.float32))  # load + compile once

    def transcribe(self, audio: np.ndarray) -> str:
        return self._mlx.transcribe(audio, path_or_hf_repo=self.repo, language="tr")["text"].strip()


class FasterWhisper:
    def __init__(self, model: str, device: str = "cuda"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device=device, compute_type="float16" if device == "cuda" else "int8")

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language="tr", beam_size=1, vad_filter=False)
        return " ".join(s.text for s in segments).strip()


class HttpSTT:
    """Remote STT service (services/stt): keeps the GPU out of the agent pod."""

    def __init__(self, url: str):
        import requests
        self.session = requests.Session()
        self.url = url.rstrip("/") + "/v1/transcribe"

    def transcribe(self, audio: np.ndarray) -> str:
        buf = io.BytesIO()
        sf.write(buf, audio, STT_RATE, subtype="PCM_16", format="WAV")
        r = self.session.post(self.url, data=buf.getvalue(), headers={"Content-Type": "audio/wav"}, timeout=15)
        r.raise_for_status()
        return r.json()["text"]


def load_stt(backend: str, model: str):
    if backend == "http":
        return HttpSTT(model)  # model = service URL
    if backend == "mlx":
        return MlxWhisper(model)
    if backend == "faster-whisper":
        return FasterWhisper(model, device="cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unknown STT backend {backend}")


class TTSClient:
    """HTTP client for the TTS service with an in-memory cache.

    Fixed sentences (greeting, tool confirmations) are pre-rendered at start-up
    with warm(), so the most common replies cost no synthesis time on a call.
    """

    def __init__(self, url: str, cache_size: int = 256):
        self.url = url.rstrip("/") + "/v1/synthesize"
        self.cache: dict[str, np.ndarray] = {}
        self.cache_size = cache_size
        self.last_hit = False

    async def warm(self, phrases: list[str]):
        for text in phrases:
            await self.synthesize(text)

    async def synthesize(self, text: str) -> np.ndarray:
        self.last_hit = text in self.cache
        if self.last_hit:
            return self.cache[text]
        audio = await self._synthesize(text)
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[text] = audio
        return audio

    async def _synthesize(self, text: str) -> np.ndarray:
        async with aiohttp.ClientSession() as s:
            async with s.post(self.url, json={"text": text, "sample_rate": SAMPLE_RATE},
                              timeout=aiohttp.ClientTimeout(total=30)) as r:
                r.raise_for_status()
                audio, _ = sf.read(io.BytesIO(await r.read()), dtype="float32")
                return audio
