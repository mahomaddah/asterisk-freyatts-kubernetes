"""STT latency on telephone audio (8 kHz prompt upsampled to 16 kHz, as the agent will see it).

    python bench/stt_bench.py
"""
import pathlib
import statistics
import time

import librosa

CLIPS = sorted(pathlib.Path("telephony/asterisk/sounds/tr/freya").glob("*.wav"))
audio = [librosa.load(c, sr=16000)[0] for c in CLIPS]


def bench(name, fn):
    fn(audio[0])  # warm-up
    times = []
    for clip, a in zip(CLIPS, audio):
        t0 = time.perf_counter()
        text = fn(a)
        times.append(time.perf_counter() - t0)
        print(f"  {name:28s} {times[-1]*1000:5.0f} ms  {len(a)/16000:4.1f}s audio | {text[:60]}")
    print(f"{name}: median {statistics.median(times)*1000:.0f} ms\n")


from faster_whisper import WhisperModel  # noqa: E402

for size in ("small", "large-v3-turbo"):
    m = WhisperModel(size, device="cpu", compute_type="int8")
    bench(f"faster-whisper {size} cpu", lambda a, m=m: " ".join(
        s.text for s in m.transcribe(a, language="tr", beam_size=1)[0]).strip())

import mlx_whisper  # noqa: E402

for repo in ("mlx-community/whisper-small-mlx", "mlx-community/whisper-large-v3-turbo"):
    bench(f"mlx {repo.split('/')[1]}", lambda a, r=repo: mlx_whisper.transcribe(
        a, path_or_hf_repo=r, language="tr")["text"].strip())
