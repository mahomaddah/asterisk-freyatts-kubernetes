"""Render IVR prompts with automatic quality control ("ASR-verified TTS").

FreyaTTS-small has no speaker embedding: the initial noise *is* the voice, and
long inputs are synthesized clause by clause. In practice two failure modes show
up on banking prompts:
  * repeated / substituted phrases   -> caught by a Whisper round-trip (CER)
  * the voice drifting mid-utterance  -> caught by per-window pitch tracking
Each prompt is rendered with a few sampler settings; the first candidate that
passes both checks wins, otherwise the best-scoring one is kept and flagged.

    python scripts/render_prompts_verified.py
"""
import json
import pathlib
import re
import sys

import jiwer
import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "FreyaTTS"))
from freyatts import FreyaTTS  # noqa: E402
from freyatts.pipeline import normalize  # noqa: E402

PROMPTS = pathlib.Path("scripts/prompts_tr.txt")
OUT_8K = pathlib.Path("telephony/asterisk/sounds/tr/freya")
REPORT = pathlib.Path("bench/results/prompt_qc.json")

MAX_CER = 0.04        # characters, spaces ignored (digits vs words do not count)
MIN_PITCH_HZ = 170    # Leyla sits around 200-330 Hz; drifted clauses fall to ~100-150
CANDIDATES = [        # (euler steps, max words per clause); sampler tweaks rarely fix drift,
    (32, 11), (32, 6), (64, 6),   # so alternative phrasings (see prompts file) are tried too
]


def canon(text: str) -> str:
    text = normalize(text)  # spells out digits the same way the TTS input was
    text = text.replace("I", "ı").replace("İ", "i").lower()
    return re.sub(r"[^\w]", "", text)


def pitch_windows(wav: np.ndarray, sr: int, win_s: float = 1.5) -> list[int]:
    y = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    win = int(win_s * 16000)
    out = []
    for i in range(0, len(y), win):
        f0, _, _ = librosa.pyin(y[i:i + win], fmin=70, fmax=400, sr=16000)
        voiced = f0[~np.isnan(f0)]
        if len(voiced) > 10:          # skip windows that are mostly silence
            out.append(int(np.median(voiced)))
    return out


def main():
    tts = FreyaTTS.from_pretrained("freyavoice/freya-tts", device="mps")
    asr = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    OUT_8K.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = {}

    for line in PROMPTS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, variants = line.split("|", 1)
        tries = [(v.strip(), s, w) for v in variants.split("||") for s, w in CANDIDATES]
        best = None
        for attempt, (text, steps, max_words) in enumerate(tries, 1):
            ref = canon(text)
            tts.max_words = max_words
            wav = tts.synthesize(text, steps=steps)
            segs, _ = asr.transcribe(librosa.resample(wav, orig_sr=48000, target_sr=16000),
                                     language="tr", beam_size=5)
            hyp_text = " ".join(s.text for s in segs).strip()
            cer = jiwer.cer(ref, canon(hyp_text))
            pitches = pitch_windows(wav, 48000)
            low = min(pitches) if pitches else 0
            ok = cer <= MAX_CER and low >= MIN_PITCH_HZ
            score = cer + (0.5 if low < MIN_PITCH_HZ else 0.0)
            print(f"{name:11s} try {attempt} steps={steps:2d} words={max_words:2d}  "
                  f"CER {cer:5.1%}  min pitch {low:3d} Hz  {'PASS' if ok else 'fail'}", flush=True)
            if best is None or score < best["score"]:
                best = dict(score=score, wav=wav, text=text, cer=cer, min_pitch=low, steps=steps,
                            max_words=max_words, attempt=attempt, heard=hyp_text, passed=ok)
            if ok:
                break

        wav8 = np.clip(librosa.resample(best.pop("wav"), orig_sr=48000, target_sr=8000), -1, 1)
        sf.write(OUT_8K / f"{name}.wav", wav8, 8000, subtype="PCM_16")
        best.pop("score")
        report[name] = best

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    passed = sum(r["passed"] for r in report.values())
    print(f"\n{passed}/{len(report)} prompts passed QC -> {REPORT}")


if __name__ == "__main__":
    main()
