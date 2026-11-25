"""Render IVR prompts with automatic quality control ("ASR-verified TTS").

FreyaTTS-small has no speaker embedding: the initial noise *is* the voice, and
long inputs are synthesized clause by clause. In practice three failure modes
show up on banking prompts:
  * repeated / substituted phrases   -> caught by a Whisper round-trip (CER)
  * the voice drifting mid-utterance  -> caught by per-window pitch tracking
  * words that only break on the line -> caught by a second round-trip at 8 kHz
The last one is easy to miss. A caller never hears the 48 kHz render: Asterisk
plays 8 kHz G.711, and that band costs us consonants. "Freya Bank'a hoş
geldiniz" passed at 48 kHz (CER 1.8%) and came back as "Fiyaya Bank" at 8 kHz
-- the model dropping the company's own name on the phone line. So every
candidate is judged at the rate it will be played at, not the rate it is
generated at. Splitting that prompt into shorter clauses (max_words=6) fixes it.

Each prompt is rendered with a few sampler settings; the first candidate that
passes all checks wins, otherwise the best-scoring one is kept and flagged.

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

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--prompts", default="scripts/prompts_tr.txt")
_ap.add_argument("--out", default="telephony/asterisk/sounds/tr/freya")
_ap.add_argument("--report", default="bench/results/prompt_qc.json")
_ap.add_argument("--manifest", action="store_true",
                 help="also write manifest.json (text -> file) for the agent's phrase cache; "
                      "its keys must match the constants in services/agent/")
_args = _ap.parse_args()
PROMPTS = pathlib.Path(_args.prompts)
OUT_8K = pathlib.Path(_args.out)
REPORT = pathlib.Path(_args.report)

MAX_CER = 0.04        # characters, spaces ignored (digits vs words do not count); both rates
LINE_SR = 8000        # what Asterisk actually plays (G.711); QC has to pass here too
MIN_PITCH_HZ = 170    # Leyla sits around 200-330 Hz; drifted clauses fall to ~100-150
CANDIDATES = [        # (euler steps, max words per clause); sampler tweaks rarely fix drift,
    (32, 11), (32, 6), (64, 6),   # so alternative phrasings (see prompts file) are tried too
]


def canon(text: str) -> str:
    text = normalize(text)  # spells out digits the same way the TTS input was
    text = text.replace("I", "ı").replace("İ", "i").lower()
    return re.sub(r"[^\w]", "", text)


def spelling(variant: str) -> tuple[str, str]:
    """Split `real text ~ what to feed the model` into (text, say).

    A Turkish-trained model reading "Freya" has to get the "Fr" cluster right, and on the
    8 kHz line it often does not. Writing "Freyya" for the model fixes the audio without
    changing what we claim was said: CER is still scored against "Freya", and the agent's
    phrase string (and manifest key) stays the real spelling."""
    text, _, say = variant.partition("~")
    return text.strip(), (say.strip() or text.strip())


def heard(asr: WhisperModel, wav: np.ndarray, sr: int) -> str:
    y = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
    segs, _ = asr.transcribe(y, language="tr", beam_size=5)
    return " ".join(s.text for s in segs).strip()


def to_line(wav: np.ndarray) -> np.ndarray:
    return np.clip(librosa.resample(wav, orig_sr=48000, target_sr=LINE_SR), -1, 1)


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
        tries = [(*spelling(v), s, w) for v in variants.split("||") for s, w in CANDIDATES]
        best = None
        for attempt, (text, say, steps, max_words) in enumerate(tries, 1):
            ref = canon(text)              # always scored against the real spelling
            tts.max_words = max_words
            wav = tts.synthesize(say, steps=steps)
            wav8 = to_line(wav)
            hyp_text = heard(asr, wav, 48000)
            hyp_line = heard(asr, wav8, LINE_SR)
            cer = jiwer.cer(ref, canon(hyp_text))
            cer_line = jiwer.cer(ref, canon(hyp_line))
            pitches = pitch_windows(wav, 48000)
            low = min(pitches) if pitches else 0
            ok = cer <= MAX_CER and cer_line <= MAX_CER and low >= MIN_PITCH_HZ
            score = cer + cer_line + (0.5 if low < MIN_PITCH_HZ else 0.0)
            print(f"{name:11s} try {attempt} steps={steps:2d} words={max_words:2d}  "
                  f"CER {cer:5.1%} studio / {cer_line:5.1%} line  min pitch {low:3d} Hz  "
                  f"{'PASS' if ok else 'fail'}{'  (spelled out)' if say != text else ''}",
                  flush=True)
            if best is None or score < best["score"]:
                best = dict(score=score, wav8=wav8, text=text, cer=cer, cer_line=cer_line,
                            min_pitch=low, steps=steps, max_words=max_words, attempt=attempt,
                            heard=hyp_text, heard_line=hyp_line, passed=ok)
                if say != text:
                    best["spoken_as"] = say
            if ok:
                break

        sf.write(OUT_8K / f"{name}.wav", best.pop("wav8"), LINE_SR, subtype="PCM_16")
        best.pop("score")
        report[name] = best

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if _args.manifest:
        manifest = {r["text"]: f"{name}.wav" for name, r in report.items()}
        (OUT_8K / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    passed = sum(r["passed"] for r in report.values())
    print(f"\n{passed}/{len(report)} prompts passed QC at 48 kHz and {LINE_SR} Hz -> {REPORT}")


if __name__ == "__main__":
    main()
