"""Intelligibility of FreyaTTS prompts, wideband vs telephone band.

Transcribes each prompt with Whisper and reports WER/CER against the input text,
once for the 48 kHz model output and once for the 8 kHz version Asterisk plays.

    python bench/tts_roundtrip.py
"""
import pathlib
import re
import sys

import jiwer
from faster_whisper import WhisperModel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "FreyaTTS"))
from freyatts.pipeline import normalize  # noqa: E402  same text the model actually read

PROMPTS = pathlib.Path("scripts/prompts_tr.txt")
SETS = {"48k": pathlib.Path("bench/out/prompts48"), "8k": pathlib.Path("telephony/asterisk/sounds/tr/freya")}


def canon(text: str) -> str:
    text = text.replace("I", "ı").replace("İ", "i").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
totals = {k: ([], []) for k in SETS}

for line in PROMPTS.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    name, text = line.split("|", 1)
    ref = canon(normalize(text))
    print(f"\n[{name}] REF: {ref}")
    for label, folder in SETS.items():
        segs, _ = model.transcribe(str(folder / f"{name}.wav"), language="tr", beam_size=5)
        hyp = canon(" ".join(s.text for s in segs))
        totals[label][0].append(ref)
        totals[label][1].append(hyp)
        print(f"  {label:>3} WER {jiwer.wer(ref, hyp):5.1%}  HYP: {hyp}")

print()
for label, (refs, hyps) in totals.items():
    print(f"{label:>3}: corpus WER {jiwer.wer(refs, hyps):.1%}  CER {jiwer.cer(refs, hyps):.1%}")
