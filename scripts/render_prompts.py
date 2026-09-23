"""Render IVR prompts through the TTS service into Asterisk's sound directory.

    python scripts/render_prompts.py [--url http://localhost:8080] [--out telephony/asterisk/sounds/tr/freya]
"""
import argparse
import pathlib
import urllib.request
import json

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8080")
ap.add_argument("--prompts", default="scripts/prompts_tr.txt")
ap.add_argument("--out", default="telephony/asterisk/sounds/tr/freya")
args = ap.parse_args()

out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)

for line in pathlib.Path(args.prompts).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    name, text = line.split("|", 1)
    req = urllib.request.Request(
        f"{args.url}/v1/synthesize",
        data=json.dumps({"text": text, "sample_rate": 8000}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        (out / f"{name}.wav").write_bytes(resp.read())
        print(f"{name:10s} audio {resp.headers['X-Audio-Seconds']}s  synth {resp.headers['X-Synthesis-Seconds']}s")
