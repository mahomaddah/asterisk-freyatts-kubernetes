"""Same text, steps and seed on CUDA (TTS service), Apple MPS and CPU, at 48 kHz.

Every clip is scored by the same Whisper model and the same pitch tracker, so differences come
from synthesis and not from the phone band. --x0-cpu draws the initial noise on the CPU on every
local device, to separate the RNG from the rest of the numerics.

    python bench/device_compare.py --tts-url http://localhost:18080
"""
import argparse, io, json, pathlib, re, sys, urllib.request

import jiwer, librosa, mlx_whisper, numpy as np, soundfile as sf, torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "FreyaTTS"))
from freyatts import FreyaTTS  # noqa: E402
from freyatts.model import LEYLA_SEED  # noqa: E402
from freyatts.pipeline import normalize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tts-url", required=True)
args = ap.parse_args()

# first phrasings only: the original sentences, before any QC rewording
texts = {}
for f in ("scripts/prompts_tr.txt", "scripts/agent_phrases_tr.txt"):
    for line in pathlib.Path(f).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, variants = line.split("|", 1)
            texts[f"{pathlib.Path(f).stem[:5]}:{name}"] = variants.split("||")[0].strip()


def canon(t):
    return re.sub(r"[^\w]", "", normalize(t).replace("I", "ı").replace("İ", "i").lower())


def score(text, wav):
    heard = mlx_whisper.transcribe(librosa.resample(wav, orig_sr=48000, target_sr=16000),
                                   path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="tr")["text"].strip()
    y, p = librosa.resample(wav, orig_sr=48000, target_sr=16000), []
    for i in range(0, len(y), 24000):
        f0, _, _ = librosa.pyin(y[i:i + 24000], fmin=70, fmax=400, sr=16000)
        v = f0[~np.isnan(f0)]
        if len(v) > 10:
            p.append(int(np.median(v)))
    return round(jiwer.cer(canon(text), canon(heard)), 3), p, heard


def remote(text):
    req = urllib.request.Request(args.tts_url + "/v1/synthesize", json.dumps({"text": text, "sample_rate": 48000}).encode(),
                                 {"Content-Type": "application/json"})
    return sf.read(io.BytesIO(urllib.request.urlopen(req, timeout=120).read()), dtype="float32")[0]


out = {}
engines = {"cuda": remote}
for dev in ("mps", "cpu"):
    g = torch.Generator(device=dev).manual_seed(int(LEYLA_SEED))
    print("x0", dev, torch.randn(1, 1, 4, device=dev, generator=g).flatten().cpu().numpy().round(4).tolist())
    engines[dev] = FreyaTTS.from_pretrained("freyavoice/freya-tts", device=dev).synthesize

for dev, fn in engines.items():
    rows = {}
    for name, text in texts.items():
        wav = np.asarray(fn(text), dtype=np.float32)
        sf.write(f"bench/out/devices/{dev}-{name.replace(':','_')}.wav", wav, 48000)
        cer, pitch, heard = score(text, wav)
        rows[name] = dict(cer=cer, pitch=pitch, heard=heard)
    same = np.array_equal(np.asarray(fn(texts["promp:welcome"])), np.asarray(fn(texts["promp:welcome"])))
    out[dev] = rows
    cers = [r["cer"] for r in rows.values()]
    drift = [n for n, r in rows.items() if r["pitch"] and min(r["pitch"]) < 170]
    print(f"{dev:5s} mean CER {np.mean(cers):.1%}  >4%: {sum(c > .04 for c in cers)}/{len(cers)}  drift(<170Hz): {len(drift)} {drift}  repeatable={same}", flush=True)
pathlib.Path("bench/results/device_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print()
for n in texts:
    print(f"{n:22s}", " | ".join(f"{d} {out[d][n]['cer']:5.1%} {min(out[d][n]['pitch'] or [0])}Hz" for d in out))
