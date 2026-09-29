"""End-to-end smoke test: a real SIP call that checks what the caller actually hears.

Registers a headless softphone (baresip) as 1001, dials an extension, speaks a sentence
(synthesized by the TTS service) after the greeting, records the audio it receives and
fails if that audio is silent. Phase 2 taught us logs can look perfect while the media
path is broken, so this test judges the media, not the logs.

    SIP_PASSWORD_1001=... python scripts/smoke_call.py --server 192.168.1.8 --ext 200 \\
        --say "Merhaba, kartımı kaybettim." --tts-url http://localhost:8080
"""
import argparse
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
import soundfile as sf

RATE = 8000


def tts(url: str, text: str) -> np.ndarray:
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/synthesize",
                                 json.dumps({"text": text, "sample_rate": RATE}).encode(),
                                 {"Content-Type": "application/json"})
    return sf.read(io.BytesIO(urllib.request.urlopen(req, timeout=60).read()), dtype="float32")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="Asterisk address (SIP UDP 5060)")
    ap.add_argument("--user", default="1001")
    ap.add_argument("--ext", default="200")
    ap.add_argument("--say", default="Merhaba, kartımı kaybettim.")
    ap.add_argument("--wait", type=float, default=6.0, help="seconds of silence before speaking (greeting)")
    ap.add_argument("--duration", type=float, default=25.0, help="call length in seconds")
    ap.add_argument("--tts-url", default="http://localhost:8080")
    ap.add_argument("--min-rms", type=float, default=0.01)
    ap.add_argument("--keep", help="copy the received audio here")
    args = ap.parse_args()

    password = os.environ.get(f"SIP_PASSWORD_{args.user}")
    if not password:
        sys.exit(f"set SIP_PASSWORD_{args.user}")
    if not shutil.which("baresip"):
        sys.exit("baresip not found (brew install baresip)")

    work = pathlib.Path(tempfile.mkdtemp(prefix="smoke-call-"))
    silence = lambda s: np.zeros(int(s * RATE), dtype=np.float32)  # noqa: E731
    speech = tts(args.tts_url, args.say) if args.say else silence(0.1)
    sf.write(work / "caller.wav", np.concatenate([silence(args.wait), speech * 0.8, silence(args.duration)]),
             RATE, subtype="PCM_16")

    modules = subprocess.run(["brew", "--prefix"], capture_output=True, text=True).stdout.strip() + "/lib/baresip/modules"
    (work / "config").write_text(f"""module_path {modules}
module stdio.so
module g711.so
module aufile.so
module account.so
module menu.so
module_app menu.so
module_app account.so
audio_player aufile,{work}/heard.wav
audio_source aufile,{work}/caller.wav
audio_alert aufile,/dev/null
sip_listen 0.0.0.0:5070
""")
    (work / "accounts").write_text(f"<sip:{args.user}@{args.server};transport=udp>;auth_pass={password};regint=60\n")

    phone = subprocess.Popen(["baresip", "-f", str(work)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
    time.sleep(3)
    phone.stdin.write(f"/dial {args.ext}\n")
    phone.stdin.flush()
    time.sleep(args.duration)
    phone.stdin.write("/hangup\n/quit\n")
    phone.stdin.flush()
    log, _ = phone.communicate(timeout=10)

    registered = "registered successfully" in log
    established = "Call established" in log
    heard, _ = sf.read(work / "heard.wav", dtype="float32") if (work / "heard.wav").exists() else (np.zeros(1), RATE)
    rms = float(np.sqrt(np.mean(heard ** 2)))
    voiced = float(np.mean(np.abs(heard) > 0.02))
    if args.keep:
        shutil.copy(work / "heard.wav", args.keep)

    ok = registered and established and rms >= args.min_rms
    print(f"registered={registered} established={established} heard_rms={rms:.4f} voiced={voiced:.0%} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
