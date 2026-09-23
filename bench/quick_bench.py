"""Quick FreyaTTS latency check on the local machine (cpu / mps / cuda)."""
import sys, time
sys.path.insert(0, "../FreyaTTS")
import numpy as np, soundfile as sf, librosa, torch
from freyatts import FreyaTTS

TEXTS = {
    "short": "Merhaba, size nasıl yardımcı olabilirim?",
    "medium": "Kartınızın son ödeme tarihi on beş ekim, asgari tutar iki bin dört yüz lira.",
    "long": "Güvenliğiniz için sizi müşteri temsilcimize aktarıyorum, lütfen hattan ayrılmayın. "
            "Beklerken kimlik numaranızın son dört hanesini hazırlayabilirsiniz.",
}

for device in sys.argv[1:] or ["cpu", "mps"]:
    t0 = time.time()
    tts = FreyaTTS.from_pretrained("freyavoice/freya-tts", device=device)
    print(f"[{device}] load {time.time()-t0:.1f}s", flush=True)
    tts.synthesize("Deneme.")  # warmup
    for name, text in TEXTS.items():
        t1 = time.time()
        wav = tts.synthesize(text)
        if device == "mps":
            torch.mps.synchronize()
        wall = time.time() - t1
        dur = len(wav) / 48000
        print(f"[{device}] {name:6s} audio {dur:5.2f}s  wall {wall:5.2f}s  RTF {wall/dur:.2f}", flush=True)
        if device == (sys.argv[1:] or ["cpu"])[0]:
            sf.write(f"bench/out/{name}_48k.wav", wav, 48000)
            # telephony leg: 8 kHz G.711 mu-law, what the caller actually hears
            nb = librosa.resample(wav, orig_sr=48000, target_sr=8000)
            sf.write(f"bench/out/{name}_8k_ulaw.wav", nb, 8000, subtype="ULAW")
