"""Same cases as llm_bench.py, but with schema-constrained JSON intents instead of tool calls.

    python bench/llm_intent_bench.py qwen3.5:2b-q4_K_M qwen3.5:4b-q4_K_M
"""
import json
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, "services/agent")
sys.path.insert(0, "bench")
from bank_tools import INTENT_PROMPT, INTENT_SCHEMA  # noqa: E402
from llm_bench import CASES  # noqa: E402

EXPECT = {None: {"smalltalk", "unknown"}}


def classify(model, text):
    body = {"model": model, "stream": False, "think": False, "format": INTENT_SCHEMA,
            "options": {"temperature": 0, "num_predict": 60},
            "messages": [{"role": "system", "content": INTENT_PROMPT}, {"role": "user", "content": text}]}
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    out = json.loads(urllib.request.urlopen(req).read())["message"]["content"]
    return time.perf_counter() - t0, json.loads(out)


for model in sys.argv[1:]:
    classify(model, "Merhaba")
    lat, correct = [], 0
    print(f"\n=== {model}")
    for text, expected in CASES:
        t, out = classify(model, text)
        ok = out["intent"] in EXPECT.get(expected, {expected})
        correct += ok
        lat.append(t)
        print(f"{'OK ' if ok else 'XX '} {t*1000:5.0f} ms  {out['intent']:17s} | {text} -> {out['reply'][:60]}")
    print(f"intent accuracy {correct}/{len(CASES)}  median latency {statistics.median(lat)*1000:.0f} ms")
