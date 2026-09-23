"""Latency and tool-choice accuracy of local LLMs for the banking voice agent.

    python bench/llm_bench.py qwen3.5:2b-q4_K_M qwen3.5:4b-q4_K_M
"""
import json
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, "services/agent")
from bank_tools import SYSTEM_PROMPT, TOOLS  # noqa: E402

OLLAMA = "http://localhost:11434/api/chat"
CASES = [  # (caller utterance as STT would deliver it, expected tool or None)
    ("Kartımı kaybettim galiba, ne yapmam lazım?", "block_card"),
    ("Cüzdanım çalındı kartım da içindeydi.", "block_card"),
    ("Bu ay ne kadar ödemem gerekiyor?", "get_statement"),
    ("Son ödeme tarihim ne zaman?", "get_statement"),
    ("Bir temsilciyle görüşmek istiyorum.", "transfer_to_agent"),
    ("Bana gerçek bir insan bağlar mısın lütfen?", "transfer_to_agent"),
    ("Merhaba, iyi günler.", None),
    ("Çalışma saatleriniz nedir?", None),
]


def chat(model, text):
    body = {"model": model, "stream": True, "think": False, "tools": TOOLS,
            "options": {"temperature": 0.2, "num_predict": 80},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]}
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, reply, tool = None, "", None
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            msg = json.loads(line).get("message", {})
            if first is None and (msg.get("content") or msg.get("tool_calls")):
                first = time.perf_counter() - t0
            reply += msg.get("content", "")
            if msg.get("tool_calls"):
                tool = msg["tool_calls"][0]["function"]["name"]
    return first or 0.0, time.perf_counter() - t0, tool, reply.strip()


def main():
    for model in sys.argv[1:]:
        chat(model, "Merhaba")  # load weights into memory
        ttfts, totals, correct = [], [], 0
        print(f"\n=== {model}")
        for text, expected in CASES:
            ttft, total, tool, reply = chat(model, text)
            ok = tool == expected
            correct += ok
            ttfts.append(ttft)
            totals.append(total)
            print(f"{'OK ' if ok else 'XX '} ttft {ttft*1000:4.0f} ms  total {total*1000:5.0f} ms  "
                  f"tool={tool}  | {text} -> {reply[:70]}")
        print(f"tool accuracy {correct}/{len(CASES)}  median TTFT {statistics.median(ttfts)*1000:.0f} ms  "
              f"median total {statistics.median(totals)*1000:.0f} ms")


if __name__ == "__main__":
    main()
