"""Accuracy and latency of the agent's actual router (guard + LLM) on the banking cases.

    python bench/router_bench.py [constrained|tool_calling] [model]
"""
import asyncio
import statistics
import sys
import time

sys.path.insert(0, "services/agent")
sys.path.insert(0, "bench")
from bank_tools import PERSONA, bank  # noqa: E402
from llm_bench import CASES  # noqa: E402
from router import ROUTERS  # noqa: E402
from tools import NONE_TOOL  # noqa: E402

EXTRA = [
    ("Merhaba kartımı kaybettim", "block_card"),
    ("Merhaba, borcumu öğrenmek istiyorum.", "get_statement"),
    ("Kartım çalındı hemen kapatın.", "block_card"),
    ("Asgari ödeme tutarım ne kadar?", "get_statement"),
    ("Yetkili biriyle görüşebilir miyim?", "transfer_to_agent"),
    ("Teşekkür ederim, iyi günler.", None),
]


async def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "constrained"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen3.5:2b-q4_K_M"
    router = ROUTERS[kind](bank, PERSONA, model, "http://localhost:11434")
    await router.decide("merhaba", [])
    correct, lat = 0, []
    for text, expected in CASES + EXTRA:
        t0 = time.perf_counter()
        d = await router.decide(text, [])
        lat.append(time.perf_counter() - t0)
        ok = d.tool == (expected or NONE_TOOL)
        correct += ok
        print(f"{'OK ' if ok else 'XX '} {lat[-1]*1000:5.0f} ms {d.source:12s} {d.tool:17s} {d.arguments} | {text} -> {d.reply[:50]}")
    n = len(CASES + EXTRA)
    print(f"{kind}/{model}: {correct}/{n}  median {statistics.median(lat)*1000:.0f} ms")

asyncio.run(main())
