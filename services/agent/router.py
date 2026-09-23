"""Turn a caller utterance into (tool, arguments, reply).

ConstrainedRouter  - small local models: JSON-schema constrained decoding forces a tool choice.
ToolCallingRouter  - larger models: native function calling (Ollama / vLLM OpenAI-compatible).
Both sit behind the same keyword guard for routes that must never depend on the model.
"""
import json
import re
from dataclasses import dataclass, field

import aiohttp

from tools import NONE_TOOL, ToolRegistry

# Routes a bank wants deterministic: asking for a human always reaches a human.
GUARDS = [
    (re.compile(r"\b(temsilci|insan|yetkili|operatör|canlı destek|biriyle görüş)", re.I), "transfer_to_agent"),
]


# Free-text replies must not state facts the bank never gave the model (hours, rates, fees...).
# A small model ignores "do not make things up" often enough that this is enforced in code.
FACT_PATTERN = re.compile(r"\d|%|\b(saat|faiz|ücret|komisyon|oran|lira|tl)\b", re.I)
YES = re.compile(r"\b(evet|olur|tamam|lütfen|bağla)", re.I)
SAFE_REPLY = "Bu konuda size en doğru bilgiyi temsilcimiz verebilir, bağlanmak ister misiniz?"


@dataclass
class Decision:
    tool: str
    arguments: dict = field(default_factory=dict)
    reply: str = ""
    source: str = "llm"  # "guard" | "llm"


class Router:
    def __init__(self, registry: ToolRegistry, persona: str, model: str, ollama_url: str):
        self.registry = registry
        self.persona = persona
        self.model = model
        self.url = ollama_url.rstrip("/") + "/api/chat"

    async def decide(self, utterance: str, history: list[dict]) -> Decision:
        for pattern, tool in GUARDS:
            if pattern.search(utterance):
                return Decision(tool, {"summary": utterance}, source="guard")
        offered_human = history and history[-1]["content"] == SAFE_REPLY
        if offered_human and YES.search(utterance):
            return Decision("transfer_to_agent", {"summary": history[-2]["content"]}, source="guard")
        decision = await self._llm(utterance, history)
        if decision.tool == NONE_TOOL and FACT_PATTERN.search(decision.reply):
            return Decision(NONE_TOOL, {}, SAFE_REPLY, source="output_guard")
        return decision

    async def _chat(self, body: dict) -> dict:
        async with aiohttp.ClientSession() as s:
            async with s.post(self.url, json=body, timeout=aiohttp.ClientTimeout(total=15)) as r:
                r.raise_for_status()
                return (await r.json())["message"]

    async def _llm(self, utterance: str, history: list[dict]) -> Decision:
        raise NotImplementedError


class ConstrainedRouter(Router):
    async def _llm(self, utterance, history):
        system = (
            f"{self.persona}\nMüşterinin son cümlesine göre tam olarak bir araç seç:\n{self.registry.describe()}\n"
            f"- {NONE_TOOL}: sadece selamlaşma, teşekkür veya bu araçlarla ilgisi olmayan sorular "
            f"(çalışma saatleri, şube, kampanya gibi)\n"
            f"Cümle selamla başlasa bile asıl isteğe göre araç seç.\n"
            f"Araç seçtiysen reply alanına sadece kısa bir bekleme cümlesi yaz, işlem yapıldı deme. "
            f"{NONE_TOOL} seçtiysen reply alanına müşteriye cevabını yaz."
        )
        message = await self._chat({
            "model": self.model, "stream": False, "think": False, "keep_alive": "60m",
            "format": self.registry.choice_schema(),
            "options": {"temperature": 0, "num_predict": 80},
            "messages": [{"role": "system", "content": system}, *history, {"role": "user", "content": utterance}],
        })
        out = json.loads(message["content"])
        return Decision(out["tool"], out.get("arguments") or {}, out.get("reply", ""))


class ToolCallingRouter(Router):
    async def _llm(self, utterance, history):
        message = await self._chat({
            "model": self.model, "stream": False, "think": False, "keep_alive": "60m",
            "tools": self.registry.function_schemas(),
            "options": {"temperature": 0.2, "num_predict": 80},
            "messages": [{"role": "system", "content": self.persona}, *history, {"role": "user", "content": utterance}],
        })
        calls = message.get("tool_calls") or []
        if calls:
            fn = calls[0]["function"]
            return Decision(fn["name"], fn.get("arguments") or {})
        return Decision(NONE_TOOL, {}, message.get("content", "").strip())


ROUTERS = {"constrained": ConstrainedRouter, "tool_calling": ToolCallingRouter}
