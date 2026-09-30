"""Turn a caller utterance into (tool, arguments, reply).

ConstrainedRouter  - small local models: JSON-schema constrained decoding forces a tool choice.
ToolCallingRouter  - larger models: native function calling (Ollama / vLLM OpenAI-compatible).
Both sit behind the same keyword guard for routes that must never depend on the model.
"""
import json
import re
from dataclasses import dataclass, field

import logging

import aiohttp

import metrics
from tools import NONE_TOOL, ToolRegistry

log = logging.getLogger("router")
LLM_TIMEOUT_S = 4  # a caller will not wait longer than this in silence

# Routes a bank wants deterministic: asking for a human always reaches a human.
GUARDS = [
    (re.compile(r"\b(temsi\w*|insan|yetkili|operatör|canlı destek|biriyle görüş)", re.I), "transfer_to_agent"),
]


# Free-text replies must not state facts the bank never gave the model (hours, rates, fees...).
# A small model ignores "do not make things up" often enough that this is enforced in code.
FACT_PATTERN = re.compile(r"\d|%|\b(saat|faiz|ücret|komisyon|oran|lira|tl)\b", re.I)
MENTIONS_HUMAN = re.compile(r"temsilci|yetkili|bağlan", re.I)
NO = re.compile(r"\b(hayır|istemiyorum|gerek yok|yok)\b", re.I)
DECLINED_HUMAN = "Peki, başka nasıl yardımcı olabilirim?"
YES = re.compile(r"\b(evet|olur|tamam|lütfen|bağla)", re.I)
SAFE_REPLY = "Bu konuda size en doğru bilgiyi temsilcimiz verebilir, bağlanmak ister misiniz?"


@dataclass
class Decision:
    tool: str
    arguments: dict = field(default_factory=dict)
    reply: str = ""
    source: str = "llm"  # "guard" | "llm" | "output_guard" | "llm_down"


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
        if offered_human and NO.search(utterance):
            return Decision(NONE_TOOL, {}, DECLINED_HUMAN, source="guard")
        if offered_human and YES.search(utterance):
            return Decision("transfer_to_agent", {"summary": history[-2]["content"]}, source="guard")
        try:
            decision = await self._llm(utterance, history)
        except Exception as exc:
            # Graceful degradation: without the LLM the guards and DTMF still work, and anything
            # else goes to a human instead of leaving the caller in silence or in a loop.
            metrics.LLM_ERRORS.inc()
            log.warning("LLM unavailable (%r); routing to a human", exc)
            return Decision("transfer_to_agent", {"summary": utterance}, source="llm_down")
        # facts the model may have invented, or a vague "you can talk to an agent" that leaves the
        # caller unsure what to do: replace with one fixed question the yes-guard understands
        if decision.tool == NONE_TOOL and (FACT_PATTERN.search(decision.reply) or MENTIONS_HUMAN.search(decision.reply)):
            return Decision(NONE_TOOL, {}, SAFE_REPLY, source="output_guard")
        return decision

    async def warm(self) -> bool:
        """Load the model into memory. Not a caller turn: a cold load may take far longer than
        LLM_TIMEOUT_S and must not count as an LLM failure."""
        try:
            await self._chat({"model": self.model, "stream": False, "think": False, "keep_alive": "60m",
                              "messages": [{"role": "user", "content": "merhaba"}],
                              "options": {"num_predict": 1}}, timeout=120)
            return True
        except Exception as exc:
            log.warning("LLM warm-up failed (%r); calls will degrade until it is reachable", exc)
            return False

    async def _chat(self, body: dict, timeout: float = LLM_TIMEOUT_S) -> dict:
        async with aiohttp.ClientSession() as s:
            async with s.post(self.url, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
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
