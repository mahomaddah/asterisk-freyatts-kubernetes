"""Voice agent: ARI application that answers calls with STT -> LLM -> TTS.

    python services/agent/main.py        (reads settings from the environment / .env)
"""
import asyncio
import logging
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from prometheus_client import start_http_server

from ari import Ari
from bank_tools import PERSONA, bank, fixed_phrases
from call import AGENT_BUSY, DTMF_TOOLS, GREETING, HOLD, UNCLEAR, Call
from console import Console
from router import ROUTERS, SAFE_REPLY
from speech import TTSClient, load_stt

log = logging.getLogger("agent")


@dataclass
class Config:
    ari_url: str = os.environ.get("ARI_URL", "http://localhost:8088")
    ari_user: str = os.environ.get("ARI_USER", "voice-agent")
    ari_password: str = os.environ.get("ARI_PASSWORD", "")
    app: str = os.environ.get("ARI_APP", "voice-agent")
    media_host: str = os.environ.get("AGENT_MEDIA_HOST", "192.168.65.254")  # how Asterisk reaches us
    agent_endpoint: str = os.environ.get("HUMAN_AGENT_ENDPOINT", "PJSIP/1002")
    tts_url: str = os.environ.get("TTS_URL", "http://localhost:8080")
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    llm_model: str = os.environ.get("LLM_MODEL", "qwen3.5:2b-q4_K_M")
    router: str = os.environ.get("ROUTER", "constrained")
    stt_backend: str = os.environ.get("STT_BACKEND", "mlx")
    stt_model: str = os.environ.get("STT_MODEL", "small")
    dynamic_sounds: pathlib.Path = pathlib.Path(os.environ.get(
        "DYNAMIC_SOUNDS_DIR", pathlib.Path(__file__).resolve().parents[2] / "telephony/asterisk/sounds/dynamic"))
    dynamic_sounds_in_asterisk: str = os.environ.get("DYNAMIC_SOUNDS_IN_ASTERISK", "/srv/dynamic-sounds")
    metrics_port: int = int(os.environ.get("METRICS_PORT", "9464"))
    console_host: str = os.environ.get("CONSOLE_HOST", "127.0.0.1")
    console_port: int = int(os.environ.get("CONSOLE_PORT", "8090"))
    console_audio: pathlib.Path = pathlib.Path(os.environ.get("CONSOLE_AUDIO_DIR", "/tmp/voice-agent-console"))


class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ari: Ari | None = None
        self.registry = bank
        self.router = ROUTERS[cfg.router](bank, PERSONA, cfg.llm_model, cfg.ollama_url)
        self.tts = TTSClient(cfg.tts_url)
        self.stt = load_stt(cfg.stt_backend, cfg.stt_model)
        self.stt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")  # one GPU stream
        self.stt_label = f"{cfg.stt_backend}/{cfg.stt_model}"
        self.console = Console(cfg.console_audio, {
            "stt": self.stt_label, "llm": cfg.llm_model, "router": cfg.router, "tts": cfg.tts_url})
        self.calls: dict[str, Call] = {}
        self.agent_legs: dict[str, Call] = {}
        self.pending_briefings: dict[str, tuple] = {}
        self._ignored: set[str] = set()
        cfg.dynamic_sounds.mkdir(parents=True, exist_ok=True)

    def ignore(self, channel_id: str):
        """Channels we created ourselves (ExternalMedia) also enter Stasis; skip them."""
        self._ignored.add(channel_id)

    async def run(self):
        await self.console.start(self.cfg.console_host, self.cfg.console_port)
        log.info("call inspector on http://%s:%d", self.cfg.console_host, self.cfg.console_port)
        await self.router.decide("merhaba", [])  # load the LLM into memory before the first call
        await self.tts.warm([GREETING, UNCLEAR, AGENT_BUSY, HOLD, SAFE_REPLY, *fixed_phrases()])
        log.info("pre-rendered %d fixed phrases", len(self.tts.cache))
        async with Ari(self.cfg.ari_url, self.cfg.ari_user, self.cfg.ari_password, self.cfg.app) as ari:
            self.ari = ari
            while True:
                try:
                    async for event in ari.events():
                        asyncio.create_task(self.handle(event))
                except Exception:
                    log.exception("ARI connection lost; reconnecting")
                await asyncio.sleep(2)

    async def handle(self, ev: dict):
        kind = ev["type"]
        channel = ev.get("channel", {})
        cid = channel.get("id")
        try:
            if kind == "StasisStart":
                args = ev.get("args", [])
                if cid in self._ignored or channel.get("name", "").startswith("UnicastRTP/"):
                    return
                if args and args[0] == "agent-leg":
                    call = self.agent_legs.get(cid)
                    if call:
                        await call.agent_answered(cid, args[2])
                    return
                call = Call(self, channel)
                self.calls[cid] = call
                await call.start()
            elif kind == "PlaybackFinished":
                pending = self.pending_briefings.pop(ev["playback"]["id"], None)
                if pending:
                    call, leg, briefing = pending
                    await call.briefing_done(leg, briefing)
            elif kind == "ChannelDtmfReceived" and cid in self.calls:
                call = self.calls[cid]
                tool = DTMF_TOOLS.get(ev["digit"])
                call.emit("dtmf", digit=ev["digit"])
                if tool and not (call.transferred or call.transferring):
                    call.barge_in()
                    await call.act(tool, {"summary": "Müşteri tuşlama ile temsilci istedi."}, "", "dtmf", f"DTMF {ev['digit']}", None)
            elif kind in ("StasisEnd", "ChannelDestroyed"):
                if cid in self.calls:
                    await self.calls.pop(cid).close()
                elif cid in self.agent_legs:
                    call = self.agent_legs.pop(cid)
                    if not call.transferred and not call.closed:
                        await call.agent_failed()
                    elif call.transferred and cid == call.agent_leg:
                        call.agent_leg = None
                        await call.agent_hung_up()
                self._ignored.discard(cid)
        except Exception:
            log.exception("error handling %s", kind)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config()
    start_http_server(cfg.metrics_port)
    log.info("loading models (STT %s/%s, LLM %s via %s router)", cfg.stt_backend, cfg.stt_model, cfg.llm_model, cfg.router)
    app = App(cfg)
    log.info("ready; metrics on :%d", cfg.metrics_port)
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
