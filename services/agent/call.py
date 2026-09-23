"""One phone call handled by the voice agent.

Media path:  caller <-> Asterisk mixing bridge <-> ExternalMedia (RTP, G.711 mu-law) <-> this object
Turn loop:   VAD end-of-speech -> STT -> router (guard / LLM) -> tool -> TTS -> RTP
Barge-in:    caller speech while the agent talks stops playback and cancels the pending turn.
Transfer:    warm - the human agent first hears a spoken summary, then joins the caller's bridge.
"""
import asyncio
import collections
import logging
import re
import time
import uuid

import numpy as np
import soundfile as sf

import metrics
from rtp import RtpEndpoint
from speech import SAMPLE_RATE, VAD, to_stt_rate
from tools import NONE_TOOL

log = logging.getLogger("call")

START_WINDOWS = 4      # 4 x 32 ms of speech  -> caller started talking
END_WINDOWS = 19       # 19 x 32 ms of silence -> caller finished (~600 ms)
PREROLL_FRAMES = 15    # keep 300 ms before the VAD trigger so first syllables are not cut
MAX_UNCLEAR = 2        # unclear turns in a row before handing over to a human

GREETING = "Freya Bank'a hoş geldiniz, ben Leyla. Size nasıl yardımcı olabilirim?"
UNCLEAR = "Kusura bakmayın, anlayamadım. Tekrar söyler misiniz?"
AGENT_BUSY = "Şu anda tüm temsilcilerimiz dolu. Size ben yardımcı olmaya devam edeyim."
HOLD = "Temsilcimiz birazdan sizinle olacak, lütfen bekleyin."
SHORT_REPLY = re.compile(r"^\W*(evet|olur|tamam|lütfen|hayır)\W*(lütfen)?\W*$", re.I)


def ringback(seconds: float, rate: int) -> np.ndarray:
    """Turkish ringback tone (450 Hz, 2 s on / 4 s off) so a waiting caller hears progress."""
    t = np.arange(int(2 * rate)) / rate
    burst = 0.15 * np.sin(2 * np.pi * 450 * t).astype(np.float32)
    cycle = np.concatenate([burst, np.zeros(int(4 * rate), dtype=np.float32)])
    return np.tile(cycle, int(np.ceil(seconds / 6)))
DTMF_TOOLS = {"0": "transfer_to_agent", "1": "get_statement", "2": "block_card"}

# Whisper hallucinates these on noise / silence in Turkish
HALLUCINATIONS = re.compile(r"altyaz|abone ol|izlediğiniz için teşekkür", re.I)


def mask_pii(text: str) -> str:
    """KVKK: never write card / ID / phone numbers to logs."""
    return re.sub(r"\d[\d\s-]{3,}\d", "***", text)


class Call:
    def __init__(self, app, channel: dict):
        self.app = app
        self.id = channel["id"]
        self.caller = channel.get("caller", {}).get("number", "?")
        self.bridge_id = None
        self.media_id = None
        self.agent_leg = None
        self.rtp: RtpEndpoint | None = None
        self.vad = VAD()
        self.preroll = collections.deque(maxlen=PREROLL_FRAMES)
        self.capture: list[np.ndarray] = []
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.turn_task: asyncio.Task | None = None
        self.history: list[dict] = []
        self.unclear = 0
        self.transferring = False  # ringing a human: no VAD turns, caller hears hold + ringback
        self.transferred = False
        self.closed = False

    def emit(self, kind: str, **data):
        self.app.console.emit(self.id, kind, **data)

    # --- setup / teardown -------------------------------------------------------
    async def start(self):
        ari = self.app.ari
        await ari.post(f"/channels/{self.id}/answer")
        self.bridge_id = (await ari.post("/bridges", type="mixing", name=f"call-{self.id}"))["id"]
        await ari.post(f"/bridges/{self.bridge_id}/addChannel", channel=self.id)

        loop = asyncio.get_running_loop()
        _, self.rtp = await loop.create_datagram_endpoint(
            lambda: RtpEndpoint(self.on_audio), local_addr=("0.0.0.0", 0))
        port = self.rtp.transport.get_extra_info("sockname")[1]
        media = await ari.post("/channels/externalMedia", app=ari.app, format="ulaw",
                               external_host=f"{self.app.cfg.media_host}:{port}")
        self.media_id = media["id"]
        self.app.ignore(self.media_id)
        # the ExternalMedia channel enters Stasis asynchronously; retry until it has
        for _ in range(20):
            try:
                await ari.post(f"/bridges/{self.bridge_id}/addChannel", channel=self.media_id)
                break
            except RuntimeError as exc:
                if "not in Stasis" not in str(exc):
                    raise
                await asyncio.sleep(0.05)
        metrics.CALLS.inc()
        metrics.ACTIVE.inc()
        log.info("call %s from %s: bridge %s, rtp port %d", self.id, self.caller, self.bridge_id, port)
        self.emit("call_start", caller=self.caller, bridge=self.bridge_id, port=port)
        await self.say(GREETING)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        metrics.ACTIVE.dec()
        if self.turn_task:
            self.turn_task.cancel()
        if self.rtp:
            self.rtp.close()
        # ARI does not hang up the other bridge members for us
        paths = [f"/channels/{c}" for c in (self.media_id, self.agent_leg) if c]
        paths += [f"/bridges/{self.bridge_id}"] if self.bridge_id else []
        for path in paths:
            try:
                await self.app.ari.delete(path)
            except RuntimeError:
                pass  # already gone
        self.emit("call_end")
        log.info("call %s closed", self.id)

    # --- audio in: VAD state machine --------------------------------------------
    def on_audio(self, pcm: np.ndarray):
        if self.transferring or self.transferred:
            return
        if self.in_speech:
            self.capture.append(pcm)
        else:
            self.preroll.append(pcm)
        for speech in self.vad.feed(pcm):
            self.speech_run = self.speech_run + 1 if speech else 0
            self.silence_run = 0 if speech else self.silence_run + 1
            if not self.in_speech and self.speech_run >= START_WINDOWS:
                self.in_speech = True
                self.capture = list(self.preroll)
                self.preroll.clear()
                self.emit("speech_start")
                self.barge_in()
            elif self.in_speech and self.silence_run >= END_WINDOWS:
                self.in_speech = False
                audio = np.concatenate(self.capture)
                self.capture = []
                self.emit("speech_end", duration_ms=round(len(audio) / SAMPLE_RATE * 1000))
                self.turn_task = asyncio.get_running_loop().create_task(self.turn(audio, time.perf_counter()))

    def barge_in(self):
        interrupted = False
        if self.rtp.playing:
            self.rtp.stop_playback()
            interrupted = True
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
            interrupted = True
        if interrupted:
            metrics.BARGE_INS.inc()
            self.emit("barge_in")
            log.info("call %s: barge-in", self.id)

    # --- one conversational turn --------------------------------------------------
    async def turn(self, audio: np.ndarray, t_end: float):
        app = self.app
        loop = asyncio.get_running_loop()
        timing = {"t_end": t_end}
        try:
            t0 = time.perf_counter()
            text = await loop.run_in_executor(app.stt_pool, app.stt.transcribe, to_stt_rate(audio))
            timing["stt"] = time.perf_counter() - t0
            metrics.STAGE.labels("stt").observe(timing["stt"])
            self.emit("stt", text=mask_pii(text), ms=timing["stt"] * 1000, model=app.stt_label,
                      audio=app.console.save_audio(audio, SAMPLE_RATE))
            if len(text) < 2 or HALLUCINATIONS.search(text):
                return
            log.info("call %s caller: %s", self.id, mask_pii(text))

            t1 = time.perf_counter()
            decision = await app.router.decide(text, self.history[-6:])
            timing["llm"] = time.perf_counter() - t1
            metrics.STAGE.labels("llm").observe(timing["llm"])
            self.emit("route", tool=decision.tool, source=decision.source, arguments=decision.arguments,
                      reply=decision.reply, ms=timing["llm"] * 1000)
            await self.act(decision.tool, decision.arguments, decision.reply, decision.source, text, timing)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("call %s: turn failed", self.id)
            self.emit("error", message=repr(exc))
            self.transferring = False  # whatever failed, keep listening to the caller
            await self.say(UNCLEAR, timing)

    async def act(self, tool: str, arguments: dict, reply: str, source: str, heard: str, timing: dict | None):
        metrics.TOOLS.labels(tool, source).inc()
        log.info("call %s decision: tool=%s source=%s", self.id, tool, source)
        if tool == NONE_TOOL:
            self.unclear = self.unclear + 1 if not reply else 0
            if self.unclear >= MAX_UNCLEAR:
                self.emit("route", tool="transfer_to_agent", source="fallback", arguments={}, reply="")
                return await self.act("transfer_to_agent", {"summary": "Müşteri anlaşılamadı."}, "", "fallback", heard, timing)
            say = reply or UNCLEAR
        else:
            self.unclear = 0
            result = self.app.registry.invoke(tool, arguments)
            say = result["say"]
            self.emit("tool", name=tool, result={k: v for k, v in result.items() if k != "say"})
        self.history += [{"role": "user", "content": heard}, {"role": "assistant", "content": say}]
        await self.say(say, timing)
        if tool == "transfer_to_agent":
            await self.warm_transfer(self.briefing_summary(arguments.get("summary") or heard))

    def briefing_summary(self, fallback: str) -> str:
        """What the human agent hears: the caller's real request, not a trailing 'evet'."""
        asks = [m["content"] for m in self.history if m["role"] == "user"
                and not m["content"].startswith("DTMF") and not SHORT_REPLY.match(m["content"])]
        return " ".join(asks[-2:]) or fallback

    async def say(self, text: str, timing: dict | None = None):
        t0 = time.perf_counter()
        audio = await self.app.tts.synthesize(text)
        tts_s = time.perf_counter() - t0
        metrics.STAGE.labels("tts").observe(tts_s)
        self.rtp.play(audio)
        self.emit("tts", text=text, ms=tts_s * 1000, cached=self.app.tts.last_hit,
                  audio=self.app.console.save_audio(audio, SAMPLE_RATE))
        if timing is not None:
            # end of speech is detected END_WINDOWS after the caller actually stopped talking
            endpoint = END_WINDOWS * 0.032
            e2e = time.perf_counter() - timing["t_end"] + endpoint
            metrics.STAGE.labels("e2e").observe(e2e)
            self.emit("turn", e2e_ms=e2e * 1000, endpoint_ms=endpoint * 1000, stt_ms=timing.get("stt", 0) * 1000,
                      llm_ms=timing.get("llm", 0) * 1000, tts_ms=tts_s * 1000)
            log.info("call %s agent (%.0f ms after caller stopped): %s", self.id, e2e * 1000, text)

    async def wait_playback(self):
        while self.rtp.playing:
            await asyncio.sleep(0.05)

    # --- warm transfer --------------------------------------------------------------
    async def warm_transfer(self, summary: str):
        self.transferring = True
        self.in_speech = False
        await self.wait_playback()
        briefing = f"Freya asistanından aktarım. Müşteri şunu söyledi: {summary}"
        audio = await self.app.tts.synthesize(briefing)
        name = f"brief-{uuid.uuid4().hex[:12]}"
        path = self.app.cfg.dynamic_sounds / f"{name}.wav"
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")  # Asterisk plays 8 kHz .wav natively

        try:
            leg = await self.app.ari.post("/channels", endpoint=self.app.cfg.agent_endpoint, app=self.app.ari.app,
                                          appArgs=f"agent-leg,{self.id},{name}", callerId="Freya AI <200>", timeout=30)
        except RuntimeError as exc:
            # e.g. "Allocation failed": no agent phone registered. Never leave the caller stuck on hold.
            path.unlink(missing_ok=True)
            self.emit("error", message=f"could not ring {self.app.cfg.agent_endpoint}: {exc}")
            return await self.agent_failed()
        self.agent_leg = leg["id"]
        self.app.agent_legs[self.agent_leg] = self
        self.emit("transfer", step=f"ringing {self.app.cfg.agent_endpoint}", detail=f"briefing: {summary}",
                  audio=self.app.console.save_audio(audio, SAMPLE_RATE))
        # caller hears a hold message, then ringback until the agent joins (or the call fails)
        self.rtp.play(await self.app.tts.synthesize(HOLD))
        self.rtp.play(ringback(40, SAMPLE_RATE))
        log.info("call %s: ringing human agent (%s)", self.id, self.agent_leg)

    async def agent_answered(self, leg_id: str, briefing: str):
        """Human agent picked up: play the briefing to them only, then join the caller."""
        self.emit("transfer", step="agent answered, playing briefing to agent only")
        self.app.pending_briefings[
            (await self.app.ari.post(f"/channels/{leg_id}/play",
                                     media=f"sound:{self.app.cfg.dynamic_sounds_in_asterisk}/{briefing}"))["id"]
        ] = (self, leg_id, briefing)

    async def briefing_done(self, leg_id: str, briefing: str):
        ari = self.app.ari
        self.transferred = True
        self.transferring = False
        self.rtp.stop_playback()
        await ari.post(f"/bridges/{self.bridge_id}/addChannel", channel=leg_id)
        await ari.delete(f"/channels/{self.media_id}")
        self.media_id = None
        (self.app.cfg.dynamic_sounds / f"{briefing}.wav").unlink(missing_ok=True)
        metrics.TRANSFERS.labels("connected").inc()
        self.emit("transfer", step="agent joined the caller, AI left the bridge")
        log.info("call %s: human agent joined, AI left the bridge", self.id)

    async def agent_hung_up(self):
        """After a transfer the human agent ended the call: end the caller's leg too."""
        try:
            await self.app.ari.delete(f"/channels/{self.id}")
        except RuntimeError:
            pass

    async def agent_failed(self):
        self.agent_leg = None
        metrics.TRANSFERS.labels("no_answer").inc()
        self.emit("transfer", step="agent did not answer")
        self.rtp.stop_playback()
        self.transferring = False
        await self.say(AGENT_BUSY)
