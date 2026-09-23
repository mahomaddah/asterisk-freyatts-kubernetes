"""Call Inspector: a live, per-call timeline of every pipeline stage.

Shows what each layer received and produced (VAD, STT text + caller audio, router
decision, tool result, TTS text + agent audio, latencies, barge-ins, transfers).
Bound to 127.0.0.1 only; audio snippets are deleted after RETENTION_MINUTES
because caller audio is personal data (KVKK).
"""
import asyncio
import json
import pathlib
import time
import uuid

import numpy as np
import soundfile as sf
from aiohttp import web

RETENTION_MINUTES = 30
MAX_CALLS = 20
PAGE = pathlib.Path(__file__).with_name("console.html")


class Console:
    def __init__(self, audio_dir: pathlib.Path, meta: dict):
        self.audio_dir = audio_dir
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.calls: dict[str, dict] = {}
        self.subscribers: set[asyncio.Queue] = set()

    # --- producers ------------------------------------------------------------
    def emit(self, call_id: str, kind: str, **data):
        call = self.calls.get(call_id)
        if call is None:
            call = self.calls[call_id] = {"id": call_id, "started": time.time(), "events": []}
            while len(self.calls) > MAX_CALLS:
                self.calls.pop(next(iter(self.calls)))
        event = {"call": call_id, "kind": kind, "t": round((time.time() - call["started"]) * 1000), **data}
        call["events"].append(event)
        for q in self.subscribers:
            q.put_nowait(event)

    def save_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        name = f"{uuid.uuid4().hex}.wav"
        sf.write(self.audio_dir / name, audio, sample_rate, subtype="PCM_16")
        return f"/audio/{name}"

    # --- web ----------------------------------------------------------------------
    async def start(self, host: str, port: int):
        app = web.Application()
        app.router.add_get("/", lambda _: web.FileResponse(PAGE))
        app.router.add_get("/api/state", self._state)
        app.router.add_get("/events", self._sse)
        app.router.add_static("/audio", self.audio_dir)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, host, port).start()
        asyncio.get_running_loop().create_task(self._expire_audio())

    async def _state(self, _):
        return web.json_response({"meta": self.meta, "calls": list(self.calls.values())})

    async def _sse(self, request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        q: asyncio.Queue = asyncio.Queue()
        self.subscribers.add(q)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15)
                    await resp.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
                except asyncio.TimeoutError:
                    await resp.write(b": keep-alive\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.subscribers.discard(q)
        return resp

    async def _expire_audio(self):
        while True:
            cutoff = time.time() - RETENTION_MINUTES * 60
            for f in self.audio_dir.glob("*.wav"):
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
            await asyncio.sleep(60)
