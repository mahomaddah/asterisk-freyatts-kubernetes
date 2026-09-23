"""Tiny async client for the Asterisk REST Interface (REST calls + event WebSocket)."""
import json
import logging

import aiohttp

log = logging.getLogger("ari")


class Ari:
    def __init__(self, base_url: str, user: str, password: str, app: str):
        self.base = base_url.rstrip("/") + "/ari"
        self.auth = aiohttp.BasicAuth(user, password)
        self.app = app
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(auth=self.auth)
        return self

    async def __aexit__(self, *exc):
        await self.session.close()

    async def request(self, method: str, path: str, **params):
        params = {k: v for k, v in params.items() if v is not None}
        async with self.session.request(method, self.base + path, params=params) as r:
            if r.status >= 400:
                body = await r.text()
                raise RuntimeError(f"ARI {method} {path} -> {r.status}: {body}")
            if r.content_type == "application/json":
                return await r.json()
            return None

    def post(self, path, **params):
        return self.request("POST", path, **params)

    def delete(self, path, **params):
        return self.request("DELETE", path, **params)

    async def events(self):
        url = self.base.replace("http", "ws", 1) + "/events"
        async with self.session.ws_connect(url, params={"app": self.app}, heartbeat=20) as ws:
            log.info("connected to ARI events for app %s", self.app)
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    yield json.loads(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
