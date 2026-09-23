"""Minimal RTP endpoint for an Asterisk ExternalMedia channel (G.711 mu-law, 20 ms frames).

Asterisk sends the bridge mix to us; we learn its address from the first packet and send the agent's speech back to that same address (symmetric RTP,
which also works through Docker / Kubernetes NAT).

mu-law (static payload type 0) rather than slin16: the phone leg is 8 kHz G.711
anyway, and Asterisk 20.6's ExternalMedia did not accept inbound slin16 on its
dynamic payload type (packets arrived but never reached the bridge).
"""
import asyncio
import random
import struct

import numpy as np

from g711 import ulaw_decode, ulaw_encode

FRAME_SAMPLES = 160               # 20 ms at 8 kHz
FRAME_BYTES = FRAME_SAMPLES       # one byte per mu-law sample
PAYLOAD_TYPE = 0                  # PCMU


class RtpEndpoint(asyncio.DatagramProtocol):
    def __init__(self, on_audio):
        self.on_audio = on_audio      # callback(np.float32 array), called per received frame
        self.transport = None
        self.remote = None
        self.seq = random.randint(0, 0xFFFF)
        self.ts = random.randint(0, 0xFFFFFFFF)
        self.ssrc = random.randint(0, 0xFFFFFFFF)
        self.out = bytearray()        # pending outbound mu-law bytes
        self.sender = None
        self.frames_in = 0
        self.frames_out = 0

    # --- inbound ---------------------------------------------------------------
    def connection_made(self, transport):
        self.transport = transport
        self.sender = asyncio.get_running_loop().create_task(self._send_loop())

    def datagram_received(self, data, addr):
        if len(data) < 12:
            return
        cc = data[0] & 0x0F
        has_ext = data[0] & 0x10
        offset = 12 + 4 * cc
        if has_ext:
            offset += 4 + 4 * struct.unpack_from("!H", data, offset + 2)[0]
        self.remote = addr
        pcm = ulaw_decode(data[offset:])
        self.frames_in += 1
        self.on_audio(pcm)

    # --- outbound --------------------------------------------------------------
    def play(self, audio: np.ndarray):
        pad = (-len(audio)) % FRAME_SAMPLES
        self.out.extend(ulaw_encode(np.concatenate([audio, np.zeros(pad, dtype=np.float32)])))

    def stop_playback(self):
        self.out.clear()

    @property
    def playing(self) -> bool:
        return len(self.out) > 0

    async def _send_loop(self):
        loop = asyncio.get_running_loop()
        next_at = loop.time()
        while True:
            next_at += 0.02
            await asyncio.sleep(max(0.0, next_at - loop.time()))
            if not self.out or self.remote is None:
                continue
            frame, self.out[:FRAME_BYTES] = bytes(self.out[:FRAME_BYTES]), b""
            header = struct.pack("!BBHII", 0x80, PAYLOAD_TYPE, self.seq, self.ts, self.ssrc)
            self.transport.sendto(header + frame, self.remote)
            self.seq = (self.seq + 1) & 0xFFFF
            self.ts = (self.ts + FRAME_SAMPLES) & 0xFFFFFFFF
            self.frames_out += 1

    def close(self):
        if self.sender:
            self.sender.cancel()
        if self.transport:
            self.transport.close()
