"""Minimal RTP endpoint for an Asterisk ExternalMedia channel (slin16, 20 ms frames).

Asterisk sends the bridge mix to us; we learn its address and payload type from the
first packet and send the agent's speech back to that same address (symmetric RTP,
which also works through Docker / Kubernetes NAT).
"""
import asyncio
import random
import struct

import numpy as np

FRAME_SAMPLES = 320               # 20 ms at 16 kHz
FRAME_BYTES = FRAME_SAMPLES * 2   # slin16 = 16-bit big-endian PCM


class RtpEndpoint(asyncio.DatagramProtocol):
    def __init__(self, on_audio):
        self.on_audio = on_audio      # callback(np.float32 array), called per received frame
        self.transport = None
        self.remote = None
        self.payload_type = None
        self.seq = random.randint(0, 0xFFFF)
        self.ts = random.randint(0, 0xFFFFFFFF)
        self.ssrc = random.randint(0, 0xFFFFFFFF)
        self.out = bytearray()        # pending outbound PCM (big-endian bytes)
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
        self.payload_type = data[1] & 0x7F
        pcm = np.frombuffer(data[offset:], dtype=">i2").astype(np.float32) / 32768.0
        self.frames_in += 1
        self.on_audio(pcm)

    # --- outbound --------------------------------------------------------------
    def play(self, audio: np.ndarray):
        pcm = (np.clip(audio, -1, 1) * 32767).astype(">i2").tobytes()
        pad = (-len(pcm)) % FRAME_BYTES
        self.out.extend(pcm + b"\0" * pad)

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
            header = struct.pack("!BBHII", 0x80, self.payload_type, self.seq, self.ts, self.ssrc)
            self.transport.sendto(header + frame, self.remote)
            self.seq = (self.seq + 1) & 0xFFFF
            self.ts = (self.ts + FRAME_SAMPLES) & 0xFFFFFFFF
            self.frames_out += 1

    def close(self):
        if self.sender:
            self.sender.cancel()
        if self.transport:
            self.transport.close()
