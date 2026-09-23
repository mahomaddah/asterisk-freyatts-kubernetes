"""G.711 mu-law codec in numpy (audioop was removed in Python 3.13)."""
import numpy as np

BIAS = 0x84
CLIP = 32635


def _decode_table() -> np.ndarray:
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = (((mantissa << 3) + BIAS) << exponent) - BIAS
    return np.where(u & 0x80, -sample, sample).astype(np.int16)


DECODE = _decode_table()


def ulaw_decode(data: bytes) -> np.ndarray:
    """mu-law bytes -> float32 in [-1, 1)."""
    return DECODE[np.frombuffer(data, dtype=np.uint8)].astype(np.float32) / 32768.0


def ulaw_encode(audio: np.ndarray) -> bytes:
    """float32 in [-1, 1] -> mu-law bytes."""
    x = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int32)
    sign = (x < 0).astype(np.int32) << 7
    x = np.minimum(np.abs(x), CLIP) + BIAS
    exponent = np.clip(np.floor(np.log2(x >> 7)).astype(np.int32), 0, 7)
    mantissa = (x >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()
