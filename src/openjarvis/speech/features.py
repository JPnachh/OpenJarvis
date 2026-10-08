"""Small, dependency-light audio features for keyword spotting.

MFCCs computed with numpy only, plus a length-normalised dynamic time warping
(DTW) distance. Together they power the personal wake word (see
``wakeword.py``): a spoken phrase is compared against recordings of the same
phrase in the user's own voice, which works for any language or accent and
needs no model download.
"""

from __future__ import annotations

import io
import math
import wave
from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16000
_FRAME = 400  # 25 ms
_HOP = 160  # 10 ms
_NFFT = 512
_MELS = 26
_CEPS = 13


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


@lru_cache(maxsize=4)
def _mel_filterbank(sample_rate: int) -> np.ndarray:
    low, high = _hz_to_mel(60.0), _hz_to_mel(min(7600.0, sample_rate / 2))
    hz = _mel_to_hz(np.linspace(low, high, _MELS + 2))
    bins = np.floor((_NFFT + 1) * hz / sample_rate).astype(int)
    bank = np.zeros((_MELS, _NFFT // 2 + 1))
    for m in range(1, _MELS + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, centre):
            bank[m - 1, k] = (k - left) / max(1, centre - left)
        for k in range(centre, right):
            bank[m - 1, k] = (right - k) / max(1, right - centre)
    return bank


@lru_cache(maxsize=1)
def _dct_matrix() -> np.ndarray:
    n = np.arange(_MELS)
    k = np.arange(_CEPS)[:, None]
    return np.cos(math.pi / _MELS * (n + 0.5) * k)


def mfcc(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return ``(frames, 13)`` mean/variance-normalised MFCCs.

    Normalising per utterance removes microphone and loudness differences, so
    a template recorded on a headset still matches the laptop mic.
    """
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size < _FRAME:
        x = np.pad(x, (0, _FRAME - x.size))
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    count = 1 + (x.size - _FRAME) // _HOP
    idx = np.arange(_FRAME)[None, :] + _HOP * np.arange(count)[:, None]
    frames = x[idx] * np.hamming(_FRAME)
    power = np.abs(np.fft.rfft(frames, _NFFT)) ** 2 / _NFFT
    energies = np.log(power @ _mel_filterbank(sample_rate).T + 1e-10)
    ceps = energies @ _dct_matrix().T
    ceps -= ceps.mean(axis=0)
    ceps /= ceps.std(axis=0) + 1e-8
    return ceps.astype(np.float32)


def dtw_distance(a: np.ndarray, b: np.ndarray, band: float = 0.35) -> float:
    """Length-normalised DTW distance between two feature sequences.

    The accumulated Euclidean cost of the best alignment is divided by
    ``n + m`` so short and long utterances are comparable. ``band`` limits how
    far the alignment may stray from the diagonal (as a share of the longer
    sequence), which both speeds things up and rules out absurd alignments.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0 or max(n, m) > 2.5 * min(n, m):
        return math.inf
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)).tolist()
    width = max(int(band * max(n, m)), abs(n - m) + 1)
    inf = math.inf
    prev = [inf] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        row = [inf] * (m + 1)
        centre = i * m / n
        lo = max(1, int(centre - width))
        hi = min(m, int(centre + width))
        costs = cost[i - 1]
        left = inf
        for j in range(lo, hi + 1):
            best = prev[j - 1]
            if prev[j] < best:
                best = prev[j]
            if left < best:
                best = left
            left = best + costs[j - 1] if best != inf else inf
            row[j] = left
        prev = row
    total = prev[m]
    return total / (n + m) if total != inf else inf


def trim_silence(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    pad_seconds: float = 0.05,
) -> np.ndarray:
    """Cut leading/trailing silence using a level relative to the loudest part."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    hop = int(0.01 * sample_rate)
    if x.size < hop * 3:
        return x
    count = x.size // hop
    rms = np.sqrt((x[: count * hop].reshape(count, hop) ** 2).mean(axis=1))
    floor = np.percentile(rms, 10)
    threshold = max(floor * 3.0, rms.max() * 0.08, 1e-4)
    voiced = np.nonzero(rms > threshold)[0]
    if voiced.size == 0:
        return x[:0]
    pad = int(pad_seconds * sample_rate)
    start = max(0, voiced[0] * hop - pad)
    end = min(x.size, (voiced[-1] + 1) * hop + pad)
    return x[start:end]


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear-interpolation resampling; plenty for speech features."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate or x.size == 0:
        return x
    duration = x.size / source_rate
    target = np.linspace(0, x.size - 1, int(round(duration * target_rate)))
    return np.interp(target, np.arange(x.size), x).astype(np.float32)


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode PCM WAV bytes to mono float32 in -1..1 plus the sample rate."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        rate = wav.getframerate()
        width = wav.getsampwidth()
        channels = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {width} bytes")
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, rate


def write_wav(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode mono float samples as 16-bit PCM WAV bytes."""
    x = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1, 1)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((x * 32767).astype("<i2").tobytes())
    return buffer.getvalue()
