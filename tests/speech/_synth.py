"""Synthetic 'spoken phrases' for wake-word tests.

Each phrase is a sequence of voiced segments (harmonics shaped by one
formant), so different phrases have different spectral trajectories, and
takes of the same phrase vary in speed, pitch, loudness and noise like real
repetitions do.
"""

from __future__ import annotations

import numpy as np

SR = 16000

HEY_JARVIS = [
    (120, 500, 0.12),
    (120, 2000, 0.15),
    (1e-4, 0, 0.05),
    (120, 700, 0.10),
    (120, 1200, 0.18),
    (120, 2500, 0.15),
]
OTHER = [(120, 900, 0.2), (120, 400, 0.15), (120, 2200, 0.2), (120, 600, 0.2)]
OTHER2 = [(120, 2000, 0.15), (120, 500, 0.12), (120, 1200, 0.18), (120, 700, 0.1)]


def phrase(segments, rng, stretch=1.0, pitch=1.0, noise=0.003, gain=0.3):
    out = []
    for f0, formant, dur in segments:
        n = int(dur * stretch * SR)
        t = np.arange(n) / SR
        sig = sum(
            np.sin(2 * np.pi * f0 * pitch * h * t)
            * np.exp(-(((f0 * pitch * h - formant) / 400) ** 2))
            for h in range(1, 15)
        )
        ramp = np.minimum(1, np.minimum(t / 0.02, (t[-1] - t) / 0.02 + 1e-9))
        out.append(sig * ramp)
    x = np.concatenate(out)
    x = x / np.max(np.abs(x)) * gain
    return (x + rng.normal(0, noise, x.size)).astype(np.float32)


def take(segments, rng):
    """One natural-ish repetition of a phrase."""
    return phrase(
        segments,
        rng,
        stretch=rng.uniform(0.88, 1.12),
        pitch=rng.uniform(0.95, 1.05),
        gain=rng.uniform(0.15, 0.5),
    )


def room(seconds, rng, level=0.003):
    return rng.normal(0, level, int(seconds * SR)).astype(np.float32)
