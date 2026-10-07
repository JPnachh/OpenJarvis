"""Double-clap detector for hands-free activation.

Works on 16 kHz mono int16 frames. A *clap* is a sharp, loud, short sound: its
energy jumps well above the running noise floor and then decays quickly. Speech,
music and sustained noise stay loud, so they are rejected by the decay check.
Two claps separated by ``min_gap``..``max_gap`` seconds trigger.

Time is measured in processed samples, not wall-clock, so the detector behaves
the same on a live microphone and on a pre-recorded file.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

_SUBFRAME_MS = 20


class ClapDetector:
    def __init__(
        self,
        sample_rate: int = 16000,
        *,
        min_peak: float = 0.30,
        ratio: float = 6.0,
        min_gap: float = 0.12,
        max_gap: float = 0.75,
        cooldown: float = 2.0,
        decay_ms: int = 140,
    ) -> None:
        self._sr = sample_rate
        self._sub = int(sample_rate * _SUBFRAME_MS / 1000)
        self._min_peak = min_peak
        self._ratio = ratio
        self._min_gap = min_gap
        self._max_gap = max_gap
        self._cooldown = cooldown
        self._decay_frames = max(1, decay_ms // _SUBFRAME_MS)
        self.reset()

    def reset(self) -> None:
        self._t = 0.0
        self._floor = 0.01
        self._prev_rms = 0.0
        self._candidate: Optional[list] = None  # [t0, rms0, frames_left]
        self._claps: deque[float] = deque()
        self._muted_until = 0.0
        self._buf = np.zeros(0, dtype=np.float32)

    def process(self, frame: np.ndarray) -> bool:
        """Feed an int16 frame; return True when a double clap was heard."""
        x = np.concatenate([self._buf, frame.astype(np.float32) / 32768.0])
        fired = False
        n = len(x) // self._sub
        for i in range(n):
            if self._step(x[i * self._sub : (i + 1) * self._sub]):
                fired = True
        self._buf = x[n * self._sub :]
        return fired

    def _step(self, sub: np.ndarray) -> bool:
        dt = len(sub) / self._sr
        self._t += dt
        rms = float(np.sqrt(np.mean(sub * sub))) + 1e-9
        peak = float(np.max(np.abs(sub)))
        fired = False

        if self._candidate is not None:
            t0, rms0, left = self._candidate
            if rms < 0.4 * rms0:  # decayed fast: it was a clap
                self._claps.append(t0)
                self._candidate = None
                fired = self._check_double()
            else:
                self._candidate[2] = left - 1
                if self._candidate[2] <= 0:  # stayed loud: speech/music/noise
                    self._candidate = None
        elif (
            self._t >= self._muted_until
            and peak >= self._min_peak
            and rms >= self._ratio * max(self._floor, 0.005)
            and rms >= 2.0 * self._prev_rms  # sharp attack
        ):
            self._candidate = [self._t, rms, self._decay_frames]
        elif rms < 3.0 * self._floor + 0.01:
            self._floor = 0.97 * self._floor + 0.03 * rms  # track quiet noise

        self._prev_rms = rms
        while self._claps and self._t - self._claps[0] > self._max_gap:
            self._claps.popleft()
        return fired

    def _check_double(self) -> bool:
        if len(self._claps) < 2:
            return False
        gap = self._claps[-1] - self._claps[-2]
        if self._min_gap <= gap <= self._max_gap:
            self._claps.clear()
            self._muted_until = self._t + self._cooldown
            return True
        return False
