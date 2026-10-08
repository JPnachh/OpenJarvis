"""Hand-clap detection on a live microphone stream.

A clap is a short, sharp burst: the signal jumps far above the background
level and decays again within a fraction of a second. Speech, music and
slammed-but-ringing objects stay loud for longer, so the detector rejects any
burst that lasts past ``max_clap_seconds``. A *pattern* (two claps by default)
is required before triggering, which keeps a single bump on the desk from
opening anything.

:class:`ClapDetector` is pure logic over per-block ``(peak, rms)`` statistics
so it can be tested without audio hardware; :func:`listen_for_claps` wires it
to the default microphone with ``sounddevice``.
"""

from __future__ import annotations

import math
import queue
from dataclasses import dataclass, field
from typing import Callable, Optional

SAMPLE_RATE = 16000
# 256 samples at 16 kHz = 16 ms. A clap's attack is a few ms and its body
# 10-50 ms, so blocks must be short enough to see it begin and end.
BLOCK_SIZE = 256


def sensitivity_to_peak(sensitivity: float) -> float:
    """Map a 0..1 sensitivity to the minimum block peak counted as a clap."""
    sensitivity = min(1.0, max(0.0, sensitivity))
    return 0.5 - 0.42 * sensitivity


@dataclass
class ClapDetector:
    """Recognise a pattern of hand claps from per-block level statistics."""

    block_seconds: float = BLOCK_SIZE / SAMPLE_RATE
    claps_required: int = 2
    #: Minimum absolute peak (full scale = 1.0) for a burst to count.
    min_peak: float = sensitivity_to_peak(0.5)
    #: How far above the background RMS a burst must jump.
    onset_ratio: float = 6.0
    #: Bursts still loud after this long are not claps.
    max_clap_seconds: float = 0.25
    #: A burst has ended once its RMS is back near the room's background
    #: level (this many times it) or below this share of its maximum. Must be
    #: close to silence: speech dips between syllables but stays well above.
    settle_ratio: float = 3.0
    decay_ratio: float = 0.1
    #: Allowed spacing between consecutive claps of one pattern.
    min_gap: float = 0.12
    max_gap: float = 0.8
    #: Ignore everything for this long after a pattern fires.
    cooldown: float = 1.0

    _t: float = field(default=0.0, init=False)
    _background: Optional[float] = field(default=None, init=False)
    _burst_start: Optional[float] = field(default=None, init=False)
    _burst_rms: float = field(default=0.0, init=False)
    _last_clap: Optional[float] = field(default=None, init=False)
    _count: int = field(default=0, init=False)
    _cooldown_until: float = field(default=0.0, init=False)
    #: When the pattern is complete, fire only if no further clap follows by
    #: then: a steady beat (music, footsteps) keeps going, real claps stop.
    _fire_at: Optional[float] = field(default=None, init=False)

    @property
    def claps_so_far(self) -> int:
        """Claps counted towards the current pattern."""
        return self._count

    def process(self, peak: float, rms: float) -> bool:
        """Feed one block; return True when the clap pattern completes."""
        self._t += self.block_seconds
        now = self._t

        if self._background is None:
            self._background = rms
            return False

        if self._fire_at is not None and now >= self._fire_at:
            self._fire_at = None
            self._count = 0
            self._last_clap = None
            self._cooldown_until = now + self.cooldown
            return True

        if self._burst_start is not None:
            self._burst_rms = max(self._burst_rms, rms)
            settled = max(
                self._background * self.settle_ratio, self._burst_rms * self.decay_ratio
            )
            if rms < settled:
                start = self._burst_start
                self._burst_start = None
                if now - start <= self.max_clap_seconds:
                    return self._register(start)
            elif now - self._burst_start > self.max_clap_seconds:
                # Sustained sound (speech, music): not a clap, and it breaks
                # any pattern in progress.
                self._burst_start = None
                self._count = 0
                self._last_clap = None
                self._fire_at = None
                # Let a lasting loud room (music, TV) raise the background.
                self._background = 0.8 * self._background + 0.2 * rms
            return False

        # Expire a half-finished pattern that waited too long.
        if (
            self._fire_at is None
            and self._last_clap is not None
            and now - self._last_clap > self.max_gap
        ):
            self._count = 0
            self._last_clap = None

        if now >= self._cooldown_until:
            floor = max(self._background, 1e-4)
            if peak >= self.min_peak and rms >= self.onset_ratio * floor:
                self._burst_start = now
                self._burst_rms = rms
                return False

        # Track the room's background level slowly, outside bursts only.
        self._background = 0.95 * self._background + 0.05 * rms
        return False

    def _register(self, clap_time: float) -> bool:
        if self._last_clap is not None:
            gap = clap_time - self._last_clap
            if gap < self.min_gap:
                return False  # echo or double-hit of the same clap
            if gap > self.max_gap:
                self._count = 0
        self._count += 1
        self._last_clap = clap_time
        if self._count == self.claps_required:
            self._fire_at = clap_time + self.max_gap
        elif self._count > self.claps_required:
            # Too many: a rhythm, not a signal. Wait for it to stop.
            self._fire_at = None
        return False


def block_stats(samples) -> tuple[float, float]:
    """Return ``(peak, rms)`` of a block of float samples in -1..1."""
    import numpy as np

    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return 0.0, 0.0
    return float(np.max(np.abs(data))), float(math.sqrt(float(np.mean(data * data))))


def listen_for_claps(
    on_pattern: Callable[[], None],
    detector: ClapDetector,
    *,
    on_block: Optional[Callable[[float, float, ClapDetector], None]] = None,
    should_stop: Callable[[], bool] = lambda: False,
    device: Optional[int | str] = None,
) -> None:
    """Listen on the microphone and call *on_pattern* for each clap pattern.

    Blocks until *should_stop* returns True or the process is interrupted.
    Raises RuntimeError when sounddevice or a microphone is unavailable.
    """
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Clap detection needs the sounddevice package and PortAudio. "
            "Install with: uv sync --extra desktop"
        ) from exc

    blocks: "queue.Queue" = queue.Queue(maxsize=256)

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        try:
            blocks.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # drop rather than block the audio thread

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            device=device,
            callback=callback,
        )
    except Exception as exc:  # PortAudioError subclasses Exception
        raise RuntimeError(f"Could not open the microphone: {exc}") from exc

    with stream:
        while not should_stop():
            try:
                block = blocks.get(timeout=0.5)
            except queue.Empty:
                continue
            peak, rms = block_stats(block)
            if on_block is not None:
                on_block(peak, rms, detector)
            if detector.process(peak, rms):
                on_pattern()
