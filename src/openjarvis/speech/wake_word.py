"""Wake-word detection ("Hey Jarvis") on top of openWakeWord.

openWakeWord ships a pre-trained English ``hey_jarvis`` model. It fires reliably
on "hey/ey Jarvis"; Spanish "oye Jarvis" scores lower, so ``threshold`` is
exposed. Requires ``pip install openwakeword`` (extra ``voice-wake``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

FRAME_SAMPLES = 1280  # 80 ms at 16 kHz, the size openWakeWord expects
RESET_COOLDOWN_FRAMES = 6  # ~0.5 s: lets a deliberate re-trigger through


class WakeWordDetector:
    def __init__(
        self,
        model_name: str = "hey_jarvis",
        *,
        threshold: float = 0.5,
        cooldown_frames: int = 25,  # ~2 s
    ) -> None:
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as exc:
            raise RuntimeError(
                "Wake word needs openwakeword: pip install 'OpenJarvis[voice-wake]'"
            ) from exc

        try:
            self._model = Model(
                wakeword_models=[model_name], inference_framework="onnx"
            )
        except Exception:
            openwakeword.utils.download_models([model_name])
            self._model = Model(
                wakeword_models=[model_name], inference_framework="onnx"
            )
        self._threshold = threshold
        self._cooldown_frames = cooldown_frames
        self._cooldown = 0
        self._buf = np.zeros(0, dtype=np.int16)
        self.last_score = 0.0

    def reset(self) -> None:
        self._model.reset()
        self._cooldown = RESET_COOLDOWN_FRAMES
        self._buf = np.zeros(0, dtype=np.int16)

    def process(self, frame: np.ndarray) -> bool:
        """Feed int16 samples; return True when the wake word was heard."""
        data = np.concatenate([self._buf, frame.astype(np.int16)])
        fired = False
        n = len(data) // FRAME_SAMPLES
        for i in range(n):
            scores = self._model.predict(
                data[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            )
            score = max(scores.values()) if scores else 0.0
            self.last_score = score
            if self._cooldown > 0:
                self._cooldown -= 1
            elif score >= self._threshold:
                fired = True
                self._cooldown = self._cooldown_frames
        self._buf = data[n * FRAME_SAMPLES :]
        return fired


def best_score(detector: WakeWordDetector, audio: np.ndarray) -> Optional[float]:
    """Highest score over *audio* (utility for calibration and tests)."""
    best = 0.0
    for i in range(0, len(audio) - FRAME_SAMPLES, FRAME_SAMPLES):
        detector.process(audio[i : i + FRAME_SAMPLES])
        best = max(best, detector.last_score)
    return best
