"""Speaker recognition ("is this my voice?") on top of sherpa-onnx.

A short WeSpeaker ResNet34 model (26 MB, downloaded once) turns speech into a
fixed-size *voiceprint*. Enrolment averages several of your samples; later audio
is accepted when its cosine similarity to that average reaches a threshold.

This is a **convenience filter, not security**: a recording or a good voice
clone of you can pass, a very similar voice may pass, and noise lowers scores.
The voiceprint is biometric data. It is stored only on this machine
(``~/.openjarvis/voiceprint.npz``) and is never sent anywhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MODEL_NAME = "wespeaker_en_voxceleb_resnet34.onnx"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"speaker-recongition-models/{MODEL_NAME}"
)
MIN_SECONDS = 1.0  # shorter audio gives unreliable embeddings
MIN_THRESHOLD = 0.30
MAX_THRESHOLD = 0.75


def _home() -> Path:
    from openjarvis.core.config import DEFAULT_CONFIG_DIR

    return Path(DEFAULT_CONFIG_DIR)


def voiceprint_path() -> Path:
    return _home() / "voiceprint.npz"


def model_path() -> Path:
    return _home() / "models" / MODEL_NAME


def ensure_model() -> Path:
    """Download the embedding model on first use."""
    path = model_path()
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    import httpx

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    logger.info("Downloading speaker model %s", MODEL_URL)
    with httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_bytes():
                fh.write(chunk)
    tmp.replace(path)
    return path


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def to_float(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


class SpeakerVerifier:
    """Embeds audio and compares it with an enrolled voiceprint."""

    def __init__(self, path: Optional[Path] = None, *, extractor=None) -> None:
        self._path = Path(path) if path else voiceprint_path()
        self._extractor = extractor
        self.centroid: Optional[np.ndarray] = None
        self.threshold: float = 0.45
        self._updates = 0
        self.load()

    # -- persistence -------------------------------------------------------
    @property
    def enrolled(self) -> bool:
        return self.centroid is not None

    def load(self) -> None:
        try:
            data = np.load(self._path)
            self.centroid = data["centroid"].astype(np.float32)
            self.threshold = float(data["threshold"])
        except (OSError, KeyError, ValueError):
            self.centroid = None

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._path, centroid=self.centroid, threshold=self.threshold)

    def clear(self) -> None:
        self.centroid = None
        try:
            self._path.unlink()
        except OSError:
            pass

    # -- model -------------------------------------------------------------
    def _get_extractor(self):
        if self._extractor is None:
            import sherpa_onnx

            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(ensure_model()), num_threads=2, provider="cpu"
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        return self._extractor

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Return a unit-length embedding for 16 kHz mono *audio*."""
        ext = self._get_extractor()
        stream = ext.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=to_float(audio))
        stream.input_finished()
        return _unit(np.asarray(ext.compute(stream), dtype=np.float32))

    # -- enrolment / verification -----------------------------------------
    def enroll(self, samples: Sequence[np.ndarray]) -> dict:
        """Build a voiceprint from several utterances and save it."""
        embs = [self.embed(s) for s in samples if len(s) >= MIN_SECONDS * SAMPLE_RATE]
        if len(embs) < 3:
            raise ValueError("Need at least 3 samples of 1 s or more to enrol")
        mat = np.stack(embs)
        self.centroid = _unit(mat.mean(axis=0))
        # How consistent are *your* samples? Leave-one-out similarity.
        loo = []
        for i in range(len(embs)):
            rest = _unit(np.delete(mat, i, axis=0).mean(axis=0))
            loo.append(float(np.dot(embs[i], rest)))
        self.threshold = float(np.clip(min(loo) - 0.10, MIN_THRESHOLD, MAX_THRESHOLD))
        self.save()
        return {
            "samples": len(embs),
            "min_similarity": min(loo),
            "mean_similarity": float(np.mean(loo)),
            "threshold": self.threshold,
        }

    def score(self, audio: np.ndarray) -> Optional[float]:
        """Cosine similarity with the voiceprint, or None if not scoreable."""
        if self.centroid is None or len(audio) < MIN_SECONDS * SAMPLE_RATE:
            return None
        return float(np.dot(self.embed(audio), self.centroid))

    def verify(self, audio: np.ndarray) -> tuple[bool, Optional[float]]:
        """Return ``(accepted, score)``; accepts when nothing is enrolled."""
        if not self.enrolled:
            return True, None
        s = self.score(audio)
        if s is None:  # too short to judge: do not lock the owner out
            return True, None
        return s >= self.threshold, s

    def adapt(self, audio: np.ndarray, score: float, margin: float = 0.15) -> None:
        """Nudge the voiceprint toward confidently-matched speech (slow EMA)."""
        if self.centroid is None or score < self.threshold + margin:
            return
        self.centroid = _unit(0.98 * self.centroid + 0.02 * self.embed(audio))
        self._updates += 1
        if self._updates % 10 == 0:
            self.save()
