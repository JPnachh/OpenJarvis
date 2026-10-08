"""Personal wake word ("Hey Jarvis") trained on the user's own voice.

Training stores a few recordings of the phrase (positives) and, optionally,
recordings of other speech (negatives) under ``~/.openjarvis/wakeword/``. The
trained profile (``profile.json``) keeps one MFCC template per positive
recording and a match threshold calibrated so the positives match each other
while the negatives do not.

At runtime :class:`WakeWordDetector` slides a window the length of each
template over live audio and fires when the DTW distance to any template
drops below the threshold. Everything runs locally with numpy; no model
download, and it works for any language or accent because it learns from
the speaker.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from openjarvis.core.config import DEFAULT_CONFIG_DIR
from openjarvis.speech.features import (
    SAMPLE_RATE,
    dtw_distance,
    mfcc,
    read_wav,
    resample,
    trim_silence,
    write_wav,
)

DEFAULT_PHRASE = "Hey Jarvis"
MIN_POSITIVES = 3
RECOMMENDED_POSITIVES = 6
PROFILE_VERSION = 1
_HOP_FRAMES = 10  # 100 ms between detection attempts
_MIN_SECONDS = 0.3
_MAX_SECONDS = 2.5


def wakeword_dir() -> Path:
    """Folder holding the training recordings and the trained profile."""
    return DEFAULT_CONFIG_DIR / "wakeword"


def _samples_dir(kind: str, root: Path | None = None) -> Path:
    if kind not in ("positive", "negative"):
        raise ValueError("kind must be 'positive' or 'negative'")
    return (root or wakeword_dir()) / "samples" / kind


# ---------------------------------------------------------------------------
# Training samples
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    id: str
    kind: str
    path: Path
    seconds: float


def add_sample(
    audio: bytes | np.ndarray,
    kind: str,
    *,
    sample_rate: int = SAMPLE_RATE,
    root: Path | None = None,
    prefix: str = "",
) -> Sample:
    """Store one training recording (WAV bytes or float samples).

    The recording is converted to 16 kHz mono and trimmed of silence. Raises
    ValueError when nothing audible remains or the length is implausible.
    """
    if isinstance(audio, (bytes, bytearray)):
        samples, sample_rate = read_wav(bytes(audio))
    else:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    samples = trim_silence(resample(samples, sample_rate, SAMPLE_RATE))
    seconds = samples.size / SAMPLE_RATE
    if seconds < 0.2:
        raise ValueError("The recording is silent or too short. Speak a bit louder.")
    limit = _MAX_SECONDS if kind == "positive" else 15.0
    if seconds > limit:
        raise ValueError(
            f"The recording is {seconds:.1f}s long; say only the phrase "
            f"(at most {limit:.0f}s)."
        )
    peak = float(np.max(np.abs(samples)))
    if peak < 0.02:
        raise ValueError("The recording is too quiet. Move closer to the microphone.")
    folder = _samples_dir(kind, root)
    folder.mkdir(parents=True, exist_ok=True)
    sample_id = f"{prefix}{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    path = folder / f"{sample_id}.wav"
    path.write_bytes(write_wav(samples))
    return Sample(sample_id, kind, path, seconds)


def list_samples(root: Path | None = None) -> list[Sample]:
    found: list[Sample] = []
    for kind in ("positive", "negative"):
        folder = _samples_dir(kind, root)
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.wav")):
            try:
                samples, rate = read_wav(path.read_bytes())
                seconds = samples.size / rate
            except (OSError, ValueError, EOFError):
                continue
            found.append(Sample(path.stem, kind, path, seconds))
    return found


def delete_sample(sample_id: str, root: Path | None = None) -> bool:
    if not sample_id or "/" in sample_id or "\\" in sample_id or ".." in sample_id:
        return False
    for kind in ("positive", "negative"):
        path = _samples_dir(kind, root) / f"{sample_id}.wav"
        if path.is_file():
            path.unlink()
            return True
    return False


def sample_audio(sample_id: str, root: Path | None = None) -> bytes | None:
    if not sample_id or "/" in sample_id or "\\" in sample_id or ".." in sample_id:
        return None
    for kind in ("positive", "negative"):
        path = _samples_dir(kind, root) / f"{sample_id}.wav"
        if path.is_file():
            return path.read_bytes()
    return None


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@dataclass
class WakeWordProfile:
    phrase: str
    threshold: float
    templates: list[np.ndarray]
    trained_at: float = field(default_factory=time.time)
    stats: dict = field(default_factory=dict)

    def save(self, root: Path | None = None) -> Path:
        folder = root or wakeword_dir()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "profile.json"
        path.write_text(
            json.dumps(
                {
                    "version": PROFILE_VERSION,
                    "phrase": self.phrase,
                    "threshold": self.threshold,
                    "sample_rate": SAMPLE_RATE,
                    "trained_at": self.trained_at,
                    "stats": self.stats,
                    "templates": [np.round(t, 4).tolist() for t in self.templates],
                }
            )
        )
        return path

    @classmethod
    def load(cls, root: Path | None = None) -> Optional["WakeWordProfile"]:
        path = (root or wakeword_dir()) / "profile.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("version") != PROFILE_VERSION:
            return None
        try:
            templates = [np.asarray(t, dtype=np.float32) for t in data["templates"]]
            return cls(
                phrase=str(data.get("phrase") or DEFAULT_PHRASE),
                threshold=float(data["threshold"]),
                templates=[t for t in templates if t.ndim == 2 and len(t) > 5],
                trained_at=float(data.get("trained_at") or 0),
                stats=dict(data.get("stats") or {}),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _load_audio(path: Path) -> np.ndarray | None:
    try:
        samples, rate = read_wav(path.read_bytes())
    except (OSError, ValueError, EOFError):
        return None
    samples = trim_silence(resample(samples, rate, SAMPLE_RATE))
    return samples if samples.size >= SAMPLE_RATE * 0.2 else None


def _best_window_distance(audio: np.ndarray, templates: list[np.ndarray]) -> float:
    """Smallest distance between any template and any window of a recording.

    Windows are cut and featurised exactly as the live detector does, so the
    calibrated threshold means the same thing at runtime.
    """
    best = math.inf
    hop = _HOP_FRAMES * 160
    for template in templates:
        needed = len(template) * 160 + 240
        if audio.size <= needed:
            best = min(best, dtw_distance(mfcc(audio), template))
            continue
        for end in range(needed, audio.size + 1, hop):
            best = min(best, dtw_distance(mfcc(audio[end - needed : end]), template))
    return best


_SHUFFLES = ((1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0), (1, 2, 3, 0), (3, 0, 1, 2))


def _shuffled(audio: np.ndarray) -> list[np.ndarray]:
    """Re-order quarter chunks of a take into impostor phrases."""
    quarter = audio.size // 4
    if quarter < 400:
        return []
    chunks = [audio[i * quarter : (i + 1) * quarter] for i in range(4)]
    return [np.concatenate([chunks[i] for i in order]) for order in _SHUFFLES]


class TrainingError(ValueError):
    """Raised when the samples cannot produce a usable profile."""


def train(phrase: str = DEFAULT_PHRASE, root: Path | None = None) -> WakeWordProfile:
    """Build and save a profile from the stored samples; return it.

    The threshold sits above how far the user's own takes are from each other
    (leave-one-out) and, when negatives exist, safely below how close other
    speech gets.
    """
    samples = list_samples(root)
    positive_audio = [_load_audio(s.path) for s in samples if s.kind == "positive"]
    negative_audio = [_load_audio(s.path) for s in samples if s.kind == "negative"]
    positives = [mfcc(a) for a in positive_audio if a is not None]
    negatives = [a for a in negative_audio if a is not None]
    if len(positives) < MIN_POSITIVES:
        raise TrainingError(
            f"Record at least {MIN_POSITIVES} samples of '{phrase}' "
            f"({RECOMMENDED_POSITIVES} recommended); you have {len(positives)}."
        )

    own = []
    for i, template in enumerate(positives):
        others = positives[:i] + positives[i + 1 :]
        own.append(min(dtw_distance(template, other) for other in others))
    own_max = max(own)
    own_mean = float(np.mean(own))

    # Impostors set the scale of "not the wake word". Real negatives are
    # best; shuffled copies of the user's own takes (same voice and sounds,
    # wrong order, the classic false alarm) are always available, so even a
    # profile without negatives, or with very consistent takes, gets a
    # sensible threshold.
    shuffled_nearest = min(
        dtw_distance(mfcc(fake), template)
        for audio in positive_audio
        if audio is not None
        for fake in _shuffled(audio)
        for template in positives
    )
    other = [_best_window_distance(neg, positives) for neg in negatives]
    other = [d for d in other if math.isfinite(d)]
    nearest = min(other) if other else None
    impostor = min([shuffled_nearest, *other])
    # Lean towards accepting the user. Shuffled takes are an approximation of
    # a false alarm, so the threshold may reach them; recorded negatives are
    # real non-wake speech, so it stays safely below those.
    threshold = own_max + 0.75 * max(0.0, impostor - own_max)
    threshold = max(threshold, own_max * 1.1)
    threshold = min(threshold, shuffled_nearest)
    if nearest is not None:
        threshold = min(threshold, nearest * 0.95)
    threshold = max(threshold, own_max * 1.02)
    margin = impostor / own_max if own_max > 0 else math.inf

    if margin < 1.1:
        quality = "poor"
        advice = (
            "The wake word is hard to tell apart from other speech. Re-record "
            "in a quieter room, say it the same way each time, or pick a "
            "longer phrase."
        )
    elif margin < 1.35:
        quality = "fair"
        advice = (
            "Works, but may confuse similar phrases. Add more samples, recorded "
            "in the room where you will use it."
        )
    else:
        quality = "good"
        advice = "Your voice profile separates the wake word clearly."
    if not other:
        advice += (
            " Record a few other sentences too, so Jarvis learns what is not "
            "the wake word."
        )

    profile = WakeWordProfile(
        phrase=phrase,
        threshold=float(threshold),
        templates=positives,
        stats={
            "positives": len(positives),
            "negatives": len(negatives),
            "own_distance_mean": round(own_mean, 4),
            "own_distance_max": round(own_max, 4),
            "nearest_negative": round(nearest, 4) if nearest is not None else None,
            "nearest_impostor": round(impostor, 4),
            "margin": round(margin, 3) if margin is not None else None,
            "quality": quality,
            "advice": advice,
        },
    )
    profile.save(root)
    return profile


def status(root: Path | None = None) -> dict:
    """Summary for the UI and CLI: where files live, counts, training state."""
    folder = root or wakeword_dir()
    samples = list_samples(root)
    profile = WakeWordProfile.load(root)
    return {
        "directory": str(folder),
        "profile_path": str(folder / "profile.json"),
        "positives": sum(1 for s in samples if s.kind == "positive"),
        "negatives": sum(1 for s in samples if s.kind == "negative"),
        "min_positives": MIN_POSITIVES,
        "recommended_positives": RECOMMENDED_POSITIVES,
        "trained": profile is not None,
        "phrase": profile.phrase if profile else DEFAULT_PHRASE,
        "trained_at": profile.trained_at if profile else None,
        "threshold": profile.threshold if profile else None,
        "stats": profile.stats if profile else None,
        "samples": [
            {"id": s.id, "kind": s.kind, "seconds": round(s.seconds, 2)}
            for s in samples
        ],
    }


# ---------------------------------------------------------------------------
# Live detection
# ---------------------------------------------------------------------------


def sensitivity_factor(sensitivity: float) -> float:
    """0..1 sensitivity → threshold multiplier (0.5 keeps the trained one)."""
    return 0.8 + 0.4 * min(1.0, max(0.0, sensitivity))


class WakeWordDetector:
    """Slide each template over the live signal; report a match."""

    def __init__(
        self,
        profile: WakeWordProfile,
        *,
        sensitivity: float = 0.5,
        cooldown: float = 2.0,
    ) -> None:
        self.profile = profile
        self.threshold = profile.threshold * sensitivity_factor(sensitivity)
        self.cooldown = cooldown
        longest = max(len(t) for t in profile.templates)
        # Keep enough audio for the longest template plus framing slack.
        self._keep = int((longest * 160 + 400) * 1.1)
        self._buffer: deque[np.ndarray] = deque()
        self._buffered = 0
        self._since_check = 0
        self._noise: Optional[float] = None
        self._t = 0.0
        self._cooldown_until = 0.0
        #: Last computed best distance, for live test meters.
        self.last_distance = math.inf
        self.last_match_audio: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._buffer.clear()
        self._buffered = 0
        self._since_check = 0

    def pause(self, seconds: float) -> None:
        """Ignore the microphone for a while (e.g. while Jarvis speaks)."""
        self._cooldown_until = max(self._cooldown_until, self._t + seconds)
        self.reset()

    def process(self, block: np.ndarray) -> bool:
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        self._t += x.size / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        if self._noise is None:
            self._noise = rms
        elif rms < self._noise * 2:
            self._noise = 0.97 * self._noise + 0.03 * rms

        self._buffer.append(x)
        self._buffered += x.size
        while self._buffered - self._buffer[0].size >= self._keep:
            self._buffered -= self._buffer.popleft().size
        self._since_check += x.size
        if self._since_check < _HOP_FRAMES * 160:
            return False
        self._since_check = 0
        if self._t < self._cooldown_until:
            return False

        audio = np.concatenate(self._buffer)
        # Only bother when the latest second holds speech-level sound.
        recent = audio[-SAMPLE_RATE // 2 :]
        level = float(np.sqrt(np.mean(recent * recent)))
        if level < max(0.008, (self._noise or 0) * 3):
            self.last_distance = math.inf
            return False

        best = math.inf
        best_window: Optional[np.ndarray] = None
        for template in self.profile.templates:
            needed = len(template) * 160 + 240
            if audio.size < needed:
                continue
            window = audio[-needed:]
            distance = dtw_distance(mfcc(window), template)
            if distance < best:
                best, best_window = distance, window
        self.last_distance = best
        if best <= self.threshold:
            #: The audio that matched, kept so a confirmed hit can be learned.
            self.last_match_audio = best_window
            self._cooldown_until = self._t + self.cooldown
            self.reset()
            return True
        return False


# ---------------------------------------------------------------------------
# Learning from use
# ---------------------------------------------------------------------------

AUTO_PREFIX = "auto-"
MAX_AUTO_SAMPLES = 6
RETRAIN_EVERY = 2


def learn_from_hit(audio: np.ndarray, root: Path | None = None) -> bool:
    """Keep a confirmed wake-word hit as a training sample; maybe retrain.

    Called only when the hit was followed by a real request, so it was the
    user talking to Jarvis. Learned takes are capped (oldest dropped) so the
    recordings the user made on purpose always carry most of the weight, and
    the profile is rebuilt every few new takes. Returns True after a retrain.
    """
    try:
        add_sample(audio, "positive", root=root, prefix=AUTO_PREFIX)
    except ValueError:
        return False
    auto = sorted(
        (s for s in list_samples(root) if s.id.startswith(AUTO_PREFIX)),
        key=lambda s: s.id,
    )
    for stale in auto[:-MAX_AUTO_SAMPLES]:
        delete_sample(stale.id, root)
    profile = WakeWordProfile.load(root)
    learned = (
        int((profile.stats or {}).get("learned_since_train", 0)) + 1 if profile else 1
    )
    if profile is not None and learned < RETRAIN_EVERY:
        profile.stats["learned_since_train"] = learned
        profile.save(root)
        return False
    try:
        retrained = train(profile.phrase if profile else DEFAULT_PHRASE, root)
    except TrainingError:
        return False
    retrained.stats["learned_since_train"] = 0
    retrained.save(root)
    return True
