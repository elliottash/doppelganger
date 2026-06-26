"""Audio loading / fixed-window framing and seeding helpers."""
from __future__ import annotations

import random
import numpy as np

from config import TARGET_SR, CLIP_SECONDS, MONO


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_audio(path: str, sr: int = TARGET_SR, mono: bool = MONO,
               seconds: float = CLIP_SECONDS) -> np.ndarray:
    """Load -> mono -> resample -> pad/centre-trim to a fixed `seconds` window.

    A fixed window matters: pooled clip embeddings shift with duration, and we do not want
    the synthetic/real gap to be confounded by a duration distribution difference. Pad short
    clips with zeros; centre-crop long ones.
    """
    import librosa
    wav, _ = librosa.load(path, sr=sr, mono=mono)
    n = int(round(seconds * sr))
    if len(wav) < n:
        wav = np.pad(wav, (0, n - len(wav)))
    elif len(wav) > n:
        start = (len(wav) - n) // 2
        wav = wav[start:start + n]
    # peak-normalise to remove trivial loudness cues (another nuisance variable)
    peak = np.max(np.abs(wav)) + 1e-9
    return (wav / peak).astype(np.float32)
