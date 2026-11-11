"""Synthetic ECG corruption (mains interference) applied to held-out windows only."""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..config import EPS, FS_ECG


CORRUPT_KINDS = ("baseline", "mains", "emg", "attenuate")
CORRUPT_LEVELS = (0.0, 0.25, 0.5, 1.0, 2.0)


CORRUPT_LABEL = {"baseline": "baseline wander", "mains": "mains interference",
                 "emg": "EMG noise", "attenuate": "contact loss"}


def corrupt(X: np.ndarray, kind: str, level: float, seed: int = 0) -> np.ndarray:
    X = np.asarray(X, np.float64)
    if level == 0.0 or kind is None:
        return X.copy()
    rng = np.random.default_rng(seed)
    n, T = X.shape
    t = np.arange(T) / FS_ECG
    q1, q3 = np.percentile(X, [25, 75], axis=-1, keepdims=True)


    mad = np.median(np.abs(X - np.median(X, axis=-1, keepdims=True)), axis=-1, keepdims=True) * 1.4826
    s = np.maximum.reduce([q3 - q1, mad, 0.05 * X.std(-1, keepdims=True),
                           np.full((n, 1), 1e-6)])

    if kind == "baseline":
        fr = rng.uniform(0.15, 0.5, (n, 1))
        ph = rng.uniform(0, 2 * np.pi, (n, 1))
        return X + 3.0 * level * s * np.sin(2 * np.pi * fr * t + ph)
    if kind == "mains":
        ph = rng.uniform(0, 2 * np.pi, (n, 1))
        w = np.sin(2 * np.pi * 50.0 * t + ph) + 0.3 * np.sin(2 * np.pi * 100.0 * t + ph)
        return X + 0.8 * level * s * w
    if kind == "emg":
        b, a = signal.butter(4, [40 / (FS_ECG / 2), min(100 / (FS_ECG / 2), 0.999)], btype="band")
        e = signal.filtfilt(b, a, rng.standard_normal((n, T)), axis=-1)
        return X + 1.2 * level * s * e / (e.std(-1, keepdims=True) + EPS)
    if kind == "attenuate":
        return np.exp(-1.2 * level) * X + 0.05 * s * rng.standard_normal((n, T))
    raise ValueError(kind)
