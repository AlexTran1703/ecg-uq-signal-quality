"""Signal-processing helpers: band-pass, QRS detectors, peak picking, beat matching, QRS excision."""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..config import EPS, FS_ECG


def _bandpass(X, lo, hi, fs, order=3):
    b, a = signal.butter(order, [lo / (fs / 2), min(hi / (fs / 2), 0.999)], btype="band")
    return signal.filtfilt(b, a, X, axis=-1)


def _mad(X):
    med = np.median(X, axis=-1, keepdims=True)
    return np.median(np.abs(X - med), axis=-1) * 1.4826 + EPS


def _detector_a(X):
    return np.abs(_bandpass(X, 5.0, 20.0, FS_ECG))


def _detector_b(X):
    x = _bandpass(X, 5.0, 25.0, FS_ECG)
    d = np.diff(x, axis=-1, prepend=x[..., :1]) ** 2
    k = max(1, int(0.12 * FS_ECG))
    return signal.oaconvolve(d, np.ones((1, k)) / k, mode="same", axes=-1)


def _peaks(d, scale_mult):
    s = np.median(np.abs(d - np.median(d))) * 1.4826 + EPS
    p, _ = signal.find_peaks(d, distance=int(0.25 * FS_ECG), prominence=scale_mult * s)
    return p


def _match(pa, pb, tol):
    if len(pa) == 0 or len(pb) == 0:
        return 0
    used = np.zeros(len(pb), bool)
    n = 0
    for t in pa:
        j = int(np.argmin(np.where(used, np.inf, np.abs(pb - t))))
        if not used[j] and abs(pb[j] - t) <= tol:
            used[j] = True
            n += 1
    return n
def _excise_qrs(B, peaks, half):
    out = B.copy()
    for i in range(len(B)):
        for r in peaks[i]:
            a, b = max(0, r - half), min(B.shape[1] - 1, r + half)
            if b > a:
                out[i, a:b + 1] = np.linspace(out[i, a], out[i, b], b - a + 1)
    return out
