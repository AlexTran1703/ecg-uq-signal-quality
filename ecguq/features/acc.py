"""Accelerometer feature groups M1-M2 and the cross-modal group M3."""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.fft import irfft, next_fast_len, rfft

from ..config import EPS, FS_ACC, FS_ECG
from .dsp import _bandpass, _detector_a, _excise_qrs, _peaks

M1_NAMES = ["ACC_ENMO", "ACC_MAD"]
M2_NAMES = ["ACC_jerk_rms", "ACC_dom_freq", "ACC_spec_entropy"]
M3_NAMES = ["CPL_vel_xcorr", "CPL_vel_coh"]


def _velocity(v: np.ndarray, fs: int = FS_ACC) -> np.ndarray:
    vel = np.cumsum(v, axis=-1) / fs
    b, a = signal.butter(2, 0.3 / (fs / 2), btype="high")
    return signal.filtfilt(b, a, vel, axis=-1)


def m1_m2_features(A: np.ndarray) -> dict[str, np.ndarray]:
    A = np.asarray(A, np.float64) / 1000.0
    mag = np.linalg.norm(A, axis=2)


    enmo = np.maximum(mag - 1.0, 0.0).mean(1)
    mad = np.abs(mag - mag.mean(1, keepdims=True)).mean(1)


    v = mag - mag.mean(1, keepdims=True)
    d = np.diff(v, axis=1) * FS_ACC
    jerk = np.log1p(np.sqrt((d ** 2).mean(1)))

    f, P = signal.welch(v, fs=FS_ACC, nperseg=min(v.shape[1], 256), axis=1)
    Pn = P / np.clip(P.sum(1, keepdims=True), EPS, None)
    ent = -(Pn * np.log(np.clip(Pn, EPS, None))).sum(1) / np.log(Pn.shape[1])
    dom = f[np.argmax(P, axis=1)]

    return {
        "ACC_ENMO": np.asarray(enmo, np.float32),
        "ACC_MAD": np.asarray(mad, np.float32),
        "ACC_jerk_rms": np.asarray(jerk, np.float32),
        "ACC_dom_freq": np.asarray(dom, np.float32),
        "ACC_spec_entropy": np.asarray(ent, np.float32),
    }


def m3_features(X: np.ndarray, A: np.ndarray, excise_qrs: bool = True) -> dict[str, np.ndarray]:
    X = np.asarray(X, np.float64)
    A = np.asarray(A, np.float64) / 1000.0
    n = len(X)

    B = _bandpass(X, 0.5, 10.0, FS_ECG)
    if excise_qrs:
        Da = _detector_a(X)
        B = _excise_qrs(B, [_peaks(Da[i], 1.5) for i in range(n)], int(0.080 * FS_ECG))

    b100 = signal.resample_poly(B, up=2, down=5, axis=-1)
    mag = np.linalg.norm(A, axis=-1)
    v = mag - mag.mean(-1, keepdims=True)
    vel = _velocity(v)

    m = min(b100.shape[1], vel.shape[1])
    b100, vel = b100[:, :m], vel[:, :m]

    nper = min(m, 256)
    f, Cxy = signal.coherence(b100, vel, fs=FS_ACC, nperseg=nper, noverlap=nper // 2, axis=-1)
    band = (f >= 0.5) & (f <= 10.0)
    coh = Cxy[..., band].mean(-1)

    a = (b100 - b100.mean(-1, keepdims=True)) / (b100.std(-1, keepdims=True) + EPS)
    c = (vel - vel.mean(-1, keepdims=True)) / (vel.std(-1, keepdims=True) + EPS)
    nfft = next_fast_len(2 * m - 1)
    cc = irfft(rfft(a, nfft, axis=-1) * np.conj(rfft(c, nfft, axis=-1)), nfft, axis=-1) / m
    lag = int(0.5 * FS_ACC)
    peak = np.abs(np.concatenate([cc[:, :lag + 1], cc[:, -lag:]], 1)).max(-1)

    return {"CPL_vel_xcorr": np.asarray(peak, np.float32),
            "CPL_vel_coh": np.asarray(coh, np.float32)}
