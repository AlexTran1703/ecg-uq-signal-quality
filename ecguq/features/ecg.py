"""ECG feature groups E1-E4 (detectability, rhythm, spectral contamination, statistics/complexity)."""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..config import EPS, FEATURES_CFG, FS_ECG
from .dsp import _detector_a, _detector_b, _match, _peaks

E1_NAMES = ["ECG_qSQI", "ECG_template_corr"]
E2_NAMES = ["ECG_HR_mean", "ECG_RR_max", "ECG_RR_ratio"]
E3_NAMES = ["ECG_pSQI", "ECG_basSQI", "ECG_HF_ratio", "ECG_line50_ratio"]
E4_NAMES = ["ECG_kSQI", "ECG_MSE"]


_MSE = FEATURES_CFG["ecg"]["multiscale_entropy"]
MSE_FS = int(_MSE["decimate_to_hz"])
MSE_SCALES = tuple(_MSE["scales"])
MSE_M = int(_MSE["m"])
MSE_R = float(_MSE["r_fraction_of_sd"])
_BANDS = FEATURES_CFG["ecg"]["bands_hz"]
WELCH_N = int(FEATURES_CFG["ecg"]["welch_nperseg"])


def spectral_ratios(f, P):
    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return P[..., m].sum(-1)

    return {f"ECG_{k}": band(*v["numerator"]) / (band(*v["denominator"]) + EPS)
            for k, v in _BANDS.items()}


def _sampen(x: np.ndarray, m: int = MSE_M, r_frac: float = MSE_R) -> float:
    n = len(x)
    if n < m + 20:
        return 0.0
    sd = float(np.std(x))
    if sd < EPS:
        return 0.0
    r = np.float32(r_frac * sd)

    xf = np.asarray(x, np.float32)
    A = np.abs(xf[:, None] - xf[None, :])

    rows_b = n - m + 1
    D = A[:rows_b, :rows_b].copy()
    for s in range(1, m):
        np.maximum(D, A[s:s + rows_b, s:s + rows_b], out=D)

    b = (int(np.count_nonzero(D <= r)) - rows_b) // 2

    rows_a = rows_b - 1
    Da = D[:rows_a, :rows_a]
    np.maximum(Da, A[m:m + rows_a, m:m + rows_a], out=Da)
    a = (int(np.count_nonzero(Da <= r)) - rows_a) // 2

    if a == 0 or b == 0:
        return 0.0
    return float(-np.log(a / b))


def _multiscale_entropy(x: np.ndarray) -> float:
    q = max(1, FS_ECG // MSE_FS)
    z = x[: (len(x) // q) * q].reshape(-1, q).mean(1)
    vals = []
    for s in MSE_SCALES:
        cg = z[: (len(z) // s) * s].reshape(-1, s).mean(1) if s > 1 else z
        vals.append(_sampen(cg))
    return float(np.mean(vals))


def ecg_features(X: np.ndarray) -> dict[str, np.ndarray]:
    X = np.asarray(X, np.float64)
    n, T = X.shape
    Da, Db = _detector_a(X), _detector_b(X)
    tol = int(0.15 * FS_ECG)
    pre, post = int(0.20 * FS_ECG), int(0.40 * FS_ECG)
    off = np.arange(-pre, post)

    qsqi = np.zeros(n)
    tcorr = np.zeros(n)
    hr_mean = np.zeros(n)
    rr_max = np.full(n, 10.0)
    rr_ratio = np.full(n, 10.0)
    beats = np.zeros(n, np.int32)

    for i in range(n):
        pa = _peaks(Da[i], 1.5)
        pb = _peaks(Db[i], 3.0)
        beats[i] = len(pa)
        if len(pa) + len(pb):
            qsqi[i] = 2.0 * _match(pa, pb, tol) / (len(pa) + len(pb))

        if len(pa) >= 3:
            rr = np.diff(pa) / FS_ECG
            hr_mean[i] = 60.0 / max(float(rr.mean()), EPS)
            rr_max[i] = float(rr.max())
            rr_ratio[i] = float(rr.max() / max(rr.min(), EPS))

            ok = pa[(pa >= pre) & (pa < T - post)]
            if len(ok) >= 3:
                B = X[i][ok[:, None] + off]
                B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + EPS)
                t = np.median(B, axis=0)
                t = (t - t.mean()) / (t.std() + EPS)
                tcorr[i] = float(np.median((B @ t) / len(t)))


    f, P = signal.welch(X, fs=FS_ECG, nperseg=min(T, WELCH_N), noverlap=min(T, WELCH_N) // 2, axis=-1)
    spec = spectral_ratios(f, P)

    mu = X.mean(-1, keepdims=True)
    sd = X.std(-1, keepdims=True) + EPS
    Z = (X - mu) / sd

    mse = np.array([_multiscale_entropy(X[i]) for i in range(n)])

    out = {
        "ECG_qSQI": qsqi,
        "ECG_template_corr": tcorr,
        "ECG_HR_mean": hr_mean,
        "ECG_RR_max": rr_max,
        "ECG_RR_ratio": rr_ratio,
        "ECG_pSQI": spec["ECG_pSQI"],
        "ECG_basSQI": spec["ECG_basSQI"],
        "ECG_HF_ratio": spec["ECG_HF_ratio"],
        "ECG_line50_ratio": spec["ECG_line50_ratio"],
        "ECG_kSQI": (Z ** 4).mean(-1),
        "ECG_MSE": mse,
    }
    out = {k: np.asarray(v, np.float32) for k, v in out.items()}
    out["_beats"] = beats
    return out


def ecg_features_subset(X, need):
    need = set(need)
    X = np.asarray(X, np.float64)
    n, T = X.shape
    out = {}

    if need & set(E1_NAMES + E2_NAMES):
        full = ecg_features(X)
        return {k: v for k, v in full.items() if k in need or k == "_beats"}

    if need & set(E3_NAMES):
        f, P = signal.welch(X, fs=FS_ECG, nperseg=min(T, WELCH_N), noverlap=min(T, WELCH_N) // 2,
                            axis=-1)
        for name, v in spectral_ratios(f, P).items():
            if name in need:
                out[name] = np.asarray(v, np.float32)

    if "ECG_kSQI" in need:
        Z = (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + EPS)
        out["ECG_kSQI"] = np.asarray((Z ** 4).mean(-1), np.float32)
    if "ECG_MSE" in need:
        out["ECG_MSE"] = np.asarray([_multiscale_entropy(X[i]) for i in range(n)], np.float32)
    return out
