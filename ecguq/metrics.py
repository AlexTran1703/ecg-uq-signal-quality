"""Weighted metrics, error-detection AUROC, risk-coverage summaries and the subject bootstrap."""
from __future__ import annotations

import numpy as np

from .config import EPS, N_BOOT, BOOT_SEED, N_CLASSES


def _confusion_w(y, yhat, w):
    return np.bincount(y * N_CLASSES + yhat, weights=w,
                       minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)


def macro_f1_w(y, yhat, w=None):
    w = np.ones(len(y)) if w is None else w
    C = _confusion_w(y, yhat, w)
    tp = np.diag(C)
    f1 = 2 * tp / np.maximum(C.sum(0) + C.sum(1), EPS)
    return float(f1.mean())


def balanced_accuracy_w(y, yhat, w=None):
    w = np.ones(len(y)) if w is None else w
    C = _confusion_w(y, yhat, w)
    n = C.sum(1)
    present = n > 0
    return float((np.diag(C)[present] / n[present]).mean()) if present.any() else np.nan


def recall_w(y, yhat, w=None):
    w = np.ones(len(y)) if w is None else w
    C = _confusion_w(y, yhat, w)
    n = C.sum(1)
    return [float(C[k, k] / n[k]) if n[k] > 0 else np.nan for k in range(N_CLASSES)]


def nll_w(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    q = np.clip(np.asarray(p, np.float64)[np.arange(len(y)), y], EPS, 1.0)
    return float(-(w * np.log(q)).sum() / w.sum())


def brier_w(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    p = np.asarray(p, np.float64)
    Y = np.zeros_like(p)
    Y[np.arange(len(y)), y] = 1.0
    return float((w * ((p - Y) ** 2).sum(1)).sum() / w.sum())


def ece_w(y, p, w=None, n_bins: int = 15):
    w = np.ones(len(y)) if w is None else w
    p = np.asarray(p, np.float64)
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    b = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    wsum = np.bincount(b, weights=w, minlength=n_bins)
    csum = np.bincount(b, weights=w * conf, minlength=n_bins)
    asum = np.bincount(b, weights=w * correct, minlength=n_bins)
    m = wsum > 0
    return float((np.abs(asum[m] - csum[m]) / w.sum()).sum())


def auroc_w(err_sorted, w_sorted, u_sorted):
    pos = w_sorted * err_sorted
    neg = w_sorted * (1.0 - err_sorted)
    npos, nneg = pos.sum(), neg.sum()
    if npos <= 0 or nneg <= 0:
        return np.nan

    grp = np.concatenate(([0], np.cumsum(np.diff(u_sorted) != 0)))
    gp = np.bincount(grp, weights=pos)
    gn = np.bincount(grp, weights=neg)
    below = np.concatenate(([0.0], np.cumsum(gn)[:-1]))
    return float((gp * (below + 0.5 * gn)).sum() / (npos * nneg))


def _rc_w(err_sorted, w_sorted):
    cw = np.cumsum(w_sorted)
    ce = np.cumsum(w_sorted * err_sorted)
    n = cw[-1]
    if n <= 0:
        return None
    cov = cw / n
    return cov, ce / np.maximum(cw, EPS), ce / n


def aurc_w(err_sorted, w_sorted):
    r = _rc_w(err_sorted, w_sorted)
    if r is None or len(r[0]) < 2:
        return np.nan
    cov, risk, _ = r
    return float(np.trapezoid(risk, cov) / (cov[-1] - cov[0]))


def augrc_w(err_sorted, w_sorted):
    r = _rc_w(err_sorted, w_sorted)
    if r is None or len(r[0]) < 2:
        return np.nan
    cov, _, grisk = r
    return float(np.trapezoid(grisk, cov) / (cov[-1] - cov[0]))


def eaurc_w(err_sorted, w_sorted, err_oracle_sorted, w_oracle_sorted):
    return aurc_w(err_sorted, w_sorted) - aurc_w(err_oracle_sorted, w_oracle_sorted)


def selective_risk_at(err_sorted, w_sorted, coverage: float):
    cw = np.cumsum(w_sorted)
    n = cw[-1]
    k = np.searchsorted(cw, coverage * n) + 1
    k = min(k, len(cw))
    return float(np.cumsum(w_sorted * err_sorted)[k - 1] / cw[k - 1])


def subject_weights(subject: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    subject = np.asarray(subject).astype(str)
    subs, code = np.unique(subject, return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        m = np.bincount(rng.integers(0, len(subs), size=len(subs)), minlength=len(subs))
        yield m[code].astype(np.float64)


def ci_from(point, vals, alpha: float = 0.05):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if not len(vals):
        return float(point), np.nan, np.nan
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def boot_stat(fn, subject, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    point = fn(np.ones(len(subject)))
    vals = [fn(w) for w in subject_weights(subject, n_boot, seed)]
    return ci_from(point, vals)


def boot_diff(fn_a, fn_b, subject, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    ones = np.ones(len(subject))
    point = fn_b(ones) - fn_a(ones)
    vals = []
    for w in subject_weights(subject, n_boot, seed):
        d = fn_b(w) - fn_a(w)
        if np.isfinite(d):
            vals.append(d)
    return ci_from(point, vals)


def boot_index(stat, subject, *arrays, n_boot: int = 500, seed: int = BOOT_SEED):
    subject = np.asarray(subject).astype(str)
    subs = np.unique(subject)
    idx_of = {s: np.flatnonzero(subject == s) for s in subs}
    rng = np.random.default_rng(seed)
    point = stat(*arrays)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[s] for s in rng.choice(subs, size=len(subs), replace=True)])
        try:
            v = stat(*(a[idx] for a in arrays))
        except Exception:
            continue
        if v is not None and np.isfinite(v):
            vals.append(v)
    return ci_from(point, vals)
