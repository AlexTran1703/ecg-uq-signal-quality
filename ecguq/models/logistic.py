"""Multinomial logistic model: MAP fit, Laplace posterior, evidence-based prior selection."""
from __future__ import annotations

import numpy as np
from scipy import linalg

from ..config import EPS, N_CLASSES

DEVICE = "cpu"
FREE = N_CLASSES - 1


def resolve_device(name: str) -> str:
    if name in ("cpu", "auto"):
        return "cpu"
    raise RuntimeError("only the CPU backend is available for the identifiable parameterisation")


def device_banner(device: str) -> str:
    return "cpu (numpy)"


def _design(Z):
    return np.hstack([np.asarray(Z, np.float64), np.ones((len(Z), 1))])


def _probs(Xa, theta):
    D = Xa.shape[1]
    W = np.asarray(theta, float).reshape(FREE, D)
    Z = np.zeros((len(Xa), N_CLASSES))
    Z[:, :FREE] = Xa @ W.T
    Z -= Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def predict_proba(Z, theta):
    return _probs(_design(Z), theta)


def laplace_precision(theta, Xa, alpha):
    P = _probs(Xa, theta)
    D = Xa.shape[1]
    H = np.zeros((FREE * D, FREE * D))
    for c in range(FREE):
        for k in range(c, FREE):
            w = P[:, c] * ((1.0 if c == k else 0.0) - P[:, k])
            blk = Xa.T @ (Xa * w[:, None])
            H[c * D:(c + 1) * D, k * D:(k + 1) * D] = blk
            if k != c:
                H[k * D:(k + 1) * D, c * D:(c + 1) * D] = blk.T
    H[np.diag_indices_from(H)] += alpha
    return H


def _map_obj(theta, Xa, Y, alpha):
    D = Xa.shape[1]
    P = _probs(Xa, theta)
    nll = -float(np.log(np.clip(P[Y.astype(bool)], EPS, None)).sum())
    G = (P[:, :FREE] - Y[:, :FREE]).T @ Xa + alpha * np.asarray(theta).reshape(FREE, D)
    return nll + 0.5 * alpha * float(np.dot(theta, theta)), G.ravel()


class _Result:

    __slots__ = ("nit", "success", "grad_inf", "fun")

    def __init__(self, nit, success, grad_inf, fun):
        self.nit, self.success, self.grad_inf, self.fun = nit, success, grad_inf, fun


def fit_map(Z, y, alpha, theta0=None, maxiter: int = 100, tol: float = 1e-9):
    Xa = _design(Z)
    Y = np.zeros((len(y), N_CLASSES))
    Y[np.arange(len(y)), y] = 1.0
    theta = (np.zeros(FREE * Xa.shape[1]) if theta0 is None
             else np.asarray(theta0, float).ravel().copy())

    f, g = _map_obj(theta, Xa, Y, alpha)
    for it in range(maxiter):
        gmax = float(np.abs(g).max())
        if gmax < tol * max(1.0, abs(f)):
            return theta, _Result(it, True, gmax, f)
        H = laplace_precision(theta, Xa, alpha)
        try:
            step = -linalg.cho_solve(linalg.cho_factor(H, lower=True), g)
        except linalg.LinAlgError:
            step = -linalg.lstsq(H, g)[0]


        s, ok = 1.0, False
        for _ in range(30):
            cand = theta + s * step
            fc, gc = _map_obj(cand, Xa, Y, alpha)
            if fc <= f - 1e-4 * s * float(np.dot(g, -step)):
                theta, f, g, ok = cand, fc, gc, True
                break
            s *= 0.5
        if not ok:
            return theta, _Result(it, True, gmax, f)
    return theta, _Result(maxiter, False, float(np.abs(g).max()), f)


def chol_pd(F, jitter0: float = 1e-10):
    scale = float(np.mean(np.diag(F))) or 1.0
    jitter = 0.0
    for _ in range(12):
        try:
            return linalg.cholesky(F + jitter * scale * np.eye(len(F)), lower=True)
        except linalg.LinAlgError:
            jitter = jitter0 if jitter == 0.0 else jitter * 10
    raise linalg.LinAlgError("precision is not positive definite even with jitter")


def sample_theta(theta, chol, n, rng):
    Zr = rng.standard_normal((n, len(theta)))
    return theta + linalg.solve_triangular(chol.T, Zr.T, lower=False).T


def log_evidence(theta, Xa, Y, alpha):
    obj, _ = _map_obj(theta, Xa, Y, alpha)
    d = len(theta)
    L = chol_pd(laplace_precision(theta, Xa, alpha))
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -obj + 0.5 * d * np.log(alpha) - 0.5 * log_det


def select_alpha_evidence(Z, y, grid, warm=True):
    Xa = _design(Z)
    Y = np.zeros((len(y), N_CLASSES))
    Y[np.arange(len(y)), y] = 1.0
    best, best_ev, theta_prev = None, -np.inf, None
    for a in grid:
        theta, _ = fit_map(Z, y, a, theta0=theta_prev if warm else None)
        ev = log_evidence(theta, Xa, Y, a)
        if ev > best_ev:
            best, best_ev = a, ev
        theta_prev = theta
    return float(best), float(best_ev)
