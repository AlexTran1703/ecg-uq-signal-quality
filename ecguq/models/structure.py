"""A2: exact spike-and-slab posterior over all feature masks."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from joblib import Parallel, delayed

from ..config import N_CLASSES
from .logistic import (FREE, _design, chol_pd, fit_map, laplace_precision, log_evidence,
                     sample_theta)


def enumerate_structures(features) -> list[tuple[int, ...]]:
    idx = tuple(range(len(features)))
    out = []
    for k in range(0, len(idx) + 1):
        out.extend(combinations(idx, k))
    return out


def log_structure_prior(sizes, d, prior="beta-binomial", rho=0.5):
    sizes = np.asarray(sizes, float)
    if prior == "bernoulli":
        return sizes * np.log(rho) + (d - sizes) * np.log1p(-rho)
    if prior == "beta-binomial":
        from scipy.special import gammaln
        return -(gammaln(d + 1) - gammaln(sizes + 1) - gammaln(d - sizes + 1))
    raise ValueError(prior)


def _fit_one(cols, Ztr, ytr, alpha):
    Zg = Ztr[:, list(cols)]
    theta, _ = fit_map(Zg, ytr, alpha)
    Xa = _design(Zg)
    Y = np.zeros((len(ytr), N_CLASSES))
    Y[np.arange(len(ytr)), ytr] = 1.0
    ev = log_evidence(theta, Xa, Y, alpha)
    chol = chol_pd(laplace_precision(theta, Xa, alpha))
    return theta, chol, ev


def fit_structures(Ztr, ytr, alpha, n_jobs: int = 1, prior: str = "beta-binomial",
                   rho: float = 0.5):
    d = Ztr.shape[1]
    structures = enumerate_structures(range(d))

    fits = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one)(cols, Ztr, ytr, alpha) for cols in structures)
    log_ev = np.array([f[2] for f in fits])

    sizes = np.array([len(s) for s in structures])
    log_post = log_ev + log_structure_prior(sizes, d, prior=prior, rho=rho)
    weights = np.exp(log_post - log_post.max())
    weights /= weights.sum()

    order = np.argsort(-weights)
    top = structures[order[0]]
    srt = np.sort(log_ev)

    incl = np.array([weights[[j for j, s in enumerate(structures) if i in s]].sum()
                     for i in range(d)])
    info = {
        "n_structures": len(structures),
        "weights": weights,
        "log_evidence": log_ev,
        "top_structure": ("+".join(str(i) for i in top) if top else "intercept-only"),
        "top_size": len(top),
        "top_weight": float(weights[order[0]]),
        "second_weight": float(weights[order[1]]) if len(order) > 1 else 0.0,
        "effective_structures": float(1.0 / np.sum(weights ** 2)),
        "logev_gap": float(srt[-1] - srt[-2]) if len(srt) > 1 else float("inf"),
        "prior": prior,
        "inclusion": incl,
        "expected_size": float(incl.sum()),
    }
    return structures, fits, weights, info


def predict_structure_averaged(structures, fits, weights, Zte, n_draws: int, seed: int):
    rng = np.random.default_rng(7000 + seed)
    picks = rng.choice(len(structures), size=n_draws, p=weights)
    members = np.empty((n_draws, len(Zte), N_CLASSES), np.float32)
    for r, k in enumerate(picks):
        theta, chol, _ = fits[k]
        cols = list(structures[k])
        draw = sample_theta(theta, chol, 1, rng)[0]
        Xa = _design(Zte[:, cols])
        Zl = np.zeros((len(Zte), N_CLASSES))
        Zl[:, :FREE] = Xa @ draw.reshape(FREE, -1).T
        Zl -= Zl.max(1, keepdims=True)
        E = np.exp(Zl)
        members[r] = E / E.sum(1, keepdims=True)
    return members, int(len(np.unique(picks)))


def fit_structure_averaged(Ztr, ytr, Zte, alpha, n_draws: int, seed: int,
                           n_jobs: int = 1, prior: str = "beta-binomial",
                           rho: float = 0.5):
    structures, fits, weights, info = fit_structures(Ztr, ytr, alpha, n_jobs, prior, rho)
    members, n_sampled = predict_structure_averaged(structures, fits, weights, Zte, n_draws, seed)
    info["n_structures_sampled"] = n_sampled
    return members, info
