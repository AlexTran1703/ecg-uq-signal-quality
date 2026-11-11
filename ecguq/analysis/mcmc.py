"""B2 MCMC convergence per LOSO fold.

Each fold reruns the Bayesian CART sampler exactly as in the main fit (8 chains, same seeds).
Tree level: rank-normalised split R-hat and bulk ESS (Vehtari et al., 2021) for the tree's log
marginal likelihood, number of leaves and depth.
Prediction level: R-hat of the predicted class probabilities of every held-out window, agreement
of each chain with the pooled prediction, and an exact split of B2 mutual information into
between-chain, between-tree (within chain) and within-tree (Dirichlet leaf) parts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import special, stats
from sklearn.metrics import f1_score

from ..config import CART_ITER, MAX_DEPTH, MIN_LEAF, N_CHAINS, N_JOBS, S_STAR
from ..models.ladder import fold_transform
from ..models.trees import BayesCART


def _log_marginal(tree, a0):
    leaves = tree["feat"] < 0
    c = tree["counts"][leaves]
    a = np.full(c.shape[1], a0)
    return float(np.sum(special.gammaln(a.sum()) - special.gammaln(a.sum() + c.sum(1))
                        + np.sum(special.gammaln(a + c) - special.gammaln(a), axis=1)))


def _split(draws):
    n = draws.shape[-1] // 2
    return np.concatenate([draws[..., :n], draws[..., n:2 * n]], axis=-2)


def _rank_normal(draws):
    flat = draws.reshape(*draws.shape[:-2], -1)
    r = stats.rankdata(flat, axis=-1, method="average")
    return stats.norm.ppf((r - 3 / 8) / (flat.shape[-1] + 1 / 4)).reshape(draws.shape)


def rhat(draws):
    """draws: (..., chains, iterations); returns R-hat over the last two axes."""
    x = _rank_normal(_split(np.asarray(draws, float)))
    n = x.shape[-1]
    w = x.var(axis=-1, ddof=1).mean(-1)
    b = n * x.mean(axis=-1).var(axis=-1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.sqrt(((n - 1) / n * w + b / n) / w)
    const = np.ptp(np.asarray(draws).reshape(*np.shape(draws)[:-2], -1), axis=-1) == 0
    return np.where(const, np.nan, r)


def ess_bulk(draws):
    x = _rank_normal(_split(np.asarray(draws, float)))
    m, n = x.shape
    if np.allclose(x.var(), 0):
        return np.nan
    xc = x - x.mean(axis=1, keepdims=True)
    acov = np.array([np.correlate(c, c, mode="full")[n - 1:] / n for c in xc]).mean(axis=0)
    mean_var = x.var(axis=1, ddof=1).mean()
    var_plus = mean_var * (n - 1) / n + x.mean(axis=1).var(ddof=1)
    rho = np.zeros(n)
    rho[0] = even = 1.0
    rho[1] = odd = 1.0 - (mean_var - acov[1]) / var_plus
    t = 1
    while t < n - 3 and even + odd > 0:
        even = 1.0 - (mean_var - acov[t + 1]) / var_plus
        odd = 1.0 - (mean_var - acov[t + 2]) / var_plus
        if even + odd >= 0:
            rho[t + 1], rho[t + 2] = even, odd
        t += 2
    max_t = t - 2
    if even > 0:
        rho[max_t + 1] = even
    t = 1
    while t <= max_t - 2:
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = rho[t + 2] = (rho[t - 1] + rho[t]) / 2.0
        t += 2
    tau = -1.0 + 2.0 * rho[:max_t + 1].sum() + rho[max_t + 1:max_t + 2].sum()
    tau = max(tau, 1.0 / np.log10(m * n))
    return float(m * n / tau)


def _entropy(p):
    p = np.clip(p, 1e-300, 1)
    return -(p * np.log(p)).sum(-1)


def _dirichlet_entropy(a):
    s = a.sum(-1)
    return special.digamma(s + 1) - (a / s[..., None] * special.digamma(a + 1)).sum(-1)


def _predictive(cart, chains, Zte, yte):
    code = cart._code_of(Zte)
    n = min(len(c) for c in chains)
    a = np.stack([np.stack([cart._dirichlet_of(t, code) for t in c[:n]]) for c in chains])
    p = a / a.sum(-1, keepdims=True)                            # (chains, draws, windows, classes)
    rh = np.nanmax(np.stack([rhat(np.moveaxis(p[..., c], (0, 1), (1, 2)))
                             for c in range(p.shape[-1])]), axis=0)
    p_chain = p.mean(1)
    p_pool = p_chain.mean(0)
    h_pool, h_chain, h_tree = _entropy(p_pool), _entropy(p_chain).mean(0), _entropy(p).mean((0, 1))
    h_leaf = _dirichlet_entropy(a).mean((0, 1))
    labels = list(range(p.shape[-1]))
    chain_f1 = [f1_score(yte, q.argmax(1), average="macro", labels=labels, zero_division=0)
                for q in p_chain]
    agree = [(q.argmax(1) == p_pool.argmax(1)).mean() for q in p_chain]
    summary = {"windows": len(yte), "pred_rhat_median": float(np.nanmedian(rh)),
               "pred_pct_rhat_below_1.1": float(np.nanmean(rh < 1.1) * 100),
               "pooled_macroF1": f1_score(yte, p_pool.argmax(1), average="macro", labels=labels,
                                          zero_division=0),
               "chain_macroF1_min": min(chain_f1), "chain_macroF1_max": max(chain_f1),
               "chain_agreement_mean": float(np.mean(agree)),
               "chain_agreement_min": float(min(agree)),
               "true_class_prob_between_chain_sd_median":
                   float(np.median(p_chain[:, np.arange(len(yte)), yte].std(0))),
               "MI_total": float((h_pool - h_leaf).mean()),
               "MI_between_chain": float((h_pool - h_chain).mean()),
               "MI_between_tree": float((h_chain - h_tree).mean()),
               "MI_within_tree": float((h_tree - h_leaf).mean())}
    return summary, rh


def _fold(X, y, subj, k, held, n_iter):
    tr, te = subj != held, subj == held
    (Ztr, Zte), _ = fold_transform(X[tr], X[te])
    cart = BayesCART(max_depth=MAX_DEPTH, min_leaf=MIN_LEAF, seed=k, n_chains=N_CHAINS)
    cart.fit(Ztr, y[tr], n_iter=n_iter, burn=n_iter // 4, thin=max(1, n_iter // 40))
    sizes = [c["n_samples"] for c in cart.chain_summary_]
    n = min(sizes)
    bounds = np.cumsum([0] + sizes)
    chains = [cart.samples_[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    traces = {"log_marginal_likelihood": [[_log_marginal(t, cart.a0) for t in c[:n]] for c in chains],
              "n_leaves": [[t["n_leaves"] for t in c[:n]] for c in chains],
              "depth": [[int(t["depth"].max()) for t in c[:n]] for c in chains]}
    rows = []
    for name, v in traces.items():
        d = np.asarray(v, float)
        rows.append({"fold": k, "subject": held, "trace": name, "n_chains": d.shape[0],
                     "draws_per_chain": d.shape[1], "rhat": float(rhat(d)), "ess_bulk": ess_bulk(d),
                     "mean": float(d.mean()), "acceptance": cart.accept_rate_})
    pred, rh = _predictive(cart, chains, Zte, y[te])
    chain_rows = [{"fold": k, "subject": held, **c} for c in cart.chain_summary_]
    return rows, chain_rows, {"fold": k, "subject": held, **pred}, rh


def b2_mcmc_diagnostics(features_csv: Path, outdir: Path, n_iter: int = CART_ITER,
                        n_jobs: int = N_JOBS) -> pd.DataFrame:
    df = pd.read_csv(features_csv, dtype={"record": str, "subject": str})
    X, y, subj = df[S_STAR].to_numpy(float), df["label"].to_numpy(int), df.subject.to_numpy(str)
    subjects = sorted(np.unique(subj))
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fold)(X, y, subj, k, s, n_iter) for k, s in enumerate(subjects))
    t = pd.DataFrame([r for rows, _, _, _ in out for r in rows])
    t.to_csv(outdir / "b2_mcmc_rhat_ess.csv", index=False)
    pd.DataFrame([c for _, chains, _, _ in out for c in chains]).to_csv(
        outdir / "b2_mcmc_chains.csv", index=False)
    pred = pd.DataFrame([p for _, _, p, _ in out])
    pred.to_csv(outdir / "b2_predictive_convergence.csv", index=False)

    summary = t.groupby("trace").agg(rhat_median=("rhat", "median"), rhat_max=("rhat", "max"),
                                     ess_median=("ess_bulk", "median"), ess_min=("ess_bulk", "min"))
    rh = np.concatenate([r for _, _, _, r in out])
    w = pred.windows / pred.windows.sum()
    mi = {c: float((pred[c] * w).sum()) for c in
          ["MI_total", "MI_between_chain", "MI_between_tree", "MI_within_tree"]}
    summary = pd.concat([summary.reset_index(), pd.DataFrame([{
        "trace": "predicted_class_probability", "rhat_median": float(np.nanmedian(rh)),
        "rhat_max": float(np.nanmax(rh)), "pct_windows_rhat_below_1.1": float(np.nanmean(rh < 1.1) * 100),
        **mi, "MI_between_chain_share": mi["MI_between_chain"] / mi["MI_total"]}])], ignore_index=True)
    summary.to_csv(outdir / "b2_mcmc_summary.csv", index=False)
    return t
