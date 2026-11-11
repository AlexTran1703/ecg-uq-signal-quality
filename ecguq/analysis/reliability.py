"""Paired gains of each posterior-derived score over the same model's max-softmax uncertainty."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..config import N_BOOT, UQ_MODELS
from ..metrics import augrc_w, auroc_w, ci_from, subject_weights

BASE = "softmax"
SCORES = ("U_pred", "U_CE", "U_MI")


def _paired_ranking_deltas(err, u, ub, subj, n_boot):
    o, ob = np.argsort(u, kind="stable"), np.argsort(ub, kind="stable")
    e, eb = err[o], err[ob]
    us, ubs = u[o], ub[ob]

    def d_auroc(w):
        return auroc_w(e, w[o], us) - auroc_w(eb, w[ob], ubs)

    def d_augrc(w):
        return augrc_w(eb, w[ob]) - augrc_w(e, w[o])

    ones = np.ones(len(err))
    va, vg = [], []
    for w in subject_weights(subj, n_boot):
        a, g = d_auroc(w), d_augrc(w)
        if np.isfinite(a):
            va.append(a)
        if np.isfinite(g):
            vg.append(g)
    return ci_from(d_auroc(ones), va), ci_from(d_augrc(ones), vg)


def _paired_rho_delta(h, u, ub, subj, n_boot):
    subj = np.asarray(subj).astype(str)
    subs = np.unique(subj)
    idx_of = {s: np.flatnonzero(subj == s) for s in subs}
    rng = np.random.default_rng(20240917)

    def rho(a, b):
        return stats.spearmanr(a, b).statistic

    point = rho(u, h) - rho(ub, h)
    vals = []
    for _ in range(min(n_boot, 500)):
        idx = np.concatenate([idx_of[s] for s in rng.choice(subs, size=len(subs), replace=True)])
        d = rho(u[idx], h[idx]) - rho(ub[idx], h[idx])
        if np.isfinite(d):
            vals.append(d)
    return ci_from(point, vals)


def incremental_over_ums(preds: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    for cid, g in preds.groupby("config", sort=False):
        model = g.model.iloc[0]
        if model not in UQ_MODELS:
            continue
        err = (g.pred.to_numpy(int) != g.y.to_numpy(int)).astype(float)
        subj = g.subject.to_numpy(str)
        ub = g[BASE].to_numpy(float)
        h = g.H_expert.to_numpy(float)
        ok = np.isfinite(h)

        for s in SCORES:
            u = g[s].to_numpy(float)
            if not np.isfinite(u).any() or np.ptp(u) == 0:
                continue
            (da, da_lo, da_hi), (dg, dg_lo, dg_hi) = _paired_ranking_deltas(
                err, u, ub, subj, n_boot)
            r = {"config": cid, "model": model, "arm": g.arm.iloc[0], "score": s,
                 "d_auroc": da, "d_auroc_lo": da_lo, "d_auroc_hi": da_hi,
                 "d_augrc_gain": dg, "d_augrc_gain_lo": dg_lo, "d_augrc_gain_hi": dg_hi}
            if ok.sum() >= 10:
                dr, dr_lo, dr_hi = _paired_rho_delta(h[ok], u[ok], ub[ok], subj[ok], n_boot)
                r.update(d_rho=dr, d_rho_lo=dr_lo, d_rho_hi=dr_hi)
            rows.append(r)
    return pd.DataFrame(rows)


def augrc_vs_random(preds: pd.DataFrame, exclude: str, n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    for cohort, keep in (("all subjects", lambda d: d), (f"excluding {exclude}", lambda d: d[d.subject != exclude])):
        for m in UQ_MODELS:
            g = keep(preds[preds.config == m + "+"]).reset_index(drop=True)
            err = (g.pred != g.y).to_numpy(float)
            W = list(subject_weights(g.subject.to_numpy(str), n_boot))
            out = {}
            for col, name in (("softmax", "U_MS"), ("U_CE", "U_CE")):
                o = np.argsort(g[col].to_numpy(float), kind="stable")
                f = lambda w, o=o: augrc_w(err[o], w[o])
                out[name] = ci_from(f(np.ones(len(g))), [f(w) for w in W])
            rnd = lambda w: float((w * err).sum() / w.sum()) / 2
            out["random"] = ci_from(rnd(np.ones(len(g))), [rnd(w) for w in W])
            o = np.argsort(g["softmax"].to_numpy(float), kind="stable")
            red = lambda w: rnd(w) - augrc_w(err[o], w[o])
            out["random_minus_U_MS"] = ci_from(red(np.ones(len(g))), [red(w) for w in W])
            rows += [{"cohort": cohort, "model": m, "endpoint": f"AUGRC({k})" if k != "random_minus_U_MS"
                      else "AUGRC reduction vs random", "value": v[0], "lo": v[1], "hi": v[2]}
                     for k, v in out.items()]
    return pd.DataFrame(rows)
