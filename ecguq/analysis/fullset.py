"""Representation sensitivity: the ladder refitted on all 18 candidate features."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from ..config import ALL_FEATURES, EPS, MECHANISMS, MODELS, N_BOOT, N_CLASSES, N_JOBS
from ..metrics import (augrc_w, auroc_w, brier_w, ci_from, ece_w, macro_f1_w, nll_w,
                      subject_weights)

GROUPS = ("E1", "E2", "E3", "E4", "M1", "M2", "M3")
K_LOCAL = 50


def fit_group_structure_averaged(Ztr, ytr, Zte, alpha, n_draws, seed, n_jobs=1, **_):
    from ..models.structure import _fit_one, log_structure_prior, predict_structure_averaged
    col = {c: i for i, c in enumerate(ALL_FEATURES)}
    units = [tuple(col[c] for c in MECHANISMS[g]) for g in GROUPS]
    subsets = [s for k in range(len(units) + 1) for s in combinations(range(len(units)), k)]
    structures = [tuple(sorted(i for u in s for i in units[u])) for s in subsets]
    fits = [_fit_one(cols, Ztr, ytr, alpha) for cols in structures]
    log_ev = np.array([f[2] for f in fits])
    log_post = log_ev + log_structure_prior(np.array([len(s) for s in subsets]), len(units))
    w = np.exp(log_post - log_post.max())
    w /= w.sum()
    members, n_sampled = predict_structure_averaged(structures, fits, w, Zte, n_draws, seed)
    incl = np.array([w[[j for j, s in enumerate(subsets) if u in s]].sum()
                     for u in range(len(units))])
    order = np.argsort(-w)
    info = {"n_structures": len(structures), "top_structure": "+".join(GROUPS[u] for u in
                                                                       subsets[order[0]]),
            "top_weight": float(w[order[0]]), "effective_structures": float(1 / np.sum(w ** 2)),
            "expected_size": float(incl.sum()), "group_inclusion": incl,
            "n_structures_sampled": n_sampled}
    return members, info


def _fold(X, y, subj, k, held, n_draws, cart_iter, max_depth, min_leaf):
    from ..models import ladder
    ladder.fit_structure_averaged = fit_group_structure_averaged
    te, tr = np.flatnonzero(subj == held), np.flatnonzero(subj != held)
    (Ztr, Zte), _ = ladder.fold_transform(X[tr], X[te])
    res = ladder.fit_fold(Ztr, y[tr], subj[tr], Zte, seed=k, n_draws=n_draws,
                          cart_iter=cart_iter, max_depth=max_depth, min_leaf=min_leaf)
    nn = NearestNeighbors(n_neighbors=K_LOCAL).fit(Ztr)
    lab = y[tr][nn.kneighbors(Zte, return_distance=False)]
    q = np.stack([(lab == c).mean(1) for c in range(N_CLASSES)], axis=1)
    h_local = -(q * np.log(q + EPS)).sum(1)
    keep = {m: {key: res[m][key] for key in ("p", "U_pred", "U_CE", "U_MI", "softmax")}
            for m in MODELS}
    info = res["A2"]["structure"]
    return te, keep, h_local, {"subject": held, "a2_top": info["top_structure"],
                               "a2_top_weight": info["top_weight"],
                               "a2_expected_groups": info["expected_size"]}


def run_loso(features_csv: Path, out_pred: Path, n_draws=100, cart_iter=2000, max_depth=12, min_leaf=10,
             n_jobs=N_JOBS):
    df = pd.read_csv(features_csv, dtype={"record": str, "subject": str})
    X, y, subj = df[ALL_FEATURES].to_numpy(float), df.label.to_numpy(int), df.subject.to_numpy(str)
    subjects = sorted(np.unique(subj))
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fold)(X, y, subj, k, s, n_draws, cart_iter, max_depth, min_leaf)
        for k, s in enumerate(subjects))
    n = len(df)
    store = {m: {key: np.zeros((n, N_CLASSES) if key == "p" else n)
                 for key in ("p", "U_pred", "U_CE", "U_MI", "softmax")} for m in MODELS}
    h_local = np.full(n, np.nan)
    diag = []
    for te, keep, h, d in out:
        h_local[te] = h
        diag.append(d)
        for m in MODELS:
            for key in store[m]:
                store[m][key][te] = keep[m][key]
    rows = []
    for m in MODELS:
        d = store[m]
        rows.append(pd.DataFrame({
            "config": m + "_full", "model": m, "record": df.record, "subject": df.subject,
            "start_sample": df.start_sample, "y": y, "pred": d["p"].argmax(1),
            "p_Q1": d["p"][:, 0], "p_Q2": d["p"][:, 1], "p_Q3": d["p"][:, 2],
            "H_expert": df.H_expert, "H_local_k50": h_local,
            "U_pred": d["U_pred"], "U_CE": d["U_CE"], "U_MI": d["U_MI"], "softmax": d["softmax"]}))
    preds = pd.concat(rows, ignore_index=True)
    preds.to_csv(out_pred, index=False)
    pd.DataFrame(diag).to_csv(out_pred.parent / "fullset_a2_group_structure.csv", index=False)
    return preds


def _paired(fn, subj, n_boot):
    return ci_from(fn(np.ones(len(subj))), [fn(w) for w in subject_weights(subj, n_boot)])


def compare(full_pred: Path, star_pred: Path, local_csv: Path, n_boot: int = N_BOOT) -> pd.DataFrame:
    full = pd.read_csv(full_pred, dtype={"record": str, "subject": str})
    star = pd.read_csv(star_pred, dtype={"record": str, "subject": str})
    star = star[star.arm == "ECG_ACC"]
    loc = pd.read_csv(local_csv, dtype={"record": str, "subject": str})
    star = star.merge(loc[["record", "start_sample", "H_local_k50"]],
                      on=["record", "start_sample"], how="left")
    key = ["record", "start_sample"]
    rows = []
    for m in MODELS:
        a = star[star.model == m].set_index(key)
        b = full[full.model == m].set_index(key).loc[a.index]
        subj = a.subject.to_numpy(str)
        y = a.y.to_numpy(int)
        pa, pb = (d[["p_Q1", "p_Q2", "p_Q3"]].to_numpy(float) for d in (a, b))
        ya, yb = a.pred.to_numpy(int), b.pred.to_numpy(int)
        q3 = (y == 2).astype(float)
        metrics = {"macro-F1": (lambda w, yh: macro_f1_w(y, yh, w), ya, yb),
                   "NLL": (lambda w, p: nll_w(y, p, w), pa, pb),
                   "Brier": (lambda w, p: brier_w(y, p, w), pa, pb),
                   "ECE": (lambda w, p: ece_w(y, p, w), pa, pb),

                   "Q3 recall": (lambda w, yh: (w * q3 * (yh == 2)).sum() / (w * q3).sum(),
                                 ya, yb)}
        no105 = subj != "105"
        rows.append({"block": "note", "model": m, "endpoint": "macro-F1 excl. 105",
                     "sstar": macro_f1_w(y[no105], ya[no105]),
                     "full": macro_f1_w(y[no105], yb[no105]),
                     "diff": np.nan, "lo": np.nan, "hi": np.nan})
        for name, (f, xa, xb) in metrics.items():
            d = _paired(lambda w: f(w, xb) - f(w, xa), subj, n_boot)
            rows.append({"block": "A", "model": m, "endpoint": name,
                         "sstar": f(np.ones(len(y)), xa), "full": f(np.ones(len(y)), xb),
                         "diff": d[0], "lo": d[1], "hi": d[2]})
        if m not in ("A2", "B2"):
            continue

        def deltas(d):
            err = (d.pred.to_numpy(int) != y).astype(float)
            ub = d.softmax.to_numpy(float)
            ob = np.argsort(ub, kind="stable")
            out = {}
            for s in ("U_pred", "U_CE", "U_MI"):
                u = d[s].to_numpy(float)
                o = np.argsort(u, kind="stable")
                out[f"dAUROC_err({s})"] = (lambda w, o=o, u=u: auroc_w(err[o], w[o], u[o])
                                           - auroc_w(err[ob], w[ob], ub[ob]))
                if s == "U_CE":
                    out["dAUGRC_gain(U_CE)"] = (lambda w, o=o: augrc_w(err[ob], w[ob])
                                                - augrc_w(err[o], w[o]))
            return out

        da, db = deltas(a), deltas(b)
        for name in da:
            d = _paired(lambda w: db[name](w) - da[name](w), subj, n_boot)
            ones = np.ones(len(y))
            rows.append({"block": "B", "model": m, "endpoint": name, "sstar": da[name](ones),
                         "full": db[name](ones), "diff": d[0], "lo": d[1], "hi": d[2]})
        ha, hb = a.H_local_k50.to_numpy(float), b.H_local_k50.to_numpy(float)
        ua, ub_ = a.U_CE.to_numpy(float), b.U_CE.to_numpy(float)
        rho = lambda u, h: stats.spearmanr(u, h).statistic
        subs = np.unique(subj)
        idx_of = {s: np.flatnonzero(subj == s) for s in subs}
        rng = np.random.default_rng(20240917)
        boots = []
        for _ in range(min(n_boot, 500)):
            idx = np.concatenate([idx_of[s] for s in rng.choice(subs, len(subs))])
            boots.append(rho(ub_[idx], hb[idx]) - rho(ua[idx], ha[idx]))
        ra, rb = rho(ua, ha), rho(ub_, hb)
        d = ci_from(rb - ra, boots)
        rows.append({"block": "B", "model": m, "endpoint": "rho(U_CE, H_local)", "sstar": ra,
                     "full": rb, "diff": d[0], "lo": d[1], "hi": d[2]})
    t = pd.DataFrame(rows)
    return t
