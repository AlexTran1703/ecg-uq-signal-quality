"""Subject-imbalance sensitivity: per-fold results, equal-subject weighting, balanced refit, subject exclusion."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from ..config import ARMS, EVAL_CFG, MECHANISMS, N_BOOT, N_CLASSES, N_JOBS
from ..metrics import (augrc_w, auroc_w, balanced_accuracy_w, brier_w, ci_from, macro_f1_w, nll_w,
                      recall_w, subject_weights)

ARM = "ECG_ACC"
MODELS = ("A1", "A2", "B1", "B2")
ALL_RUNGS = ("A0", "A1", "A2", "B0", "B1", "B2")
MIN_RHO_N = int(EVAL_CFG["equal_subject"]["min_windows_for_correlation"])


def load_predictions(path: Path, local_csv: Path) -> pd.DataFrame:
    p = pd.read_csv(path, dtype={"record": str, "subject": str})
    p = p[p.arm == ARM].copy()
    loc = pd.read_csv(local_csv, dtype={"record": str, "subject": str})
    return p.merge(loc[["record", "start_sample", "H_local_k50"]],
                   on=["record", "start_sample"], how="left")


def _equal_subject_w(subj: np.ndarray) -> np.ndarray:
    _, code, cnt = np.unique(subj, return_inverse=True, return_counts=True)
    return 1.0 / cnt[code]


def _endpoints(g: pd.DataFrame):
    y, yhat = g.y.to_numpy(int), g.pred.to_numpy(int)
    p = g[["p_Q1", "p_Q2", "p_Q3"]].to_numpy(float)
    err = (yhat != y).astype(float)
    ub = g["softmax"].to_numpy(float)
    ob = np.argsort(ub, kind="stable")

    def delta(score):
        u = g[score].to_numpy(float)
        o = np.argsort(u, kind="stable")
        da = lambda w: auroc_w(err[o], w[o], u[o]) - auroc_w(err[ob], w[ob], ub[ob])
        dg = lambda w: augrc_w(err[ob], w[ob]) - augrc_w(err[o], w[o])
        return da, dg

    out = {"macro-F1": lambda w: macro_f1_w(y, yhat, w),
           "NLL": lambda w: nll_w(y, p, w)}
    for s in ("U_CE", "U_MI"):
        da, dg = delta(s)
        out[f"dAUROC_err({s})"] = da
        out[f"dAUGRC_gain({s})"] = dg
    return out


def _rho_pooled(u, h):
    return float(stats.spearmanr(u, h).statistic)


def _rho_by_subject(u, h, subj):
    zs, keep = {}, []
    for s in np.unique(subj):
        m = (subj == s) & np.isfinite(h)
        if m.sum() < MIN_RHO_N or np.ptp(u[m]) == 0 or np.ptp(h[m]) == 0:
            continue
        r = stats.spearmanr(u[m], h[m]).statistic
        if np.isfinite(r):
            zs[s] = np.arctanh(np.clip(r, -0.999, 0.999))
    return zs


def _fisher_mean(zs: dict, n_boot: int, seed: int = 20261006):
    z = np.array(list(zs.values()))
    if not len(z):
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    boots = [np.tanh(z[rng.integers(0, len(z), len(z))].mean()) for _ in range(n_boot)]
    pt, lo, hi = ci_from(np.tanh(z.mean()), boots)
    return pt, lo, hi, len(z)


def per_fold_table(p: pd.DataFrame) -> pd.DataFrame:
    base = p[p.config == "A2+"]
    total = np.bincount(base.y, minlength=N_CLASSES)
    rows = []
    for s, g in base.groupby("subject", sort=False):
        n = np.bincount(g.y, minlength=N_CLASSES)
        tr = total - n
        r = {"subject": s, "n": len(g), "Q1": int(n[0]), "Q2": int(n[1]), "Q3": int(n[2]),
             "Q3_in_training": int(tr[2])}
        for c in range(N_CLASSES):
            r[f"train_Q{c + 1}_pct"] = 100 * tr[c] / tr.sum()
        for m in MODELS:
            gm = p[(p.config == m + "+") & (p.subject == s)]
            y, yhat = gm.y.to_numpy(int), gm.pred.to_numpy(int)
            pr = gm[["p_Q1", "p_Q2", "p_Q3"]].to_numpy(float)
            rec = recall_w(y, yhat)
            for c in range(N_CLASSES):
                r[f"{m}_Q{c + 1}_recall"] = rec[c] if n[c] > 0 else np.nan
            r[f"{m}_balanced_acc"] = balanced_accuracy_w(y, yhat)
            r[f"{m}_NLL"] = nll_w(y, pr)
        rows.append(r)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def weighting_table(p: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    for m in ALL_RUNGS:
        g = p[p.config == m + "+"].reset_index(drop=True)
        subj = g.subject.to_numpy(str)
        w_eq = _equal_subject_w(subj)
        for name, fn in _endpoints(g).items():
            if m not in MODELS and name not in ("macro-F1", "NLL"):
                continue
            pooled = ci_from(fn(np.ones(len(g))),
                             [fn(w) for w in subject_weights(subj, n_boot)])
            equal = ci_from(fn(w_eq),
                            [fn(w * w_eq) for w in subject_weights(subj, n_boot)])
            rows.append({"model": m, "endpoint": name, "rq": "RQ1" if name in ("macro-F1", "NLL")
                         else "RQ2", "pooled": pooled[0], "pooled_lo": pooled[1],
                         "pooled_hi": pooled[2], "equal": equal[0], "equal_lo": equal[1],
                         "equal_hi": equal[2], "n_subjects": len(np.unique(subj))})
        if m not in MODELS:
            continue
        for probe, hcol in (("H_local", "H_local_k50"), ("H_expert", "H_expert")):
            h = g[hcol].to_numpy(float)
            for s in ("U_CE", "U_MI"):
                u = g[s].to_numpy(float)
                ok = np.isfinite(h)
                subs = np.unique(subj[ok])
                idx_of = {k: np.flatnonzero((subj == k) & ok) for k in subs}
                rng = np.random.default_rng(20240917)
                boots = []
                for _ in range(min(n_boot, 500)):
                    idx = np.concatenate([idx_of[k] for k in rng.choice(subs, len(subs))])
                    boots.append(_rho_pooled(u[idx], h[idx]))
                pooled = ci_from(_rho_pooled(u[ok], h[ok]), boots)
                eq, lo, hi, k = _fisher_mean(_rho_by_subject(u, h, subj), min(n_boot, 500))
                rows.append({"model": m, "endpoint": f"rho({s}, {probe})", "rq": "RQ3",
                             "pooled": pooled[0], "pooled_lo": pooled[1], "pooled_hi": pooled[2],
                             "equal": eq, "equal_lo": lo, "equal_hi": hi, "n_subjects": k})
    return pd.DataFrame(rows)


def ms_table(p: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    for m in ALL_RUNGS:
        g = p[p.config == m + "+"].reset_index(drop=True)
        subj = g.subject.to_numpy(str)
        w_eq = _equal_subject_w(subj)
        err = (g.pred.to_numpy(int) != g.y.to_numpy(int)).astype(float)
        u = g["softmax"].to_numpy(float)
        o = np.argsort(u, kind="stable")
        fns = {"AUROC_err(U_MS)": lambda w: auroc_w(err[o], w[o], u[o]),
               "AUGRC(U_MS)": lambda w: augrc_w(err[o], w[o])}
        for name, fn in fns.items():
            pooled = ci_from(fn(np.ones(len(g))), [fn(w) for w in subject_weights(subj, n_boot)])
            equal = ci_from(fn(w_eq), [fn(w * w_eq) for w in subject_weights(subj, n_boot)])
            rows.append({"model": m, "endpoint": name, "rq": "RQ2", "pooled": pooled[0],
                         "pooled_lo": pooled[1], "pooled_hi": pooled[2], "equal": equal[0],
                         "equal_lo": equal[1], "equal_hi": equal[2],
                         "n_subjects": len(np.unique(subj))})
    return pd.DataFrame(rows)


def exclusion_table(p: pd.DataFrame, subject: str = "105", n_boot: int = N_BOOT) -> pd.DataFrame:
    from ..metrics import selective_risk_at

    rows = []
    for m in MODELS:
        g = p[(p.config == m + "+") & (p.subject != subject)].reset_index(drop=True)
        subj = g.subject.to_numpy(str)
        W = list(subject_weights(subj, n_boot))
        base = {"model": m, "n_subjects": len(np.unique(subj)), "n_windows": len(g)}
        for name, fn in _endpoints(g).items():
            if not name.startswith("d"):
                continue
            v = ci_from(fn(np.ones(len(g))), [fn(w) for w in W])
            rows.append({**base, "endpoint": name, "value": v[0], "lo": v[1], "hi": v[2]})
        err = (g.pred.to_numpy(int) != g.y.to_numpy(int)).astype(float)
        rnd = ci_from(err.mean(), [float((w * err).sum() / w.sum()) for w in W])
        rows.append({**base, "endpoint": "risk50(random)", "value": rnd[0], "lo": rnd[1],
                     "hi": rnd[2]})
        for s in ("softmax", "U_pred", "U_CE", "U_MI"):
            u = g[s].to_numpy(float)
            o = np.argsort(u, kind="stable")
            fn = lambda w, o=o: selective_risk_at(err[o], w[o], 0.5)
            v = ci_from(fn(np.ones(len(g))), [fn(w) for w in W])
            name = "U_MS" if s == "softmax" else s
            rows.append({**base, "endpoint": f"risk50({name})", "value": v[0], "lo": v[1],
                         "hi": v[2]})

            red = lambda w, fn=fn: float((w * err).sum() / w.sum()) - fn(w)
            v = ci_from(red(np.ones(len(g))), [red(w) for w in W])
            rows.append({**base, "endpoint": f"risk50_reduction({name})", "value": v[0],
                         "lo": v[1], "hi": v[2]})
    return pd.DataFrame(rows)


def _sample(subj_tr: np.ndarray, rng, mode: str) -> np.ndarray:
    subs, cnt = np.unique(subj_tr, return_counts=True)
    cap = int(cnt.min())
    bal = np.concatenate([rng.choice(np.flatnonzero(subj_tr == s), cap, replace=False)
                          for s in subs])
    if mode == "balanced":
        return np.sort(bal)
    return np.sort(rng.choice(len(subj_tr), len(bal), replace=False))


def _refit_fold(X, y, subj, k, held, mode, seed, n_draws, cart_iter, max_depth, min_leaf,
                cols, mech):
    from ..models.ladder import fit_fold, fold_transform
    te = np.flatnonzero(subj == held)
    tr_all = np.flatnonzero(subj != held)
    pick = tr_all[_sample(subj[tr_all], np.random.default_rng(10_000 * seed + k), mode)]
    (Ztr, Zte), _ = fold_transform(X[pick], X[te])
    res = fit_fold(Ztr, y[pick], subj[pick], Zte, seed=k, n_draws=n_draws, cart_iter=cart_iter,
                   max_depth=max_depth, min_leaf=min_leaf, columns=cols, mechanisms=mech)
    return mode, seed, te, len(pick), {m: {key: res[m][key] for key in
                                           ("p", "U_pred", "U_CE", "U_MI", "softmax")}
                                       for m in MODELS}


def refit(features_csv: Path, local_csv: Path, seeds=(0, 1, 2), n_draws: int = 100,
          cart_iter: int = 2000, max_depth: int = 12, min_leaf: int = 10,
          n_jobs: int = N_JOBS) -> pd.DataFrame:
    df = pd.read_csv(features_csv, dtype={"record": str, "subject": str})
    loc = pd.read_csv(local_csv, dtype={"record": str, "subject": str})
    df = df.merge(loc[["record", "start_sample", "H_local_k50"]],
                  on=["record", "start_sample"], how="left")
    cols = ARMS[ARM]
    mech = {g: c for g, c in MECHANISMS.items() if set(c) <= set(cols)}
    X, y, subj = df[cols].to_numpy(float), df["label"].to_numpy(int), df.subject.to_numpy(str)
    subjects = sorted(np.unique(subj))
    jobs = [(mode, s, k, held) for mode in ("balanced",) for s in seeds
            for k, held in enumerate(subjects)]
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_refit_fold)(X, y, subj, k, held, mode, s, n_draws, cart_iter, max_depth,
                             min_leaf, cols, mech) for mode, s, k, held in jobs)

    rows = []
    for mode in ("balanced",):
        for s in seeds:
            got = [o for o in out if o[0] == mode and o[1] == s]
            n_train = float(np.mean([o[3] for o in got]))
            for m in MODELS:
                store = {key: np.zeros((len(df), N_CLASSES) if key == "p" else len(df))
                         for key in ("p", "U_pred", "U_CE", "U_MI", "softmax")}
                for _, _, te, _, res in got:
                    for key in store:
                        store[key][te] = res[m][key]
                g = pd.DataFrame({"y": y, "pred": store["p"].argmax(1), "subject": subj,
                                  "p_Q1": store["p"][:, 0], "p_Q2": store["p"][:, 1],
                                  "p_Q3": store["p"][:, 2], "U_CE": store["U_CE"],
                                  "U_MI": store["U_MI"], "softmax": store["softmax"],
                                  "H_expert": df.H_expert.to_numpy(float),
                                  "H_local_k50": df.H_local_k50.to_numpy(float)})
                ones = np.ones(len(g))
                r = {"mode": mode, "seed": s, "model": m, "n_train_mean": n_train}
                for name, fn in _endpoints(g).items():
                    r[name] = fn(ones)
                for probe, hcol in (("H_local", "H_local_k50"), ("H_expert", "H_expert")):
                    h = g[hcol].to_numpy(float)
                    ok = np.isfinite(h)
                    for sc in ("U_CE", "U_MI"):
                        r[f"rho({sc}, {probe})"] = _rho_pooled(g[sc].to_numpy(float)[ok], h[ok])
                y3 = g.y.to_numpy(int) == 2
                r["Q3_recall_fold105"] = float((g.pred.to_numpy(int)[y3 & (subj == "105")] == 2)
                                               .mean())
                rows.append(r)
    return pd.DataFrame(rows)
