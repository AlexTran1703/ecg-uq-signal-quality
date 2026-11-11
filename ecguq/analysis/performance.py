"""Pooled performance, reliability, rung contrasts, expert agreement and risk-coverage curves."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..config import EPS, N_BOOT, N_CLASSES, WIN_S, ARMS, MODELS
from .reliability import incremental_over_ums
from ..models.ladder import CONTROL_OF, MODEL_UQ, config_id
from ..metrics import (_rc_w, aurc_w, augrc_w, auroc_w, balanced_accuracy_w, boot_diff, boot_stat,
                      boot_index, brier_w, ece_w, macro_f1_w, nll_w, recall_w,
                      selective_risk_at)


def _scores_of(model):
    if not MODEL_UQ[model]:
        return ["U_pred", "U_CE", "softmax"]
    return ["U_pred", "U_CE", "U_MI", "softmax"]


def pooled_metrics(preds: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for cid, g in preds.groupby("config", sort=False):
        y = g.y.to_numpy(int)
        p = g[["p_Q1", "p_Q2", "p_Q3"]].to_numpy(float)
        yhat = g.pred.to_numpy(int)
        subj = g.subject.to_numpy(str)
        rec = recall_w(y, yhat)
        r = {"config": cid, "model": g.model.iloc[0], "arm": g.arm.iloc[0], "n": len(g),
             "recall_Q1": rec[0], "recall_Q2": rec[1], "recall_Q3": rec[2]}
        for name, fn in [("macro_F1", lambda w: macro_f1_w(y, yhat, w)),
                         ("balanced_acc", lambda w: balanced_accuracy_w(y, yhat, w)),
                         ("NLL", lambda w: nll_w(y, p, w)),
                         ("Brier", lambda w: brier_w(y, p, w)),
                         ("ECE", lambda w: ece_w(y, p, w))]:
            pt, lo, hi = boot_stat(fn, subj, n_boot)
            r[name], r[f"{name}_lo"], r[f"{name}_hi"] = pt, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def reliability_metrics(preds: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for cid, g in preds.groupby("config", sort=False):
        y, yhat = g.y.to_numpy(int), g.pred.to_numpy(int)
        err = (yhat != y).astype(float)
        subj = g.subject.to_numpy(str)
        oracle_order = np.argsort(err, kind="stable")
        for s in _scores_of(g.model.iloc[0]):
            u = g[s].to_numpy(float)
            if not np.isfinite(u).any() or np.ptp(u) == 0:
                continue
            order = np.argsort(u, kind="stable")
            e_s, u_s = err[order], u[order]
            e_o = err[oracle_order]
            r = {"config": cid, "model": g.model.iloc[0], "arm": g.arm.iloc[0], "score": s}
            for name, fn in [
                ("error_AUROC", lambda w: auroc_w(e_s, w[order], u_s)),
                ("AURC", lambda w: aurc_w(e_s, w[order])),
                ("AUGRC", lambda w: augrc_w(e_s, w[order])),
                ("E_AURC", lambda w: aurc_w(e_s, w[order]) - aurc_w(e_o, w[oracle_order])),
            ]:
                pt, lo, hi = boot_stat(fn, subj, n_boot)
                r[name], r[f"{name}_lo"], r[f"{name}_hi"] = pt, lo, hi
            for cov in (0.5, 0.8, 0.9):
                r[f"risk_at_{int(cov*100)}"] = selective_risk_at(e_s, np.ones(len(e_s)), cov)
            rows.append(r)
    return pd.DataFrame(rows)


def per_subject_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cid, s), g in preds.groupby(["config", "subject"], sort=False):
        y, yhat = g.y.to_numpy(int), g.pred.to_numpy(int)
        p = g[["p_Q1", "p_Q2", "p_Q3"]].to_numpy(float)
        err = (yhat != y).astype(float)
        u = g["U_pred"].to_numpy(float)
        order = np.argsort(u, kind="stable")
        present = sorted(np.unique(y).tolist())
        rows.append({
            "config": cid, "model": g.model.iloc[0], "arm": g.arm.iloc[0], "subject": s,
            "n": len(g), "classes_present": len(present),
            "macro_F1_present": macro_f1_w(y, yhat) if len(present) == N_CLASSES else np.nan,
            "balanced_acc": balanced_accuracy_w(y, yhat), "NLL": nll_w(y, p), "Brier": brier_w(y, p),
            "error_AUROC": auroc_w(err[order], np.ones(len(g)), u[order]),
            "AUGRC": augrc_w(err[order], np.ones(len(g))),
        })
    return pd.DataFrame(rows)


def per_subject_scores(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cid, subj), g in preds.groupby(["config", "subject"], sort=False):
        y, yhat = g.y.to_numpy(int), g.pred.to_numpy(int)
        err = (yhat != y).astype(float)
        h = g.H_expert.to_numpy(float)
        ok = np.isfinite(h)
        w1 = np.ones(len(g))
        for s in _scores_of(g.model.iloc[0]):
            u = g[s].to_numpy(float)
            if np.ptp(u) == 0:
                continue
            order = np.argsort(u, kind="stable")
            r = {"config": cid, "model": g.model.iloc[0], "arm": g.arm.iloc[0],
                 "subject": subj, "score": s, "n": len(g),
                 "error_AUROC": auroc_w(err[order], w1, u[order]),
                 "AUGRC": augrc_w(err[order], w1),
                 "AURC": aurc_w(err[order], w1),
                 "U_correct": float(u[err == 0].mean()) if (err == 0).any() else np.nan,
                 "U_incorrect": float(u[err == 1].mean()) if (err == 1).any() else np.nan,
                 "U_sd": float(u.std())}

            r["separation"] = ((r["U_incorrect"] - r["U_correct"]) / r["U_sd"]
                               if r["U_sd"] > 1e-9 else np.nan)
            if ok.sum() > 10 and np.ptp(h[ok]) > 0:
                r["spearman_H"] = float(stats.spearmanr(u[ok], h[ok]).statistic)
                split = h[ok] > 0
                r["U_unanimous"] = float(u[ok][~split].mean()) if (~split).any() else np.nan
                r["U_split"] = float(u[ok][split].mean()) if split.any() else np.nan
                r["n_split"] = int(split.sum())
            rows.append(r)
    return pd.DataFrame(rows)


def risk_coverage_curves(preds: pd.DataFrame, n_grid: int = 201) -> pd.DataFrame:
    grid = np.linspace(1e-3, 1.0, n_grid)
    rows = []
    for cid, g in preds.groupby("config", sort=False):
        y, yhat = g.y.to_numpy(int), g.pred.to_numpy(int)
        err = (yhat != y).astype(float)
        w = np.ones(len(g))
        base = err.mean()

        def curve(order, name):
            cov, risk, grisk = _rc_w(err[order], w)
            rows.append(pd.DataFrame({
                "config": cid, "model": g.model.iloc[0], "arm": g.arm.iloc[0], "score": name,
                "coverage": grid,
                "risk": np.interp(grid, cov, risk),
                "generalized_risk": np.interp(grid, cov, grisk)}))

        for s in _scores_of(g.model.iloc[0]):
            u = g[s].to_numpy(float)
            if np.ptp(u) == 0:
                continue
            curve(np.argsort(u, kind="stable"), s)
        curve(np.argsort(err, kind="stable"), "oracle")
        rows.append(pd.DataFrame({
            "config": cid, "model": g.model.iloc[0], "arm": g.arm.iloc[0], "score": "random",
            "coverage": grid, "risk": base, "generalized_risk": base * grid}))
    return pd.concat(rows, ignore_index=True)


def contrasts(preds: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    idx = {c: g for c, g in preds.groupby("config", sort=False)}
    rows = []

    def add(kind, a_id, b_id):
        ga, gb = idx.get(a_id), idx.get(b_id)
        if ga is None or gb is None:
            return
        ya, yb = ga.y.to_numpy(int), gb.y.to_numpy(int)
        pa = ga[["p_Q1", "p_Q2", "p_Q3"]].to_numpy(float)
        pb = gb[["p_Q1", "p_Q2", "p_Q3"]].to_numpy(float)
        yha, yhb = ga.pred.to_numpy(int), gb.pred.to_numpy(int)
        ea, eb = (yha != ya).astype(float), (yhb != yb).astype(float)
        subj = ga.subject.to_numpy(str)

        def emit(name, fa, fb, sa="-", sb="-"):
            d, lo, hi = boot_diff(fa, fb, subj, n_boot)
            rows.append({"kind": kind, "contrast": f"{b_id} - {a_id}", "metric": name,
                         "score_a": sa, "score_b": sb, "diff": d, "lo": lo, "hi": hi,
                         "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))})

        emit("macro_F1", lambda w: macro_f1_w(ya, yha, w), lambda w: macro_f1_w(yb, yhb, w))
        emit("NLL", lambda w: nll_w(ya, pa, w), lambda w: nll_w(yb, pb, w))
        emit("Brier", lambda w: brier_w(ya, pa, w), lambda w: brier_w(yb, pb, w))
        emit("ECE", lambda w: ece_w(ya, pa, w), lambda w: ece_w(yb, pb, w))


        for sb in _scores_of(gb.model.iloc[0]):
            sa = sb if sb in _scores_of(ga.model.iloc[0]) else "softmax"
            ua, ub = ga[sa].to_numpy(float), gb[sb].to_numpy(float)
            if np.ptp(ua) == 0 or np.ptp(ub) == 0:
                continue
            oa, ob = np.argsort(ua, kind="stable"), np.argsort(ub, kind="stable")
            emit("AUGRC", lambda w: augrc_w(ea[oa], w[oa]), lambda w: augrc_w(eb[ob], w[ob]), sa, sb)
            emit("error_AUROC", lambda w: auroc_w(ea[oa], w[oa], ua[oa]),
                 lambda w: auroc_w(eb[ob], w[ob], ub[ob]), sa, sb)

    for b, a in CONTROL_OF.items():
        for arm in ARMS:
            add("mechanism", config_id(a, arm), config_id(b, arm))
    for m in MODELS:
        add("accelerometer", config_id(m, "ECG"), config_id(m, "ECG_ACC"))
    return pd.DataFrame(rows)


def expert_agreement(preds: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for cid, g in preds.groupby("config", sort=False):
        h = g.H_expert.to_numpy(float)
        ok = np.isfinite(h)
        if ok.sum() < 10:
            continue
        subj = g.subject.to_numpy(str)[ok]
        split = (h[ok] > 0).astype(float)
        for s in _scores_of(g.model.iloc[0]):
            u = g[s].to_numpy(float)[ok]
            if np.ptp(u) == 0:
                continue
            rho, lo_r, hi_r = boot_index(
                lambda a, b: stats.spearmanr(a, b).statistic, subj, u, h[ok],
                n_boot=min(n_boot, 500))

            def mean_diff(w):
                a = (w * split * u).sum() / max((w * split).sum(), EPS)
                b = (w * (1 - split) * u).sum() / max((w * (1 - split)).sum(), EPS)
                return a - b

            d, lo_d, hi_d = boot_stat(mean_diff, subj, n_boot)
            rows.append({"config": cid, "model": g.model.iloc[0], "arm": g.arm.iloc[0], "score": s,
                         "n_unanimous": int((split == 0).sum()), "n_split": int(split.sum()),
                         "spearman_H": rho, "spearman_lo": lo_r, "spearman_hi": hi_r,
                         "mean_U_split_minus_unanimous": d, "diff_lo": lo_d, "diff_hi": hi_d})
    return pd.DataFrame(rows)


def dataset_table(features_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(features_csv, dtype={"record": str, "subject": str})
    g = df.groupby("subject")
    out = pd.DataFrame({
        "records": g.record.nunique(), "windows": g.size(),
        "Q1": g.label.apply(lambda s: int((s == 0).sum())),
        "Q2": g.label.apply(lambda s: int((s == 1).sum())),
        "Q3": g.label.apply(lambda s: int((s == 2).sum())),
        "H_expert_available": g.H_expert.apply(lambda s: int(s.notna().sum())),
        "unanimous": g.H_expert.apply(lambda s: int((s == 0).sum())),
    }).reset_index()
    out["hours"] = out.windows * WIN_S / 3600
    return out.sort_values("windows", ascending=False)
