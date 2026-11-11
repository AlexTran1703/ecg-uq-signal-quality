"""Fold preprocessing, the six model rungs (A0-A2, B0-B2) and the leave-one-subject-out loop."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from ..config import (ALPHA_GRID, ALPHA_MLE, ARMS, EPS, MECHANISMS, MODELS,
                     N_CLASSES, N_JOBS, S_STAR, UQ_MODELS)
from ..uncertainty import decompose, point_scores, softmax_uncertainty
from .logistic import (_design, chol_pd, fit_map, laplace_precision,
                            predict_proba, sample_theta, select_alpha_evidence)
from .structure import fit_structure_averaged
from .trees import BayesCART, dirichlet_leaf_members


MODEL_UQ = {m: m in UQ_MODELS for m in MODELS}
ARM_SUFFIX = {"ECG": "", "ECG_ACC": "+"}


CONTROL_OF = {"A1": "A0", "A2": "A1", "B1": "B0", "B2": "B1"}
SCORES = ("U_pred", "U_CE", "U_MI", "softmax")


def config_id(model, arm):
    return model + ARM_SUFFIX[arm]


def fold_transform(Xtr_raw, *others):
    Xtr_raw = np.where(np.isfinite(Xtr_raw), Xtr_raw, np.nan)
    med = np.nanmedian(Xtr_raw, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr_raw), Xtr_raw, med)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9

    def apply(X):
        X = np.where(np.isfinite(X), X, med)
        return (X - mu) / sd

    return ((Xtr - mu) / sd, *(apply(o) for o in others)), (med, mu, sd)


def prepare_fold(train: pd.DataFrame, test: pd.DataFrame, cols):
    (Ztr, Zte), _ = fold_transform(train[cols].to_numpy(float), test[cols].to_numpy(float))
    return Ztr, Zte


def _members_from(Zte, theta):
    return predict_proba(Zte, theta)


def fit_fold(Ztr, ytr, gtr, Zte, seed: int, n_draws: int,
             cart_iter: int, max_depth: int, min_leaf: int, alpha=None, verbose=False,
             device=None, columns=None, mechanisms=None):
    out = {}
    if alpha is None:
        alpha, _ = select_alpha_evidence(Ztr, ytr, ALPHA_GRID)

    Xa_tr = _design(Ztr)


    theta0, r0 = fit_map(Ztr, ytr, ALPHA_MLE)
    out["A0"] = {**point_scores(predict_proba(Zte, theta0)),
                 "alpha": ALPHA_MLE, "mle_iters": int(r0.nit)}


    theta1, _ = fit_map(Ztr, ytr, alpha)
    chol = chol_pd(laplace_precision(theta1, Xa_tr, alpha))
    rng = np.random.default_rng(1000 + seed)
    draws = sample_theta(theta1, chol, n_draws, rng)
    members = np.stack([_members_from(Zte, th) for th in draws]).astype(np.float32)
    out["A1"] = {**decompose(members), "alpha": alpha}


    a2_members, a2_info = fit_structure_averaged(Ztr, ytr, Zte, alpha, n_draws, seed)
    out["A2"] = {**decompose(a2_members), "alpha": alpha, "structure": a2_info}


    t0 = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf,
                                random_state=seed).fit(Ztr, ytr)
    p0 = np.full((len(Zte), N_CLASSES), EPS)
    p0[:, t0.classes_.astype(int)] = t0.predict_proba(Zte)
    out["B0"] = point_scores(p0 / p0.sum(1, keepdims=True))
    out["B1"] = decompose(dirichlet_leaf_members(t0, Ztr, ytr, Zte, n_draws, seed=2000 + seed))


    cart = BayesCART(max_depth=max_depth, min_leaf=min_leaf, seed=seed)
    cart.fit(Ztr, ytr, n_iter=cart_iter, burn=cart_iter // 4, thin=max(1, cart_iter // 40),
             verbose=verbose)
    out["B2"] = {**decompose(cart.predict_members(Zte, n_draws, seed=3000 + seed)),
                 "accept": cart.accept_rate_,
                 "chains": getattr(cart, "chain_summary_", None)}

    for m in out:
        out[m]["softmax"] = softmax_uncertainty(out[m]["p"])
    return out


def _loso_job(X, y, subj, te_mask, k, held, arm, n_draws, cart_iter, max_depth, min_leaf,
              device="cpu", columns=None, mechanisms=None):
    t0 = time.time()
    tr, te = np.flatnonzero(~te_mask), np.flatnonzero(te_mask)
    (Ztr, Zte), _ = fold_transform(X[tr], X[te])
    res = fit_fold(Ztr, y[tr], subj[tr], Zte, seed=k, n_draws=n_draws,
                   cart_iter=cart_iter, max_depth=max_depth, min_leaf=min_leaf, device=device,
                   columns=columns, mechanisms=mechanisms)
    keep = {m: {key: res[m][key] for key in
                ("p", "U_pred", "U_CE", "U_MI", "softmax")} for m in MODELS}
    struct = res["A2"].get("structure") or {}
    return (k, held, arm, te, keep,
            {"arm": arm, "subject": held, "n_test": len(te), "alpha": res["A1"]["alpha"],
             "cart_accept": res["B2"]["accept"],
             "a2_structures": struct.get("n_structures"),
             "a2_top_size": struct.get("top_size"),
             "a2_top_weight": struct.get("top_weight"),
             "a2_effective_structures": struct.get("effective_structures"),
             "a2_expected_size": struct.get("expected_size"),
             "a2_logev_gap": struct.get("logev_gap"),
             "b2_chains": len(res["B2"].get("chains") or []),
             "b2_leaves_mean": (float(np.mean([c["leaves_mean"] for c in res["B2"]["chains"]]))
                                if res["B2"].get("chains") else None),
             "b2_leaves_spread": (float(np.ptp([c["leaves_mean"] for c in res["B2"]["chains"]]))
                                  if res["B2"].get("chains") else None),
             "b2_depth_mean": (float(np.mean([c["depth_mean"] for c in res["B2"]["chains"]]))
                               if res["B2"].get("chains") else None),
             "b2_distinct": (int(sum(c["n_distinct"] for c in res["B2"]["chains"]))
                             if res["B2"].get("chains") else None),
             "seconds": time.time() - t0})


def run_loso(features_csv: Path, outdir: Path, n_draws: int = 100,
             cart_iter: int = 2000, max_depth: int = 6, min_leaf: int = 20,
             folds: int | None = None, n_jobs: int = N_JOBS, device: str = "cpu") -> pd.DataFrame:
    df = pd.read_csv(features_csv, dtype={"record": str, "subject": str})
    subjects = sorted(df.subject.unique())
    if folds:
        subjects = subjects[:folds]
        df = df[df.subject.isin(subjects)].reset_index(drop=True)
    y = df["label"].to_numpy(int)
    subj = df.subject.to_numpy(str)
    n = len(df)
    if n_jobs is None:
        n_jobs = 1 if device != "cpu" else N_JOBS
    print(f"  {n:,} windows, {len(subjects)} subjects, {len(S_STAR)} features, "
          f"{len(subjects) * len(ARMS)} fold-by-arm fits, "
          f"{n_jobs if n_jobs > 0 else os.cpu_count()} worker(s)", flush=True)

    store = {config_id(m, a): {k: np.zeros((n, N_CLASSES) if k == "p" else n)
                               for k in ("p", "U_pred", "U_CE", "U_MI", "softmax")}
             for m in MODELS for a in ARMS}
    fold_of = np.zeros(n, int)

    jobs = []
    for arm, cols in ARMS.items():
        X = df[cols].to_numpy(float)


        mech = {g: c for g, c in MECHANISMS.items() if set(c) <= set(cols)}
        for k, held in enumerate(subjects):
            jobs.append((X, (subj == held), k, held, arm, list(cols), mech))
    if n_jobs == 1:


        out = []
        for i, (X, mask, k, held, arm) in enumerate(jobs, 1):
            r = _loso_job(X, y, subj, mask, k, held, arm, n_draws, cart_iter,
                          max_depth, min_leaf, device)
            print(f"    [{i:2d}/{len(jobs)}] {arm:8s} fold {k:2d} held-out {held}: "
                  f"{len(r[3]):6d} windows, {r[5]['seconds']:.1f} s", flush=True)
            out.append(r)
    else:
        out = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_loso_job)(X, y, subj, mask, k, held, arm, n_draws, cart_iter,
                               max_depth, min_leaf, device, cols, mech)
            for X, mask, k, held, arm, cols, mech in jobs)

    diag = []
    for k, held, arm, te, keep, d in out:
        fold_of[te] = k
        for m in MODELS:
            cid = config_id(m, arm)
            for key in store[cid]:
                store[cid][key][te] = keep[m][key]
        diag.append(d)
        print(f"    {arm:8s} fold {k:2d} held-out {held}: {len(te):6d} windows, "
              f"lambda {d['alpha']:g}, {d['seconds']:.0f} s", flush=True)

    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid, d in store.items():
        rows.append(pd.DataFrame({
            "config": cid, "model": cid.rstrip("+"),
            "arm": "ECG_ACC" if cid.endswith("+") else "ECG",
            "record": df.record.to_numpy(), "subject": df.subject.to_numpy(),
            "start_sample": df.start_sample.to_numpy(), "fold": fold_of,
            "y": y, "pred": d["p"].argmax(1),
            "p_Q1": d["p"][:, 0], "p_Q2": d["p"][:, 1], "p_Q3": d["p"][:, 2],
            "H_expert": df.H_expert.to_numpy(),
            "U_pred": d["U_pred"], "U_CE": d["U_CE"], "U_MI": d["U_MI"],
            "softmax": d["softmax"],
        }))
    preds = pd.concat(rows, ignore_index=True)
    preds.to_csv(outdir / "predictions.csv.gz", index=False)
    pd.DataFrame(diag).to_csv(outdir / "t00_fold_diagnostics.csv", index=False)
    print(f"\n  wrote {len(preds):,} out-of-subject predictions -> {outdir/'predictions.csv.gz'}",
          flush=True)
    return preds
