"""Reduced training information, NSTDB noise stress and synthetic mains interference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from ..config import S_STAR, ALPHA_GRID, ARMS, MODELS, N_CLASSES, N_JOBS
from .corrupt import CORRUPT_LEVELS, corrupt
from ..data.nstdb import (NOISE_KINDS, SNR_DB, add_noise, available as nstdb_available,
                    describe as nstdb_describe)
from ..features.ecg import ecg_features_subset
from ..models.ladder import fit_fold, fold_transform, prepare_fold
from ..models.logistic import (_design, chol_pd, fit_map, laplace_precision,
                            predict_proba, sample_theta, select_alpha_evidence)
from ..models.structure import fit_structures, predict_structure_averaged
from ..models.trees import BayesCART, dirichlet_leaf_members
from ..uncertainty import decompose, softmax_uncertainty
from sklearn.tree import DecisionTreeClassifier
from ..metrics import macro_f1_w, nll_w


def sanity_training_size(features_csv: Path, outdir: Path, fractions=(0.25, 0.5, 0.75, 1.0),
                         n_draws: int = 100, cart_iter: int = 1200, folds: int | None = None):
    df = pd.read_csv(features_csv, dtype={"record": str, "subject": str})
    subjects = sorted(df.subject.unique())
    if folds:
        subjects = subjects[:folds]
        df = df[df.subject.isin(subjects)].reset_index(drop=True)
    y = df["label"].to_numpy(int)
    rows = []

    def _one(frac, k, held):
        te_mask = (df.subject == held).to_numpy()
        tr, te = np.flatnonzero(~te_mask), np.flatnonzero(te_mask)
        sub = np.sort(np.random.default_rng(7 + k).permutation(tr)
                      [: max(200, int(round(frac * len(tr))))])
        Ztr, Zte = prepare_fold(df.iloc[sub], df.iloc[te], S_STAR)
        res = fit_fold(Ztr, y[sub], df.subject.to_numpy()[sub], Zte, seed=k, n_draws=n_draws,
                       cart_iter=cart_iter, max_depth=12, min_leaf=10)
        return len(sub), {m: res[m] for m in MODELS}, y[te]

    for frac in fractions:
        acc = {m: [] for m in MODELS}
        ys, n_tr = [], 0
        out = Parallel(n_jobs=N_JOBS)(
            delayed(_one)(frac, k, held) for k, held in enumerate(subjects))
        for n_sub, res, yte in out:
            n_tr += n_sub
            for m in MODELS:
                acc[m].append(res[m])
            ys.append(yte)
        yy = np.concatenate(ys)
        for m in MODELS:
            p = np.concatenate([r["p"] for r in acc[m]])
            row = {"model": m, "arm": "ECG_ACC", "fraction": frac,
                   "n_train": int(round(n_tr / len(subjects))),
                   "macro_F1": macro_f1_w(yy, p.argmax(1)), "NLL": nll_w(yy, p)}
            for s in ("U_pred", "U_CE", "U_MI"):
                row[s] = float(np.mean(np.concatenate([r[s] for r in acc[m]])))
            rows.append(row)
            print(f"    {m} frac {frac:.2f}: F1 {row['macro_F1']:.3f} "
                  f"U_MI {row['U_MI']:.4f} U_CE {row['U_CE']:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "s3_training_size.csv", index=False)
    return out


STRESS_MODELS = ("A1", "A2", "B1", "B2")


STRESS_ARMS = ("ECG_ACC", "ECG")


def _noised_ecg_features(X, kind, snr_db, seed, need):
    Xc = add_noise(X, kind, snr_db, seed=seed)
    f = ecg_features_subset(Xc, need)
    f.pop("_beats", None)
    return f


def _mains_ecg_features(X, level, seed, need):
    f = ecg_features_subset(corrupt(X, "mains", level, seed=seed), need)
    f.pop("_beats", None)
    return f


def _fit_arm(df, y, subjects, rows_idx, cols, cart_iter, label):
    fitted, tests = {}, {}
    for k, held in enumerate(subjects):
        te_mask = (df.subject == held).to_numpy()
        tr = np.flatnonzero(~te_mask)
        sel = np.flatnonzero(te_mask[rows_idx])
        if not len(sel):
            continue
        tests[k] = sel
        (Ztr,), tf = fold_transform(df.iloc[tr][cols].to_numpy(float))
        alpha, _ = select_alpha_evidence(Ztr, y[tr], ALPHA_GRID)
        theta, _ = fit_map(Ztr, y[tr], alpha)
        cart = BayesCART(max_depth=12, min_leaf=10, seed=k).fit(
            Ztr, y[tr], n_iter=cart_iter, burn=cart_iter // 4, thin=max(1, cart_iter // 40))
        t0 = DecisionTreeClassifier(max_depth=12, min_samples_leaf=10,
                                    random_state=k).fit(Ztr, y[tr])
        chol = chol_pd(laplace_precision(theta, _design(Ztr), alpha))


        sa = fit_structures(Ztr, y[tr], alpha)[:3]
        fitted[k] = (tf, theta, chol, cart, t0, Ztr, y[tr], sa)
        print(f"      {label} fold {k:2d} held-out {held}: ready", flush=True)
    print(f"    {label}: {len(fitted)} folds ready on {len(cols)} columns", flush=True)
    return fitted, tests


def _score_condition(fitted, tests, feats, df, rows_idx, cols, n_draws):
    acc = {m: {s: [] for s in ("S_Q", "U_pred", "U_CE", "U_MI", "softmax")}
           for m in STRESS_MODELS}
    dist = []
    sev = np.arange(N_CLASSES, dtype=float)
    for (k, sel), f in zip(tests.items(), feats):
        (med, mu, sd), theta, chol, cart, t0, Ztr_k, ytr_k, sa_k = fitted[k]
        F = np.zeros((len(sel), len(cols)))
        for j, name in enumerate(cols):


            F[:, j] = (f[name] if name in f
                       else df.iloc[rows_idx[sel]][name].to_numpy(float))
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        Zte = (np.where(np.isfinite(F), F, med) - mu) / sd


        dist.append(np.linalg.norm(Zte, axis=1))

        rng = np.random.default_rng(5000 + k)
        mem = np.stack([predict_proba(Zte, th)
                        for th in sample_theta(theta, chol, n_draws, rng)])
        parts = {
            "A1": decompose(mem),
            "A2": decompose(predict_structure_averaged(*sa_k, Zte, n_draws, 8000 + k)[0]),
            "B1": decompose(dirichlet_leaf_members(t0, Ztr_k, ytr_k, Zte, n_draws, seed=6000 + k)),
            "B2": decompose(cart.predict_members(Zte, n_draws, seed=7000 + k)),
        }
        for m, d in parts.items():
            acc[m]["S_Q"].append(d["p"] @ sev)
            acc[m]["softmax"].append(softmax_uncertainty(d["p"]))
            for s in ("U_pred", "U_CE", "U_MI"):
                acc[m][s].append(d[s])

    D = float(np.mean(np.concatenate(dist)))
    return {m: {s: float(np.mean(np.concatenate(v))) for s, v in d.items()}
            for m, d in acc.items()}, D


def noise_stress(features_csv: Path, outdir: Path, n_draws: int = 100, cart_iter: int = 1200,
                 folds: int | None = None):
    wav_path = features_csv.with_name("waveforms.npz")
    if not wav_path.exists():
        print("  no waveforms.npz -- rerun `features` with --wave-cache > 0; skipping", flush=True)
        return None
    if not nstdb_available():
        print("  NSTDB not found -- fetch bw/em/ma into dataset/nstdb; skipping", flush=True)
        return None

    df = pd.read_csv(features_csv, dtype={"record": str, "subject": str})
    z = np.load(wav_path)
    rows_idx, ECG = z["rows"], z["ecg"]
    subjects = sorted(df.subject.unique())
    if folds:
        subjects = subjects[:folds]
    y = df["label"].to_numpy(int)
    print(f"    noise library: {nstdb_describe()}", flush=True)

    rows = []
    for arm in STRESS_ARMS:
        cols = ARMS[arm]
        need_ecg = [c for c in cols if c.startswith("ECG_")]
        fitted, tests = _fit_arm(df, y, subjects, rows_idx, cols, cart_iter, arm)
        if not fitted:
            continue
        for kind in NOISE_KINDS:
            for snr in SNR_DB:
                feats = Parallel(n_jobs=N_JOBS)(
                    delayed(_noised_ecg_features)(ECG[sel], kind, snr, k, need_ecg)
                    for k, sel in tests.items())
                res, D = _score_condition(fitted, tests, feats, df, rows_idx, cols, n_draws)
                for m, v in res.items():
                    rows.append({"arm": arm, "model": m, "kind": kind, "snr_db": snr,
                                 "distance": D, **v})
                tag = "clean" if not np.isfinite(snr) else f"{snr:+.0f} dB"
                print(f"    {arm:8s} {kind} {tag:>8s}  " + "  ".join(
                    f"{m} S_Q {res[m]['S_Q']:.3f} U {res[m]['U_pred']:.3f}"
                    for m in STRESS_MODELS), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "s2_noise_stress.csv", index=False)
    return out


def mains_sensitivity(features_csv: Path, outdir: Path, n_draws: int = 100, cart_iter: int = 1200,
                      folds: int | None = None):
    wav_path = features_csv.with_name("waveforms.npz")
    if not wav_path.exists():
        return None
    df = pd.read_csv(features_csv, dtype={"record": str, "subject": str})
    z = np.load(wav_path)
    rows_idx, ECG = z["rows"], z["ecg"]
    subjects = sorted(df.subject.unique())
    if folds:
        subjects = subjects[:folds]
    y = df["label"].to_numpy(int)

    cols = ARMS["ECG_ACC"]
    need_ecg = [c for c in cols if c.startswith("ECG_")]
    fitted, tests = _fit_arm(df, y, subjects, rows_idx, cols, cart_iter, "mains")
    if not fitted:
        return None

    rows = []
    for level in CORRUPT_LEVELS:
        feats = Parallel(n_jobs=N_JOBS)(
            delayed(_mains_ecg_features)(ECG[sel], level, k, need_ecg)
            for k, sel in tests.items())
        res, D = _score_condition(fitted, tests, feats, df, rows_idx, cols, n_draws)
        for m, v in res.items():
            rows.append({"model": m, "level": level, "distance": D, **v})
        print(f"    mains level {level:.2f}  " + "  ".join(
            f"{m} S_Q {res[m]['S_Q']:.3f} U {res[m]['U_pred']:.3f}" for m in STRESS_MODELS),
            flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "s4_mains_sensitivity.csv", index=False)
    return out
