"""Local class-overlap entropy in the locked feature space and its association with uncertainty."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from ..config import ARMS, EPS, K_PRIMARY, K_SWEEP, N_BOOT, N_BOOT_RANK, N_CLASSES, UQ_MODELS
from ..models.ladder import fold_transform
from ..metrics import ci_from

K_GRID = (25, 50, 100)


TERMS = ("U_CE", "U_MI", "U_pred", "softmax")


def local_label_entropy(features_csv: Path, arm: str = "ECG_ACC",
                        ks=K_GRID) -> pd.DataFrame:
    df = pd.read_csv(features_csv, dtype={"record": str, "subject": str})
    cols = ARMS[arm]
    y = df["label"].to_numpy(int)
    out = {f"H_local_k{k}": np.full(len(df), np.nan) for k in ks}

    for held in sorted(df.subject.unique()):
        te = (df.subject == held).to_numpy()
        tr = ~te
        (Ztr, Zte), _ = fold_transform(df.loc[tr, cols].to_numpy(float),
                                       df.loc[te, cols].to_numpy(float))
        ytr = y[tr]
        nn = NearestNeighbors(n_neighbors=max(ks)).fit(Ztr)
        idx = nn.kneighbors(Zte, return_distance=False)
        for k in ks:
            lab = ytr[idx[:, :k]]
            q = np.stack([(lab == c).mean(1) for c in range(N_CLASSES)], axis=1)
            out[f"H_local_k{k}"][te] = -(q * np.log(q + EPS)).sum(1)

    res = df[["record", "subject", "start_sample"]].copy()
    for k, v in out.items():
        res[k] = v
    return res


def _boot_rho(u, h, subj, n_boot=500, seed=20240917):
    subj = np.asarray(subj).astype(str)
    subs = np.unique(subj)
    idx_of = {s: np.flatnonzero(subj == s) for s in subs}
    rng = np.random.default_rng(seed)
    point = stats.spearmanr(u, h).statistic
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[s] for s in rng.choice(subs, size=len(subs), replace=True)])
        v = stats.spearmanr(u[idx], h[idx]).statistic
        if np.isfinite(v):
            vals.append(v)
    return ci_from(point, vals)


def build(features_csv: Path, preds_path: Path, outdir: Path, local_csv: Path, arm: str = "ECG_ACC",
          n_boot: int = N_BOOT) -> pd.DataFrame:
    hl = local_label_entropy(features_csv, arm)
    hl.to_csv(local_csv, index=False)
    frac = float((hl[f"H_local_k{K_PRIMARY}"] > 0).mean())
    print(f"  H_local: {frac:.1%} of windows sit in a mixed-label neighbourhood (k={K_PRIMARY})",
          flush=True)

    preds = _predictions(preds_path, arm)
    key = ["record", "subject", "start_sample"]
    preds = preds.merge(hl, on=key, how="left")

    rows = []
    for cid, g in preds.groupby("config", sort=False):
        model = g.model.iloc[0]
        if model not in UQ_MODELS:
            continue
        subj = g.subject.to_numpy(str)
        for k in K_GRID:
            h = g[f"H_local_k{k}"].to_numpy(float)
            ok = np.isfinite(h)
            for s in TERMS:
                u = g[s].to_numpy(float)
                if not np.isfinite(u).any() or np.ptp(u[ok]) == 0:
                    continue
                rho, lo, hi = _boot_rho(u[ok], h[ok], subj[ok], n_boot=min(n_boot, N_BOOT_RANK))
                rows.append({"config": cid, "model": model, "arm": arm, "k": k, "score": s,
                             "spearman_local": rho, "lo": lo, "hi": hi})
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "local_overlap.csv", index=False)
    print(f"  t10_local_overlap.csv: {len(t)} correlations", flush=True)
    return t


def _predictions(path: Path, arm: str) -> pd.DataFrame:
    p = pd.read_csv(path, dtype={"record": str, "subject": str})
    return p[p.arm == arm]


def _boot_delta(u_ce, u_mi, h, subj, n_boot, seed=20260825):
    subj = np.asarray(subj).astype(str)
    idx_of = {s: np.flatnonzero(subj == s) for s in np.unique(subj)}
    subs = np.array(sorted(idx_of))
    rng = np.random.default_rng(seed)
    point = (stats.spearmanr(u_ce, h).statistic - stats.spearmanr(u_mi, h).statistic)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[s] for s in rng.choice(subs, size=len(subs), replace=True)])
        d = (stats.spearmanr(u_ce[idx], h[idx]).statistic
             - stats.spearmanr(u_mi[idx], h[idx]).statistic)
        if np.isfinite(d):
            vals.append(d)
    return ci_from(point, vals)


def k_sensitivity(features_csv: Path, preds_path: Path, outdir: Path, arm: str = "ECG_ACC",
                  ks=K_SWEEP, n_boot: int = 300) -> pd.DataFrame:
    hl = local_label_entropy(features_csv, arm, ks=ks)
    preds = _predictions(preds_path, arm)
    key = ["record", "subject", "start_sample"]
    preds = preds.merge(hl, on=key, how="left")

    rows = []
    for cid, g in preds.groupby("config", sort=False):
        model = g.model.iloc[0]
        if model not in UQ_MODELS or not cid.endswith("+"):
            continue
        subj = g.subject.to_numpy(str)
        ce, mi = g["U_CE"].to_numpy(float), g["U_MI"].to_numpy(float)
        for k in ks:
            h = g[f"H_local_k{k}"].to_numpy(float)
            ok = np.isfinite(h) & np.isfinite(ce) & np.isfinite(mi)
            r_ce = stats.spearmanr(ce[ok], h[ok]).statistic
            r_mi = stats.spearmanr(mi[ok], h[ok]).statistic
            d, lo, hi = _boot_delta(ce[ok], mi[ok], h[ok], subj[ok], n_boot)
            rows.append({"model": model, "arm": arm, "k": k, "rho_CE": r_ce, "rho_MI": r_mi,
                         "delta": d, "delta_lo": lo, "delta_hi": hi,
                         "primary": k == K_PRIMARY})
            print(f"    {model} k={k:>3}  rho_CE {r_ce:+.3f}  rho_MI {r_mi:+.3f}  "
                  f"delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)

    t = pd.DataFrame(rows).sort_values(["model", "k"]).reset_index(drop=True)
    outdir.mkdir(parents=True, exist_ok=True)
    t.to_csv(outdir / "s6_hlocal_ksweep.csv", index=False)
    print(f"  s6_hlocal_ksweep.csv: {len(t)} rows over k = {ks[0]}-{ks[-1]}", flush=True)
    return t


if __name__ == "__main__":
    build()
