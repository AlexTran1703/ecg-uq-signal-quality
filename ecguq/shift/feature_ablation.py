"""Feature behaviour under muscle artifact and leave-one-feature-out refitting of A2 (Supplementary S6.3).

The six ECG features are recomputed for the cached held-out windows under clean and corrupted
conditions, using the same NSTDB noise seeds as the stress test (the fold index). A2 is then
refitted in every fold with each ECG feature removed in turn, and the clean-to-corrupted change in
conditional entropy is compared with that of the six-feature model (D_j).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..config import ALPHA_GRID, BOOT_SEED, N_BOOT, N_DRAWS, N_JOBS, S_STAR_ECG, SHIFT_CFG
from ..data.nstdb import add_noise
from ..features.ecg import ecg_features_subset
from ..metrics import macro_f1_w, nll_w
from ..models.ladder import fold_transform
from ..models.logistic import select_alpha_evidence
from ..models.structure import fit_structures, predict_structure_averaged
from ..uncertainty import decompose

CFG = SHIFT_CFG["feature_diagnostics"]
KIND = CFG["artifact"]
SNRS = [np.inf if s == "clean" else float(s) for s in CFG["snr_db"]]
CONTRAST = float(CFG["ablation_contrast_db"])


def _features(X, snr, seed):
    Xc = add_noise(X, KIND, snr, seed=seed) if np.isfinite(snr) else X
    f = ecg_features_subset(Xc, S_STAR_ECG)
    return pd.DataFrame({c: np.asarray(f[c], float) for c in S_STAR_ECG})


def corrupted_features(features_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(features_csv, dtype={"record": str, "subject": str})
    z = np.load(features_csv.with_name("waveforms.npz"))
    rows, ecg = z["rows"], z["ecg"]
    subs = sorted(df.subject.unique())
    csub = df.subject.to_numpy()[rows]
    labels = df.label.to_numpy()[rows]
    out = []
    for snr in SNRS:
        res = Parallel(n_jobs=N_JOBS)(delayed(_features)(ecg[csub == s], snr, k)
                                      for k, s in enumerate(subs))
        for s, r in zip(subs, res):
            r["subject"], r["label"], r["snr_db"] = s, labels[csub == s], snr
            out.append(r)
    return pd.concat(out, ignore_index=True)


def _fold(df, feats, k, s, cols):
    tr = df[df.subject != s]
    ytr = tr.label.to_numpy(int)
    blocks = [feats[(feats.subject == s) & (feats.snr_db == snr)][cols].to_numpy(float) for snr in SNRS]
    (Ztr, *Zte), _ = fold_transform(tr[cols].to_numpy(float), *blocks)
    alpha, _ = select_alpha_evidence(Ztr, ytr, ALPHA_GRID)
    sa = fit_structures(Ztr, ytr, alpha)[:3]
    return s, [(snr, decompose(predict_structure_averaged(*sa, Z, N_DRAWS, 8000 + k)[0]))
               for snr, Z in zip(SNRS, Zte)]


def leave_one_feature_out(features_csv: Path, feats: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(features_csv, dtype={"record": str, "subject": str})
    subs = sorted(df.subject.unique())
    configs = [("all six features", list(S_STAR_ECG))] + [
        (f"without {c}", [x for x in S_STAR_ECG if x != c]) for c in S_STAR_ECG]
    rows = []
    for name, cols in configs:
        res = Parallel(n_jobs=N_JOBS)(delayed(_fold)(df, feats, k, s, cols) for k, s in enumerate(subs))
        for s, out in res:
            y = feats[(feats.subject == s) & (feats.snr_db == SNRS[0])].label.to_numpy(int)
            for snr, d in out:
                for i in range(len(y)):
                    rows.append((name, s, snr, i, y[i], *d["p"][i], d["U_CE"][i]))
    return pd.DataFrame(rows, columns=["feature_set", "subject", "snr_db", "i", "y",
                                       "p_Q1", "p_Q2", "p_Q3", "U_CE"])


def summarise(lofo: pd.DataFrame) -> pd.DataFrame:
    lofo = lofo.assign(S_Q=lofo.p_Q2 + 2 * lofo.p_Q3)
    clean, bad = lofo[lofo.snr_db == SNRS[0]], lofo[lofo.snr_db == CONTRAST]
    key = ["feature_set", "subject", "i"]
    m = clean.merge(bad, on=key, suffixes=("_clean", "_bad"))
    m["dU"] = m.U_CE_bad - m.U_CE_clean
    full = m[m.feature_set == "all six features"].set_index(["subject", "i"]).dU
    subs = np.array(sorted(m.subject.unique()))
    rng = np.random.default_rng(BOOT_SEED)
    boots = [rng.choice(len(subs), len(subs)) for _ in range(N_BOOT)]
    rows = []
    for name, g in m.groupby("feature_set", sort=False):
        g = g.set_index(["subject", "i"])
        D = (g.dU - full)
        per = D.groupby(level=0).agg(["sum", "count"]).loc[subs]
        bd = [per["sum"].values[b].sum() / per["count"].values[b].sum() for b in boots]
        P = g[["p_Q1_clean", "p_Q2_clean", "p_Q3_clean"]].to_numpy()
        yc = g.y_clean.to_numpy(int)
        rows.append({"feature_set": name, "clean_macro_F1": macro_f1_w(yc, P.argmax(1)),
                     "clean_NLL": nll_w(yc, P), "U_CE_clean": g.U_CE_clean.mean(),
                     "U_CE_corrupted": g.U_CE_bad.mean(), "delta_U_CE": g.dU.mean(),
                     "S_Q_corrupted": g.S_Q_bad.mean(), "D_j": D.mean(),
                     "D_j_lo": np.percentile(bd, 2.5), "D_j_hi": np.percentile(bd, 97.5)})
    return pd.DataFrame(rows)


def run(features_csv: Path, outdir: Path):
    feats = corrupted_features(features_csv)
    feats.to_csv(outdir / "muscle_artifact_features.csv", index=False)
    lofo = leave_one_feature_out(features_csv, feats)
    lofo.to_csv(outdir / "leave_one_feature_out_windows.csv.gz", index=False)
    summarise(lofo).to_csv(outdir / "leave_one_feature_out.csv", index=False)
