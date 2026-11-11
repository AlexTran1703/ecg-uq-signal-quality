"""Representation screening: all 127 mechanism subsets with the point-estimate model S0."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from ..config import ALPHA_GRID, MECHANISMS, MECH_DESC, MECH_SOURCE, N_JOBS, SCREENING_CFG
from ..models.ladder import fold_transform
from ..metrics import brier_w, ece_w, macro_f1_w, nll_w
from ..models.logistic import fit_map, predict_proba, select_alpha_evidence

N_SPLITS = int(SCREENING_CFG["n_splits"])
TIES = SCREENING_CFG["ties"]
TOP_FRACTION = float(SCREENING_CFG["prevalence_top_fraction"])
KEYS = tuple(MECHANISMS)


def all_subsets(keys=KEYS) -> list[str]:
    out = []
    for k in range(1, len(keys) + 1):
        out.extend("+".join(c) for c in combinations(keys, k))
    return out


def columns_of(group: str) -> list[str]:
    return [c for g in group.split("+") for c in MECHANISMS[g]]


def _screen_one(group: str, X: np.ndarray, y: np.ndarray, subj: np.ndarray, splits) -> dict:
    p = np.zeros((len(y), int(y.max()) + 1))
    for tr, va in splits:
        (Ztr, Zva), _ = fold_transform(X[tr], X[va])
        alpha, _ = select_alpha_evidence(Ztr, y[tr], ALPHA_GRID)
        theta, _ = fit_map(Ztr, y[tr], alpha)
        p[va] = predict_proba(Zva, theta)
    w = np.ones(len(y))
    return {"group": group,
            "n_mech": group.count("+") + 1,
            "n_features": X.shape[1],
            "NLL": nll_w(y, p, w),
            "Brier": brier_w(y, p, w),
            "ECE": ece_w(y, p, w),
            "macro_F1": macro_f1_w(y, p.argmax(1), w)}


def select_star(t: pd.DataFrame, nll_tol: float = TIES["nll_tolerance"]) -> str:
    best = t.NLL.min()
    tied = t[t.NLL <= best + nll_tol]
    tied = tied[tied.Brier <= tied.Brier.min() + TIES["brier_tolerance"]]
    tied = tied[tied.ECE <= tied.ECE.min() + TIES["ece_tolerance"]]
    tied = tied.sort_values(["n_features", "NLL"])
    return str(tied.iloc[0].group)


def run_screen(features_csv: Path, outdir: Path, n_jobs: int = N_JOBS) -> str:
    df = pd.read_csv(features_csv, dtype={"record": str, "subject": str})
    y = df["label"].to_numpy(int)
    if y.min() == 1:
        y = y - 1
    subj = df.subject.to_numpy(str)
    splits = list(GroupKFold(n_splits=N_SPLITS).split(df, y, groups=subj))

    groups = all_subsets()
    print(f"  {len(groups)} mechanism subsets, {N_SPLITS}-fold subject-grouped split", flush=True)
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_screen_one)(g, df[columns_of(g)].to_numpy(float), y, subj, splits)
        for g in groups)

    t = pd.DataFrame(rows).sort_values("NLL").reset_index(drop=True)
    star = select_star(t)
    t["selected"] = t.group == star

    t.to_csv(outdir / "screen_all_subsets.csv", index=False)
    pd.DataFrame({"selected_representation": [star],
                  "columns": [" ".join(columns_of(star))]}).to_csv(outdir / "selected_representation.csv", index=False)


    k = max(2, int(round(TOP_FRACTION * len(t))))
    top = t.nsmallest(k, "NLL")
    pi0 = float(np.mean([g.count("+") + 1 for g in top.group]) / len(KEYS))
    pres = pd.DataFrame([
        {"mechanism": m, "source": MECH_SOURCE[m], "desc": MECH_DESC[m],
         "n_features": len(MECHANISMS[m]),
         "share_top": float(top.group.str.split("+").apply(lambda c: m in c).mean()),
         "null": pi0}
        for m in KEYS]).sort_values("share_top", ascending=False)
    pres.to_csv(outdir / "screen_mechanism_prevalence.csv", index=False)

    print(f"\n  S* = {star}  ({len(columns_of(star))} features)", flush=True)
    print(t.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    return star
