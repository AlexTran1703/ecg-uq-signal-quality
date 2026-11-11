"""RQ0 - data and representation: feature table, cohort summary, LOSO folds and the mechanism screen.

Outputs (results/rq_0_result): features.csv, waveforms.npz, dataset_summary.csv (Table 1),
candidate_features.csv (Table 2), loso_folds.csv, screen_all_subsets.csv (Table S2),
screen_mechanism_prevalence.csv (Figure 2), selected_representation.csv.
"""
from __future__ import annotations

import pandas as pd

from ..analysis.performance import dataset_table
from ..analysis.screening import run_screen
from ..config import FEATURES_CFG, result_dir
from .common import features_csv, log


def run():
    out = result_dir(0)
    f = features_csv()
    dataset_table(f).to_csv(out / "dataset_summary.csv", index=False)

    rows = [{"mechanism": g, "source": v["source"], "description": v["description"], "feature": c}
            for g, v in FEATURES_CFG["mechanisms"].items() for c in v["columns"]]
    pd.DataFrame(rows).to_csv(out / "candidate_features.csv", index=False)

    df = pd.read_csv(f, dtype={"record": str, "subject": str})
    subjects = sorted(df.subject.unique())
    folds = df.groupby("subject").agg(windows=("label", "size"), Q1=("label", lambda s: int((s == 0).sum())),
                                      Q2=("label", lambda s: int((s == 1).sum())),
                                      Q3=("label", lambda s: int((s == 2).sum())))
    folds.insert(0, "fold", [subjects.index(s) for s in folds.index])
    folds.reset_index().sort_values("fold").to_csv(out / "loso_folds.csv", index=False)

    log("screening 127 mechanism subsets")
    star = run_screen(f, out)
    locked = FEATURES_CFG["selected_representation"]
    if star != locked:
        log(f"note: the screen selected {star}; config/features.json locks {locked}")
