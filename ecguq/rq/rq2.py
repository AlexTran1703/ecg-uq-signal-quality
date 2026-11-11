"""RQ2 - operational reliability of posterior-derived uncertainty against same-model max-softmax.

Outputs (results/rq_2_result): reliability.csv (Table 4), incremental_gains.csv (Figure 4),
risk_coverage_curves.csv (Figure S2), per_subject_scores.csv, subject_weighting.csv and
balanced_refit.csv (Table S7), exclude_subject.csv (Table S7, last column),
augrc_vs_random.csv (Results 3.3 and Section S4).
"""
from __future__ import annotations

from ..analysis.performance import per_subject_scores, reliability_metrics, risk_coverage_curves
from ..analysis.reliability import augrc_vs_random, incremental_over_ums
from ..analysis.weighting import exclusion_table
from ..config import EXCLUDED_SUBJECT, N_BOOT, result_dir
from .common import balanced_refit, load_predictions, log, subject_weighting

RQ2_REFIT = ["mode", "seed", "model", "dAUROC_err(U_CE)", "dAUGRC_gain(U_CE)",
             "dAUROC_err(U_MI)", "dAUGRC_gain(U_MI)"]


def run():
    out = result_dir(2)
    preds = load_predictions()

    log("absolute reliability, paired gains over max-softmax and risk-coverage curves")
    reliability_metrics(preds, N_BOOT).to_csv(out / "reliability.csv", index=False)
    incremental_over_ums(preds, N_BOOT).to_csv(out / "incremental_gains.csv", index=False)
    risk_coverage_curves(preds).to_csv(out / "risk_coverage_curves.csv", index=False)
    per_subject_scores(preds).to_csv(out / "per_subject_scores.csv", index=False)

    log("subject-balance sensitivity")
    w = subject_weighting()
    w[(w.rq == "RQ2") | w.endpoint.str.contains("U_MS")].to_csv(out / "subject_weighting.csv", index=False)
    balanced_refit()[RQ2_REFIT].to_csv(out / "balanced_refit.csv", index=False)

    acc = preds[preds.arm == "ECG_ACC"]
    exclusion_table(acc, EXCLUDED_SUBJECT, N_BOOT).to_csv(out / "exclude_subject.csv", index=False)
    augrc_vs_random(acc, EXCLUDED_SUBJECT, N_BOOT).to_csv(out / "augrc_vs_random.csv", index=False)
