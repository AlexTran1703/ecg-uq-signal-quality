"""RQ1 - predictive comparability and posterior structure.

Outputs (results/rq_1_result): loso_predictions.csv.gz, loso_fold_diagnostics.csv,
pooled_performance.csv (Table 3), per_subject_performance.csv (Figure S1), rung_contrasts.csv,
per_fold_recall.csv (Tables S4-S5), subject_weighting.csv and balanced_refit.csv (Table S6),
posterior_structure/ (Figure 3), estimator_diagnostics.csv, b2_mcmc_rhat_ess.csv,
b2_mcmc_chains.csv, b2_predictive_convergence.csv, b2_mcmc_summary.csv (Section S2), representation_sensitivity.csv (Table S3).
"""
from __future__ import annotations

from ..analysis.diagnostics import run_diagnostics
from ..analysis.fullset import compare, run_loso as run_fullset
from ..analysis.mcmc import b2_mcmc_diagnostics
from ..analysis.performance import contrasts, per_subject_metrics, pooled_metrics
from ..analysis.posterior import posterior_shapes
from ..analysis.weighting import per_fold_table
from ..config import CART_ITER, MAX_DEPTH, MIN_LEAF, N_BOOT, N_DRAWS, result_dir
from .common import (balanced_refit, features_csv, load_predictions, local_entropy_csv, log,
                     predictions_csv, subject_weighting)

RQ1_REFIT = ["mode", "seed", "model", "n_train_mean", "macro-F1", "NLL", "Q3_recall_fold105"]


def run():
    out = result_dir(1)
    f = features_csv()
    preds = load_predictions()

    log("pooled performance, per-subject performance and rung contrasts")
    pooled_metrics(preds, N_BOOT).to_csv(out / "pooled_performance.csv", index=False)
    per_subject_metrics(preds).to_csv(out / "per_subject_performance.csv", index=False)
    contrasts(preds, N_BOOT).to_csv(out / "rung_contrasts.csv", index=False)
    per_fold_table(preds[preds.arm == "ECG_ACC"]).to_csv(out / "per_fold_recall.csv", index=False)

    w = subject_weighting()
    w[w.rq == "RQ1"].to_csv(out / "subject_weighting.csv", index=False)
    balanced_refit()[RQ1_REFIT].to_csv(out / "balanced_refit.csv", index=False)

    log("posterior structure (A2 inclusion, B2 feature use and topology)")
    sub = out / "posterior_structure"
    sub.mkdir(exist_ok=True)
    posterior_shapes(f, sub, n_draws=max(200, N_DRAWS), cart_iter=CART_ITER,
                     max_depth=MAX_DEPTH, min_leaf=MIN_LEAF)

    log("computational checks and B2 MCMC convergence")
    run_diagnostics(f, out)
    b2_mcmc_diagnostics(f, out)

    log("representation sensitivity: all 18 candidate features")
    full_pred = out / "fullset_predictions.csv.gz"
    if not full_pred.exists():
        run_fullset(f, full_pred, N_DRAWS, CART_ITER, MAX_DEPTH, MIN_LEAF)
    compare(full_pred, predictions_csv(), local_entropy_csv()).to_csv(
        out / "representation_sensitivity.csv", index=False)
