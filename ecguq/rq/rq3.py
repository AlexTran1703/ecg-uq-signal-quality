"""RQ3 - construct validity of the designated aleatoric (U_CE) and epistemic (U_MI) components.

Outputs (results/rq_3_result): expert_agreement.csv (Figure 5a), local_entropy.csv,
local_overlap.csv (Figure 5b), local_overlap_k_sweep.csv (Figure S4), reduced_training.csv
(Figure 5c, Figure S3), subject_weighting.csv and balanced_refit.csv (Table S8).
"""
from __future__ import annotations

from ..analysis.overlap import k_sensitivity
from ..analysis.performance import expert_agreement
from ..config import CART_ITER_STRESS, EVAL_CFG, N_BOOT, N_DRAWS, result_dir
from ..shift.stress import sanity_training_size
from .common import (balanced_refit, features_csv, load_predictions, local_entropy_csv, log,
                     predictions_csv, subject_weighting)

RQ3_REFIT = ["mode", "seed", "model", "rho(U_CE, H_local)", "rho(U_MI, H_local)",
             "rho(U_CE, H_expert)", "rho(U_MI, H_expert)"]


def run():
    out = result_dir(3)
    f = features_csv()
    preds = load_predictions()

    log("expert disagreement probe")
    expert_agreement(preds, N_BOOT).to_csv(out / "expert_agreement.csv", index=False)

    log("local class-overlap probe and neighbourhood-size sweep")
    local_entropy_csv()
    k_sensitivity(f, predictions_csv(), out)
    (out / "s6_hlocal_ksweep.csv").replace(out / "local_overlap_k_sweep.csv")

    log("reduced training information")
    sanity_training_size(f, out, tuple(EVAL_CFG["reduced_training_fractions"]),
                         n_draws=N_DRAWS, cart_iter=CART_ITER_STRESS)
    (out / "s3_training_size.csv").replace(out / "reduced_training.csv")

    w = subject_weighting()
    w[w.rq == "RQ3"].to_csv(out / "subject_weighting.csv", index=False)
    balanced_refit()[RQ3_REFIT].to_csv(out / "balanced_refit.csv", index=False)
