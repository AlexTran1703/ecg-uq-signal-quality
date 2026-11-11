"""RQ4 - behaviour under controlled ECG corruption and preliminary zero-shot transfer to LTSTDB.

Outputs (results/rq_4_result): noise_stress.csv (Figure 6, Figure S6), synthetic_mains.csv
(Figure S5), muscle_artifact_features.csv (Figure S7), leave_one_feature_out.csv (Table S9),
ltstdb_windows.csv, ltstdb_transfer.csv, ltstdb_state_levels.csv (Figure 7a-b),
ltstdb_input_novelty.csv (Figure 7c), ltstdb_subject_contrasts.csv, ltstdb_episodes.csv,
ltstdb_mains_by_record.csv and ltstdb_mains_by_window.csv.gz (Section 2.7).
"""
from __future__ import annotations

import pandas as pd

from ..config import CART_ITER_STRESS, N_DRAWS, result_dir
from ..shift import feature_ablation, mains_check, transfer
from ..shift.stress import mains_sensitivity, noise_stress
from .common import features_csv, log


def run():
    out = result_dir(4)
    f = features_csv()

    log("NSTDB noise stress on held-out windows")
    noise_stress(f, out, n_draws=N_DRAWS, cart_iter=CART_ITER_STRESS)
    (out / "s2_noise_stress.csv").replace(out / "noise_stress.csv")
    log("synthetic mains interference")
    mains_sensitivity(f, out, n_draws=N_DRAWS, cart_iter=CART_ITER_STRESS)
    (out / "s4_mains_sensitivity.csv").replace(out / "synthetic_mains.csv")

    log("feature behaviour under muscle artifact and leave-one-feature-out refitting")
    feature_ablation.run(f, out)

    log("zero-shot transfer to LTSTDB")
    windows = out / "ltstdb_windows.csv"
    if not windows.exists():
        transfer.build_features(transfer.positive_records()).to_csv(windows, index=False)
    tr = transfer.transfer(f, windows, out / "ltstdb_transfer.csv")
    transfer.subject_stats(tr).to_csv(out / "ltstdb_subject_contrasts.csv", index=False)
    transfer.state_levels(tr).to_csv(out / "ltstdb_state_levels.csv", index=False)
    transfer.episode_stats(tr).to_csv(out / "ltstdb_episodes.csv", index=False)
    but = pd.read_csv(f, dtype={"record": str, "subject": str})
    w = pd.read_csv(windows, dtype={"record": str, "subject": str})
    transfer.novelty_table(but, w).to_csv(out / "ltstdb_input_novelty.csv", index=False)
    mains_check.run(windows, out)
