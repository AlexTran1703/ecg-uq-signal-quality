"""Shared inputs for the research questions; each is built once and reused if it already exists."""
from __future__ import annotations

import pandas as pd

from ..config import (BUTQDB_ROOT, CACHE, CART_ITER, MAX_DEPTH, MIN_LEAF, N_BOOT, N_DRAWS,
                      WAVE_CACHE, EVAL_CFG, result_dir)


def log(msg):
    print(msg, flush=True)


def features_csv():
    path = result_dir(0) / "features.csv"
    if not path.exists():
        from ..data.butqdb import build_features
        log(f"building the BUT QDB feature table from {BUTQDB_ROOT}")
        build_features(BUTQDB_ROOT, path, WAVE_CACHE)
    return path


def predictions_csv():
    path = result_dir(1) / "loso_predictions.csv.gz"
    if not path.exists():
        from ..models.ladder import run_loso
        log("fitting the six rungs in leave-one-subject-out folds")
        run_loso(features_csv(), result_dir(1), N_DRAWS, CART_ITER, MAX_DEPTH, MIN_LEAF)
        (result_dir(1) / "predictions.csv.gz").rename(path)
        (result_dir(1) / "t00_fold_diagnostics.csv").rename(result_dir(1) / "loso_fold_diagnostics.csv")
    return path


def local_entropy_csv():
    path = result_dir(3) / "local_entropy.csv"
    if not path.exists():
        from ..analysis.overlap import build
        log("computing local class-overlap entropy")
        build(features_csv(), predictions_csv(), result_dir(3), path, n_boot=N_BOOT)
    return path


def subject_weighting():
    path = CACHE / "subject_weighting_all_endpoints.csv"
    if not path.exists():
        from ..analysis.weighting import load_predictions, ms_table, weighting_table
        log("subject-weighting sensitivity for all endpoints")
        CACHE.mkdir(parents=True, exist_ok=True)
        p = load_predictions(predictions_csv(), local_entropy_csv())
        pd.concat([weighting_table(p, N_BOOT), ms_table(p, N_BOOT)],
                  ignore_index=True).to_csv(path, index=False)
    return pd.read_csv(path)


def balanced_refit():
    path = CACHE / "balanced_refit_all_endpoints.csv"
    if not path.exists():
        from ..analysis.weighting import refit
        log("subject-balanced refits (three seeds)")
        CACHE.mkdir(parents=True, exist_ok=True)
        refit(features_csv(), local_entropy_csv(), seeds=tuple(EVAL_CFG["balanced_refit"]["seeds"]),
              n_draws=N_DRAWS, cart_iter=CART_ITER, max_depth=MAX_DEPTH,
              min_leaf=MIN_LEAF).to_csv(path, index=False)
    return pd.read_csv(path)


def load_predictions():
    return pd.read_csv(predictions_csv(), dtype={"record": str, "subject": str})
