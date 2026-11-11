"""Load the JSON files in config/ and expose the settings used by every module."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "config"


def _load(name):
    return json.loads((CONFIG_DIR / f"{name}.json").read_text(encoding="utf-8"))


PATHS = _load("paths.local") if (CONFIG_DIR / "paths.local.json").exists() else _load("paths")
DATA_CFG = _load("data")
FEATURES_CFG = _load("features")
SCREENING_CFG = _load("screening")
MODELS_CFG = _load("models")
EVAL_CFG = _load("evaluation")
SHIFT_CFG = _load("shift")


def repo_path(p):
    p = Path(p)
    return p if p.is_absolute() else REPO / p


BUTQDB_ROOT = repo_path(PATHS["butqdb_root"])
LTSTDB_ROOT = repo_path(PATHS["ltstdb_root"])
NSTDB_ROOT = repo_path(PATHS["nstdb_root"])
RESULTS = repo_path(PATHS["results_root"])
CACHE = RESULTS / "_cache"


def result_dir(rq: int) -> Path:
    d = RESULTS / f"rq_{rq}_result"
    d.mkdir(parents=True, exist_ok=True)
    return d


EPS = float(DATA_CFG["eps"])
N_CLASSES = int(DATA_CFG["n_classes"])
FS_ECG_RAW = int(DATA_CFG["fs_ecg_raw_hz"])
FS_ECG = int(DATA_CFG["fs_ecg_hz"])
FS_ACC = int(DATA_CFG["fs_acc_hz"])
WIN_S = float(DATA_CFG["window_s"])
DECIM = FS_ECG_RAW // FS_ECG
WN_ECG = int(WIN_S * FS_ECG)
WN_ACC = int(WIN_S * FS_ACC)
MIN_PURITY = float(DATA_CFG["label_rule"]["min_modal_class_purity"])
N_JOBS = int(DATA_CFG["n_jobs"])
BLOCK = int(DATA_CFG["feature_block_windows"])
WAVE_CACHE = int(DATA_CFG["waveform_cache_per_record"])


def record_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 6)


def subject_of(record: str) -> str:
    return record[: int(DATA_CFG["subject_id_digits"])]


MECHANISMS = {g: v["columns"] for g, v in FEATURES_CFG["mechanisms"].items()}
MECH_DESC = {g: v["description"] for g, v in FEATURES_CFG["mechanisms"].items()}
MECH_SOURCE = {g: v["source"] for g, v in FEATURES_CFG["mechanisms"].items()}
ALL_FEATURES = [c for g in MECHANISMS for c in MECHANISMS[g]]
S_STAR_MECHANISMS = tuple(FEATURES_CFG["selected_representation"].split("+"))
S_STAR = [c for g in S_STAR_MECHANISMS for c in MECHANISMS[g]]
S_STAR_ECG = [c for g in S_STAR_MECHANISMS if MECH_SOURCE[g] == "ECG" for c in MECHANISMS[g]]
ARMS = {"ECG": S_STAR_ECG, "ECG_ACC": S_STAR}

MODELS = tuple(MODELS_CFG["rungs"])
UQ_MODELS = tuple(MODELS_CFG["posterior_rungs"])
N_DRAWS = int(MODELS_CFG["posterior_draws"])
ALPHA_GRID = tuple(MODELS_CFG["logistic"]["prior_precision_grid"])
ALPHA_MLE = float(MODELS_CFG["logistic"]["a0_prior_precision"])
A2_PRIOR = MODELS_CFG["logistic"]["a2_structure_prior"]
MAX_DEPTH = int(MODELS_CFG["trees"]["max_depth"])
MIN_LEAF = int(MODELS_CFG["trees"]["min_leaf"])
N_CHAINS = int(MODELS_CFG["trees"]["b2_mcmc"]["n_chains"])
CART_ITER = int(MODELS_CFG["trees"]["b2_mcmc"]["n_iter"])
CART_ITER_STRESS = int(MODELS_CFG["trees"]["b2_mcmc_iter_for_shift_and_training_size"])

N_BOOT = int(EVAL_CFG["bootstrap"]["n_resamples"])
N_BOOT_RANK = int(EVAL_CFG["bootstrap"]["n_resamples_rank_correlation"])
BOOT_SEED = int(EVAL_CFG["bootstrap"]["seed"])
K_PRIMARY = int(EVAL_CFG["local_overlap"]["k_primary"])
K_SWEEP = tuple(EVAL_CFG["local_overlap"]["k_sweep"])
EXCLUDED_SUBJECT = str(EVAL_CFG["excluded_subject"])

SNR_DB = tuple(float("inf") if s == "clean" else float(s) for s in SHIFT_CFG["nstdb"]["snr_db"])
NOISE_KINDS = tuple(SHIFT_CFG["nstdb"]["records"])
