# ECG signal-quality uncertainty: code and settings

Every result is a CSV written to `results/rq_<N>_result/`. Every setting that controls a result is
in `config/*.json`.

## 1. Install

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Tested with Python 3.13.

## 2. Data

Download the three PhysioNet databases and put them under `data/` (or edit `config/paths.json`):

| Database | Used for | Expected path |
|---|---|---|
| [BUT QDB](https://physionet.org/content/butqdb/1.0.0/) | development and LOSO evaluation | `data/but-qdb/` (folders `100001/` … `126001/`) |
| [NSTDB](https://physionet.org/content/nstdb/1.0.0/) | noise library for controlled corruption | `data/nstdb/` (`bw`, `em`, `ma` records) |
| [LTSTDB](https://physionet.org/content/ltstdb/1.0.0/) | zero-shot transfer | `data/ltstdb/` (with `RECORDS`) |

To point at data elsewhere without editing `paths.json`, create `config/paths.local.json` with the
same keys; it is used instead when present and is ignored by git.

## 3. Run

```bash
python run.py --rq 0      # features, cohort, LOSO folds, representation screen
python run.py --rq 1      # RQ1  predictive comparability and posterior structure
python run.py --rq 2      # RQ2  operational reliability
python run.py --rq 3      # RQ3  construct validity of U_CE and U_MI
python run.py --rq 4      # RQ4  controlled ECG corruption and zero-shot transfer
python run.py --rq all    # everything, in order

python plot_figures.py    # figures -> results/figures/
```

A step builds what it needs from earlier steps if it is missing (for example `--rq 2` fits the
LOSO models of RQ1 first). Shared intermediate tables are kept in `results/_cache/`.
`plot_figures.py` plots from whichever `rq_<N>_result` folders exist and skips the rest.

## 4. Outputs

| Folder | Main files | Manuscript |
|---|---|---|
| `rq_0_result` | `features.csv`, `dataset_summary.csv`, `candidate_features.csv`, `loso_folds.csv`, `screen_all_subsets.csv`, `screen_mechanism_prevalence.csv` | Tables 1–2, Fig. 2, Table S2 |
| `rq_1_result` | `loso_predictions.csv.gz`, `pooled_performance.csv`, `rung_contrasts.csv`, `per_fold_recall.csv`, `subject_weighting.csv`, `balanced_refit.csv`, `posterior_structure/`, `estimator_diagnostics.csv`, `b2_mcmc_rhat_ess.csv`, `b2_predictive_convergence.csv`, `b2_mcmc_summary.csv`, `representation_sensitivity.csv` | Table 3, Fig. 3, Tables S3–S6, Section S2 |
| `rq_2_result` | `reliability.csv`, `incremental_gains.csv`, `risk_coverage_curves.csv`, `subject_weighting.csv`, `balanced_refit.csv`, `exclude_subject.csv`, `augrc_vs_random.csv` | Table 4, Fig. 4, Fig. S2, Table S7 |
| `rq_3_result` | `expert_agreement.csv`, `local_overlap.csv`, `local_overlap_k_sweep.csv`, `reduced_training.csv`, `subject_weighting.csv`, `balanced_refit.csv` | Fig. 5, Figs S3–S4, Table S8 |
| `rq_4_result` | `noise_stress.csv`, `synthetic_mains.csv`, `muscle_artifact_features.csv`, `leave_one_feature_out.csv`, `ltstdb_state_levels.csv`, `ltstdb_input_novelty.csv`, `ltstdb_mains_by_record.csv` | Figs 6–7, Figs S5–S7, Table S9 |

**Fold definitions.** Leave-one-subject-out over the 15 BUT QDB subjects; fold `k` holds out the
`k`-th subject in sorted order (`rq_0_result/loso_folds.csv`).

### Model estimation and limitations

| Model | Estimation | Limitation |
|---|---|---|
| A0 | Maximum-likelihood multinomial logistic regression | Point estimate; no posterior uncertainty |
| A1 | Gaussian Laplace posterior over coefficients | Approximate posterior; no MCMC |
| A2 | Exact enumeration of all 512 feature subsets, each with a Laplace approximation | Structure enumeration is exact; coefficient posteriors and model evidence are approximate |
| B0 | Single CART tree with empirical leaves | Point estimate; no posterior uncertainty |
| B1 | The B0 tree with conjugate Dirichlet leaf posteriors | Exact given the tree; no tree-structure uncertainty |
| B2 | Bayesian CART, 8 MCMC chains | Chains do not mix (see below) |

B2 uses 8 chains × 2,000 iterations (burn-in 500, thinning 50) for the main analysis and
8 × 1,000 iterations for the noise stress test, synthetic-mains check and reduced-training
experiment
(`b2_mcmc_iter_for_shift_and_training_size` in `config/models.json`).

**B2 convergence.** `rq_1_result/b2_mcmc_rhat_ess.csv` gives rank-normalised split R-hat and bulk ESS
per fold for the tree log marginal likelihood, number of leaves and depth: median R-hat 2.5–3.3
(maximum 5.5), bulk ESS about 12 of 240 retained draws. `b2_predictive_convergence.csv` repeats the
check on the predicted class probabilities of held-out windows (no window with R-hat below 1.1),
reports each chain's agreement with the pooled prediction, and splits B2 mutual information into
between-chain (85%), between-tree (12%) and within-tree (2%) parts. `b2_mcmc_summary.csv`
summarises both. Running the chains ten times longer does not change this; to check, call
`b2_mcmc_diagnostics(features_csv, outdir, n_iter=20000)` from `ecguq/analysis/mcmc.py`.

B2 results are therefore pooled estimates from non-converged chains, not a converged posterior
sample, and B2 mutual information and structural variability mainly reflect differences between
chains. B2 predictive performance remains a held-out evaluation of the pooled model. The
convergence limitation is specific to B2.

## 5. Settings

| File | Contents |
|---|---|
| `config/data.json` | sampling rates, 10-s windows, label rule (full consensus coverage, ≥ 90% modal class) |
| `config/features.json` | the seven mechanism groups and 18 features, the locked representation `E3+E4+M2`, QRS detectors, spectral bands, multiscale entropy (decimation to 50 Hz, scales 1–3, m = 2, r = 0.15·SD), accelerometer and cross-modal settings |
| `config/screening.json` | 5-fold subject-grouped screen, ranking (NLL, Brier, ECE, size) and tie tolerances |
| `config/models.json` | prior-precision grid, A2 prior, tree depth and leaf size, B2 MCMC (8 chains × 2,000 iterations), posterior draws, seeds |
| `config/evaluation.json` | subject bootstrap (1,000; 500 for rank correlations), local-overlap k, training fractions, balanced-refit seeds, excluded subject |
| `config/shift.json` | NSTDB records, SNR ladder and mixing rule, synthetic mains, feature diagnostics, LTSTDB lead, guard interval, state rule and novelty k |

NSTDB mixing: each 10-s ECG window `x` receives an independently offset NSTDB segment `n`,
scaled as `x + λn` with `λ = sqrt(P_x / (P_n · 10^(SNR/10)))` (powers about the window mean).

## 6. Code layout

```
run.py                       runs one research question (or all)
plot_figures.py              plots figures from the results folders
ecguq/config.py              reads config/*.json
ecguq/data/                  BUT QDB windows and labels, NSTDB noise
ecguq/features/              ECG (E1-E4), accelerometer (M1-M2) and cross-modal (M3) features
ecguq/models/                logistic A0-A2, trees B0-B2, the LOSO ladder
ecguq/uncertainty.py         U_pred, U_CE, U_MI, U_MS
ecguq/metrics.py             metrics, risk-coverage, subject bootstrap
ecguq/analysis/              screening, performance, reliability, overlap, weighting, diagnostics, MCMC
ecguq/shift/                 noise stress, feature ablation, LTSTDB transfer, mains check
ecguq/rq/                    one module per research question (rq0 ... rq4)
```
