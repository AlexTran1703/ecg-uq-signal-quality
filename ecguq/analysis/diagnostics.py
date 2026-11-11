"""Computational checks: optimiser, Laplace curvature, A2 enumeration and B2 MCMC chains."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ALPHA_GRID, ALPHA_MLE, N_CLASSES, N_JOBS, S_STAR
from ..models.ladder import prepare_fold
from ..models.logistic import (_design, _map_obj, fit_map, laplace_precision,
                            predict_proba, select_alpha_evidence)
from ..models.structure import fit_structure_averaged
from ..models.trees import BayesCART

MAXITER = 500
N_RESTARTS = 5
N_DRAWS = 100


def _fold(df, y, held, seed, cart_iter=2000, max_depth=12, min_leaf=10):
    te = (df.subject == held).to_numpy()
    Ztr, Zte = prepare_fold(df[~te], df[te], S_STAR)
    ytr = y[~te]
    Xa = _design(Ztr)
    Y = np.zeros((len(ytr), N_CLASSES))
    Y[np.arange(len(ytr)), ytr] = 1.0
    row = {"subject": held, "n_train": int(len(ytr))}


    th0, r0 = fit_map(Ztr, ytr, ALPHA_MLE)
    _, g0 = _map_obj(th0, Xa, Y, ALPHA_MLE)
    H0 = laplace_precision(th0, Xa, ALPHA_MLE)
    lam0 = float(np.linalg.eigvalsh(H0).min())
    p0 = predict_proba(Zte, th0)


    alpha, ev = select_alpha_evidence(Ztr, ytr, ALPHA_GRID)
    th1, r1 = fit_map(Ztr, ytr, alpha)
    _, g1 = _map_obj(th1, Xa, Y, alpha)
    H1 = laplace_precision(th1, Xa, alpha)
    e1 = np.linalg.eigvalsh(H1)
    p1 = predict_proba(Zte, th1)

    rng = np.random.default_rng(90000 + seed)
    starts = [np.zeros_like(th1), th1 * (1.0 + 1e-12)]
    while len(starts) < N_RESTARTS:
        starts.append(rng.normal(0.0, 0.5, size=th1.shape))
    TH, objs, P = [], [], []
    for s in starts:
        th, _ = fit_map(Ztr, ytr, alpha, theta0=s)
        TH.append(th)
        objs.append(_map_obj(th, Xa, Y, alpha)[0])
        P.append(predict_proba(Zte, th))
    TH, P = np.stack(TH), np.stack(P)

    row.update({
        "a0_iters": int(r0.nit), "a0_hit_cap": bool(r0.nit >= MAXITER),
        "a0_grad_inf": float(np.abs(g0).max()),
        "a0_hess_lam_min": lam0,
        "a0_theta_norm": float(np.linalg.norm(th0)),
        "a0_vs_a1_agree": float(np.mean(p0.argmax(1) == p1.argmax(1))),
        "a0_vs_a1_maxdp": float(np.abs(p0 - p1).max()),
        "alpha": alpha, "alpha_log_evidence": ev,
        "a1_iters": int(r1.nit),
        "a1_grad_inf": float(np.abs(g1).max()),
        "a1_theta_norm": float(np.linalg.norm(th1)),
        "a1_shrinkage": float(np.linalg.norm(th1) / max(np.linalg.norm(th0), 1e-12)),
        "a1_hess_lam_min": float(e1.min()), "a1_hess_lam_max": float(e1.max()),
        "a1_hess_cond": float(e1.max() / e1.min()) if e1.min() > 0 else float("inf"),
        "a1_hess_pd": bool(e1.min() > 0),
        "a1_obj_spread": float(max(objs) - min(objs)),
        "a1_theta_dist_max": float(np.linalg.norm(TH[:, None] - TH[None, :], axis=-1).max()),
        "a1_pred_spread_max": float(np.abs(P[:, None] - P[None, :]).max()),
        "a1_argmax_disagree": float(np.mean(P.argmax(-1) != P.argmax(-1)[0:1])),
    })


    _, info = fit_structure_averaged(Ztr, ytr, Zte, alpha, N_DRAWS, seed,
                                     n_jobs=N_JOBS)
    row.update({
        "a2_n_structures": info["n_structures"],
        "a2_top_size": info["top_size"],
        "a2_top_weight": info["top_weight"],
        "a2_second_weight": info["second_weight"],
        "a2_effective_structures": info["effective_structures"],
        "a2_expected_size": info["expected_size"],
        "a2_logev_gap": info["logev_gap"],
        "a2_sampled": info["n_structures_sampled"],
        "a2_min_inclusion": float(np.min(info["inclusion"])),
    })


    cart = BayesCART(max_depth=max_depth, min_leaf=min_leaf, seed=seed)


    cart.fit(Ztr, ytr, n_iter=cart_iter, burn=cart_iter // 4,
             thin=max(1, cart_iter // 40))
    cs = pd.DataFrame(cart.chain_summary_)
    row.update({
        "b2_chains": len(cs),
        "b2_accept": float(cs.accept.mean()),
        "b2_distinct_total": int(cs.n_distinct.sum()),
        "b2_leaves_mean": float(cs.leaves_mean.mean()),
        "b2_leaves_between_chain_sd": float(cs.leaves_mean.std(ddof=0)),
        "b2_leaves_spread": float(cs.leaves_mean.max() - cs.leaves_mean.min()),
        "b2_depth_mean": float(cs.depth_mean.mean()),
        "b2_depth_spread": float(cs.depth_mean.max() - cs.depth_mean.min()),
    })
    return row, cart.chain_summary_


def run_diagnostics(features_csv, out_dir) -> pd.DataFrame:
    df = pd.read_csv(features_csv, dtype={"record": str, "subject": str})
    y = df["label"].to_numpy(int)
    if y.min() == 1:
        y = y - 1
    subjects = sorted(df.subject.unique())

    rows, chains = [], []
    for k, held in enumerate(subjects):
        r, cs = _fold(df, y, held, k)
        rows.append(r)
        for c in cs:
            chains.append({"subject": held, **c})
        print("  fold %-5s A0 iters %3d%s lam_min %.3g | A1 alpha %-5g cond %.2g dp %.1e | "
              "A2 eff %.2f E[size] %.2f | B2 leaves %.1f +- %.1f"
              % (held, r["a0_iters"], "*" if r["a0_hit_cap"] else " ", r["a0_hess_lam_min"],
                 r["alpha"], r["a1_hess_cond"], r["a1_pred_spread_max"],
                 r["a2_effective_structures"], r["a2_expected_size"],
                 r["b2_leaves_mean"], r["b2_leaves_between_chain_sd"]), flush=True)

    t = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir / "estimator_diagnostics.csv", index=False)
    pd.DataFrame(chains).to_csv(out_dir / "b2_chain_diagnostics.csv", index=False)

    print("\n  A0 unpenalised Hessian positive definite in %d/%d folds; cap reached in %d"
          % (int((t.a0_hess_lam_min > 0).sum()), len(t), int(t.a0_hit_cap.sum())))
    print("  A1 Hessian positive definite in %d/%d; max prediction spread across restarts %.2e"
          % (int(t.a1_hess_pd.sum()), len(t), t.a1_pred_spread_max.max()))
    print("  A2 effective structures %.2f-%.2f (median %.2f); expected active features %.1f-%.1f"
          % (t.a2_effective_structures.min(), t.a2_effective_structures.max(),
             t.a2_effective_structures.median(),
             t.a2_expected_size.min(), t.a2_expected_size.max()))
    print("  B2 between-chain SD of mean leaf count %.2f-%.2f"
          % (t.b2_leaves_between_chain_sd.min(), t.b2_leaves_between_chain_sd.max()))
    print("\n  wrote", out_dir / "estimator_diagnostics.csv")
    return t
