"""Posterior structure summaries: A2 feature inclusion and B2 feature use and topology."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta
from sklearn.tree import DecisionTreeClassifier

from ..config import ALPHA_GRID, EPS, N_CLASSES, N_JOBS, S_STAR
from ..models.ladder import prepare_fold
from ..models.logistic import (_design, chol_pd, fit_map, laplace_precision, sample_theta,
                            select_alpha_evidence)
from ..models.structure import fit_structures
from ..models.trees import BayesCART


FEATURE_TEX = {
    "ECG_pSQI": r"$\mathrm{pSQI}$",
    "ECG_basSQI": r"$\mathrm{basSQI}$",
    "ECG_HF_ratio": r"$r_{\mathrm{HF}}$",
    "ECG_line50_ratio": r"$r_{50}$",
    "ECG_kSQI": r"$\mathrm{kSQI}$",
    "ECG_MSE": r"$\mathrm{MSE}$",
    "ACC_jerk_rms": r"$\mathrm{jerk}$",
    "ACC_dom_freq": r"$f_{\mathrm{dom}}$",
    "ACC_spec_entropy": r"$H_{\mathrm{spec}}$",
}
DIRICHLET_ALPHA = 1.0
LEAF_QUANTILES = (0.05, 0.50, 0.95)
LEAF_KIND = ("pure", "moderate", "diffuse")


def _entropy(p, axis=-1):
    p = np.clip(np.asarray(p, float), EPS, 1.0)
    return -(p * np.log(p)).sum(axis)


def _linear_fold(Ztr, ytr, n_draws, rng, n_jobs):
    alpha, _ = select_alpha_evidence(Ztr, ytr, ALPHA_GRID)
    theta, _ = fit_map(Ztr, ytr, alpha)
    chol = chol_pd(laplace_precision(theta, _design(Ztr), alpha))
    draws = sample_theta(theta, chol, n_draws, rng)

    d = Ztr.shape[1]

    W = draws.reshape(len(draws), N_CLASSES - 1, d + 1)[:, :, :d]

    structures, _, weights, info = fit_structures(Ztr, ytr, alpha, n_jobs=n_jobs)
    sizes = np.array([len(s) for s in structures])
    size_mass = np.array([weights[sizes == k].sum() for k in range(d + 1)])
    return W, np.asarray(info["inclusion"], float), size_mass, alpha


def _b1_leaves(Ztr, ytr, max_depth, min_leaf, seed):
    t0 = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf,
                                random_state=seed).fit(Ztr, ytr)
    leaf = t0.apply(Ztr)
    ids = np.unique(leaf)
    counts = np.zeros((len(ids), N_CLASSES))
    for i, lid in enumerate(ids):
        counts[i] = np.bincount(ytr[leaf == lid], minlength=N_CLASSES)

    a0 = DIRICHLET_ALPHA / N_CLASSES
    post = counts + a0
    mean = post / post.sum(1, keepdims=True)
    H = _entropy(mean)


    n = counts.sum(1)
    order = np.argsort(H)
    cum = np.cumsum(n[order]) / n.sum()
    picks = [int(order[np.searchsorted(cum, q)]) for q in LEAF_QUANTILES]
    return post, mean, H, n, picks, len(ids)


def _b2_structure(Ztr, ytr, cart_iter, max_depth, min_leaf, seed):
    cart = BayesCART(max_depth=max_depth, min_leaf=min_leaf, seed=seed).fit(
        Ztr, ytr, n_iter=cart_iter, burn=cart_iter // 4, thin=max(1, cart_iter // 40))
    S = cart.samples_
    d = Ztr.shape[1]

    root = np.zeros(d)
    used = np.zeros(d)
    for s in S:
        root[int(s["feat"][0])] += 1
        for j in {int(f) for f in s["feat"] if f >= 0}:
            used[j] += 1

    key = [(tuple(int(v) for v in s["feat"]), tuple(int(v) for v in s["t"])) for s in S]
    freq = Counter(key)


    by_root: dict[int, list] = {}
    for s, k in zip(S, key):
        by_root.setdefault(int(s["feat"][0]), []).append((freq[k], s))
    leading = sorted(by_root, key=lambda j: -root[j])[:3]
    top = [(j, root[j] / len(S), max(by_root[j], key=lambda t: t[0])[1]) for j in leading]
    return cart, root / len(S), used / len(S), top, len(S), len(freq)


def _sketch(tree, thr, depth_limit=1):
    rows = []

    def rec(node, depth, slot):
        feat, left, right = int(tree["feat"][node]), int(tree["left"][node]), int(tree["right"][node])
        leaf = feat < 0
        below = int((tree["feat"][_subtree(tree, node)] < 0).sum())
        rows.append({"depth": depth, "slot": slot, "leaf": leaf,
                     "feature": None if leaf else S_STAR[feat],
                     "threshold": None if leaf else float(thr[feat][int(tree["t"][node])]),
                     "leaves_below": below,
                     "truncated": (not leaf) and depth == depth_limit})
        if leaf or depth == depth_limit:
            return
        rec(left, depth + 1, 2 * slot)
        rec(right, depth + 1, 2 * slot + 1)

    rec(0, 0, 0)
    return rows


def _subtree(tree, node):
    out, stack = [], [node]
    while stack:
        i = stack.pop()
        out.append(i)
        if tree["feat"][i] >= 0:
            stack += [int(tree["left"][i]), int(tree["right"][i])]
    return np.array(out, int)


def posterior_shapes(features_csv: Path, outdir: Path, n_draws: int = 400,
                     cart_iter: int = 2000, max_depth: int = 12, min_leaf: int = 10,
                     folds: int | None = None) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv, dtype={"record": str, "subject": str})
    y = df["label"].to_numpy(int)
    if y.min() == 1:
        y = y - 1
    subjects = sorted(df.subject.unique())
    if folds:
        subjects = subjects[:folds]
    d = len(S_STAR)

    coef, incl, size, n_train = [], [], [], []
    root_freq, used_freq = [], []
    for k, held in enumerate(subjects):
        te = (df.subject == held).to_numpy()
        Ztr, _ = prepare_fold(df[~te], df[te], S_STAR)
        ytr = y[~te]
        rng = np.random.default_rng(31000 + k)

        W, pip, size_mass, alpha = _linear_fold(Ztr, ytr, n_draws, rng, N_JOBS)
        coef.append(W)
        incl.append(pip)
        size.append(size_mass)
        n_train.append(len(ytr))

        _, rf, uf, _, n_s, n_d = _b2_structure(Ztr, ytr, cart_iter, max_depth, min_leaf, k)
        root_freq.append(rf)
        used_freq.append(uf)
        print(f"  fold {held:<5} alpha {alpha:<6g} min PIP {pip.min():.3f} "
              f"E[size] {float((np.arange(d + 1) * size_mass).sum()):.2f} | "
              f"B2 {n_d}/{n_s} distinct topologies, top root split "
              f"{S_STAR[int(np.argmax(rf))]} at {rf.max():.2f}", flush=True)


    ref_i = int(np.argsort(n_train)[len(n_train) // 2])
    ref = subjects[ref_i]
    te = (df.subject == ref).to_numpy()
    Ztr, _ = prepare_fold(df[~te], df[te], S_STAR)
    ytr = y[~te]
    print(f"\n  reference fold for the leaf and topology panels: held-out {ref} "
          f"({n_train[ref_i]} training windows, the median fold)", flush=True)

    post, lmean, lH, ln, picks, n_leaves = _b1_leaves(Ztr, ytr, max_depth, min_leaf, ref_i)
    cart, rf, uf, top, n_s, n_d = _b2_structure(Ztr, ytr, cart_iter, max_depth, min_leaf, ref_i)


    C = np.concatenate(coef, axis=0)
    rows = []
    for c in range(N_CLASSES - 1):
        for j in range(d):
            v = C[:, c, j]
            rows.append({"feature": S_STAR[j], "class": c,
                         "q025": float(np.percentile(v, 2.5)),
                         "q250": float(np.percentile(v, 25)),
                         "q500": float(np.percentile(v, 50)),
                         "q750": float(np.percentile(v, 75)),
                         "q975": float(np.percentile(v, 97.5))})
    pd.DataFrame(rows).to_csv(outdir / "p01_a1_coefficients.csv", index=False)


    I = np.stack(incl)
    pd.DataFrame({"feature": S_STAR, "pip_mean": I.mean(0),
                  "pip_lo": I.min(0), "pip_hi": I.max(0)}).to_csv(
        outdir / "p02_a2_inclusion.csv", index=False)
    Sz = np.stack(size)
    pd.DataFrame({"size": np.arange(d + 1), "weight_mean": Sz.mean(0),
                  "weight_lo": Sz.min(0), "weight_hi": Sz.max(0)}).to_csv(
        outdir / "p03_a2_model_size.csv", index=False)


    rows = []
    for kind, q, li in zip(LEAF_KIND, LEAF_QUANTILES, picks):
        a = post[li]
        tot = a.sum()
        for c in range(N_CLASSES):
            m = a[c] / tot

            lo, hi = _beta.ppf([0.025, 0.975], a[c], tot - a[c])
            rows.append({"kind": kind, "quantile": q, "leaf": int(li), "n_train": int(ln[li]),
                         "entropy": float(lH[li]), "class": c, "mean": float(m),
                         "lo": float(lo), "hi": float(hi)})
    pd.DataFrame(rows).to_csv(outdir / "p04_b1_leaves.csv", index=False)


    rows = []
    for rank, (j, w, tree) in enumerate(top):
        for r in _sketch(tree, cart.thr_):
            rows.append({"rank": rank, "root": S_STAR[j], "p_root": w,
                         "n_leaves": int(tree["n_leaves"]),
                         "max_depth": int(tree["depth"].max()), **r})
    pd.DataFrame(rows).to_csv(outdir / "p05_b2_trees.csv", index=False)

    R, U = np.stack(root_freq), np.stack(used_freq)
    pd.DataFrame({"feature": S_STAR, "p_root_mean": R.mean(0), "p_root_lo": R.min(0),
                  "p_root_hi": R.max(0), "p_used_mean": U.mean(0)}).to_csv(
        outdir / "p06_b2_splits.csv", index=False)

    meta = {"n_folds": len(subjects), "n_draws": n_draws, "ref_subject": ref,
            "ref_n_train": int(n_train[ref_i]), "b0_leaves": int(n_leaves),
            "b2_samples": int(n_s), "b2_distinct": int(n_d),
            "b2_top_root": S_STAR[top[0][0]], "b2_top_root_p": float(top[0][1]),
            "b2_top_weight": float(max(Counter(
                (tuple(int(v) for v in t["feat"]), tuple(int(v) for v in t["t"]))
                for t in cart.samples_).values()) / n_s),
            "a2_min_pip": float(I.mean(0).min()), "a2_max_pip": float(I.mean(0).max()),
            "a2_expected_size": float((np.arange(d + 1) * Sz.mean(0)).sum())}
    pd.DataFrame([meta]).to_csv(outdir / "p07_meta.csv", index=False)

    print(f"\n  A2 inclusion {meta['a2_min_pip']:.2f}-{meta['a2_max_pip']:.2f}, "
          f"E[|gamma|] {meta['a2_expected_size']:.2f} of {d}")
    print(f"  B2 {meta['b2_distinct']} distinct topologies in {meta['b2_samples']} draws; "
          f"the leading one holds {100 * meta['b2_top_weight']:.1f}%; "
          f"leading root split {meta['b2_top_root']} at {meta['b2_top_root_p']:.2f}")
    print(f"  posterior-shape tables -> {outdir}")
    return meta
