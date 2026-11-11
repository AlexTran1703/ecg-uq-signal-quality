"""Plot the figures from whichever results/rq_<N>_result folders exist.

    python plot_figures.py

A figure is skipped, with a message, when the RQ it needs has not been run. Output: results/figures/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ecguq.config import K_PRIMARY, MODELS, RESULTS, UQ_MODELS
from ecguq.data.nstdb import NOISE_KINDS, NOISE_LABEL, SNR_DB

ARM = "ECG_ACC"


BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#ffffff"
SCORE_COLOR = {"U_pred": BLUE, "U_CE": ORANGE, "U_MI": AQUA, "softmax": MUTED}
SCORE_MARKER = {"U_pred": "s", "U_CE": "o", "U_MI": "^", "softmax": "D"}
SCORE_LABEL = {"U_pred": r"$U_{\mathrm{pred}}$", "U_CE": r"$U_{\mathrm{CE}}$",
               "U_MI": r"$U_{\mathrm{MI}}$", "softmax": r"$U_{\mathrm{MS}}$"}
LINEAR_RAMP = ("#efa079", "#eb6834", "#8f3a17")
TREE_RAMP = ("#86b6ef", "#2a78d6", "#104281")
MODEL_COLOR = {"A0": LINEAR_RAMP[0], "A1": LINEAR_RAMP[1], "A2": LINEAR_RAMP[2],
               "B0": TREE_RAMP[0], "B1": TREE_RAMP[1], "B2": TREE_RAMP[2]}
FONT_SCALE = 1.35
LINE_W = {"A1": 4.4, "A2": 1.9, "B1": 2.2, "B2": 2.2}
FIGURES = RESULTS / "figures"


def style():
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 8.5 * FONT_SCALE, "axes.labelsize": 8.5 * FONT_SCALE,
        "axes.titlesize": 9.0 * FONT_SCALE, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 7.5 * FONT_SCALE, "ytick.labelsize": 7.5 * FONT_SCALE,
        "legend.frameon": False, "legend.fontsize": 7.5 * FONT_SCALE,
        "grid.color": GRID, "grid.linewidth": 0.7, "lines.linewidth": 2.0,
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", pad=6)


def save(fig, name):
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"  figures/{name}.png", flush=True)


SOURCE_COLOR = {"ECG": BLUE, "ACC": ORANGE, "cross-modal": MUTED}


def screening(src: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style()
    t = pd.read_csv(src / "screen_all_subsets.csv")
    pres = pd.read_csv(src / "screen_mechanism_prevalence.csv")
    star = str(t[t.selected].iloc[0].group)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.4))


    ax = axes[0]
    grid(ax)
    ax.scatter(t.n_features, t.NLL, s=16, facecolor=MUTED, alpha=0.45, edgecolor=SURFACE,
               linewidths=0.5, zorder=3)
    best = t.loc[t.groupby("n_features").NLL.idxmin()].sort_values("n_features")
    ax.plot(best.n_features, best.NLL, color=AXIS, lw=1.2, zorder=4)
    s = t[t.selected].iloc[0]
    ax.scatter([s.n_features], [s.NLL], s=64, marker="D", facecolor=LINEAR_RAMP[2],
               edgecolor=SURFACE, linewidths=0.8, zorder=6)
    ax.annotate(r"$S^{\star}$", (s.n_features, s.NLL), xytext=(7, -2),
                textcoords="offset points", fontsize=8 * FONT_SCALE, color=LINEAR_RAMP[2], weight="semibold")
    ax.set_xlabel("number of features")
    ax.set_ylabel("NLL  (lower is better)")
    panel(ax, "a", "all 127 combinations")


    ax = axes[1]
    grid(ax, "x")
    pres = pres.sort_values("share_top")
    yy = np.arange(len(pres))
    cols = [SOURCE_COLOR[s] for s in pres.source]
    ax.barh(yy, pres.share_top, height=0.62, color=cols, edgecolor=SURFACE, linewidth=0.6,
            zorder=3)
    null = float(pres.null.iloc[0])
    ax.axvline(null, color=AXIS, lw=1.2, zorder=4)
    ax.annotate(f"size-matched\nnull {null:.2f}", (null, 2.5), xytext=(5, 0), va="center",
                textcoords="offset points", fontsize=6.4 * FONT_SCALE, color=INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{m}  {d}" for m, d in zip(pres.mechanism, pres.desc)], fontsize=7.0 * FONT_SCALE,
                       color=INK)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("share of the strongest subsets containing it")
    panel(ax, "b", "prevalence among leading models")
    ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=6, color=c, label=k)
                       for k, c in SOURCE_COLOR.items()],
              loc="lower right", fontsize=6.8 * FONT_SCALE)


    ax = axes[2]
    grid(ax, "x")
    d = t.nsmallest(10, "NLL").iloc[::-1]
    yy = np.arange(len(d))
    cols = [LINEAR_RAMP[2] if g == star else TREE_RAMP[1] for g in d.group]
    ax.barh(yy, d.NLL, height=0.62, color=cols, edgecolor=SURFACE, linewidth=0.6, zorder=3)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.annotate(f"{r.ECE:.3f}", (r.NLL, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=6.2 * FONT_SCALE, color=INK2)
    ax.set_xlim(0, float(d.NLL.max()) * 1.20)
    lbl = [g + (r"  $S^{\star}$" if g == star else "") for g in d.group]
    ax.set_yticks(yy)
    ax.set_yticklabels(lbl, fontsize=6.4 * FONT_SCALE, color=INK)
    ax.set_xlabel("NLL  (lower is better)")
    panel(ax, "c", "leading combinations")

    fig.suptitle("Mechanism-level representation screening", x=0.005, ha="left",
                 fontsize=10 * FONT_SCALE, weight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig2_screening")


SCORES = ("U_pred", "U_CE", "U_MI")
PANELS = [("d_auroc", "a", r"$\Delta$AUROC$_{\mathrm{err}}$", "error detection"),
          ("d_augrc_gain", "b", r"$\Delta$AUGRC$_{\mathrm{gain}}$", "selective prediction")]


def operational_reliability(src: Path, arm: str = "ECG_ACC"):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t = pd.read_csv(src / "incremental_gains.csv")
    t = t[t.arm == arm]
    models = [m for m in UQ_MODELS if (t.model == m).any()]

    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharex=True)
    for ax, (col, letter, ylab, what) in zip(axes, PANELS):
        grid(ax)
        if col not in t.columns:
            continue
        for j, s in enumerate(SCORES):
            xs, ys, los, his = [], [], [], []
            for i, m in enumerate(models):
                r = t[(t.model == m) & (t.score == s)]
                if not len(r) or not np.isfinite(r[col].iloc[0]):
                    continue
                xs.append(i + (j - 1) * 0.22)
                ys.append(float(r[col].iloc[0]))
                los.append(float(r[f"{col}_lo"].iloc[0]))
                his.append(float(r[f"{col}_hi"].iloc[0]))
            if not xs:
                continue
            ys = np.asarray(ys)
            ax.errorbar(xs, ys, yerr=[ys - np.asarray(los), np.asarray(his) - ys],
                        fmt=SCORE_MARKER[s], ms=6, lw=0, elinewidth=1.4, capsize=2.6,
                        color=SCORE_COLOR[s], mec=SURFACE, mew=0.9, zorder=3)

        ax.axhline(0, color=INK, lw=1.5, zorder=2)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.set_ylabel(ylab)
        panel(ax, letter, what)

    h = [Line2D([], [], marker=SCORE_MARKER[s], ls="", ms=6, color=SCORE_COLOR[s], mec=SURFACE,
                label=SCORE_LABEL[s]) for s in SCORES]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.045), columnspacing=2.2)
    fig.suptitle(r"Operational reliability relative to same-model $U_{\mathrm{MS}}$  (above zero = better)",
                 x=0.005, ha="left", fontsize=10 * FONT_SCALE, weight="semibold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    save(fig, "fig4_operational")


TERMS = ("U_CE", "U_MI")


def _expert(work: Path):
    t = pd.read_csv(work / "expert_agreement.csv")
    t = t[t.config.str.endswith("+")].copy()
    t["model"] = t.config.str.rstrip("+")
    return t.set_index(["model", "score"])


def _overlap(work: Path):
    src = work / "local_overlap.csv"
    if not src.exists():
        return None
    t = pd.read_csv(src)
    t = t[(t.arm == ARM) & (t.k == K_PRIMARY)]
    return t.set_index(["model", "score"])


def _training(work: Path):
    src = work / "reduced_training.csv"
    if not src.exists():
        return None
    t = pd.read_csv(src)
    t = t[t.arm == ARM]
    lo, hi = t.fraction.min(), t.fraction.max()
    out = {}
    for m in UQ_MODELS:
        g = t[t.model == m]
        a, b = g[g.fraction == lo], g[g.fraction == hi]
        if len(a) and len(b):


            out[m] = {s: float(a[s].iloc[0] - b[s].iloc[0]) for s in TERMS}
    return out, lo, hi


def _corr_panel(ax, table, models, col, lo_col, hi_col, ylab, letter, title):
    grid(ax)
    ax.axhline(0, color=INK, lw=1.5, zorder=2)
    for j, s in enumerate(TERMS):
        xs, ys, lo, hi = [], [], [], []
        for i, m in enumerate(models):
            if (m, s) not in table.index:
                continue
            r = table.loc[(m, s)]
            xs.append(i + (j - 0.5) * 0.24)
            ys.append(float(r[col]))
            lo.append(float(r[lo_col]))
            hi.append(float(r[hi_col]))
        if not xs:
            continue
        ys = np.asarray(ys)
        ax.errorbar(xs, ys, yerr=[ys - np.asarray(lo), np.asarray(hi) - ys], fmt=SCORE_MARKER[s], ms=6,
                    lw=0, elinewidth=1.4, capsize=2.6, color=SCORE_COLOR[s], mec=SURFACE,
                    mew=0.9, zorder=3)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylabel(ylab)
    panel(ax, letter, title)


def construct_validity(work: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    exp = _expert(work)
    ovl = _overlap(work)
    tr = _training(work)
    if ovl is None or tr is None:
        print("  (fig5 needs t10_local_overlap.csv and s3_training_size.csv -- run `overlap` "
              "and the stress stage)", flush=True)
        return
    tr, frac_lo, frac_hi = tr
    models = list(UQ_MODELS)

    style()
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.8))

    _corr_panel(axes[0], exp, models, "spearman_H", "spearman_lo", "spearman_hi",
                r"$\rho(U, H_{\mathrm{expert}})$", "a", "annotator ambiguity")
    _corr_panel(axes[1], ovl, models, "spearman_local", "lo", "hi",
                r"$\rho(U, H_{\mathrm{local}})$", "b",
                f"local class overlap ($k={K_PRIMARY}$)")

    ax = axes[2]
    grid(ax)
    ax.axhline(0, color=INK, lw=1.5, zorder=2)


    for j, s in enumerate(TERMS):
        xs = np.array([i + (j - 0.5) * 0.24 for i in range(len(models))])
        ys = np.array([tr.get(m, {}).get(s, np.nan) for m in models])
        ax.vlines(xs, 0, ys, color=SCORE_COLOR[s], lw=1.4, zorder=3)
        ax.plot(xs, ys, SCORE_MARKER[s], ms=6, color=SCORE_COLOR[s], mec=SURFACE, mew=0.9,
                zorder=4)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylabel(r"$\Delta U$ (%d%% vs %d%% training)"
                  % (round(frac_lo * 100), round(frac_hi * 100)))
    panel(ax, "c", "reduced training information")

    h = [Line2D([], [], marker=SCORE_MARKER[s], ls="", ms=6, color=SCORE_COLOR[s], mec=SURFACE,
                label=SCORE_LABEL[s]) for s in TERMS]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.045), columnspacing=2.4)
    fig.suptitle("Construct validity of the posterior uncertainty decomposition", x=0.005,
                 ha="left", fontsize=10 * FONT_SCALE, weight="semibold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    save(fig, "fig5_construct")


def _axis(ax, levels):
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(["clean" if not np.isfinite(s) else f"{s:g}" for s in levels],
                       fontsize=7.2 * FONT_SCALE)
    ax.set_xlim(-0.25, len(levels) - 0.75)


def noise_stress(work: Path, arm: str = ARM):
    _noise(work, (arm, "ECG"), keep=(0, 2), name="fig6_noise_stress",
           title="Sensitivity of controlled ECG corruption to model input representation "
                 "(noise added to the ECG only)")


def _noise(work: Path, arm, keep: tuple, name: str, title: str):
    import matplotlib.pyplot as plt

    src = work / "noise_stress.csv"
    if not src.exists():
        print("  (no s2_noise_stress.csv -- run the stress stage)", flush=True)
        return
    d = pd.read_csv(src)
    arms = (arm,) if isinstance(arm, str) else tuple(arm)
    d = d[d.arm.isin(arms)]
    if not len(d):
        return
    kinds = [k for k in NOISE_KINDS if (d.kind == k).any()]
    levels = [s for s in SNR_DB if (d.snr_db == s).any() or (not np.isfinite(s)
                                                             and d.snr_db.isna().any())]

    d["snr_db"] = d.snr_db.fillna(np.inf)
    pos = {s: i for i, s in enumerate(levels)}


    ROWS = ((0, "S_Q", r"$S_Q=0\,p(Q1)+1\,p(Q2)+2\,p(Q3)$", False),
            (1, "U_CE", r"$U_{\mathrm{CE}}$ (nats)", False),
            (2, "U_CE", r"$\Delta U_{\mathrm{CE}}$ vs. clean (nats)", True))
    ROWS = tuple((i, *r[1:]) for i, r in enumerate(r for r in ROWS if r[0] in keep))

    style()
    fig, axes = plt.subplots(len(ROWS), len(kinds), figsize=(2.9 * len(kinds),
                                                              2.5 * len(ROWS) + 0.6),
                             sharex=True, sharey="row", squeeze=False)
    for col, kind in enumerate(kinds):
        g = d[d.kind == kind]
        for r, ycol, ylab, relative in ROWS:
            ax = axes[r][col]
            grid(ax)
            if relative:
                ax.axhline(0, color=INK, lw=1.4, zorder=2)
            for z, (a_i, m) in enumerate((a_, m_) for a_ in arms for m_ in UQ_MODELS):
                gg = g[(g.model == m) & (g.arm == a_i)].copy()
                if not len(gg):
                    continue
                dashed = a_i != arms[0]
                gg["x"] = gg.snr_db.map(pos)
                gg = gg.sort_values("x")
                y = gg[ycol].to_numpy(float)
                if relative:
                    base = gg.loc[~np.isfinite(gg.snr_db), ycol]
                    if not len(base):
                        continue
                    y = y - float(base.iloc[0])
                ax.plot(gg.x, y, marker="o", ms=3.6 if dashed else 4.5, lw=LINE_W[m],
                        color=MODEL_COLOR[m], ls=(0, (3.0, 1.6)) if dashed else "-",
                        mfc=SURFACE if dashed else MODEL_COLOR[m],
                        solid_capstyle="round", zorder=3 + z)
            _axis(ax, levels)
            if r == 0:
                panel(ax, "abc"[col], NOISE_LABEL[kind])
            elif len(ROWS) > 1:
                panel(ax, "def"[col], "")
            if r == len(ROWS) - 1:
                ax.set_xlabel("SNR (dB)")
            if col == 0:
                ax.set_ylabel(ylab)

    h = [plt.Line2D([], [], color=MODEL_COLOR[m], lw=LINE_W[m], marker="o", ms=4.5, label=m)
         for m in UQ_MODELS]
    if len(arms) > 1:
        h += [plt.Line2D([], [], color=INK2, lw=1.6, label=r"solid: multimodal model $S^{\star}$ (ACC unchanged)"),
              plt.Line2D([], [], color=INK2, lw=1.6, ls=(0, (3.0, 1.6)), marker="o", ms=3.6,
                         mfc=SURFACE, label=r"dashed: ECG-only model $S^{\star}_{\mathrm{ECG}}$")]
    fig.legend(handles=h, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, -0.018),
               frameon=False, columnspacing=2.0)
    fig.suptitle(title, x=0.005, ha="left", fontsize=10 * FONT_SCALE, weight="semibold")
    fig.tight_layout(rect=(0, 0.05 if len(ROWS) < 3 else 0.03, 1, 0.94 if len(ROWS) < 3
                           else 0.955))
    save(fig, name)


STATES = ["C", "T", "U"]
STATE_LABEL = ["clean", "transition", "unreadable"]


def _levels(ax, lv, score, ylabel, letter, title):
    grid(ax, "y")
    for m in MODELS:
        d = lv[(lv.score == score) & (lv.model == m)]
        y = [d[d.state == st]["median"].median() for st in STATES]
        ax.plot(range(3), y, marker="o", ms=6, lw=1.8, color=MODEL_COLOR[m],
                mec=SURFACE, mew=1.2, zorder=3, label=m)
    ax.set_xticks(range(3))
    ax.set_xticklabels(STATE_LABEL)
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylabel(ylabel)
    panel(ax, letter, title)


def _novelty(ax, nov, letter):
    grid(ax, "y")
    P = nov.pivot(index="subject", columns="state", values="novelty_pct").reindex(
        columns=STATES).to_numpy(float)
    x = np.arange(3)
    for row in P:
        ok = np.isfinite(row)
        ax.plot(x[ok], row[ok], ls="-" if ok.sum() >= 2 else "none", color=MUTED, lw=0.9,
                alpha=0.55, marker="o", ms=3.0, zorder=2)
    ax.plot(x, np.nanmedian(P, axis=0), color=INK, lw=2.2, marker="o", ms=6,
            mec=SURFACE, mew=1.2, zorder=4, label="median over subjects")
    ax.axhline(0.5, color=AXIS, lw=0.9, ls=(0, (4.0, 3.0)), zorder=1)
    ax.annotate("median held-out BUT QDB window", (2.22, 0.5), xytext=(0, -3),
                textcoords="offset points", ha="right", va="top", fontsize=6.4 * FONT_SCALE,
                color=MUTED)
    ax.set_xticks(range(3))
    ax.set_xticklabels(STATE_LABEL)
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("distance percentile to BUT QDB")
    ax.legend(loc="upper left", fontsize=6.8 * FONT_SCALE, frameon=False)
    panel(ax, letter, "input novelty relative to BUT QDB")


def ltstdb_transfer(src: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style()
    lv = pd.read_csv(src / "ltstdb_state_levels.csv", dtype={"subject": str})
    nov = pd.read_csv(src / "ltstdb_input_novelty.csv", dtype={"subject": str})
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(9.2, 3.8))
    _levels(ax_a, lv, "S_Q", r"$S_Q=0\,p(Q1)+1\,p(Q2)+2\,p(Q3)$", "a", "transferred severity")
    _levels(ax_b, lv, "U_pred", r"$U_{\mathrm{pred}}$ (nats)", "b", "predictive uncertainty")
    _novelty(ax_c, nov, "c")
    h = [Line2D([], [], color=MODEL_COLOR[m], marker="o", ms=6, lw=1.8, mec=SURFACE, label=m)
         for m in MODELS]
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False,
               bbox_to_anchor=(0.36, -0.045), columnspacing=1.8)
    fig.suptitle("Zero-shot transfer from BUT QDB to LTSTDB: prediction, uncertainty and input novelty",
                 x=0.005, ha="left", fontsize=10 * FONT_SCALE, weight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save(fig, "fig7_ltstdb_transfer")


PLOTS = [
    (screening, 0, ["screen_all_subsets.csv", "screen_mechanism_prevalence.csv"]),
    (operational_reliability, 2, ["incremental_gains.csv"]),
    (construct_validity, 3, ["expert_agreement.csv", "local_overlap.csv", "reduced_training.csv"]),
    (noise_stress, 4, ["noise_stress.csv"]),
    (ltstdb_transfer, 4, ["ltstdb_state_levels.csv", "ltstdb_input_novelty.csv"]),
]


def main():
    for fn, rq, needs in PLOTS:
        src = RESULTS / f"rq_{rq}_result"
        missing = [f for f in needs if not (src / f).exists()]
        if missing:
            print(f"skip {fn.__name__}: run `python run.py --rq {rq}` first")
            continue
        try:
            fn(src)
        except Exception as e:
            print(f"skip {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
