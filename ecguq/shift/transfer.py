"""Zero-shot transfer to LTSTDB: windows, states, frozen models, state summaries and input novelty."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (CART_ITER, FS_ECG, K_PRIMARY, LTSTDB_ROOT, MAX_DEPTH, MIN_LEAF, MODELS,
                      N_CLASSES, N_DRAWS, S_STAR, S_STAR_ECG, SHIFT_CFG, WIN_S)
from ..features.ecg import ecg_features
from ..models.ladder import MODEL_UQ, fit_fold, fold_transform

LTSTDB = LTSTDB_ROOT
LEAD = int(SHIFT_CFG["ltstdb"]["lead"])


GUARD_S = float(SHIFT_CFG["ltstdb"]["guard_s"])


WIN = int(WIN_S * FS_ECG)


def quality_annotations(rec: str, lead: int = LEAD):
    import wfdb
    a = wfdb.rdann(str(LTSTDB / rec), "sta")
    hdr = wfdb.rdheader(str(LTSTDB / rec))
    urd, noi, open_at = [], [], None
    for s, aux in zip(a.sample, a.aux_note):
        if re.fullmatch(rf"\(urd{lead}", aux):
            open_at = s
        elif re.fullmatch(rf"urd{lead}\)", aux) and open_at is not None:
            urd.append((open_at, s))
            open_at = None
        elif re.match(rf"noi{lead}", aux):
            noi.append(s)
    if open_at is not None:
        urd.append((open_at, hdr.sig_len))
    return urd, noi, hdr.sig_len


def label_windows(rec: str, lead: int = LEAD) -> pd.DataFrame:
    urd, noi, n = quality_annotations(rec, lead)
    guard = int(GUARD_S * FS_ECG)
    starts = np.arange(0, n - WIN + 1, WIN)
    rows = []
    for st in starts:
        en = st + WIN
        cov = sum(max(0, min(en, b) - max(st, a)) for a, b in urd)
        r_u = cov / WIN
        near = any(abs(st - b) < guard or abs(en - a) < guard for a, b in urd) or \
               any(st - guard < p < en + guard for p in noi)
        if r_u >= 1.0:
            lab = "U"
        elif r_u == 0.0:
            lab = "C" if not near else "drop"
        else:


            lab = "T"
        rows.append((st, r_u, lab))
    df = pd.DataFrame(rows, columns=["start_sample", "r_U", "label"])

    for p in noi:
        st = int(p - WIN // 2)
        if 0 <= st <= n - WIN:
            df.loc[len(df)] = (st, 0.0, "N")
    df["record"] = rec
    df["subject"] = rec[2:5]
    return df


def _read_lead(rec: str, lead: int = LEAD) -> np.ndarray:
    import wfdb
    r = wfdb.rdrecord(str(LTSTDB / rec))
    return np.asarray(r.p_signal[:, lead], np.float64)


def build_features(records: list[str], chunk: int = 400) -> pd.DataFrame:
    out = []
    for k, rec in enumerate(records, 1):
        w = label_windows(rec)
        w = w[w.label.isin(["C", "T", "U", "N"])].reset_index(drop=True)
        sig = _read_lead(rec)
        feats = {c: np.empty(len(w)) for c in ECG_FEATURES}
        for i in range(0, len(w), chunk):
            sl = w.iloc[i:i + chunk]
            X = np.stack([sig[s:s + WIN] for s in sl.start_sample])
            f = ecg_features(X)
            f.pop("_beats", None)
            for c in ECG_FEATURES:
                feats[c][i:i + len(sl)] = f[c]
        for c in ECG_FEATURES:
            w[c] = feats[c]
        out.append(w)
        print(f"  [{k}/{len(records)}] {rec}: "
              f"{(w.label == 'C').sum():5d} C  {(w.label == 'U').sum():4d} U  "
              f"{(w.label == 'T').sum():3d} T  {(w.label == 'N').sum():2d} N", flush=True)
    return pd.concat(out, ignore_index=True)


def positive_records() -> list[str]:
    recs = [l.strip() for l in open(LTSTDB / "RECORDS") if l.strip()]
    keep = []
    for r in recs:
        urd, noi, _ = quality_annotations(r)
        if urd or noi:
            keep.append(r)
    return keep


def transfer(but_features: Path, windows_csv: Path, out_csv: Path) -> pd.DataFrame:
    but = pd.read_csv(but_features, dtype={"record": str, "subject": str})
    lts = pd.read_csv(windows_csv, dtype={"record": str, "subject": str})


    cols = S_STAR_ECG
    Xtr = but[cols].to_numpy(float)
    ytr = but["label"].to_numpy(int)
    gtr = but["subject"].to_numpy()
    Xte = lts[cols].to_numpy(float)
    (Ztr, Zte), _ = fold_transform(Xtr, Xte)
    print(f"transfer: S*_ECG = {len(cols)} of S*'s {len(S_STAR)} columns "
          f"({len(S_STAR) - len(cols)} accelerometer columns have no LTSTDB counterpart)")
    print(f"  fit on {len(Ztr):,} BUT windows, score {len(Zte):,} LTSTDB windows")

    res = fit_fold(Ztr, ytr, gtr, Zte, seed=0, n_draws=N_DRAWS,
                   cart_iter=CART_ITER, max_depth=MAX_DEPTH, min_leaf=MIN_LEAF)

    out = lts[["record", "subject", "start_sample", "r_U", "label"]].copy()
    for m in MODELS:
        p = res[m]["p"]
        out[f"{m}|p_Q3"] = p[:, 2]


        out[f"{m}|S_Q"] = p @ np.arange(N_CLASSES, dtype=float)
        for s in ("U_pred", "U_CE", "U_MI", "softmax"):
            if s == "U_MI" and not MODEL_UQ[m]:
                continue
            out[f"{m}|{s}"] = res[m][s]
    out.to_csv(out_csv, index=False)
    return out


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return np.nan
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def subject_stats(tr: pd.DataFrame, min_n: int = 20, min_t: int = 4) -> pd.DataFrame:
    cols = [c for c in tr.columns if "|" in c]
    pairs = [("C", "T"), ("T", "U"), ("C", "U")]
    need = {"C": min_n, "U": min_n, "T": min_t}
    rows = []
    for subj, g in tr.groupby("subject"):
        have = {k: g[g.label == k] for k in ("C", "T", "U")}
        for lo, hi in pairs:
            a, b = have[lo], have[hi]
            if len(a) < need[lo] or len(b) < need[hi]:
                continue
            for col in cols:
                m, sc = col.split("|")
                va = a[col].to_numpy(float); vb = b[col].to_numpy(float)
                va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
                if not len(va) or not len(vb):
                    continue
                sd = np.std(np.concatenate([va, vb])) or np.nan
                rows.append({"subject": subj, "model": m, "score": sc,
                             "contrast": f"{hi}-{lo}", "n_lo": len(va), "n_hi": len(vb),
                             "med_lo": np.median(va), "med_hi": np.median(vb),
                             "delta": np.median(vb) - np.median(va),
                             "d_std": (vb.mean() - va.mean()) / sd,
                             "auroc": _auroc(vb, va)})
    return pd.DataFrame(rows)


def state_levels(tr: pd.DataFrame, min_win: int = 4) -> pd.DataFrame:
    cols = [c for c in tr.columns if "|" in c]
    rows = []
    for subj, g in tr.groupby("subject"):
        for st in ("C", "T", "U"):
            h = g[g.label == st]
            if len(h) < min_win:
                continue
            for col in cols:
                m, sc = col.split("|")
                v = h[col].to_numpy(float); v = v[np.isfinite(v)]
                if len(v):
                    rows.append({"subject": subj, "state": st, "model": m, "score": sc,
                                 "n": len(v), "median": np.median(v)})
    return pd.DataFrame(rows)


def episode_stats(tr: pd.DataFrame, model: str = "B2", score: str = "U_pred") -> pd.DataFrame:
    col = f"{model}|{score}"
    rows = []
    for rec, g in tr.groupby("record"):
        clean = g[g.label == "C"][col].to_numpy(float)
        if not len(clean):
            continue
        thr = np.nanpercentile(clean, 95)
        urd, _, _ = quality_annotations(rec)
        idx = g[g.label == "U"].set_index("start_sample")[col]
        for a, b in urd:
            v = idx[(idx.index >= a) & (idx.index + WIN <= b)].to_numpy(float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            rows.append({"record": rec, "subject": rec[2:5], "start_h": a / FS_ECG / 3600,
                         "dur_min": (b - a) / FS_ECG / 60, "n_win": len(v),
                         "median": np.median(v), "max": v.max(),
                         "alarm": bool((v > thr).any()),
                         "frac_over": float((v > thr).mean())})
    return pd.DataFrame(rows)


def input_novelty(but: pd.DataFrame, lt: pd.DataFrame, k: int = K_PRIMARY):
    from sklearn.neighbors import NearestNeighbors

    ref = but[S_STAR_ECG].to_numpy(float)
    med = np.median(ref, axis=0)
    scale = 1.4826 * np.median(np.abs(ref - med), axis=0)
    scale[scale <= 0] = 1.0
    Zb = (ref - med) / scale
    Zl = (lt[S_STAR_ECG].to_numpy(float) - med) / scale

    d_lt = NearestNeighbors(n_neighbors=k).fit(Zb).kneighbors(Zl)[0].mean(1)


    subj = but.subject.to_numpy().astype(str)
    d_but = np.empty(len(Zb))
    for s in np.unique(subj):
        held = subj == s
        d_but[held] = NearestNeighbors(n_neighbors=k).fit(
            Zb[~held]).kneighbors(Zb[held])[0].mean(1)

    ecdf = np.sort(d_but)
    return np.searchsorted(ecdf, d_lt, side="right") / len(ecdf), d_lt, d_but


STATES = ["C", "T", "U"]
MIN_STATE_WINDOWS = int(SHIFT_CFG["ltstdb"]["min_windows_per_state"])


def novelty_table(but: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    pct, _, _ = input_novelty(but, w, k=int(SHIFT_CFG["ltstdb"]["novelty_k"]))
    rows = []
    for subj, idx in w.groupby("subject").groups.items():
        ii = w.index.get_indexer(idx)
        lab = w.loc[idx].label
        for st in STATES:
            m = (lab == st).to_numpy()
            if m.sum() >= MIN_STATE_WINDOWS:
                rows.append({"subject": subj, "state": st, "n": int(m.sum()),
                             "novelty_pct": float(np.median(pct[ii][m]))})
    return pd.DataFrame(rows)
