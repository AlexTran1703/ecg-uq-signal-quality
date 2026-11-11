"""Window-level 50 Hz and 60 Hz power in the LTSTDB transfer recordings (lead used for transfer)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from ..config import LTSTDB_ROOT, SHIFT_CFG, WIN_S

CFG = SHIFT_CFG["ltstdb"]["mains_check"]
LEAD = int(SHIFT_CFG["ltstdb"]["lead"])


def run(windows_csv: Path, outdir: Path) -> pd.DataFrame:
    import wfdb
    w = pd.read_csv(windows_csv, dtype={"record": str})
    rows = []
    for rec, g in w.groupby("record"):
        r = wfdb.rdrecord(str(LTSTDB_ROOT / rec), channels=[LEAD])
        x, fs = r.p_signal[:, 0], r.fs
        n = int(WIN_S * fs)
        for _, row in g.iterrows():
            seg = x[int(row.start_sample):int(row.start_sample) + n]
            seg = seg[np.isfinite(seg)]
            if len(seg) < n // 2:
                continue
            f, p = signal.welch(seg - seg.mean(), fs=fs, nperseg=min(len(seg), int(2 * fs)))
            total = p[(f >= 0.5) & (f <= 100)].sum()
            out = {"record": rec, "subject": row.subject, "state": row.label}
            for hz in CFG["line_hz"]:
                band = (f >= hz - CFG["half_width_hz"]) & (f <= hz + CFG["half_width_hz"])
                out[f"power_{hz}hz_pct"] = 100 * p[band].sum() / total
            rows.append(out)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "ltstdb_mains_by_window.csv.gz", index=False)
    thr = 100 * CFG["threshold_fraction"]
    s = t.groupby("record").agg(**{f"share_{hz}hz_above_threshold_pct": (f"power_{hz}hz_pct",
                                   lambda v, thr=thr: 100 * float((v > thr).mean())) for hz in CFG["line_hz"]},
                                **{f"p95_{hz}hz_pct": (f"power_{hz}hz_pct", lambda v: float(np.percentile(v, 95)))
                                   for hz in CFG["line_hz"]})
    s.reset_index().to_csv(outdir / "ltstdb_mains_by_record.csv", index=False)
    return t
