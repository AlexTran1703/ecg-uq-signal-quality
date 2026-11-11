"""Feature table: all 18 candidate features for a block of windows, in the configured column order."""
from __future__ import annotations

import numpy as np

from ..config import ALL_FEATURES
from .acc import m1_m2_features, m3_features
from .ecg import ecg_features


def feature_block(X, A, excise_qrs: bool = True) -> np.ndarray:
    f = {**ecg_features(X), **m1_m2_features(A), **m3_features(X, A, excise_qrs)}
    return np.stack([np.asarray(f[c], np.float32) for c in ALL_FEATURES], axis=1)
