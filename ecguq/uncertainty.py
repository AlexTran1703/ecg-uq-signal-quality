"""Predictive uncertainty: entropy decomposition (U_pred, U_CE, U_MI) and max-softmax (U_MS)."""
from __future__ import annotations

import numpy as np

from .config import EPS


def softmax(Z, axis=-1):
    Z = np.asarray(Z, np.float64)
    Z = Z - Z.max(axis=axis, keepdims=True)
    E = np.exp(Z)
    return E / np.clip(E.sum(axis=axis, keepdims=True), EPS, None)


def entropy(p, axis=-1):
    p = np.asarray(p, np.float64)
    return -(p * np.log(np.clip(p, EPS, None))).sum(axis)


def decompose(members: np.ndarray) -> dict[str, np.ndarray]:
    members = np.asarray(members, np.float64)
    p = members.mean(0)
    u_pred = entropy(p)
    u_ce = entropy(members).mean(0)
    return {"p": p, "U_pred": u_pred, "U_CE": u_ce, "U_MI": np.maximum(u_pred - u_ce, 0.0)}


def point_scores(p: np.ndarray) -> dict[str, np.ndarray]:
    p = np.asarray(p, np.float64)
    h = entropy(p)
    return {"p": p, "U_pred": h, "U_CE": h, "U_MI": np.zeros(len(p))}


def softmax_uncertainty(p: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(p, np.float64).max(-1)
