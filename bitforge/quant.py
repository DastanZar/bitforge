"""Quantization to Q15 and the saturation audit."""
from __future__ import annotations

import numpy as np

SCALE = 1 << 15


def q15(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    r = np.round(a * SCALE)
    assert np.abs(r).max() < SCALE, "value outside [-1,1): scale your weights"
    return r.astype(np.int16)


def deq15(q: np.ndarray):
    return q.astype(np.float64) / SCALE


def quantize_row(w) -> tuple:
    """Per-row scale-and-quantize (absmax). Returns (int16 codes, float scales).
    Scales live OUTSIDE the forward loop — the inference path stays integer."""
    a = np.asarray(w, dtype=np.float64)
    s = np.abs(a).max(axis=-1, keepdims=True).clip(min=1e-12)
    return q15(a / s), s.squeeze(-1)
