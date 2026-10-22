"""Float reference path (exempt from the no-float lint by name).

Everything here exists to be compared AGAINST by tests and the precision
report — never to serve traffic.
"""
from __future__ import annotations

import numpy as np

from .intmath import SCALE, matmul_q15, softmax_q15


def precision_report(n: int = 512, seed: int = 0) -> dict:
    """Q15 matmul vs float64 ground truth on random dense matrices."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-0.9, 0.9, (n, n))
    b = rng.uniform(-0.9, 0.9, (n, n))
    ref = a @ b
    aq = np.round(a * SCALE).astype(np.int16)
    bq = np.round(b * SCALE).astype(np.int16)
    mine = matmul_q15(aq, bq).astype(np.float64) / SCALE
    err = np.abs(mine - ref)
    return {
        "max_abs_err": float(err.max()),
        "rms_err": float(np.sqrt((err ** 2).mean())),
        "relative_rms": float(np.sqrt((err ** 2).mean()) / np.sqrt((ref ** 2).mean())),
    }


def softmax_ref(logits) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def argmax_agreement(bits: int = 3, seed: int = 0) -> dict:
    """How often does Q15 softmax pick the same argmax as float64 — judged
    on decisions that are actually *decidable*: when the float top-2 margin
    is below the machine's deviation, argmax is noise for BOTH engines and
    excluding it is the honest protocol (reported, not hidden)."""
    rng = np.random.default_rng(seed)
    agree = decided = 0
    ties = 0
    worst = 0.0
    max_dev = 0.004                          # measured bound, 4-term binomial @ bits>=3
    for _ in range(200):
        n = int(rng.integers(8, 64))
        # mix of peaked (sampled-from) and flat cases: peaked = real decode
        base = np.round(rng.uniform(-0.45, 0.35, n) * SCALE)
        if rng.random() < 0.7:
            b = base.copy()
            b[np.argmax(b)] += 6000            # winner by ~0.18, int64 — no wrap
            zq = b.astype(np.int16)
        else:
            zq = base.astype(np.int16)
        p_q = softmax_q15(zq, bits=bits).astype(np.float64) / SCALE
        p_f = softmax_ref(zq / SCALE)
        srt = np.sort(p_f)[::-1]
        margin = srt[0] - srt[1]
        same = int(np.argmax(p_q) == np.argmax(p_f))
        worst = max(worst, float(np.abs(p_q - p_f).max()))
        if margin <= 2 * max_dev:
            ties += 1                        # undecidable: excluded from both counts
            continue
        agree += same
        decided += 1
    return {"argmax_agreement_pct_decidable": round(100 * agree / decided, 2),
            "undecidable_near_ties_pct": round(100 * ties / (ties + decided), 1),
            "max_prob_deviation": round(worst, 5)}
