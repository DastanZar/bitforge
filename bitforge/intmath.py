"""Q15 fixed-point kernels in pure int64 numpy.

Convention: value v in [-1, 1) is stored as integer V = round(v * 2^15).
Products: (V·U + 2^14) >> 15 — round-to-nearest, integer only.

Operand/output widths mirror real integer tensor units: operands are int16,
the accumulator is int64 (T4-int8/int16 units accumulate in int32; we widen
for headroom) so a K-length dot of int16 products NEVER overflows and the
*result* keeps accumulator width — the Q15 int16 range is a storage
convention for activations/weights (|v|<1), not a law of arithmetic for
inner products. Saturation asserts mark every place the narrower domain is
restored (relu/gemm feeding a norm-boundary), not silent wraps.
"""
from __future__ import annotations

import numpy as np

Q = 15
SCALE = 1 << Q          # 32768
HALF = 1 << (Q - 1)     # rounding bias


def _assert_int16(x: np.ndarray, name: str) -> None:
    assert x.dtype == np.int16, f"{name} must be int16, got {x.dtype}"


def matmul_q15(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """int16 [M,K] @ int16 [K,N] -> int64 Q15 accumulator (exact for our K).

    The result is returned in accumulator width: requantize_to_int16() at
    the next storage boundary (that's what hardware's rescale/epilogue does).
    """
    _assert_int16(a, "a")
    _assert_int16(b, "b")
    assert a.shape[1] == b.shape[0] and a.shape[1] <= 4096, "declared K limit"
    acc = a.astype(np.int64) @ b.astype(np.int64)
    return (acc + HALF) >> Q


def gemv_q15(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """int16 [N,K] @ int16 [K] -> int64 Q15 accumulator."""
    _assert_int16(w, "w")
    _assert_int16(x, "x")
    assert w.shape[1] == x.shape[0] and w.shape[1] <= 4096, "declared K limit"
    acc = w.astype(np.int64) @ x.astype(np.int64)
    return (acc + HALF) >> Q


def requantize_to_int16(acc: np.ndarray) -> np.ndarray:
    """Back into the Q16 storage domain with saturation *checked*, never
    wrapped: the caller must scale its accumulator to |v|<1 first
    (engines do this with a fixed right-shift chosen by calibration)."""
    assert np.abs(acc).max() <= 32767, "accumulator exceeds int16: apply shift first"
    return acc.astype(np.int16)


def relu_q15(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0).astype(np.int16)


def exp_neg_q15(x: np.ndarray, bits: int = 4) -> np.ndarray:
    """exp(x/2^15) for x ∈ [-2^15, 0] (Q15 in, Q15 out).

    Integer binomial identity: exp(y) = (1 + y/n)^n, n = 2^bits, evaluated as
    bits squarings of base = 1 + x/2^bits in Q15. For |y| ≤ 1 the relative
    bias is bounded: (1 + y/16)^16 vs e^y is within ~3.3% at y=-1, less near
    0 (where softmax mass lives). No clamping, no branches, no floats — the
    whole domain of softmax inputs after the row-max shift is [-1, 0).
    """
    x = x.astype(np.int64)
    assert (x >= -SCALE).all() and (x <= 0).all(), "exp_neg: expects x in [-1, 0] as Q15"
    base = SCALE + (x >> bits)                   # in [SCALE - SCALE/2^bits, SCALE]
    e = base
    for _ in range(bits):
        e = (e * e + HALF) >> Q
    return e.astype(np.int64)


def softmax_q15(logits: np.ndarray, bits: int = 4,
                tally: dict | None = None) -> np.ndarray:
    """Row softmax on Q15 logits whose *real* range fits (-1, 1): engines
    pre-divide by a temperature into this domain (argmax-invariant).

    1. shift rows by their max (softmax invariance, pure int subtraction)
    2. exp_neg every entry
    3. normalize: integer reciprocal per row, q = (e · ceil(2^30/Σ)) >> 30
    """
    x = logits.astype(np.int64)
    assert np.abs(x).max() <= 32767
    x = x - x.max(axis=-1, keepdims=True)
    e = exp_neg_q15(x, bits)
    s = e.sum(axis=-1, keepdims=True)
    r = _div_q15(e, s)
    return np.clip(r, 0, 32767).astype(np.int16)


def _div_q15(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Per-row Q15 division: q15 = floor(num·2^15/den) computed as
    ((num<<15) · ceil(2^30/den)) >> 30 — exact integer ops throughout;
    the ceil biases by <1 ulp at the 2^30 scale, never truncates a
    representable probability to zero."""
    den = np.maximum(den.astype(np.int64), 1)
    inv = (1 << 30) // den + ((1 << 30) % den > 0).astype(np.int64)
    q = ((num.astype(np.int64) << Q) * inv) >> 30
    return q
