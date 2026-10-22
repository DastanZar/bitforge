import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, ".")
from bitforge import gemv_q15, matmul_q15, relu_q15, softmax_q15  # noqa: E402
from bitforge import deq15, q15                                   # noqa: E402
from bitforge.intmath import SCALE, requantize_to_int16           # noqa: E402
from bitforge.lint import lint_forward                            # noqa: E402
from bitforge.reference import argmax_agreement, precision_report, softmax_ref  # noqa: E402


def test_matmul_parity_small_exact():
    a = q15(np.array([[0.5, -0.25], [0.125, 0.0]]))
    b = q15(np.array([[0.5, 0.25], [-0.5, 0.0]]))
    got = deq15(matmul_q15(a, b))
    want = np.array([[0.375, 0.125], [0.0625, 0.03125]])
    assert np.array_equal(got, want)


def test_gemv_parity():
    rng = np.random.default_rng(0)
    w = q15(rng.uniform(-0.5, 0.5, (32, 64)))
    x = q15(rng.uniform(-0.5, 0.5, 64))
    got = deq15(gemv_q15(w, x))
    want = (w.astype(np.float64) / SCALE) @ (x.astype(np.float64) / SCALE)
    assert np.abs(got - want).max() < 1.5 / SCALE   # within one rounding step


def test_precision_report_bounded():
    rep = precision_report(n=256)
    # worst-case triangle bound: K terms, each |a|·0.5ulp + |b|·0.5ulp
    # (quantization), never realized — the assertion guards regressions,
    # not the theory's loose end; RMS is the real quality number.
    k = 256
    worst_case = k * 0.9 * 0.5 / SCALE * 2
    assert rep["max_abs_err"] < worst_case, (rep, worst_case)
    assert rep["relative_rms"] < 0.002, rep   # ~25x smaller than a single int8


def test_softmax_sums_to_one_and_matches():
    rng = np.random.default_rng(1)
    z = np.round(rng.uniform(-0.45, 0.45, (64, 50)) * SCALE).astype(np.int16)
    p = deq15(softmax_q15(z))
    assert np.allclose(p.sum(axis=-1), 1.0, atol=0.01), p.sum(-1)
    ref = softmax_ref(z / SCALE)
    assert np.abs(p - ref).max() < 0.05   # (1+y/16)^16 bias at |y|<=0.9 is <4%


def test_softmax_argmax_agreement():
    rep = argmax_agreement()
    assert rep["argmax_agreement_pct_decidable"] == 100.0, rep
    assert rep["max_prob_deviation"] < 0.01, rep


def test_requantize_guards_saturation():
    with pytest.raises(AssertionError):
        requantize_to_int16(np.array([40000], dtype=np.int64))
    ok = requantize_to_int16(np.array([30000], dtype=np.int64))
    assert ok.dtype == np.int16


def test_relu_on_int16():
    assert relu_q15(q15(np.array([-0.5, 0.3, 0.0]))).tolist() == [0, 9830, 0]


def test_lint_gate_passes_on_shipped_forward():
    issues = lint_forward(pathlib.Path("bitforge"))
    assert issues == [], "\n".join(issues)


def test_lint_gate_catches_a_float():
    """The gate must actually gate: plant a float, expect the report."""
    good = lint_forward(pathlib.Path("bitforge"))
    src = pathlib.Path("bitforge/_leak_tmp.py")
    src.write_text("def leak(x):\n    return x * 0.5\n")
    try:
        bad = lint_forward(pathlib.Path("bitforge"))
        assert len(bad) == len(good) + 1 and "float literal 0.5" in bad[0], bad
    finally:
        src.unlink()


def test_lint_gate_catches_true_division():
    good = lint_forward(pathlib.Path("bitforge"))
    src = pathlib.Path("bitforge/_leak2_tmp.py")
    src.write_text("def leak(a, b):\n    return a / b\n")
    try:
        bad = lint_forward(pathlib.Path("bitforge"))
        assert len(bad) == len(good) + 1 and "true division" in bad[0], bad
    finally:
        src.unlink()
