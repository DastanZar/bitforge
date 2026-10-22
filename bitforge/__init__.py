"""bitforge — the forward path with no floats in it.

Deterministic inference is a deployment requirement people keep discovering
the hard way (same model + same prompt + GPU reduction nondeterminism =
flaky evals and un-auditable agents). The only real cure is an integer-only
forward pass: Q15 fixed-point everywhere, exact integer arithmetic, and —
the part that actually sticks — a CI LINT that fails the build if any float
operation appears in the forward path. Not a comment asking devs to be
careful: an AST gate that fails.

This repo ships the int matmul (Q15), the quantizer, the lint, and the
precision-vs-bitwidth study the lint protects.
"""
from .intmath import gemv_q15, matmul_q15, relu_q15, softmax_q15
from .quant import q15, deq15, quantize_row

__all__ = ["q15", "deq15", "quantize_row", "matmul_q15", "gemv_q15", "relu_q15", "softmax_q15"]
