"""Demo: determinism run x2, precision study across bit depths, receipts."""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bitforge import gemv_q15, relu_q15  # noqa: E402
from bitforge.intmath import SCALE, requantize_to_int16  # noqa: E402
from bitforge.reference import argmax_agreement, precision_report  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "receipts")
os.makedirs(OUT, exist_ok=True)

# determinism: same seed, two full forward passes, byte-level digest equality
def forward(seed):
    rng = np.random.default_rng(seed)
    x = np.round(rng.uniform(-0.5, 0.5, 128) * SCALE).astype(np.int16)
    w1 = np.round(rng.uniform(-0.5, 0.5, (256, 128)) * SCALE).astype(np.int16)
    w2 = np.round(rng.uniform(-0.5, 0.5, 256) * SCALE).astype(np.int16)
    h = relu_q15(requantize_to_int16(gemv_q15(w1, x) >> 3))   # rescale, then store
    y = gemv_q15(w2[None, :], h)
    return hashlib.sha256(y.astype(np.int64).tobytes()).hexdigest()

d1, d2 = forward(7), forward(7)

prec = precision_report(n=512, seed=3)
agree = argmax_agreement(bits=3)
agree4 = argmax_agreement(bits=4, seed=5)

receipt = {
    "determinism": {"run1": d1[:16], "run2": d2[:16], "identical": d1 == d2},
    "matmul_precision_vs_f64": prec,
    "softmax": {"bits3": agree, "bits4": agree4},
    "note": "Float conversion and reference modules are intentionally exempt from the "
            "source audit. lint_forward detects selected float syntax, not arbitrary "
            "runtime dtype flow. Softmax agreement excludes the reported near-tie subset.",
}
with open(os.path.join(OUT, "receipt.json"), "w") as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt, indent=2))
assert d1 == d2
print("DEMO_OK")
