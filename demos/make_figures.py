"""Render measured precision and repeatability without GPU claims."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "assets"
FIG.mkdir(exist_ok=True)
receipt = json.loads((ROOT / "receipts/receipt.json").read_text())
plt.rcParams.update({"figure.dpi": 140, "font.size": 10})
rng = np.random.default_rng(2)
a = rng.uniform(-0.9, 0.9, (128, 256))
b = rng.uniform(-0.9, 0.9, (256, 64))
ref = a @ b
rows = []
for bits in range(6, 17):
    scale = 1 << bits
    quantized = (np.round(a * scale) / scale) @ (np.round(b * scale) / scale)
    relative = float(np.linalg.norm(quantized - ref) / np.linalg.norm(ref))
    rows.append({"fractional_bits": bits, "relative_rms": relative})
(ROOT / "receipts/bit_sweep.json").write_text(json.dumps({
    "seed": 2, "shapes": [[128, 256], [256, 64]],
    "method": "Input rounding followed by float64 matmul; not the integer kernel",
    "measurements": rows,
}, indent=2) + "\n")
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.semilogy([r["fractional_bits"] for r in rows],
            [r["relative_rms"] for r in rows], "o-", color="#0f766e")
ax.axvline(15, color="#b45309", linestyle="--", label="Q15 fractional precision")
ax.set(xlabel="Fractional bits", ylabel="Relative RMS error",
       title="Input-rounding sensitivity (float64 accumulation)")
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "bit_curve.png")
plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.axis("off")
d = receipt["determinism"]
ax.text(0.04, 0.78, "REPEATED INTEGER FORWARD PASS", weight="bold", fontsize=15)
ax.text(0.04, 0.51, f"run A: {d['run1']}\nrun B: {d['run2']}", family="monospace")
ax.text(0.04, 0.19, f"Same-host digest equality: {d['identical']}\n"
        "Source audit catches selected float syntax; it is not a runtime proof.")
fig.tight_layout()
fig.savefig(FIG / "lint_gate.png")
plt.close(fig)
print("FIGURES_OK")
