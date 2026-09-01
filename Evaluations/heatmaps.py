import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/rrame12/Desktop/Research/DWT_IR/runs_pr_ablation/results_grid_eval")

levels = [2, 3, 5]
filters = [11, 101, 1001]
stems = ["Vocal", "Bass", "Drums"]

data = {stem: np.full((len(levels), len(filters)), np.nan, dtype=float) for stem in stems}

for i, L in enumerate(levels):
    for j, F in enumerate(filters):
        jpath = ROOT / f"L{L}_F{F}" / "mimo_metrics_summary.json"
        with open(jpath, "r") as f:
            mimo = json.load(f)

        for stem in stems:
            data[stem][i, j] = float(mimo["output_after"]["SIRB"][stem]["mean"])

for stem in stems:
    plt.figure(figsize=(5.2, 3.6))
    im = plt.imshow(data[stem], aspect="auto")
    plt.xticks(range(len(filters)), [str(f) for f in filters])
    plt.yticks(range(len(levels)), [str(l) for l in levels])
    plt.xlabel("Filter length F")
    plt.ylabel("Levels L")
    plt.title(f"{stem} SIR(B) (dB)")
    plt.colorbar(im)

    for i in range(len(levels)):
        for j in range(len(filters)):
            plt.text(j, i, f"{data[stem][i,j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(ROOT / f"heatmap_sirb_{stem.lower()}.png", dpi=200, bbox_inches="tight")
    plt.close()

print("Saved:")
for stem in stems:
    print(ROOT / f"heatmap_sirb_{stem.lower()}.png")