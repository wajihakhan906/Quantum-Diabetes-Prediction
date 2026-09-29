"""Figures from Results/results.json + circuit diagrams."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from vqc import FEATURE_MAPS
from qiskit.circuit.library import real_amplitudes

ROOT = Path(__file__).resolve().parent.parent


def main():
    runs = json.load(open(ROOT / "Results" / "results.json"))["runs"]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [r["model"] + (f"\n{r['qubits']}q, {r['layers']}L" if r["qubits"] else "") for r in runs]
    colors = ["#4C72B0" if r["qubits"] else "#999999" for r in runs]
    bars = ax.bar(range(len(runs)), [r["accuracy"] for r in runs], color=colors)
    ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(range(len(runs))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.5, 0.95); ax.set_ylabel("Test accuracy"); ax.set_title("PIMA diabetes: quantum (blue) vs classical (grey)")
    fig.tight_layout(); fig.savefig(ROOT / "Figures" / "accuracy_comparison.png", dpi=200)

    fig, ax = plt.subplots(figsize=(6, 4))
    for r in runs:
        if r.get("loss_history"):
            ax.plot(r["loss_history"], label=f"{r['qubits']} qubits, {r['layers']} layers")
    ax.set_xlabel("COBYLA iteration"); ax.set_ylabel("Cross-entropy loss"); ax.set_title("VQC training"); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "Figures" / "training_loss.png", dpi=200)

    FEATURE_MAPS["zz"](8).draw("mpl", filename=str(ROOT / "Figures" / "zz_feature_map_8q.png"), fold=30)
    real_amplitudes(8, reps=6, entanglement="linear").draw("mpl", filename=str(ROOT / "Figures" / "ansatz_8q_6layers.png"), fold=40)


if __name__ == "__main__":
    main()
