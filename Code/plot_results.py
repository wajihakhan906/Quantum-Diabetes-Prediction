"""Figures from Results/results.json + circuit diagrams."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from qiskit.circuit.library import real_amplitudes, zz_feature_map

ROOT = Path(__file__).resolve().parent.parent


def main():
    runs = [r for r in json.load(open(ROOT / "Results" / "results.json"))["runs"] if r["model"] != "validation-grid"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    labels = [r["model"] if not r["qubits"] else f"VQC\n{r['qubits']}q, {r['layers']}L" for r in runs]
    colors = ["#999999" if not r["qubits"] else "#4C72B0" if r["qubits"] == 4 else "#C44E52" for r in runs]
    bars = ax.bar(range(len(runs)), [r["accuracy"] for r in runs], color=colors)
    ax.bar_label(bars, fmt="%.3f", fontsize=7.5)
    ax.set_xticks(range(len(runs))); ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
    ax.axhline(0.649, color="k", ls=":", lw=1); ax.text(len(runs) - 0.5, 0.655, "majority class", ha="right", fontsize=7)
    ax.set_ylim(0.5, 0.82); ax.set_ylabel("Test accuracy")
    ax.set_title("PIMA test accuracy: classical (grey), VQC 4 qubits (blue), VQC 8 qubits (red)")
    fig.tight_layout(); fig.savefig(ROOT / "Figures" / "accuracy_comparison.png", dpi=200)

    fig, ax = plt.subplots(figsize=(6, 4))
    for r in runs:
        if r.get("loss_history"):
            ax.plot(r["loss_history"], label=f"{r['qubits']} qubits, {r['layers']} layers")
    ax.set_xlabel("Adam step"); ax.set_ylabel("Binary cross-entropy"); ax.set_title("VQC training loss"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "Figures" / "training_loss.png", dpi=200)

    zz_feature_map(8, reps=2, entanglement="linear").draw("mpl", filename=str(ROOT / "Figures" / "zz_feature_map_8q.png"), fold=30)
    real_amplitudes(8, reps=6, entanglement="linear").draw("mpl", filename=str(ROOT / "Figures" / "ansatz_8q_6layers.png"), fold=40)


if __name__ == "__main__":
    main()
