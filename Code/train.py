"""Train and evaluate VQC variants on PIMA and compare with classical baselines.

    python train.py                                   # 4 and 8 qubits, 2/4/6 layers, Adam (default)
    python train.py --optimizer cobyla --maxiter 300  # qiskit-machine-learning VQC with COBYLA

Input scaling ("bandwidth"): features are mapped to [0, bandwidth * pi] before the ZZ feature map. With the
full [0, pi] range the ZZ map scrambles the data and the VQC stays near chance; the bandwidth and number of
layers are therefore chosen on a validation split of the training data (never on the test set).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data import PAPER_4_ATTRIBUTES, encoder, load_pima, split
from vqc_torch import TorchVQCClassifier

RESULTS = Path(__file__).resolve().parent.parent / "Results"


def metrics(y, pred, score=None):
    m = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
         "recall": recall_score(y, pred), "f1": f1_score(y, pred)}
    if score is not None:
        m["roc_auc"] = roc_auc_score(y, score)
    return {k: round(float(v), 4) for k, v in m.items()}


def make_vqc(n, layers, args):
    if args.optimizer == "adam":
        return TorchVQCClassifier(n, layers, epochs=args.epochs, lr=args.lr, seed=args.seed)
    from vqc import build_vqc
    return build_vqc(n, layers, "zz", args.maxiter, args.seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--bandwidths", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--optimizer", choices=["adam", "cobyla"], default="adam")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--attributes", choices=["paper", "select"], default="paper",
                    help="4-qubit inputs: the paper's Glucose/Age/Insulin/BMI, or ANOVA F-test selection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(RESULTS / "results.json"))
    args = ap.parse_args()

    df = load_pima()
    Xtr, Xte, ytr, yte = split(df, seed=args.seed)
    runs = []

    enc8 = encoder(8).fit(Xtr)
    for name, clf in {"Logistic Regression": LogisticRegression(max_iter=1000),
                      "Decision Tree": DecisionTreeClassifier(random_state=args.seed),
                      "Gradient Boosting": GradientBoostingClassifier(random_state=args.seed),
                      "Random Forest": RandomForestClassifier(n_estimators=300, random_state=args.seed),
                      "SVM (RBF)": SVC(probability=True, random_state=args.seed)}.items():
        clf.fit(enc8.transform(Xtr), ytr)
        runs.append({"model": name, "qubits": None,
                     **metrics(yte, clf.predict(enc8.transform(Xte)), clf.predict_proba(enc8.transform(Xte))[:, 1])})
        print(runs[-1])

    for n in args.qubits:
        if n == 4 and args.attributes == "paper":
            Xtr_n, Xte_n, _, _ = split(df, seed=args.seed, attributes=PAPER_4_ATTRIBUTES)
            enc = encoder(n, reduction="none").fit(Xtr_n)
            attrs = PAPER_4_ATTRIBUTES
        else:
            Xtr_n, Xte_n = Xtr, Xte
            enc = encoder(n).fit(Xtr_n, ytr)
            attrs = "all 8" if n == 8 else "F-test"
        Qtr, Qte = enc.transform(Xtr_n), enc.transform(Xte_n)

        # model selection on a validation split of the training data
        a, v, ya, yv = train_test_split(Qtr, ytr, test_size=0.2, stratify=ytr, random_state=args.seed)
        grid = []
        for bw in args.bandwidths:
            for layers in args.layers:
                m = make_vqc(n, layers, args).fit(a * bw, ya)
                grid.append({"bandwidth": bw, "layers": layers,
                             "val_accuracy": round(float(accuracy_score(yv, np.asarray(m.predict(v * bw)).ravel())), 4)})
                print(f"  {n} qubits  bandwidth {bw:<4}  layers {layers}  val acc {grid[-1]['val_accuracy']:.3f}")

        for layers in args.layers:  # best bandwidth per depth, refit on the full training set, test once
            best = max((g for g in grid if g["layers"] == layers), key=lambda g: g["val_accuracy"])
            bw = best["bandwidth"]
            t0 = time.time()
            m = make_vqc(n, layers, args).fit(Qtr * bw, ytr)
            pred = np.asarray(m.predict(Qte * bw)).ravel()
            score = np.asarray(m.predict_proba(Qte * bw))[:, 1]
            runs.append({"model": "VQC-ZZ", "qubits": n, "layers": layers, "attributes": attrs, "bandwidth": bw,
                         "optimizer": args.optimizer, "val_accuracy": best["val_accuracy"],
                         "train_accuracy": round(float(accuracy_score(ytr, np.asarray(m.predict(Qtr * bw)).ravel())), 4),
                         **metrics(yte, pred, score), "seconds": round(time.time() - t0, 1),
                         "loss_history": getattr(m, "loss_history", [])})
            print({k: v for k, v in runs[-1].items() if k != "loss_history"})
        runs.append({"model": "validation-grid", "qubits": n, "grid": grid})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"dataset": "PIMA Indian Diabetes", "test_fraction": 0.2, "seed": args.seed,
                                          "config": vars(args), "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
