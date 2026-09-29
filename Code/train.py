"""Train and evaluate VQC variants on PIMA and compare with classical baselines.

    python train.py                                   # 4 and 8 qubits, ZZ map, 6 layers
    python train.py --qubits 8 --layers 1 2 4 6 --feature-maps zz z pauli
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC

from data import encoder, load_pima, split
from vqc import build_vqc

RESULTS = Path(__file__).resolve().parent.parent / "Results"


def metrics(y, pred, score=None):
    m = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
         "recall": recall_score(y, pred), "f1": f1_score(y, pred)}
    if score is not None:
        m["roc_auc"] = roc_auc_score(y, score)
    return {k: round(float(v), 4) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--layers", type=int, nargs="+", default=[6])
    ap.add_argument("--feature-maps", nargs="+", default=["zz"])
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(RESULTS / "results.json"))
    args = ap.parse_args()

    Xtr, Xte, ytr, yte = split(load_pima(), seed=args.seed)
    runs = []

    for name, clf in {"Logistic Regression": LogisticRegression(max_iter=1000),
                      "SVM (RBF)": SVC(probability=True, random_state=args.seed),
                      "Random Forest": RandomForestClassifier(n_estimators=300, random_state=args.seed)}.items():
        enc = encoder(8).fit(Xtr)
        clf.fit(enc.transform(Xtr), ytr)
        runs.append({"model": name, "qubits": None, **metrics(yte, clf.predict(enc.transform(Xte)),
                                                               clf.predict_proba(enc.transform(Xte))[:, 1])})
        print(runs[-1])

    for n in args.qubits:
        enc = encoder(n).fit(Xtr, ytr)
        Qtr, Qte = enc.transform(Xtr), enc.transform(Xte)
        for fm in args.feature_maps:
            for layers in args.layers:
                history = []
                vqc = build_vqc(n, layers, fm, args.maxiter, args.seed, callback=lambda w, v: history.append(float(v)))
                t0 = time.time()
                vqc.fit(Qtr, ytr)
                pred = np.asarray(vqc.predict(Qte)).ravel()
                score = np.asarray(vqc.predict_proba(Qte))[:, 1]
                runs.append({"model": f"VQC-{fm.upper()}", "qubits": n, "layers": layers, "feature_map": fm,
                             "train_accuracy": round(float(accuracy_score(ytr, np.asarray(vqc.predict(Qtr)).ravel())), 4),
                             **metrics(yte, pred, score), "seconds": round(time.time() - t0, 1), "loss_history": history})
                print({k: v for k, v in runs[-1].items() if k != "loss_history"})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"dataset": "PIMA Indian Diabetes", "test_fraction": 0.2, "seed": args.seed,
                                          "maxiter": args.maxiter, "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
