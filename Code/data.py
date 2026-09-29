"""PIMA Indian Diabetes dataset: loading, cleaning and scaling for angle encoding."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
           "DiabetesPedigreeFunction", "Age", "Outcome"]
# a value of 0 is physiologically impossible for these and marks a missing measurement
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "pima-indians-diabetes.csv"


def load_pima(path=DEFAULT_PATH):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df[ZERO_IS_MISSING] = df[ZERO_IS_MISSING].replace(0, np.nan)
    return df


def split(df, test_size=0.2, seed=42):
    X, y = df.drop(columns="Outcome"), df["Outcome"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    medians = Xtr.median()  # impute with training medians only (no test leakage)
    return Xtr.fillna(medians).to_numpy(), Xte.fillna(medians).to_numpy(), ytr, yte


def encoder(n_qubits, reduction="select"):
    """Standardise -> reduce 8 features to `n_qubits` -> scale to [0, pi]."""
    steps = [("std", StandardScaler())]
    if n_qubits < 8:
        steps.append(("reduce", SelectKBest(f_classif, k=n_qubits) if reduction == "select" else PCA(n_qubits)))
    steps.append(("angle", MinMaxScaler(feature_range=(0, np.pi))))
    return Pipeline(steps)
