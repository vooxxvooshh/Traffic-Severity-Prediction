import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0
        ),
    }


def evaluate_model(model, X_test, y_test, label: str = "model") -> dict:
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics["label"] = label

    print(f"\n===== {label.upper()} =====")
    print("Accuracy       :", round(metrics["accuracy"], 4))
    print("Precision (macro):", round(metrics["precision_macro"], 4))
    print("Recall (macro)   :", round(metrics["recall_macro"], 4))
    print("F1 (macro)       :", round(metrics["f1_macro"], 4))
    print("\nClassification report:\n", metrics["classification_report"])
    print("Confusion matrix:\n", np.array(metrics["confusion_matrix"]))

    return metrics


def save_comparison_table(rows: list[dict], path: str = "model_comparison.csv") -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    df.to_csv(path, index=False)
    print(f"\nComparison table saved to {path}")
    print(df.to_string(index=False))
    return df


def save_metrics_json(metrics: dict, path: str = "best_model_metrics.json") -> None:
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
