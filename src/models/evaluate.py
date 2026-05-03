import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_ks(y_true, y_score) -> float:
    """
    Calculate KS statistic.
    """
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_score": y_score,
        }
    )

    df = df.sort_values("y_score", ascending=False)

    total_bad = df["y_true"].sum()
    total_good = len(df) - total_bad

    df["cum_bad"] = df["y_true"].cumsum() / total_bad
    df["cum_good"] = (1 - df["y_true"]).cumsum() / total_good

    return float((df["cum_bad"] - df["cum_good"]).abs().max())


def calculate_gini(auc: float) -> float:
    """
    Calculate Gini from AUC.
    """
    return float(2 * auc - 1)


def evaluate_binary_classifier(
    y_true,
    y_score,
    threshold: float = 0.5,
) -> dict:
    """
    Calculate common binary classification and probability metrics.

    y_score should be predicted probabilities for class 1.
    """
    y_pred = (np.array(y_score) >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_score)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "auc": float(auc),
        "gini": calculate_gini(auc),
        "ks": calculate_ks(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "log_loss": float(log_loss(y_true, y_score)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "threshold": float(threshold),
    }

    return metrics