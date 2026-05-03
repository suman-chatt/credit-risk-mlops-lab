import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve



def generate_calibration_table(
    y_true,
    y_score,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Build calibration table (predicted vs actual by probability bins).
    """

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_score,
        n_bins=n_bins,
        strategy="quantile",
    )

    return pd.DataFrame(
        {
            "predicted_prob": prob_pred,
            "actual_rate": prob_true,
            "bin": range(1, len(prob_pred) + 1),
        }
    )

def calculate_gains_lift(
    y_true,
    y_score,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Build gains / lift table using quantile bins.
    """

    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_score": y_score,
        }
    )

    df["bucket"] = pd.qcut(df["y_score"], q=n_bins, duplicates="drop")

    df = df.sort_values("y_score", ascending=False)

    grouped = df.groupby("bucket", observed=False)

    summary = grouped.agg(
        total=("y_true", "count"),
        bad=("y_true", "sum"),
    ).reset_index()

    summary["good"] = summary["total"] - summary["bad"]
    summary["bad_rate"] = summary["bad"] / summary["total"]

    summary = summary.sort_values("bucket", ascending=False).reset_index(drop=True)

    summary["cum_bad"] = summary["bad"].cumsum()
    summary["cum_total"] = summary["total"].cumsum()

    total_bad = summary["bad"].sum()
    total_total = summary["total"].sum()

    summary["cum_bad_pct"] = summary["cum_bad"] / total_bad
    summary["cum_total_pct"] = summary["cum_total"] / total_total

    summary["lift"] = summary["cum_bad_pct"] / summary["cum_total_pct"]

    return summary


def generate_roc_curve_data(y_true, y_score) -> pd.DataFrame:
    """
    Return ROC curve points.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    return pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }
    )


def generate_ks_curve_data(
    y_true,
    y_score,
    max_points: int = 1000,
) -> pd.DataFrame:
    """
    Generate downsampled KS curve data.

    Full row-level KS curves are too large for Excel export, so this keeps
    representative points while preserving the max KS point.
    """
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_score": y_score,
        }
    )

    df = df.sort_values("y_score", ascending=False).reset_index(drop=True)

    total_bad = df["y_true"].sum()
    total_good = len(df) - total_bad

    df["cum_bad_pct"] = df["y_true"].cumsum() / total_bad
    df["cum_good_pct"] = (1 - df["y_true"]).cumsum() / total_good
    df["ks"] = df["cum_bad_pct"] - df["cum_good_pct"]
    df["row_number"] = df.index + 1
    df["population_pct"] = df["row_number"] / len(df)

    if len(df) <= max_points:
        return df

    sample_idx = set(
        np.linspace(0, len(df) - 1, max_points, dtype=int)
    )

    max_ks_idx = int(df["ks"].abs().idxmax())
    sample_idx.add(max_ks_idx)

    keep_cols = [
        "row_number",
        "population_pct",
        "y_score",
        "cum_bad_pct",
        "cum_good_pct",
        "ks",
    ]

    return df.loc[sorted(sample_idx), keep_cols].reset_index(drop=True)


def calculate_pva(
    df: pd.DataFrame,
    y_true_col: str,
    y_score_col: str,
    time_col: str,
) -> pd.DataFrame:
    """
    Predicted vs Actual aggregation by time.
    """

    grouped = df.groupby(time_col, observed=False).agg(
        actual_rate=(y_true_col, "mean"),
        predicted_rate=(y_score_col, "mean"),
        count=(y_true_col, "count"),
    )

    grouped = grouped.reset_index().sort_values(time_col)

    return grouped
