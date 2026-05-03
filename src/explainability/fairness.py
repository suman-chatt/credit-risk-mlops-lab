import pandas as pd


def compute_group_metrics(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    score_col: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute basic fairness metrics by group.

    Metrics:
    - default rate (actual)
    - approval rate (based on threshold)
    - TPR / Recall
    - FPR
    """

    rows = []

    for group, gdf in df.groupby(group_col):

        y_true = gdf[target_col]
        y_score = gdf[score_col]
        y_pred = (y_score >= threshold).astype(int)

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        total = len(gdf)

        rows.append(
            {
                "group": group,
                "count": total,
                "default_rate": y_true.mean(),
                "approval_rate": (y_pred == 0).mean(),
                "tpr_recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            }
        )

    return pd.DataFrame(rows)


def compute_disparity_ratios(
    group_metrics: pd.DataFrame,
    reference_group: str,
) -> pd.DataFrame:
    """
    Compute disparity ratios vs reference group.
    """

    ref_row = group_metrics[group_metrics["group"] == reference_group]

    if ref_row.empty:
        raise ValueError(f"Reference group {reference_group} not found.")

    ref = ref_row.iloc[0]

    df = group_metrics.copy()

    df["approval_rate_ratio"] = df["approval_rate"] / ref["approval_rate"]
    df["tpr_ratio"] = df["tpr_recall"] / ref["tpr_recall"]
    df["fpr_ratio"] = df["fpr"] / ref["fpr"]

    return df


def assign_fairness_flags(
    df: pd.DataFrame,
    threshold_low: float = 0.8,
    threshold_high: float = 1.25,
) -> pd.DataFrame:
    """
    Assign fairness flags using common 80% rule.
    """

    def flag(val):
        if val < threshold_low or val > threshold_high:
            return "review"
        return "ok"

    df = df.copy()

    df["approval_flag"] = df["approval_rate_ratio"].apply(flag)
    df["tpr_flag"] = df["tpr_ratio"].apply(flag)
    df["fpr_flag"] = df["fpr_ratio"].apply(flag)

    return df


def run_fairness_analysis(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    score_col: str,
    threshold: float = 0.5,
    reference_group: str | None = None,
) -> pd.DataFrame:
    """
    Full fairness pipeline.
    """

    group_metrics = compute_group_metrics(
        df=df,
        group_col=group_col,
        target_col=target_col,
        score_col=score_col,
        threshold=threshold,
    )

    if reference_group is None:
        reference_group = group_metrics.iloc[0]["group"]

    disparity = compute_disparity_ratios(
        group_metrics,
        reference_group=reference_group,
    )

    flagged = assign_fairness_flags(disparity)

    return flagged