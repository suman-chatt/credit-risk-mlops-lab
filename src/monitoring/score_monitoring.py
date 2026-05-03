import pandas as pd

from src.monitoring.drift import monitor_score_drift


def monitor_model_scores(
    reference_scores: pd.DataFrame,
    actual_scores: pd.DataFrame,
    score_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monitor drift for multiple model scores.
    """

    summary_rows = []
    bucket_tables = []

    for col in score_cols:

        if col not in reference_scores.columns or col not in actual_scores.columns:
            continue

        summary, buckets = monitor_score_drift(
            expected_scores=reference_scores[col],
            actual_scores=actual_scores[col],
            score_name=col,
        )

        summary_rows.append(summary)
        bucket_tables.append(buckets)

    summary_df = pd.concat(summary_rows, ignore_index=True)

    if bucket_tables:
        bucket_df = pd.concat(bucket_tables, ignore_index=True)
    else:
        bucket_df = pd.DataFrame()

    return summary_df, bucket_df