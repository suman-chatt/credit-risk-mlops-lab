import pandas as pd


def create_time_index(
    df: pd.DataFrame,
    date_col: str,
    freq: str = "M",
) -> pd.Series:
    """
    Convert date column into time buckets (monthly by default).
    """

    dates = pd.to_datetime(df[date_col])
    return dates.dt.to_period(freq).astype(str)


def score_stability_over_time(
    df: pd.DataFrame,
    score_col: str,
    time_col: str,
) -> pd.DataFrame:
    """
    Track score distribution over time.
    """

    grouped = df.groupby(time_col)

    rows = []

    for period, gdf in grouped:

        rows.append(
            {
                "period": period,
                "count": len(gdf),
                "mean_score": gdf[score_col].mean(),
                "std_score": gdf[score_col].std(),
                "p10": gdf[score_col].quantile(0.10),
                "p50": gdf[score_col].quantile(0.50),
                "p90": gdf[score_col].quantile(0.90),
            }
        )

    return pd.DataFrame(rows).sort_values("period")


def performance_stability_over_time(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    time_col: str,
) -> pd.DataFrame:
    """
    Track model performance drift over time.
    """

    from sklearn.metrics import roc_auc_score

    rows = []

    for period, gdf in df.groupby(time_col):

        if gdf[target_col].nunique() < 2:
            continue

        auc = roc_auc_score(gdf[target_col], gdf[score_col])

        rows.append(
            {
                "period": period,
                "count": len(gdf),
                "default_rate": gdf[target_col].mean(),
                "auc": auc,
            }
        )

    return pd.DataFrame(rows).sort_values("period")