import pandas as pd


DEFAULT_RISK_BANDS = {
    "Very Low Risk": (0.00, 0.20),
    "Low Risk": (0.20, 0.40),
    "Medium Risk": (0.40, 0.70),
    "High Risk": (0.70, 0.90),
    "Very High Risk": (0.90, 1.00),
}


def add_score_percentiles(
    df: pd.DataFrame,
    score_col: str,
    percentile_col: str = "score_percentile",
) -> pd.DataFrame:
    """
    Add percentile rank based on model score.

    Higher score = higher risk.
    """

    df_out = df.copy()

    df_out[percentile_col] = df_out[score_col].rank(
        method="average",
        pct=True,
    )

    return df_out


def assign_risk_bands(
    df: pd.DataFrame,
    percentile_col: str = "score_percentile",
    band_col: str = "risk_band",
    risk_bands: dict | None = None,
) -> pd.DataFrame:
    """
    Assign risk bands using score percentiles.
    """

    if risk_bands is None:
        risk_bands = DEFAULT_RISK_BANDS

    df_out = df.copy()

    df_out[band_col] = "Unassigned"

    for band_name, (lower, upper) in risk_bands.items():
        mask = (
            (df_out[percentile_col] > lower)
            & (df_out[percentile_col] <= upper)
        )

        if lower == 0:
            mask = (
                (df_out[percentile_col] >= lower)
                & (df_out[percentile_col] <= upper)
            )

        df_out.loc[mask, band_col] = band_name

    return df_out


def create_risk_band_summary(
    df: pd.DataFrame,
    score_col: str,
    band_col: str = "risk_band",
) -> pd.DataFrame:
    """
    Summarize score distribution by risk band.
    """

    summary = (
        df.groupby(band_col, observed=False)
        .agg(
            applicant_count=(score_col, "count"),
            min_score=(score_col, "min"),
            mean_score=(score_col, "mean"),
            max_score=(score_col, "max"),
        )
        .reset_index()
    )

    total_count = summary["applicant_count"].sum()
    summary["applicant_pct"] = summary["applicant_count"] / total_count

    return summary


def create_score_summary(
    df: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    """
    Summarize score columns.
    """

    rows = []

    for col in score_cols:
        rows.append(
            {
                "score_col": col,
                "count": df[col].count(),
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "p01": df[col].quantile(0.01),
                "p05": df[col].quantile(0.05),
                "p25": df[col].quantile(0.25),
                "p50": df[col].quantile(0.50),
                "p75": df[col].quantile(0.75),
                "p95": df[col].quantile(0.95),
                "p99": df[col].quantile(0.99),
                "max": df[col].max(),
            }
        )

    return pd.DataFrame(rows)


def create_score_correlation_table(
    df: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    """
    Create score correlation matrix in long format.
    """

    corr = df[score_cols].corr()

    return (
        corr.reset_index()
        .melt(
            id_vars="index",
            var_name="score_col_2",
            value_name="correlation",
        )
        .rename(columns={"index": "score_col_1"})
    )


def build_scored_output(
    predictions: pd.DataFrame,
    production_model_id: str,
) -> pd.DataFrame:
    """
    Build final scored output using selected production model.
    """

    if production_model_id not in predictions.columns:
        raise ValueError(
            f"Production model score not found in predictions: {production_model_id}"
        )

    df_scored = predictions.copy()

    df_scored = df_scored.rename(
        columns={production_model_id: "production_pd"}
    )

    df_scored = add_score_percentiles(
        df_scored,
        score_col="production_pd",
        percentile_col="score_percentile",
    )

    df_scored = assign_risk_bands(
        df_scored,
        percentile_col="score_percentile",
        band_col="risk_band",
    )

    return df_scored