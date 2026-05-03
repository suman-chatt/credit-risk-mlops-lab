import pandas as pd


DEFAULT_REASON_MAP = {
    "RevolvingUtilization": "High revolving credit utilization",
    "NumberOfTime30-59DaysPastDueNotWorse": "Recent 30-59 day delinquency history",
    "NumberOfTime60-89DaysPastDueNotWorse": "Recent 60-89 day delinquency history",
    "NumberOfTimes90DaysLate": "Severe delinquency history",
    "DebtRatio": "High debt burden",
    "MonthlyIncome": "Income / affordability profile",
    "NumberOfOpenCreditLinesAndLoans": "Credit line exposure",
    "NumberRealEstateLoansOrLines": "Real estate loan exposure",
    "NumberOfDependents": "Household dependent burden",
    "age": "Borrower age profile",
}


def map_feature_to_reason(
    feature: str,
    reason_map: dict | None = None,
) -> str:
    """
    Map model feature name to business-friendly reason code.
    """

    if reason_map is None:
        reason_map = DEFAULT_REASON_MAP

    for key, reason in reason_map.items():
        if key in feature:
            return reason

    return "Other model signal"


def create_reason_code_table_from_feature_importance(
    feature_importance: pd.DataFrame,
    reason_map: dict | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Create global reason-code table from feature importance.
    """

    df = feature_importance.copy()

    df["reason_code"] = df["feature"].apply(
        lambda x: map_feature_to_reason(x, reason_map)
    )

    reason_summary = (
        df.groupby("reason_code", observed=False)
        .agg(
            total_importance=("importance", "sum"),
            max_importance=("importance", "max"),
            feature_count=("feature", "nunique"),
        )
        .reset_index()
        .sort_values("total_importance", ascending=False)
    )

    return reason_summary.head(top_n)


def create_applicant_reason_codes_from_scores(
    scored_df: pd.DataFrame,
    score_cols: list[str],
    production_score_col: str = "production_pd",
) -> pd.DataFrame:
    """
    Lightweight applicant-level reason proxy.

    Since full SHAP is handled separately, this function summarizes
    model agreement / disagreement for each scored applicant.
    """

    df = scored_df.copy()

    available_score_cols = [col for col in score_cols if col in df.columns]

    if production_score_col not in df.columns:
        raise ValueError(f"Missing production score column: {production_score_col}")

    if not available_score_cols:
        raise ValueError("No score columns available for reason-code proxy.")

    df["mean_challenger_score"] = df[available_score_cols].mean(axis=1)
    df["score_disagreement"] = (
        df[available_score_cols].max(axis=1)
        - df[available_score_cols].min(axis=1)
    )

    output_cols = [
        "row_id",
        production_score_col,
        "score_percentile",
        "risk_band",
        "mean_challenger_score",
        "score_disagreement",
    ]

    output_cols = [col for col in output_cols if col in df.columns]

    return df[output_cols].copy()