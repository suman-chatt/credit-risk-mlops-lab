import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_SCALE_FEATURES = [
    "RevolvingUtilization_capped",
    "age",
    "age_sq",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio_capped",
    "MonthlyIncome_median",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents_median",
    "RevolvingUtilization_log",
    "DebtRatio_log",
    "MonthlyIncome_missing_flag",
    "NumberOfDependents_missing_flag",
    "DebtRatio_high_flag",
    "RevolvingUtilization_high_flag",
]


def fit_scaler(
    df: pd.DataFrame,
    scale_features: list[str] | None = None,
) -> StandardScaler:
    """
    Fit a StandardScaler on selected numeric features.
    """
    if scale_features is None:
        scale_features = DEFAULT_SCALE_FEATURES

    missing_features = [col for col in scale_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing scale features: {missing_features}")

    scaler = StandardScaler()
    scaler.fit(df[scale_features])

    return scaler


def transform_with_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    scale_features: list[str] | None = None,
    suffix: str = "_scaled",
) -> pd.DataFrame:
    """
    Apply fitted scaler and append scaled columns.
    """
    if scale_features is None:
        scale_features = DEFAULT_SCALE_FEATURES

    missing_features = [col for col in scale_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing scale features: {missing_features}")

    df_out = df.copy()

    scaled_values = scaler.transform(df_out[scale_features])

    scaled_cols = [f"{col}{suffix}" for col in scale_features]

    df_scaled = pd.DataFrame(
        scaled_values,
        columns=scaled_cols,
        index=df_out.index,
    )

    df_out = pd.concat([df_out, df_scaled], axis=1)

    return df_out


def fit_transform_scaler(
    df: pd.DataFrame,
    scale_features: list[str] | None = None,
    suffix: str = "_scaled",
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Fit scaler and return dataframe with appended scaled columns.
    """
    scaler = fit_scaler(df, scale_features)

    df_scaled = transform_with_scaler(
        df=df,
        scaler=scaler,
        scale_features=scale_features,
        suffix=suffix,
    )

    return df_scaled, scaler