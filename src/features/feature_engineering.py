import pandas as pd

from src.features.binning import FINAL_BIN_COLS


TARGET_COL = "SeriousDlqin2yrs"


TREE_FEATURES = [
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


SCALED_FEATURES = [f"{col}_scaled" for col in TREE_FEATURES]


WOE_FEATURES = [f"{col}_woe" for col in FINAL_BIN_COLS]


BINNED_LOGIT_FEATURES = FINAL_BIN_COLS.copy()


def validate_features(df: pd.DataFrame, features: list[str], dataset_name: str) -> None:
    missing = [col for col in features if col not in df.columns]

    if missing:
        raise ValueError(f"{dataset_name} missing features: {missing}")


def build_model_ready_datasets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Create slim model-ready datasets from the engineered dataframe.

    No transformations happen here.
    This function only selects validated final columns.

    Outputs:
    - tree: raw / capped / log / flag features for tree-based models
    - scaled: scaled numeric features for regularized logistic and NN-style models
    - woe: WOE-transformed features for scorecard-style logistic regression
    - binned_logit: categorical binned features for interpretable binned logistic regression
    """

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    validate_features(df, TREE_FEATURES, "tree dataset")
    validate_features(df, SCALED_FEATURES, "scaled dataset")
    validate_features(df, WOE_FEATURES, "WOE dataset")
    validate_features(df, BINNED_LOGIT_FEATURES, "binned logistic dataset")

    df_tree = df[[TARGET_COL] + TREE_FEATURES].copy()
    df_scaled = df[[TARGET_COL] + SCALED_FEATURES].copy()
    df_woe = df[[TARGET_COL] + WOE_FEATURES].copy()
    df_binned_logit = df[[TARGET_COL] + BINNED_LOGIT_FEATURES].copy()

    return {
        "tree": df_tree,
        "scaled": df_scaled,
        "woe": df_woe,
        "binned_logit": df_binned_logit,
    }