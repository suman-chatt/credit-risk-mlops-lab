import numpy as np
import pandas as pd

from src.preprocessing.preprocessing_config import PreprocessingConfig
from src.utils.validation import validate_required_columns


REQUIRED_COLUMNS = [
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfDependents",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
]


def remove_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove index-style unnamed columns created by CSV exports."""
    return df.loc[:, ~df.columns.str.contains("^Unnamed")].copy()


def fit_preprocessor(df: pd.DataFrame) -> dict:
    """
    Learn preprocessing values from the training dataset.

    This mirrors Notebook 02:
    - remove unnamed ID column first
    - remove duplicate rows
    - remove invalid ages
    - learn medians after those row filters
    """
    df_fit = remove_unnamed_columns(df)

    validate_required_columns(df_fit, REQUIRED_COLUMNS)

    df_fit = df_fit.drop_duplicates().reset_index(drop=True)
    df_fit = df_fit[df_fit["age"] > 0].reset_index(drop=True)

    fitted_values = {
        "number_of_dependents_median": df_fit["NumberOfDependents"].median(),
        "monthly_income_median": df_fit["MonthlyIncome"].median(),
    }

    return fitted_values


def transform_preprocessor(
    df: pd.DataFrame,
    fitted_values: dict,
    config: PreprocessingConfig | None = None,
) -> pd.DataFrame:
    """
    Apply Notebook 02 preprocessing logic using fitted values.

    This function intentionally preserves the notebook column names because
    downstream feature engineering currently depends on them.
    """
    if config is None:
        config = PreprocessingConfig()

    df_out = remove_unnamed_columns(df)

    validate_required_columns(df_out, REQUIRED_COLUMNS)

    # Remove duplicate rows after removing the unnamed index column
    df_out = df_out.drop_duplicates().reset_index(drop=True)

    # Remove invalid age rows
    df_out = df_out[df_out["age"] > 0].reset_index(drop=True)

    # NumberOfDependents median imputation
    df_out["NumberOfDependents_missing_flag"] = (
        df_out["NumberOfDependents"].isna().astype(int)
    )

    df_out["NumberOfDependents_median"] = df_out["NumberOfDependents"].fillna(
        fitted_values["number_of_dependents_median"]
    )

    # DebtRatio treatment
    df_out["DebtRatio_high_flag"] = (
        df_out["DebtRatio"] > config.debt_ratio_cap
    ).astype(int)

    df_out["DebtRatio_capped"] = np.where(
        df_out["DebtRatio"] > config.debt_ratio_cap,
        config.debt_ratio_cap,
        df_out["DebtRatio"],
    )

    df_out["DebtRatio_log"] = np.log1p(df_out["DebtRatio_capped"])

    # MonthlyIncome median imputation
    df_out["MonthlyIncome_missing_flag"] = df_out["MonthlyIncome"].isna().astype(int)

    df_out["MonthlyIncome_median"] = df_out["MonthlyIncome"].fillna(
        fitted_values["monthly_income_median"]
    )

    # Age nonlinear feature
    df_out["age_sq"] = df_out["age"] ** 2

    # Revolving utilization treatment
    raw_util_col = "RevolvingUtilizationOfUnsecuredLines"

    df_out["RevolvingUtilization_high_flag"] = (
        df_out[raw_util_col] > config.revolving_utilization_cap
    ).astype(int)

    df_out["RevolvingUtilization_capped"] = df_out[raw_util_col].clip(
        lower=0,
        upper=config.revolving_utilization_cap,
    )

    df_out["RevolvingUtilization_log"] = np.log1p(
        df_out["RevolvingUtilization_capped"]
    )

    return df_out


def preprocess_training_data(
    df: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Fit preprocessing values on training data and return preprocessed dataframe.
    """
    fitted_values = fit_preprocessor(df)

    df_preprocessed = transform_preprocessor(
        df=df,
        fitted_values=fitted_values,
        config=config,
    )

    return df_preprocessed, fitted_values