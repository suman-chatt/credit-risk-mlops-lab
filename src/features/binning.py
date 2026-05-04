import numpy as np
import pandas as pd


FINAL_BIN_COLS = [
    "RevolvingUtilization_final_bin",
    "age_final_bin_v2",
    "MonthlyIncome_final_bin",
    "MonthlyIncome_missing_flag_bin",
    "DebtRatio_high_flag_bin",
    "DebtRatio_explicit_bin",
    "NumberOfDependents_final_bin",
    "NumberOfDependents_missing_flag_bin",
    "RealEstateLoans_final_bin",
    "NumberOfTime30-59DaysPastDueNotWorse_final_bin",
    "NumberOfTime60-89DaysPastDueNotWorse_final_bin",
    "NumberOfTimes90DaysLate_final_bin",
]


FINAL_WOE_VARS = {
    "RevolvingUtilizationOfUnsecuredLines": "RevolvingUtilization_final_bin",
    "age": "age_final_bin_v2",
    "MonthlyIncome_median": "MonthlyIncome_final_bin",
    "MonthlyIncome_missing_flag": "MonthlyIncome_missing_flag_bin",
    "DebtRatio_high_flag": "DebtRatio_high_flag_bin",
    "DebtRatio": "DebtRatio_explicit_bin",
    "NumberOfDependents_median": "NumberOfDependents_final_bin",
    "NumberOfDependents_missing_flag": "NumberOfDependents_missing_flag_bin",
    "NumberRealEstateLoansOrLines": "RealEstateLoans_final_bin",
    "NumberOfTime30-59DaysPastDueNotWorse": "NumberOfTime30-59DaysPastDueNotWorse_final_bin",
    "NumberOfTime60-89DaysPastDueNotWorse": "NumberOfTime60-89DaysPastDueNotWorse_final_bin",
    "NumberOfTimes90DaysLate": "NumberOfTimes90DaysLate_final_bin",
}


def apply_final_binning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply final approved binning rules from Notebook 03.

    These bins were manually finalized after reviewing:
    1. WOE / IV behavior
    2. monotonicity and stability
    3. practical credit-risk business interpretation

    These are not arbitrary technical splits. They intentionally avoid
    strange cutoffs such as age 32 or income 94,783 and instead use
    explainable borrower / affordability / delinquency bands.
    """

    df_out = df.copy()
    # Revolving utilization
    df_out["RevolvingUtilization_final_bin"] = pd.cut(
        df_out["RevolvingUtilizationOfUnsecuredLines"],
        bins=[-0.001, 0, 0.05, 0.15, 0.30, 0.50, 0.75, 1.00, np.inf],
        labels=[
            "=0",
            "0-5%",
            "5-15%",
            "15-30%",
            "30-50%",
            "50-75%",
            "75-100%",
            "100%+",
        ],
        right=True,
        include_lowest=True,
    )

    # Age
    df_out["age_final_bin_v2"] = pd.cut(
        df_out["age"],
        bins=[20, 39, 49, 59, 69, np.inf],
        labels=[
            "21-39",
            "40-49",
            "50-59",
            "60-69",
            "70+",
        ],
        right=True,
        include_lowest=True,
    )

    # Open credit lines
    # Retained for consistency with Notebook 03 WOE review.
    # Not currently included in FINAL_BIN_COLS.
    df_out["OpenCreditLines_final_bin_v2"] = pd.cut(
        df_out["NumberOfOpenCreditLinesAndLoans"],
        bins=[-0.001, 0, 1, 2, 5, 9, 14, np.inf],
        labels=[
            "0",
            "1",
            "2",
            "3-5",
            "6-9",
            "10-14",
            "15+",
        ],
        right=True,
        include_lowest=True,
    )

    # Monthly income
    df_out["MonthlyIncome_final_bin"] = pd.cut(
        df_out["MonthlyIncome_median"],
        bins=[-0.001, 5000, 10000, np.inf],
        labels=[
            "0-5000",
            "5000-10000",
            "10000+",
        ],
        right=True,
        include_lowest=True,
    )

    # DebtRatio

    df_out["DebtRatio_final_bin"] = pd.cut(
        df_out["DebtRatio"],
        bins=[-0.001, 0.20, 0.40, 0.60, 1.00, 2.00, np.inf],
        labels=[
            "0-0.20",
            "0.20-0.40",
            "0.40-0.60",
            "0.60-1.00",
            "1.00-2.00",
            "2.00+",
        ],
        right=True,
        include_lowest=True,
    )

    df_out["DebtRatio_explicit_bin"] = pd.cut(
        df_out["DebtRatio"],
        bins=[-0.001, 0.20, 0.40, 0.60, 1.00, 2.00, np.inf],
        labels=[
            "0-0.20",
            "0.20-0.40",
            "0.40-0.60",
            "0.60-1.00",
            "1.00-2.00",
            "2.00+",
        ],
        right=True,
        include_lowest=True,
    )

    # Real estate loans / lines
    df_out["RealEstateLoans_final_bin"] = pd.cut(
        df_out["NumberRealEstateLoansOrLines"],
        bins=[-0.001, 0, 2, np.inf],
        labels=[
            "0",
            "1-2",
            "3+",
        ],
        right=True,
        include_lowest=True,
    )

    # Number of dependents
    df_out["NumberOfDependents_final_bin"] = np.select(
        [
            df_out["NumberOfDependents_median"] == 0,
            df_out["NumberOfDependents_median"] == 1,
            df_out["NumberOfDependents_median"] == 2,
            df_out["NumberOfDependents_median"] >= 3,
        ],
        ["0", "1", "2", "3+"],
        default="unknown",
    )

    # Delinquency count variables
    delinquency_vars = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]

    for col in delinquency_vars:
        df_out[col + "_final_bin"] = np.select(
            [
                df_out[col] == 0,
                df_out[col] == 1,
                df_out[col] == 2,
                df_out[col] >= 3,
            ],
            ["0", "1", "2", "3+"],
            default="unknown",
        )
    # Binary flags as string bins
    binary_flags = [
        "DebtRatio_high_flag",
        "MonthlyIncome_missing_flag",
        "NumberOfDependents_missing_flag",
    ]

    for col in binary_flags:
        df_out[col + "_bin"] = df_out[col].astype(int).astype(str)

    return df_out


def validate_final_bins(df: pd.DataFrame) -> None:
    """
    Validate that all final bin columns exist and contain no missing values.
    """

    missing_cols = [col for col in FINAL_BIN_COLS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing final bin columns: {missing_cols}")

    missing_values = df[FINAL_BIN_COLS].isna().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        raise ValueError(
            f"Final bin columns contain missing values: {missing_values.to_dict()}"
        )