import pandas as pd

from src.config import DATA_DIR, OUTPUTS_SRC_DIR


NOTEBOOK_BIN_NAME_MAP = {
    "RevolvingUtilization_final_bin": "RevolvingUtilization_final_bin",
    "age_final_bin_v2": "age_final_bin_v2",
    "MonthlyIncome_final_bin": "MonthlyIncome_final_bin",
    "MonthlyIncome_missing_flag_bin": "MonthlyIncome_missing_flag_bin",
    "DebtRatio_high_flag_bin": "DebtRatio_high_flag_bin",
    "NumberOfDependents_final_bin": "NumberOfDependents_final_bin",
    "NumberOfDependents_missing_flag_bin": "NumberOfDependents_missing_flag_bin",
    "RealEstateLoans_final_bin": "RealEstateLoans_final_bin",
    "NumberOfTime30-59DaysPastDueNotWorse_final_bin": "NumberOfTime30-59DaysPastDueNotWorse_final_bin",
    "NumberOfTime60-89DaysPastDueNotWorse_final_bin": "NumberOfTime60-89DaysPastDueNotWorse_final_bin",
    "NumberOfTimes90DaysLate_final_bin": "NumberOfTimes90DaysLate_final_bin",
}


def main():
    notebook_path = DATA_DIR / "df_logit_bins_ready.csv"
    src_path = OUTPUTS_SRC_DIR / "features" / "df_model_binned_src.csv"

    df_notebook = pd.read_csv(notebook_path)
    df_src = pd.read_csv(src_path)

    print("\n=== BINNED DATASET SHAPES ===")
    print(f"Notebook: {df_notebook.shape}")
    print(f"src:      {df_src.shape}")

    print("\n=== BIN COUNT COMPARISON ===")

    for src_col, notebook_col in NOTEBOOK_BIN_NAME_MAP.items():
        print(f"\n{src_col}")

        if notebook_col not in df_notebook.columns:
            print(f"  Notebook missing column: {notebook_col}")
            continue

        if src_col not in df_src.columns:
            print(f"  src missing column: {src_col}")
            continue

        notebook_counts = (
            df_notebook[notebook_col]
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
        )

        src_counts = (
            df_src[src_col]
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
        )

        compare = pd.concat(
            [notebook_counts, src_counts],
            axis=1,
            keys=["notebook_count", "src_count"],
        ).fillna(0)

        compare["diff"] = compare["src_count"] - compare["notebook_count"]

        print(compare)

    print("\n=== SRC-ONLY EXTRA BIN CHECK ===")
    extra_col = "DebtRatio_explicit_bin"

    if extra_col in df_src.columns:
        print(df_src[extra_col].astype(str).value_counts(dropna=False).sort_index())
    else:
        print(f"{extra_col} not found in src.")


if __name__ == "__main__":
    main()