import json

import pandas as pd

from src.config import OUTPUTS_DIR, OUTPUTS_SRC_DIR


FEATURE_NAME_MAP = {
    "RevolvingUtilization_final_bin": "RevolvingUtilizationOfUnsecuredLines",
    "age_final_bin_v2": "age",
    "MonthlyIncome_final_bin": "MonthlyIncome_median",
    "MonthlyIncome_missing_flag_bin": "MonthlyIncome_missing_flag",
    "DebtRatio_high_flag_bin": "DebtRatio_high_flag",
    "DebtRatio_explicit_bin": "DebtRatio",
    "NumberOfDependents_final_bin": "NumberOfDependents_median",
    "NumberOfDependents_missing_flag_bin": "NumberOfDependents_missing_flag",
    "RealEstateLoans_final_bin": "NumberRealEstateLoansOrLines",
    "NumberOfTime30-59DaysPastDueNotWorse_final_bin": "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse_final_bin": "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate_final_bin": "NumberOfTimes90DaysLate",
}


def main():
    notebook_path = OUTPUTS_DIR / "final_iv_ranking.csv"
    src_path = OUTPUTS_SRC_DIR / "woe" / "iv_values.json"

    df_notebook = pd.read_csv(notebook_path)

    with open(src_path, "r") as f:
        src_iv_raw = json.load(f)

    src_iv = {
        FEATURE_NAME_MAP.get(feature, feature): iv
        for feature, iv in src_iv_raw.items()
    }

    df_src = pd.DataFrame(
        [{"feature": feature, "src_iv_total": iv} for feature, iv in src_iv.items()]
    )

    df_compare = df_notebook.merge(
        df_src,
        on="feature",
        how="outer",
    )

    df_compare["iv_diff"] = df_compare["src_iv_total"] - df_compare["iv_total"]
    df_compare["abs_iv_diff"] = df_compare["iv_diff"].abs()

    df_compare["src_iv_rank"] = df_compare["src_iv_total"].rank(
        method="dense",
        ascending=False,
    )

    df_compare["rank_diff"] = df_compare["src_iv_rank"] - df_compare["iv_rank"]

    df_compare["iv_close_flag"] = df_compare["abs_iv_diff"] < 1e-3

    df_compare["rank_match_flag"] = (
        df_compare["rank_diff"].fillna(0) == 0
    )

    df_compare = df_compare.sort_values(
        ["rank_match_flag", "iv_rank", "src_iv_rank"],
        ascending=[True, True, True],
    )

    display_cols = [
        "feature",
        "iv_total",
        "iv_rank",
        "src_iv_total",
        "src_iv_rank",
        "iv_diff",
        "abs_iv_diff",
        "rank_diff",
        "iv_close_flag",
        "rank_match_flag",
    ]

    print("\n=== IV COMPARISON: NOTEBOOK vs SRC ===\n")
    print(df_compare[display_cols].to_string(index=False))

    original_features = df_compare[df_compare["iv_rank"].notna()]
    extra_src_features = df_compare[df_compare["iv_rank"].isna()]

    rank_mismatches = original_features[
        ~original_features["rank_match_flag"]
    ]

    iv_large_diffs = original_features[
        ~original_features["iv_close_flag"]
    ]

    print("\n=== SUMMARY ===")
    print(f"Notebook IV rows: {len(df_notebook)}")
    print(f"src IV rows: {len(df_src)}")
    print(f"Original feature rank mismatches: {len(rank_mismatches)}")
    print(f"Original feature IV diffs >= 0.001: {len(iv_large_diffs)}")
    print(f"Extra src-only features: {len(extra_src_features)}")

    if not extra_src_features.empty:
        print("\nExtra src-only features:")
        print(extra_src_features[["feature", "src_iv_total", "src_iv_rank"]].to_string(index=False))

    if not rank_mismatches.empty:
        print("\nRank mismatches:")
        print(rank_mismatches[display_cols].to_string(index=False))

    if not iv_large_diffs.empty:
        print("\nLarge IV differences:")
        print(iv_large_diffs[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()