from pathlib import Path

import pandas as pd


SPLIT = "validation"
N_BINS = 10

SCORES_PATH = Path("outputs_src/scoring/model_scores_wide.csv")
ENSEMBLE_REGISTRY_PATH = Path("outputs_src/ensemble_registry.xlsx")
CALIBRATION_PATH = Path("outputs_src/diagnostics/calibration_tables.csv")


def build_calibration_table(scores: pd.DataFrame, model_id: str) -> pd.DataFrame:
    model_df = scores[["split", "y_true", model_id]].dropna().copy()
    model_df = model_df.rename(columns={model_id: "predicted_pd"})

    model_df["bin"] = pd.qcut(
        model_df["predicted_pd"],
        q=N_BINS,
        labels=False,
        duplicates="drop",
    )

    rows = []

    for bin_id, group in model_df.groupby("bin", observed=True):
        rows.append(
            {
                "table_name": "calibration_table",
                "model_id": model_id,
                "split": SPLIT,
                "bin": int(bin_id),
                "count": len(group),
                "predicted_prob": group["predicted_pd"].mean(),
                "actual_rate": group["y_true"].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)


def main():
    scores = pd.read_csv(SCORES_PATH)
    ensemble_registry = pd.read_excel(ENSEMBLE_REGISTRY_PATH)

    scores["split"] = scores["split"].astype(str).str.lower().str.strip()
    scores = scores[scores["split"] == SPLIT].copy()

    if "y_true" not in scores.columns:
        raise ValueError("y_true column not found in model_scores_wide.csv")

    ensemble_model_ids = (
        ensemble_registry["model_id"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .unique()
        .tolist()
    )

    available_models = set(scores.columns)
    usable_models = [m for m in ensemble_model_ids if m in available_models]
    missing_models = [m for m in ensemble_model_ids if m not in available_models]

    if not usable_models:
        raise ValueError("No usable ensemble model columns found in model_scores_wide.csv")

    if CALIBRATION_PATH.exists():
        calibration_df = pd.read_csv(CALIBRATION_PATH)
    else:
        calibration_df = pd.DataFrame()

    existing_ensemble_models = []
    if not calibration_df.empty and "model_id" in calibration_df.columns:
        calibration_df["model_id"] = (
            calibration_df["model_id"].astype(str).str.upper().str.strip()
        )
        existing_ensemble_models = sorted(
            set(calibration_df["model_id"]).intersection(usable_models)
        )

    print("Usable ensemble models found in scoring file:")
    print(usable_models)

    print("\nEnsemble models already present in calibration_tables.csv:")
    print(existing_ensemble_models)

    if missing_models:
        print("\nSkipped ensemble models missing from model_scores_wide.csv:")
        print(missing_models)

    new_rows = []
    for model_id in usable_models:
        new_rows.append(build_calibration_table(scores, model_id))

    ensemble_calibration = pd.concat(new_rows, ignore_index=True)

    if not calibration_df.empty and "model_id" in calibration_df.columns:
        calibration_df = calibration_df[
            ~calibration_df["model_id"].isin(usable_models)
        ].copy()

    final_df = pd.concat([calibration_df, ensemble_calibration], ignore_index=True)

    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(CALIBRATION_PATH, index=False)

    print(f"\nUpdated: {CALIBRATION_PATH}")
    print(f"Added/refreshed calibration rows for {len(usable_models)} ensemble models.")
    print(f"Final row count: {len(final_df):,}")


if __name__ == "__main__":
    main()