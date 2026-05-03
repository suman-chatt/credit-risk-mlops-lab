from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_curve


SPLIT = "validation"
N_POINTS_PER_MODEL = 75

SCORES_PATH = Path("outputs_src/scoring/model_scores_wide.csv")
ENSEMBLE_REGISTRY_PATH = Path("outputs_src/ensemble_registry.xlsx")
KS_SAMPLE_PATH = Path("outputs_src/diagnostics/ks_curves_sample.csv")


def build_ks_curve_sample(scores: pd.DataFrame, model_id: str) -> pd.DataFrame:
    model_df = scores[["split", "y_true", model_id]].dropna().copy()
    model_df = model_df.rename(columns={model_id: "predicted_pd"})

    fpr, tpr, _ = roc_curve(model_df["y_true"], model_df["predicted_pd"])

    ks_df = pd.DataFrame(
        {
            "table_name": "ks_curve",
            "model_id": model_id,
            "split": SPLIT,
            "population_pct": range(len(tpr)),
            "cum_bad_pct": tpr,
            "cum_good_pct": fpr,
        }
    )

    ks_df["ks"] = ks_df["cum_bad_pct"] - ks_df["cum_good_pct"]

    if len(ks_df) > N_POINTS_PER_MODEL:
        ks_df = (
            ks_df.iloc[
                pd.Series(range(len(ks_df)))
                .sample(n=N_POINTS_PER_MODEL, random_state=42)
                .sort_values()
                .index
            ]
            .copy()
        )

    ks_df["population_pct"] = (
        pd.Series(range(len(ks_df))) / max(len(ks_df) - 1, 1)
    )

    return ks_df.reset_index(drop=True)


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

    if KS_SAMPLE_PATH.exists():
        ks_sample = pd.read_csv(KS_SAMPLE_PATH)
    else:
        ks_sample = pd.DataFrame()

    existing_ensemble_models = []
    if not ks_sample.empty and "model_id" in ks_sample.columns:
        existing_ensemble_models = sorted(
            set(ks_sample["model_id"].astype(str).str.upper()).intersection(usable_models)
        )

    print("Usable ensemble models found in scoring file:")
    print(usable_models)

    print("\nEnsemble models already present in ks_curves_sample.csv:")
    print(existing_ensemble_models)

    if missing_models:
        print("\nSkipped ensemble models missing from model_scores_wide.csv:")
        print(missing_models)

    if not usable_models:
        raise ValueError("No usable ensemble model columns found in model_scores_wide.csv")

    new_rows = []
    for model_id in usable_models:
        new_rows.append(build_ks_curve_sample(scores, model_id))

    ensemble_ks = pd.concat(new_rows, ignore_index=True)

    if not ks_sample.empty and "model_id" in ks_sample.columns:
        ks_sample["model_id"] = ks_sample["model_id"].astype(str).str.upper().str.strip()
        ks_sample = ks_sample[~ks_sample["model_id"].isin(usable_models)].copy()

    final_df = pd.concat([ks_sample, ensemble_ks], ignore_index=True)
    KS_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(KS_SAMPLE_PATH, index=False)

    print(f"\nUpdated: {KS_SAMPLE_PATH}")
    print(f"Added/refreshed ensemble KS rows for {len(usable_models)} models.")
    print(f"Final row count: {len(final_df):,}")


if __name__ == "__main__":
    main()