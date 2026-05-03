import json

import pandas as pd

from src.config import OUTPUTS_SRC_DIR
from src.models.data_split import prepare_training_datasets
from src.models.ensemble import run_ensemble_tournament
from src.models.model_registry import (
    add_split_metadata,
    create_empty_registry,
    export_registry_artifacts,
    export_registry_to_excel,
)


MODEL_READY_DIR = OUTPUTS_SRC_DIR / "model_ready"
REGISTRY_DIR = OUTPUTS_SRC_DIR / "registry"
ENSEMBLE_OUTPUT_PATH = OUTPUTS_SRC_DIR / "ensemble_registry.xlsx"

ensemble_model_dir = OUTPUTS_SRC_DIR / "ensemble" / "models"
saved_ensemble_models = sorted(ensemble_model_dir.glob("*.joblib"))

print(f"Ensemble model artifacts saved to: {ensemble_model_dir}")
print(f"Saved ensemble model artifacts: {len(saved_ensemble_models)}")

expected_models = {
    "EASTACK001",
    "WASTACK001",
    "WASTACK002",
    "WASTACK003",
    "WASTACK004",
    "RASTACK001",
    "LRSTACK001",
    "RIDGESTACK001",
    "LASSOSTACK001",
    "RFSTACK001",
    "NNSTACK001",
    "XGBSTACK001",
}

saved_ids = {p.stem for p in saved_ensemble_models}
missing_ids = sorted(expected_models - saved_ids)

if missing_ids:
    print("\nWARNING: Missing expected ensemble model artifacts:")
    for model_id in missing_ids:
        print(f"  - {model_id}")
else:
    print("All expected ensemble model artifacts were saved.")


def main():
    df_tree = pd.read_csv(MODEL_READY_DIR / "df_tree_model_ready_src.csv")
    df_scaled = pd.read_csv(MODEL_READY_DIR / "df_scaled_model_ready_src.csv")
    df_woe = pd.read_csv(MODEL_READY_DIR / "df_logit_woe_ready_src.csv")
    df_binned = pd.read_csv(MODEL_READY_DIR / "df_logit_bins_ready_src.csv")

    datasets = prepare_training_datasets(
        df_tree=df_tree,
        df_scaled=df_scaled,
        df_woe=df_woe,
        df_binned_logit=df_binned,
        validation_size=0.30,
        random_state=42,
    )

    model_summary = pd.read_parquet(REGISTRY_DIR / "model_summary.parquet")

    with open(REGISTRY_DIR / "artifact_paths.json", "r") as f:
        artifact_paths = json.load(f)

    registry = create_empty_registry()

    registry = add_split_metadata(
        registry,
        {
            "validation_size": 0.30,
            "random_state": 42,
            "train_rows": len(datasets["train_idx"]),
            "validation_rows": len(datasets["val_idx"]),
            "train_target_rate": datasets["tree"]["y_train"].mean(),
            "validation_target_rate": datasets["tree"]["y_val"].mean(),
            "source_registry": str(REGISTRY_DIR / "model_summary.parquet"),
        },
    )

    registry = run_ensemble_tournament(
        registry=registry,
        model_summary=model_summary,
        artifact_paths=artifact_paths,
        datasets=datasets,
    )

    export_registry_to_excel(
        registry=registry,
        output_path=ENSEMBLE_OUTPUT_PATH,
    )

    export_registry_artifacts(
        registry=registry,
        output_dir=OUTPUTS_SRC_DIR / "ensemble",
        save_models=True,
    )

    summary = pd.DataFrame(registry["summary"])

    print("\nEnsemble training complete.")
    print(f"Registry saved to: {ENSEMBLE_OUTPUT_PATH}")
    print(f"Ensemble models trained: {len(summary)}")

    print("\nTop ensemble models by validation AUC:")
    print(
        summary[
            [
                "model_id",
                "model_name",
                "feature_count",
                "validation_auc",
                "validation_ks",
                "validation_log_loss",
                "validation_brier_score",
            ]
        ]
        .sort_values("validation_auc", ascending=False)
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()