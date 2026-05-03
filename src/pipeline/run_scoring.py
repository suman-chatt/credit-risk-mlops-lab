from pathlib import Path

import pandas as pd

from src.models.data_split import one_hot_encode_binned_dataset
from src.scoring.risk_bands import (
    build_scored_output,
    create_risk_band_summary,
    create_score_correlation_table,
    create_score_summary,
)
from src.scoring.score_models import (
    load_artifact_paths,
    load_model_registry,
    score_models,
    select_champion_models,
)


BASE_DIR = Path("outputs_src")

REGISTRY_PATH = BASE_DIR / "registry" / "model_summary.parquet"
ARTIFACT_PATHS = BASE_DIR / "registry" / "artifact_paths.json"

ENSEMBLE_REGISTRY_PATH = BASE_DIR / "ensemble" / "registry" / "model_summary.parquet"
ENSEMBLE_ARTIFACT_PATHS = BASE_DIR / "ensemble" / "registry" / "artifact_paths.json"

SCORING_OUTPUT_DIR = BASE_DIR / "scoring"


def load_scoring_datasets() -> dict[str, pd.DataFrame]:
    """
    Load model-ready scoring datasets already created by the src pipeline.
    """

    model_ready_dir = BASE_DIR / "model_ready"

    df_tree = pd.read_csv(model_ready_dir / "df_tree_model_ready_src.csv")
    df_scaled = pd.read_csv(model_ready_dir / "df_scaled_model_ready_src.csv")
    df_woe = pd.read_csv(model_ready_dir / "df_logit_woe_ready_src.csv")
    df_binned = pd.read_csv(model_ready_dir / "df_logit_bins_ready_src.csv")

    df_binned_ohe = one_hot_encode_binned_dataset(df_binned)

    target_col = "SeriousDlqin2yrs"

    return {
        "tree": df_tree.drop(columns=[target_col], errors="ignore"),
        "scaled": df_scaled.drop(columns=[target_col], errors="ignore"),
        "woe": df_woe.drop(columns=[target_col], errors="ignore"),
        "binned_logit": df_binned.drop(columns=[target_col], errors="ignore"),
        "binned_ohe": df_binned_ohe.drop(columns=[target_col], errors="ignore"),
    }


def add_base_model_predictions_for_ensembles(
    full_registry: pd.DataFrame,
    full_artifacts: dict,
    datasets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Create base-model prediction matrix needed to score ensemble models.
    """

    ensemble_rows = full_registry[
        full_registry["dataset_type"] == "base_model_predictions"
    ].copy()

    if ensemble_rows.empty:
        return datasets

    required_base_model_ids = set()

    for _, row in ensemble_rows.iterrows():
        required_base_model_ids.update(
            [x.strip() for x in row["features"].split(",") if x.strip()]
        )

    base_predictions = score_models(
        model_registry=full_registry,
        artifact_paths=full_artifacts,
        datasets=datasets,
        model_ids=list(required_base_model_ids),
    )

    datasets = datasets.copy()
    datasets["base_model_predictions"] = base_predictions.drop(
        columns=["row_id"],
        errors="ignore",
    )

    return datasets


def main() -> None:
    SCORING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model registries...")

    base_registry = load_model_registry(REGISTRY_PATH)
    ensemble_registry = load_model_registry(ENSEMBLE_REGISTRY_PATH)

    base_artifacts = load_artifact_paths(ARTIFACT_PATHS)
    ensemble_artifacts = load_artifact_paths(ENSEMBLE_ARTIFACT_PATHS)

    full_registry = pd.concat(
        [base_registry, ensemble_registry],
        ignore_index=True,
    )

    full_artifacts = {**base_artifacts, **ensemble_artifacts}

    print("Selecting champion models...")

    champions = select_champion_models(full_registry)

    print("Champions selected:")
    for champion_type, model_id in champions.items():
        print(f"{champion_type}: {model_id}")

    print("Loading scoring datasets...")

    datasets = load_scoring_datasets()

    print("Creating base-model predictions for ensemble scoring...")

    datasets = add_base_model_predictions_for_ensembles(
        full_registry=full_registry,
        full_artifacts=full_artifacts,
        datasets=datasets,
    )

    print("Scoring champion models...")

    model_ids_to_score = list(champions.values())

    predictions = score_models(
        model_registry=full_registry,
        artifact_paths=full_artifacts,
        datasets=datasets,
        model_ids=model_ids_to_score,
    )

    production_model = champions["performance_champion"]

    print(f"Using production model: {production_model}")

    scored_df = build_scored_output(
        predictions=predictions,
        production_model_id=production_model,
    )

    print("Generating summaries...")

    risk_band_summary = create_risk_band_summary(
        scored_df,
        score_col="production_pd",
    )

    score_summary = create_score_summary(
        predictions,
        score_cols=model_ids_to_score,
    )

    correlation_table = create_score_correlation_table(
        predictions,
        score_cols=model_ids_to_score,
    )

    print("Saving outputs...")

    scored_df.to_csv(SCORING_OUTPUT_DIR / "scored_applicants.csv", index=False)
    predictions.to_csv(SCORING_OUTPUT_DIR / "all_model_scores.csv", index=False)
    risk_band_summary.to_csv(
        SCORING_OUTPUT_DIR / "risk_band_summary.csv",
        index=False,
    )
    score_summary.to_csv(SCORING_OUTPUT_DIR / "score_summary.csv", index=False)
    correlation_table.to_csv(
        SCORING_OUTPUT_DIR / "score_correlations.csv",
        index=False,
    )

    print("Scoring pipeline complete.")


if __name__ == "__main__":
    main()