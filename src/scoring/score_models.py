import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm


def parse_features(feature_string: str) -> list[str]:
    """
    Convert registry feature string into a clean feature list.
    """
    if pd.isna(feature_string):
        return []

    return [x.strip() for x in str(feature_string).split(",") if x.strip()]


def load_artifact_paths(path) -> dict:
    """
    Load model artifact paths from JSON.
    """
    with open(path, "r") as f:
        return json.load(f)


def load_model_registry(path) -> pd.DataFrame:
    """
    Load model registry summary from parquet or csv.
    """
    path = Path(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported registry file type: {path}")


def load_model_artifact(model_id: str, artifact_paths: dict):
    """
    Load fitted model object for a model_id.
    """
    if model_id not in artifact_paths:
        raise ValueError(f"No artifact path found for model_id: {model_id}")

    return joblib.load(artifact_paths[model_id])


def align_one_hot_columns(
    X: pd.DataFrame,
    expected_columns: list[str],
) -> pd.DataFrame:
    """
    Align one-hot encoded scoring columns to training columns.
    """
    return X.reindex(columns=expected_columns, fill_value=0)


def select_model_input(
    model_row: pd.Series,
    datasets: dict,
) -> pd.DataFrame:
    """
    Select the correct scoring dataset based on model registry metadata.

    Expected datasets keys:
    - tree
    - scaled
    - woe
    - binned_logit
    - binned_ohe
    """

    dataset_type = model_row["dataset_type"]
    features = parse_features(model_row["features"])

    if dataset_type == "tree":
        return datasets["tree"][features].copy()

    if dataset_type in ["scaled", "scaled_pruned"]:
        return datasets["scaled"][features].copy()

    if dataset_type in ["woe", "woe_pruned"]:
        return datasets["woe"][features].copy()

    if dataset_type == "binned_ohe":
        return datasets["binned_ohe"][features].copy()

    if dataset_type == "binned_logit":
        X_raw = datasets["binned_logit"][features].astype(str)
        return pd.get_dummies(X_raw, drop_first=True, dtype=int)

    if dataset_type == "base_model_predictions":
        return datasets["base_model_predictions"][features].copy()

    raise ValueError(f"Unsupported dataset_type for scoring: {dataset_type}")


def score_single_model(
    model,
    model_row: pd.Series,
    datasets: dict,
) -> np.ndarray:
    """
    Score one fitted model using registry metadata.
    """

    model_id = model_row["model_id"]
    model_name = model_row["model_name"]

    X = select_model_input(model_row, datasets)

    if model_id.startswith(("WLOG", "BLOG")) or "staged logistic" in model_name:
        expected_cols = [x for x in model.params.index if x != "const"]
        X = align_one_hot_columns(X, expected_cols)

        X_const = sm.add_constant(X, has_constant="add")
        return np.asarray(model.predict(X_const))

    return model.predict_proba(X)[:, 1]


def score_models(
    model_registry: pd.DataFrame,
    artifact_paths: dict,
    datasets: dict,
    model_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Score selected models and return one dataframe of predictions.

    Output columns:
    - row_id
    - one prediction column per model_id
    """

    if model_ids is not None:
        registry_subset = model_registry[
            model_registry["model_id"].isin(model_ids)
        ].copy()
    else:
        registry_subset = model_registry.copy()

    if registry_subset.empty:
        raise ValueError("No models selected for scoring.")

    first_dataset = next(iter(datasets.values()))
    predictions = pd.DataFrame({"row_id": first_dataset.index})

    for _, model_row in registry_subset.iterrows():
        model_id = model_row["model_id"]
        model = load_model_artifact(model_id, artifact_paths)

        predictions[model_id] = score_single_model(
            model=model,
            model_row=model_row,
            datasets=datasets,
        )

    return predictions


def select_top_models_by_auc(
    model_registry: pd.DataFrame,
    top_n: int = 10,
    exclude_model_families: list[str] | None = None,
) -> list[str]:
    """
    Select top model_ids by validation AUC.
    """

    df = model_registry.copy()

    if exclude_model_families:
        df = df[~df["model_family"].isin(exclude_model_families)]

    return (
        df.sort_values("validation_auc", ascending=False)
        .head(top_n)["model_id"]
        .tolist()
    )


def select_champion_models(model_registry: pd.DataFrame) -> dict:
    """
    Select key champion / challenger models for scoring and dashboarding.
    """

    champions = {}

    def best_model_id(mask):
        subset = model_registry[mask].copy()
        if subset.empty:
            return None

        return (
            subset.sort_values("validation_auc", ascending=False)
            .iloc[0]["model_id"]
        )

    champions["performance_champion"] = best_model_id(
        model_registry["model_family"].isin(["ensemble"])
    )

    champions["practical_champion"] = best_model_id(
        model_registry["model_family"].isin(["boosting"])
    )

    champions["neural_network_challenger"] = best_model_id(
        model_registry["model_family"].eq("neural_network")
    )

    champions["simple_challenger"] = best_model_id(
        model_registry["model_id"].str.startswith("BNB")
    )

    champions["interpretable_woe_challenger"] = best_model_id(
        model_registry["model_id"].str.startswith("WLOG")
    )

    champions["interpretable_binned_challenger"] = best_model_id(
        model_registry["model_id"].str.startswith("BLOG")
    )

    return {
        key: value
        for key, value in champions.items()
        if value is not None
    }