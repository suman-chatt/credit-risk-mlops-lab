from pathlib import Path

import pandas as pd
import streamlit as st
import ast
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs_src"


@st.cache_data
def load_base_registry() -> pd.DataFrame:
    path = OUTPUTS_DIR / "registry" / "model_summary.parquet"
    return pd.read_parquet(path)


@st.cache_data
def load_ensemble_registry() -> pd.DataFrame:
    path = OUTPUTS_DIR / "ensemble" / "registry" / "model_summary.parquet"
    return pd.read_parquet(path)

def extract_ensemble_weight_info(row):
    """
    Add ensemble interpretation for every ensemble model.

    Weighted ensembles get true weight-based diversity.
    Equal/rank averages get equal-share diversity.
    Stacking models are marked as meta-learners because base-model
    dominance is not directly available unless meta coefficients/importances
    are separately logged.
    """

    if row.get("model_family") != "ensemble":
        return pd.Series(
            {
                "ensemble_method_type": None,
                "dominant_base_model": None,
                "max_weight_share": None,
                "ensemble_diversity_score": None,
                "ensemble_note": None,
            }
        )

    model_id = str(row.get("model_id", ""))
    model_name = str(row.get("model_name", ""))
    params_text = str(row.get("params", ""))
    features_text = str(row.get("features", ""))

    base_models = [x.strip() for x in features_text.split(",") if x.strip()]
    n_base = len(base_models)

    params_text_clean = re.sub(
        r"np\.float64\(([^)]+)\)",
        r"\1",
        params_text,
    )

    try:
        params_dict = ast.literal_eval(params_text_clean)
    except Exception:
        params_dict = {}

    weights = params_dict.get("weights")

    if isinstance(weights, dict) and len(weights) > 0:
        dominant_model = max(weights, key=weights.get)
        max_weight = float(weights[dominant_model])
        diversity_score = 1 - max_weight

        if max_weight >= 0.95:
            note = f"Effectively behaves like {dominant_model}"
        elif max_weight >= 0.75:
            note = f"Dominated by {dominant_model}"
        else:
            note = "Diversified weighted ensemble"

        return pd.Series(
            {
                "ensemble_method_type": "weighted_average",
                "dominant_base_model": dominant_model,
                "max_weight_share": max_weight,
                "ensemble_diversity_score": diversity_score,
                "ensemble_note": note,
            }
        )

    if "Equal average" in model_name:
        max_weight = 1 / n_base if n_base else None
        return pd.Series(
            {
                "ensemble_method_type": "equal_average",
                "dominant_base_model": "No single dominant model",
                "max_weight_share": max_weight,
                "ensemble_diversity_score": 1 - max_weight if max_weight else None,
                "ensemble_note": "Equal contribution from all base models",
            }
        )

    if "Rank average" in model_name:
        max_weight = 1 / n_base if n_base else None
        return pd.Series(
            {
                "ensemble_method_type": "rank_average",
                "dominant_base_model": "No single dominant model",
                "max_weight_share": max_weight,
                "ensemble_diversity_score": 1 - max_weight if max_weight else None,
                "ensemble_note": "Rank-based averaging across base models",
            }
        )

    if "stack" in model_id.lower() or "stacking" in model_name.lower():
        return pd.Series(
            {
                "ensemble_method_type": "meta_learner_stacking",
                "dominant_base_model": "Not directly available",
                "max_weight_share": None,
                "ensemble_diversity_score": None,
                "ensemble_note": "Stacking model; base dominance requires meta-model coefficients/importances",
            }
        )

    return pd.Series(
        {
            "ensemble_method_type": "ensemble",
            "dominant_base_model": "Unknown",
            "max_weight_share": None,
            "ensemble_diversity_score": None,
            "ensemble_note": "No ensemble weighting metadata available",
        }
    )

@st.cache_data
def load_full_model_registry() -> pd.DataFrame:
    base = load_base_registry()
    ensemble = load_ensemble_registry()

    registry = pd.concat([base, ensemble], ignore_index=True)

    numeric_cols = [
        "train_auc",
        "validation_auc",
        "train_ks",
        "validation_ks",
        "validation_log_loss",
        "validation_brier_score",
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1",
    ]

    for col in numeric_cols:
        if col in registry.columns:
            registry[col] = pd.to_numeric(registry[col], errors="coerce")

    if "validation_auc" in registry.columns:
        registry["auc_rank"] = registry["validation_auc"].rank(
            ascending=False,
            method="dense",
        )

    if {"train_auc", "validation_auc"}.issubset(registry.columns):
        registry["auc_gap"] = registry["train_auc"] - registry["validation_auc"]

    ensemble_info = registry.apply(extract_ensemble_weight_info, axis=1)
    registry = pd.concat([registry, ensemble_info], axis=1)

    return registry