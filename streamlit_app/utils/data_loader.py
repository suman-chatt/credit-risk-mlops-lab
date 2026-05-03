from pathlib import Path
import ast
import re

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs_src"


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".xlsx":
        return pd.read_excel(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    return pd.DataFrame()


@st.cache_data
def load_base_registry() -> pd.DataFrame:
    candidate_paths = [
        OUTPUTS_DIR / "registry" / "model_summary.csv",
        OUTPUTS_DIR / "registry" / "model_summary.xlsx",
        OUTPUTS_DIR / "registry" / "model_summary.parquet",
        OUTPUTS_DIR / "model_registry.xlsx",
    ]

    for path in candidate_paths:
        df = read_table(path)
        if not df.empty:
            return df

    return pd.DataFrame()


@st.cache_data
def load_ensemble_registry() -> pd.DataFrame:
    candidate_paths = [
        OUTPUTS_DIR / "ensemble_registry.xlsx",
        OUTPUTS_DIR / "ensemble_registry.csv",
        OUTPUTS_DIR / "ensemble_registry.parquet",
        OUTPUTS_DIR / "ensemble" / "registry" / "model_summary.csv",
        OUTPUTS_DIR / "ensemble" / "registry" / "model_summary.xlsx",
        OUTPUTS_DIR / "ensemble" / "registry" / "model_summary.parquet",
    ]

    for path in candidate_paths:
        df = read_table(path)
        if not df.empty:
            return df

    return pd.DataFrame()


def extract_ensemble_weight_info(row: pd.Series) -> pd.Series:
    if str(row.get("model_family", "")).lower() != "ensemble":
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

    params_text_clean = re.sub(r"np\.float64\(([^)]+)\)", r"\1", params_text)

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

    if "equal average" in model_name.lower():
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

    if "rank average" in model_name.lower():
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

    frames = [df for df in [base, ensemble] if not df.empty]

    if not frames:
        return pd.DataFrame()

    registry = pd.concat(frames, ignore_index=True)

    if "model_id" in registry.columns:
        registry = registry.drop_duplicates(subset=["model_id"], keep="last")

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
        "selection_score",
        "feature_count",
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
    registry = pd.concat([registry.reset_index(drop=True), ensemble_info], axis=1)

    return registry.reset_index(drop=True)