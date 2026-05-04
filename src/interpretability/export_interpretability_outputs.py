from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Paths

PROJECT_ROOT = Path.cwd()
OUTPUTS_DIR = PROJECT_ROOT / "outputs_src"

REGISTRY_PATH = OUTPUTS_DIR / "model_registry.xlsx"
BASE_REGISTRY_PATH = OUTPUTS_DIR / "registry" / "model_summary.parquet"
ENSEMBLE_REGISTRY_PATH = OUTPUTS_DIR / "ensemble_registry.xlsx"

MODEL_DIRS = [
    OUTPUTS_DIR / "models",
    OUTPUTS_DIR / "ensemble" / "models",
]

INTERP_DIR = OUTPUTS_DIR / "interpretability"

SUMMARY_DIR = INTERP_DIR / "summary"
LOGISTIC_DIR = INTERP_DIR / "logistic"
FEATURE_IMPORTANCE_DIR = INTERP_DIR / "feature_importance"
SHAP_DIR = INTERP_DIR / "shap"
PERMUTATION_DIR = INTERP_DIR / "permutation_importance"
ENSEMBLE_DIR = INTERP_DIR / "ensemble"
GOVERNANCE_DIR = INTERP_DIR / "governance"

MODELERS_DEFAULT_MODEL_ID = "CAT002"
BEST_AUC_MODEL_ID = "XGBSTACK001"

# Setup

def ensure_dirs() -> None:
    for path in [
        INTERP_DIR,
        SUMMARY_DIR,
        LOGISTIC_DIR,
        FEATURE_IMPORTANCE_DIR,
        SHAP_DIR,
        PERMUTATION_DIR,
        ENSEMBLE_DIR,
        GOVERNANCE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def find_model_path(model_id: str) -> Path | None:
    for model_dir in MODEL_DIRS:
        candidate = model_dir / f"{model_id}.joblib"
        if candidate.exists():
            return candidate
    return None


def load_model(model_id: str) -> Any | None:
    path = find_model_path(model_id)
    if path is None:
        return None
    return joblib.load(path)


def load_registry() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if REGISTRY_PATH.exists():
        frames.append(pd.read_excel(REGISTRY_PATH))

    if BASE_REGISTRY_PATH.exists():
        frames.append(pd.read_parquet(BASE_REGISTRY_PATH))

    if ENSEMBLE_REGISTRY_PATH.exists():
        frames.append(pd.read_excel(ENSEMBLE_REGISTRY_PATH))

    if not frames:
        raise FileNotFoundError(
            "No model registry found. Expected one of: "
            f"{REGISTRY_PATH}, {BASE_REGISTRY_PATH}, {ENSEMBLE_REGISTRY_PATH}"
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["model_id"], keep="last").reset_index(drop=True)

    if "auc_gap" not in df.columns and {"train_auc", "validation_auc"}.issubset(df.columns):
        df["auc_gap"] = df["train_auc"] - df["validation_auc"]

    return df

# Utility functions

def safe_parse_params(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, dict):
        return value

    if not isinstance(value, str):
        return value

    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def parse_features(value: Any) -> list[str]:
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, tuple):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]

    return []


def significance_stars(p_value: float | None) -> str:
    if p_value is None or pd.isna(p_value):
        return ""

    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "."
    return ""


def direction_from_coef(coef: float | None) -> str:
    if coef is None or pd.isna(coef):
        return ""

    if coef > 0:
        return "positive"
    if coef < 0:
        return "negative"
    return "zero"


def add_selection_score(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    required = [
        "validation_auc",
        "validation_ks",
        "validation_log_loss",
        "validation_brier_score",
        "auc_gap",
        "feature_count",
    ]

    if not all(c in data.columns for c in required):
        data["selection_score"] = np.nan
        return data

    def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")

        if s.max() == s.min():
            return pd.Series(1.0, index=s.index)

        score = (s - s.min()) / (s.max() - s.min())
        return score if higher_is_better else 1 - score

    auc_score = minmax_score(data["validation_auc"], True)
    ks_score = minmax_score(data["validation_ks"], True)
    logloss_score = minmax_score(data["validation_log_loss"], False)
    brier_score = minmax_score(data["validation_brier_score"], False)
    gap_score = minmax_score(data["auc_gap"].abs(), False)
    simplicity_score = minmax_score(data["feature_count"], False)

    data["selection_score"] = (
        0.30 * auc_score
        + 0.20 * ks_score
        + 0.20 * logloss_score
        + 0.15 * brier_score
        + 0.10 * gap_score
        + 0.05 * simplicity_score
    )

    return data


def infer_interpretability_type(model_family: str, model_id: str) -> str:
    family = str(model_family).lower()
    model_id = str(model_id).upper()

    if family == "logistic":
        return "coefficients_pvalues_odds_ratios"

    if family in {"tree", "boosting"}:
        return "feature_importance"

    if family == "neural_network":
        return "permutation_importance_required"

    if family == "naive_bayes":
        return "class_probability_review"

    if family == "ensemble" or "STACK" in model_id:
        return "base_model_contribution_or_meta_model_importance"

    return "general_model_review"


def assign_governance_flag(row: pd.Series) -> str:
    family = str(row.get("model_family", "")).lower()
    model_id = str(row.get("model_id", ""))
    auc_gap = row.get("auc_gap", np.nan)
    feature_count = row.get("feature_count", np.nan)
    model_path = find_model_path(model_id)

    if model_path is None:
        return "review_required_missing_model_artifact"

    if family == "ensemble":
        return "review_required_high_complexity"

    if pd.notna(auc_gap) and abs(auc_gap) > 0.03:
        return "review_required_possible_overfit"

    if pd.notna(feature_count) and feature_count > 50:
        return "review_required_high_feature_count"

    if family == "logistic":
        return "governance_friendly_interpretable"

    if family in {"boosting", "tree"}:
        return "acceptable_with_feature_importance_review"

    return "acceptable_with_supporting_diagnostics"


def get_model_feature_names(model: Any, registry_features: list[str]) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        try:
            return [str(x) for x in list(model.feature_names_in_)]
        except Exception:
            pass

    if hasattr(model, "feature_names_"):
        try:
            return [str(x) for x in list(model.feature_names_)]
        except Exception:
            pass

    return registry_features

# Summary export

def export_summary_outputs(df: pd.DataFrame) -> None:
    data = add_selection_score(df)

    summary_cols = [
        "model_id",
        "model_family",
        "model_name",
        "dataset_type",
        "feature_count",
        "features",
        "train_auc",
        "validation_auc",
        "validation_ks",
        "validation_log_loss",
        "validation_brier_score",
        "auc_gap",
        "selection_score",
    ]

    summary_cols = [c for c in summary_cols if c in data.columns]
    summary = data[summary_cols].copy()

    summary["model_artifact_path"] = summary["model_id"].apply(
        lambda x: str(find_model_path(str(x))) if find_model_path(str(x)) else ""
    )

    summary["model_artifact_available"] = summary["model_artifact_path"].ne("")

    summary["interpretability_type"] = summary.apply(
        lambda r: infer_interpretability_type(r.get("model_family"), r.get("model_id")),
        axis=1,
    )

    summary["governance_flag"] = summary.apply(assign_governance_flag, axis=1)

    summary["reference_model_role"] = ""
    summary.loc[summary["model_id"] == MODELERS_DEFAULT_MODEL_ID, "reference_model_role"] = "modeler_default"
    summary.loc[summary["model_id"] == BEST_AUC_MODEL_ID, "reference_model_role"] = "best_validation_auc"

    summary.to_csv(SUMMARY_DIR / "interpretability_model_summary.csv", index=False)
    summary.to_excel(SUMMARY_DIR / "interpretability_model_summary.xlsx", index=False)

# Logistic coefficients

def export_logistic_outputs(df: pd.DataFrame) -> None:
    logistic_df = df[df["model_family"].astype(str).str.lower() == "logistic"].copy()

    records: list[dict[str, Any]] = []

    for _, row in logistic_df.iterrows():
        model_id = str(row.get("model_id"))
        model = load_model(model_id)
        registry_features = parse_features(row.get("features"))

        if model is None:
            records.append(
                {
                    "model_id": model_id,
                    "feature": None,
                    "coefficient": None,
                    "std_error": None,
                    "z_value": None,
                    "p_value": None,
                    "significance": None,
                    "odds_ratio": None,
                    "actual_direction": None,
                    "expected_direction": None,
                    "direction_check": "not_assessed",
                    "source": "missing_model_artifact",
                    "status": "missing_model_artifact",
                }
            )
            continue

        # statsmodels BinaryResultsWrapper
        if all(hasattr(model, attr) for attr in ["params", "bse", "pvalues", "tvalues"]):
            params = model.params
            bse = model.bse
            pvalues = model.pvalues
            tvalues = model.tvalues

            for feature in params.index:
                coef = float(params.loc[feature])
                p_value = float(pvalues.loc[feature])

                records.append(
                    {
                        "model_id": model_id,
                        "feature": str(feature),
                        "coefficient": coef,
                        "std_error": float(bse.loc[feature]),
                        "z_value": float(tvalues.loc[feature]),
                        "p_value": p_value,
                        "significance": significance_stars(p_value),
                        "odds_ratio": float(np.exp(coef)),
                        "actual_direction": direction_from_coef(coef),
                        "expected_direction": "",
                        "direction_check": "not_assessed",
                        "source": "statsmodels",
                        "status": "available",
                    }
                )

        # sklearn LogisticRegression
        elif hasattr(model, "coef_"):
            feature_names = get_model_feature_names(model, registry_features)
            coefs = np.asarray(model.coef_).ravel()

            if len(feature_names) != len(coefs):
                feature_names = [f"feature_{i}" for i in range(len(coefs))]

            for feature, coef in zip(feature_names, coefs):
                coef = float(coef)

                records.append(
                    {
                        "model_id": model_id,
                        "feature": str(feature),
                        "coefficient": coef,
                        "std_error": None,
                        "z_value": None,
                        "p_value": None,
                        "significance": "",
                        "odds_ratio": float(np.exp(coef)),
                        "actual_direction": direction_from_coef(coef),
                        "expected_direction": "",
                        "direction_check": "not_assessed",
                        "source": "sklearn_logistic",
                        "status": "available_no_pvalues",
                    }
                )

            if hasattr(model, "intercept_"):
                intercept = float(np.asarray(model.intercept_).ravel()[0])
                records.append(
                    {
                        "model_id": model_id,
                        "feature": "intercept",
                        "coefficient": intercept,
                        "std_error": None,
                        "z_value": None,
                        "p_value": None,
                        "significance": "",
                        "odds_ratio": float(np.exp(intercept)),
                        "actual_direction": direction_from_coef(intercept),
                        "expected_direction": "",
                        "direction_check": "not_assessed",
                        "source": "sklearn_logistic",
                        "status": "available_no_pvalues",
                    }
                )

        else:
            records.append(
                {
                    "model_id": model_id,
                    "feature": None,
                    "coefficient": None,
                    "std_error": None,
                    "z_value": None,
                    "p_value": None,
                    "significance": None,
                    "odds_ratio": None,
                    "actual_direction": None,
                    "expected_direction": None,
                    "direction_check": "not_assessed",
                    "source": type(model).__name__,
                    "status": "unsupported_logistic_object",
                }
            )

    out = pd.DataFrame(records)

    if out.empty:
        out = pd.DataFrame(
            [
                {
                    "model_id": None,
                    "feature": None,
                    "coefficient": None,
                    "std_error": None,
                    "z_value": None,
                    "p_value": None,
                    "significance": None,
                    "odds_ratio": None,
                    "actual_direction": None,
                    "expected_direction": None,
                    "direction_check": "not_assessed",
                    "source": None,
                    "status": "no_logistic_models_found",
                }
            ]
        )

    out.to_csv(LOGISTIC_DIR / "logistic_coefficients_summary.csv", index=False)
    out.to_excel(LOGISTIC_DIR / "logistic_coefficients_summary.xlsx", index=False)

# Feature importance

def extract_feature_importance(model: Any, feature_names: list[str]) -> tuple[np.ndarray | None, str]:
    # CatBoost
    if hasattr(model, "get_feature_importance"):
        try:
            return np.asarray(model.get_feature_importance(), dtype=float), "catboost_feature_importance"
        except Exception:
            pass

    # Tree / boosting sklearn-style models
    if hasattr(model, "feature_importances_"):
        try:
            return np.asarray(model.feature_importances_, dtype=float), "model_feature_importances"
        except Exception:
            pass

    # Logistic / linear models
    if hasattr(model, "coef_"):
        try:
            return np.abs(np.asarray(model.coef_).ravel()), "absolute_coefficient"
        except Exception:
            pass

    # Neural network / MLPClassifier
    if hasattr(model, "coefs_"):
        try:
            first_layer = np.asarray(model.coefs_[0], dtype=float)
            return np.sum(np.abs(first_layer), axis=1), "neural_network_first_layer_input_sensitivity"
        except Exception:
            pass

    # Gaussian Naive Bayes
    if hasattr(model, "theta_") and hasattr(model, "var_"):
        try:
            means = np.asarray(model.theta_, dtype=float)
            variances = np.asarray(model.var_, dtype=float)

            if means.shape[0] == 2:
                pooled_std = np.sqrt(np.mean(variances, axis=0))
                pooled_std = np.where(pooled_std == 0, 1e-6, pooled_std)
                importance = np.abs(means[1] - means[0]) / pooled_std
                return importance, "naive_bayes_class_separation"
        except Exception:
            pass

    # Bernoulli / Multinomial Naive Bayes
    if hasattr(model, "feature_log_prob_"):
        try:
            probs = np.asarray(model.feature_log_prob_, dtype=float)

            if probs.shape[0] == 2:
                importance = np.abs(probs[1] - probs[0])
                return importance, "naive_bayes_log_probability_difference"
        except Exception:
            pass

    # Saved weighted ensemble objects
    if hasattr(model, "weights_"):
        try:
            return np.asarray(model.weights_, dtype=float), "ensemble_weight"
        except Exception:
            pass

    return None, "not_available"


def export_feature_importance_outputs(df: pd.DataFrame) -> None:
    model_df = df.copy()
    records: list[dict[str, Any]] = []

    for _, row in model_df.iterrows():
        model_id = str(row.get("model_id"))
        model_family = str(row.get("model_family"))
        registry_features = parse_features(row.get("features"))
        model = load_model(model_id)

        if model is None:
            records.append(
                {
                    "model_id": model_id,
                    "model_family": model_family,
                    "feature": None,
                    "importance": None,
                    "importance_normalized": None,
                    "importance_type": None,
                    "rank": None,
                    "status": "missing_model_artifact",
                }
            )
            continue

        feature_names = get_model_feature_names(model, registry_features)
        importances, importance_type = extract_feature_importance(model, feature_names)

        if importances is None:
            records.append(
                {
                    "model_id": model_id,
                    "model_family": model_family,
                    "feature": None,
                    "importance": None,
                    "importance_normalized": None,
                    "importance_type": importance_type,
                    "rank": None,
                    "status": "importance_not_available_for_model_type",
                }
            )
            continue

        importances = np.asarray(importances, dtype=float).ravel()

        if len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        total_importance = float(np.sum(np.abs(importances)))

        for feature, importance in zip(feature_names, importances):
            importance = float(importance)
            normalized = (
                float(abs(importance) / total_importance)
                if total_importance > 0
                else np.nan
            )

            records.append(
                {
                    "model_id": model_id,
                    "model_family": model_family,
                    "feature": str(feature),
                    "importance": importance,
                    "importance_normalized": normalized,
                    "importance_type": importance_type,
                    "rank": None,
                    "status": "available",
                }
            )

    out = pd.DataFrame(records)

    if not out.empty and "importance_normalized" in out.columns:
        out["rank"] = (
            out.groupby("model_id")["importance_normalized"]
            .rank(method="first", ascending=False)
        )

    out.to_csv(FEATURE_IMPORTANCE_DIR / "feature_importance_summary.csv", index=False)
    out.to_excel(FEATURE_IMPORTANCE_DIR / "feature_importance_summary.xlsx", index=False)

# Ensemble interpretability

def export_ensemble_outputs(df: pd.DataFrame) -> None:
    ensemble_df = df[df["model_family"].astype(str).str.lower() == "ensemble"].copy()

    summary_records: list[dict[str, Any]] = []
    weight_records: list[dict[str, Any]] = []
    meta_records: list[dict[str, Any]] = []

    for _, row in ensemble_df.iterrows():
        model_id = str(row.get("model_id"))
        model = load_model(model_id)
        params = safe_parse_params(row.get("params"))
        features = parse_features(row.get("features"))

        summary_records.append(
            {
                "model_id": model_id,
                "model_name": row.get("model_name"),
                "features": row.get("features"),
                "params": row.get("params"),
                "validation_auc": row.get("validation_auc"),
                "validation_ks": row.get("validation_ks"),
                "validation_log_loss": row.get("validation_log_loss"),
                "validation_brier_score": row.get("validation_brier_score"),
                "auc_gap": row.get("auc_gap"),
                "model_artifact_available": model is not None,
                "model_artifact_path": str(find_model_path(model_id)) if find_model_path(model_id) else "",
                "status": "available" if model is not None else "missing_model_artifact",
            }
        )

        # Weighted/equal ensembles from saved object
        if model is not None and hasattr(model, "weights_"):
            feature_names = get_model_feature_names(model, features)
            weights = np.asarray(model.weights_, dtype=float).ravel()

            if len(feature_names) != len(weights):
                feature_names = [f"base_model_{i}" for i in range(len(weights))]

            for base_model, weight in zip(feature_names, weights):
                weight_records.append(
                    {
                        "ensemble_model_id": model_id,
                        "base_model_id": str(base_model),
                        "weight": float(weight),
                        "source": "saved_model_object",
                        "status": "available",
                    }
                )

        # Weights from params fallback
        elif isinstance(params, dict) and isinstance(params.get("weights"), dict):
            for base_model, weight in params["weights"].items():
                weight_records.append(
                    {
                        "ensemble_model_id": model_id,
                        "base_model_id": str(base_model),
                        "weight": float(weight),
                        "source": "registry_params",
                        "status": "available",
                    }
                )

        # Stacking meta-model coefficients / feature importances
        if model is not None:
            feature_names = get_model_feature_names(model, features)

            if hasattr(model, "coef_"):
                coefs = np.asarray(model.coef_).ravel()

                if len(feature_names) != len(coefs):
                    feature_names = [f"base_model_{i}" for i in range(len(coefs))]

                for base_model, coef in zip(feature_names, coefs):
                    meta_records.append(
                        {
                            "stack_model_id": model_id,
                            "base_model_id": str(base_model),
                            "meta_importance": float(coef),
                            "meta_importance_abs": float(abs(coef)),
                            "importance_type": "stacking_coefficient",
                            "status": "available",
                        }
                    )

            elif hasattr(model, "feature_importances_"):
                imps = np.asarray(model.feature_importances_, dtype=float).ravel()

                if len(feature_names) != len(imps):
                    feature_names = [f"base_model_{i}" for i in range(len(imps))]

                for base_model, imp in zip(feature_names, imps):
                    meta_records.append(
                        {
                            "stack_model_id": model_id,
                            "base_model_id": str(base_model),
                            "meta_importance": float(imp),
                            "meta_importance_abs": float(abs(imp)),
                            "importance_type": "stacking_feature_importance",
                            "status": "available",
                        }
                    )
            elif hasattr(model, "coefs_"):
                first_layer = np.asarray(model.coefs_[0], dtype=float)
                imps = np.sum(np.abs(first_layer), axis=1)

                if len(feature_names) != len(imps):
                    feature_names = [f"base_model_{i}" for i in range(len(imps))]

                for base_model, imp in zip(feature_names, imps):
                    meta_records.append(
                        {
                            "stack_model_id": model_id,
                            "base_model_id": str(base_model),
                            "meta_importance": float(imp),
                            "meta_importance_abs": float(abs(imp)),
                            "importance_type": "neural_network_stacking_input_sensitivity",
                            "status": "available",
                        }
                    )

    summary_out = pd.DataFrame(summary_records)
    weight_out = pd.DataFrame(weight_records)
    meta_out = pd.DataFrame(meta_records)

    if summary_out.empty:
        summary_out = pd.DataFrame(
            [{"model_id": None, "status": "no_ensemble_models_found"}]
        )

    if weight_out.empty:
        weight_out = pd.DataFrame(
            [
                {
                    "ensemble_model_id": None,
                    "base_model_id": None,
                    "weight": None,
                    "source": None,
                    "status": "no_fixed_weight_ensembles_found",
                }
            ]
        )

    if meta_out.empty:
        meta_out = pd.DataFrame(
            [
                {
                    "stack_model_id": None,
                    "base_model_id": None,
                    "meta_importance": None,
                    "meta_importance_abs": None,
                    "importance_type": None,
                    "status": "no_stacking_meta_importance_found",
                }
            ]
        )

    summary_out.to_csv(ENSEMBLE_DIR / "ensemble_interpretability_summary.csv", index=False)
    summary_out.to_excel(ENSEMBLE_DIR / "ensemble_interpretability_summary.xlsx", index=False)

    weight_out.to_csv(ENSEMBLE_DIR / "ensemble_base_model_weights.csv", index=False)
    weight_out.to_excel(ENSEMBLE_DIR / "ensemble_base_model_weights.xlsx", index=False)

    meta_out.to_csv(ENSEMBLE_DIR / "stacking_meta_model_importance.csv", index=False)
    meta_out.to_excel(ENSEMBLE_DIR / "stacking_meta_model_importance.xlsx", index=False)


# --------------------------------------------------
# SHAP / permutation status
# --------------------------------------------------

def export_shap_status(df: pd.DataFrame) -> None:
    records = []

    for _, row in df.iterrows():
        model_id = str(row.get("model_id"))
        family = str(row.get("model_family")).lower()
        model_path = find_model_path(model_id)

        if family in {"boosting", "tree", "ensemble", "neural_network"}:
            records.append(
                {
                    "model_id": model_id,
                    "model_family": family,
                    "shap_available": False,
                    "mean_abs_shap_file": None,
                    "summary_plot_file": None,
                    "status": "not_generated",
                    "message": (
                        "SHAP requires model-ready validation matrices aligned to each model. "
                        "Feature importance and ensemble contributions are exported in this script."
                    ),
                    "model_artifact_available": model_path is not None,
                }
            )

    out = pd.DataFrame(records)

    if out.empty:
        out = pd.DataFrame(
            [
                {
                    "model_id": None,
                    "model_family": None,
                    "shap_available": False,
                    "mean_abs_shap_file": None,
                    "summary_plot_file": None,
                    "status": "no_shap_candidate_models_found",
                    "message": None,
                    "model_artifact_available": False,
                }
            ]
        )

    out.to_csv(SHAP_DIR / "shap_export_status.csv", index=False)


# --------------------------------------------------
# Governance checks
# --------------------------------------------------

def export_governance_checks(df: pd.DataFrame) -> None:
    data = add_selection_score(df)
    records = []

    for _, row in data.iterrows():
        model_id = str(row.get("model_id"))
        family = str(row.get("model_family", "")).lower()
        auc_gap = row.get("auc_gap", np.nan)
        feature_count = row.get("feature_count", np.nan)
        model_path = find_model_path(model_id)

        checks = []

        if model_path is None:
            checks.append("Missing saved model artifact; cannot fully reproduce or interpret model.")

        if pd.notna(auc_gap) and abs(auc_gap) > 0.03:
            checks.append("Large train-validation AUC gap; review overfitting risk.")

        if pd.notna(feature_count) and feature_count > 50:
            checks.append("High feature count; review deployment and monitoring complexity.")

        if family == "ensemble":
            checks.append("Ensemble model; review base-model contribution, dominance, and incremental value.")

        if family in {"boosting", "tree"}:
            checks.append("Tree/boosting model; review feature importance and SHAP when available.")

        if family == "logistic":
            checks.append("Logistic model; review coefficient direction, p-value, and odds ratio.")

        if family == "neural_network":
            checks.append("Neural network model; require model-agnostic interpretability before governance approval.")

        if family == "naive_bayes":
            checks.append("Naive Bayes model; review class-separation behavior and probability calibration.")

        if not checks:
            checks.append("No major governance red flags from registry-level checks.")

        records.append(
            {
                "model_id": model_id,
                "model_family": row.get("model_family"),
                "validation_auc": row.get("validation_auc"),
                "validation_ks": row.get("validation_ks"),
                "validation_log_loss": row.get("validation_log_loss"),
                "validation_brier_score": row.get("validation_brier_score"),
                "auc_gap": auc_gap,
                "feature_count": feature_count,
                "selection_score": row.get("selection_score"),
                "model_artifact_available": model_path is not None,
                "model_artifact_path": str(model_path) if model_path else "",
                "governance_flag": assign_governance_flag(row),
                "governance_checks": " | ".join(checks),
            }
        )

    out = pd.DataFrame(records)
    out.to_csv(GOVERNANCE_DIR / "model_governance_checks.csv", index=False)
    out.to_excel(GOVERNANCE_DIR / "model_governance_checks.xlsx", index=False)


# --------------------------------------------------
# Manifest
# --------------------------------------------------

def write_manifest() -> None:
    manifest = {
        "output_root": str(INTERP_DIR),
        "model_search_paths": [str(x) for x in MODEL_DIRS],
        "files": {
            "summary": str(SUMMARY_DIR / "interpretability_model_summary.csv"),
            "logistic_coefficients": str(LOGISTIC_DIR / "logistic_coefficients_summary.csv"),
            "feature_importance": str(FEATURE_IMPORTANCE_DIR / "feature_importance_summary.csv"),
            "ensemble_summary": str(ENSEMBLE_DIR / "ensemble_interpretability_summary.csv"),
            "ensemble_weights": str(ENSEMBLE_DIR / "ensemble_base_model_weights.csv"),
            "stacking_meta_importance": str(ENSEMBLE_DIR / "stacking_meta_model_importance.csv"),
            "shap_status": str(SHAP_DIR / "shap_export_status.csv"),
            "governance_checks": str(GOVERNANCE_DIR / "model_governance_checks.csv"),
        },
        "reference_models": {
            "modeler_default": MODELERS_DEFAULT_MODEL_ID,
            "best_validation_auc": BEST_AUC_MODEL_ID,
        },
        "note": (
            "This export extracts real model interpretability artifacts where available. "
            "SHAP is intentionally tracked as status-only until model-specific validation matrices "
            "are wired into the export layer."
        ),
    }

    with open(INTERP_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    ensure_dirs()

    registry = load_registry()

    export_summary_outputs(registry)
    export_logistic_outputs(registry)
    export_feature_importance_outputs(registry)
    export_ensemble_outputs(registry)
    export_shap_status(registry)
    export_governance_checks(registry)
    write_manifest()

    print("Interpretability export complete.")
    print(f"Output directory: {INTERP_DIR}")
    print("Generated:")
    print(f"- {SUMMARY_DIR / 'interpretability_model_summary.csv'}")
    print(f"- {LOGISTIC_DIR / 'logistic_coefficients_summary.csv'}")
    print(f"- {FEATURE_IMPORTANCE_DIR / 'feature_importance_summary.csv'}")
    print(f"- {ENSEMBLE_DIR / 'ensemble_base_model_weights.csv'}")
    print(f"- {ENSEMBLE_DIR / 'stacking_meta_model_importance.csv'}")
    print(f"- {GOVERNANCE_DIR / 'model_governance_checks.csv'}")


if __name__ == "__main__":
    main()