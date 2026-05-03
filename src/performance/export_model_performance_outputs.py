from pathlib import Path
import ast
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.models.data_split import prepare_training_datasets


PROJECT_ROOT = Path.cwd()
OUTPUTS_DIR = PROJECT_ROOT / "outputs_src"

MODEL_READY_DIR = OUTPUTS_DIR / "model_ready"
BASE_REGISTRY_DIR = OUTPUTS_DIR / "registry"
ENSEMBLE_REGISTRY_DIR = OUTPUTS_DIR / "ensemble" / "registry"
ENSEMBLE_MODELS_DIR = OUTPUTS_DIR / "ensemble" / "models"
PERF_DIR = OUTPUTS_DIR / "performance"

BASE_DIAG_DIR = OUTPUTS_DIR / "diagnostics"
ENSEMBLE_DIAG_DIR = OUTPUTS_DIR / "ensemble" / "diagnostics"

PERF_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load inputs
# --------------------------------------------------

def load_inputs() -> tuple[dict, pd.DataFrame, dict, pd.DataFrame, dict]:
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

    base_summary = pd.read_parquet(BASE_REGISTRY_DIR / "model_summary.parquet")

    with open(BASE_REGISTRY_DIR / "artifact_paths.json", "r", encoding="utf-8") as f:
        base_artifact_paths = json.load(f)

    ensemble_summary_path = ENSEMBLE_REGISTRY_DIR / "model_summary.parquet"
    if ensemble_summary_path.exists():
        ensemble_summary = pd.read_parquet(ensemble_summary_path)
    elif (OUTPUTS_DIR / "ensemble_registry.xlsx").exists():
        ensemble_summary = pd.read_excel(OUTPUTS_DIR / "ensemble_registry.xlsx")
    else:
        ensemble_summary = pd.DataFrame()

    ensemble_artifact_paths_path = ENSEMBLE_REGISTRY_DIR / "artifact_paths.json"
    if ensemble_artifact_paths_path.exists():
        with open(ensemble_artifact_paths_path, "r", encoding="utf-8") as f:
            ensemble_artifact_paths = json.load(f)
    else:
        ensemble_artifact_paths = {
            p.stem: str(p)
            for p in ENSEMBLE_MODELS_DIR.glob("*.joblib")
        }

    return (
        datasets,
        base_summary,
        base_artifact_paths,
        ensemble_summary,
        ensemble_artifact_paths,
    )


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def parse_features(feature_string: Any) -> list[str]:
    if pd.isna(feature_string):
        return []
    if isinstance(feature_string, list):
        return [str(x).strip() for x in feature_string if str(x).strip()]
    return [x.strip() for x in str(feature_string).split(",") if x.strip()]


def safe_parse_params(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def get_xy_for_row(row: pd.Series, datasets: dict, split_name: str) -> tuple[pd.DataFrame | None, pd.Series | None]:
    dataset_type = str(row.get("dataset_type", "")).lower()
    features = parse_features(row.get("features"))

    if not features:
        return None, None

    if "tree" in dataset_type:
        split = datasets["tree"]
    elif "scaled" in dataset_type:
        split = datasets["scaled"]
    elif "woe" in dataset_type:
        split = datasets["woe"]
    elif "binned_logit" in dataset_type:
        split = datasets["binned_logit"]
    elif "binned_ohe" in dataset_type:
        split = datasets.get("binned_ohe")
        if split is None:
            return None, None
    else:
        return None, None

    x_key = f"X_{split_name}"
    y_key = f"y_{split_name}"

    if x_key not in split or y_key not in split:
        return None, None

    missing = [c for c in features if c not in split[x_key].columns]
    if missing:
        return None, None

    X = split[x_key][features].copy()
    y = split[y_key].copy()

    return X.reset_index(drop=True), y.reset_index(drop=True)


def align_to_sklearn_features(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        X = X.reindex(columns=expected_cols, fill_value=0)
    return X


def align_to_statsmodels_params(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    X_const = sm.add_constant(X, has_constant="add")

    if hasattr(model, "params"):
        expected_cols = list(model.params.index)
        X_const = X_const.reindex(columns=expected_cols, fill_value=0)

    return X_const


def is_statsmodels_model(model: Any) -> bool:
    return "statsmodels" in str(type(model)).lower() or hasattr(model, "params")


def predict_base_model(
    model: Any,
    row: pd.Series,
    datasets: dict,
    split_name: str,
) -> tuple[np.ndarray | None, pd.Series | None]:
    X, y = get_xy_for_row(row, datasets, split_name)

    if X is None or y is None:
        return None, None

    dataset_type = str(row.get("dataset_type", "")).lower()

    if "binned_logit" in dataset_type:
        X = X.astype(str)
        X = pd.get_dummies(X, drop_first=True, dtype=int)

    try:
        if is_statsmodels_model(model):
            X_model = align_to_statsmodels_params(model, X)
            preds = np.asarray(model.predict(X_model), dtype=float)
        else:
            X_model = align_to_sklearn_features(model, X)
            if hasattr(model, "predict_proba"):
                preds = np.asarray(model.predict_proba(X_model)[:, 1], dtype=float)
            else:
                preds = np.asarray(model.predict(X_model), dtype=float)

        preds = np.clip(preds, 1e-6, 1 - 1e-6)

        if len(preds) != len(y):
            return None, None

        return preds, y

    except Exception as exc:
        print(f"[Prediction skipped] {row.get('model_id')} / {split_name}: {exc}")
        return None, None


# --------------------------------------------------
# Base prediction cache
# --------------------------------------------------

def build_base_prediction_cache(
    datasets: dict,
    base_summary: pd.DataFrame,
    base_artifact_paths: dict,
) -> dict[tuple[str, str], tuple[np.ndarray, pd.Series]]:
    cache = {}

    for _, row in base_summary.iterrows():
        model_id = str(row["model_id"])
        path = base_artifact_paths.get(model_id)

        if path is None:
            continue

        try:
            model = joblib.load(path)
        except Exception as exc:
            print(f"[Load skipped] {model_id}: {exc}")
            continue

        for split_name in ["train", "val"]:
            preds, y = predict_base_model(model, row, datasets, split_name)

            if preds is not None and y is not None:
                cache[(model_id, split_name)] = (preds, y)

    return cache


# --------------------------------------------------
# Ensemble prediction
# --------------------------------------------------

def build_ensemble_matrix(
    ensemble_row: pd.Series,
    split_name: str,
    base_prediction_cache: dict[tuple[str, str], tuple[np.ndarray, pd.Series]],
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    base_model_ids = parse_features(ensemble_row.get("features"))

    if not base_model_ids:
        return None, None

    pred_dict = {}
    y_ref = None

    for base_model_id in base_model_ids:
        key = (base_model_id, split_name)

        if key not in base_prediction_cache:
            return None, None

        preds, y = base_prediction_cache[key]
        pred_dict[base_model_id] = preds

        if y_ref is None:
            y_ref = y.reset_index(drop=True)

    X = pd.DataFrame(pred_dict)

    return X, y_ref


def predict_ensemble_model(
    model: Any,
    row: pd.Series,
    split_name: str,
    base_prediction_cache: dict[tuple[str, str], tuple[np.ndarray, pd.Series]],
) -> tuple[np.ndarray | None, pd.Series | None]:
    X, y = build_ensemble_matrix(row, split_name, base_prediction_cache)

    if X is None or y is None:
        return None, None

    params = safe_parse_params(row.get("params"))

    try:
        if model is not None and hasattr(model, "predict_proba"):
            X_model = align_to_sklearn_features(model, X)
            preds = np.asarray(model.predict_proba(X_model)[:, 1], dtype=float)

        elif model is not None and hasattr(model, "predict"):
            X_model = align_to_sklearn_features(model, X)
            preds = np.asarray(model.predict(X_model), dtype=float)

        elif isinstance(params, dict) and params.get("weights") == "equal":
            preds = X.mean(axis=1).to_numpy(dtype=float)

        elif isinstance(params, dict) and isinstance(params.get("weights"), dict):
            weights = pd.Series(params["weights"], dtype=float)
            weights = weights.reindex(X.columns, fill_value=0.0)

            if weights.sum() == 0:
                return None, None

            weights = weights / weights.sum()
            preds = np.dot(X.values, weights.values)

        elif str(row.get("model_id", "")).upper().startswith("RASTACK"):
            preds = X.rank(pct=True).mean(axis=1).to_numpy(dtype=float)

        else:
            return None, None

        preds = np.clip(preds, 1e-6, 1 - 1e-6)

        if len(preds) != len(y):
            return None, None

        return preds, y

    except Exception as exc:
        print(f"[Ensemble prediction skipped] {row.get('model_id')} / {split_name}: {exc}")
        return None, None


# --------------------------------------------------
# Build prediction table
# --------------------------------------------------

def build_prediction_table(
    base_prediction_cache: dict,
    ensemble_summary: pd.DataFrame,
    ensemble_artifact_paths: dict,
) -> pd.DataFrame:
    records = []

    for (model_id, split_name), (preds, y) in base_prediction_cache.items():
        for i, pred in enumerate(preds):
            records.append(
                {
                    "model_id": model_id,
                    "split": "validation" if split_name == "val" else split_name,
                    "row_number": i,
                    "y_true": int(y.iloc[i]),
                    "y_pred": float(pred),
                }
            )

    if not ensemble_summary.empty:
        for _, row in ensemble_summary.iterrows():
            model_id = str(row["model_id"])
            path = ensemble_artifact_paths.get(model_id)

            model = None
            if path is not None and Path(path).exists():
                try:
                    model = joblib.load(path)
                except Exception as exc:
                    print(f"[Ensemble load skipped] {model_id}: {exc}")

            for split_name in ["train", "val"]:
                preds, y = predict_ensemble_model(
                    model=model,
                    row=row,
                    split_name=split_name,
                    base_prediction_cache=base_prediction_cache,
                )

                if preds is None or y is None:
                    continue

                for i, pred in enumerate(preds):
                    records.append(
                        {
                            "model_id": model_id,
                            "split": "validation" if split_name == "val" else split_name,
                            "row_number": i,
                            "y_true": int(y.iloc[i]),
                            "y_pred": float(pred),
                        }
                    )

    out = pd.DataFrame(records)

    out.to_csv(PERF_DIR / "model_predictions.csv", index=False)

    validation_out = out[out["split"] == "validation"].copy()
    validation_out.to_csv(PERF_DIR / "model_validation_predictions.csv", index=False)

    return out


# --------------------------------------------------
# Threshold analysis
# --------------------------------------------------

def build_threshold_analysis(predictions: pd.DataFrame) -> None:
    thresholds = np.round(np.linspace(0.01, 0.99, 99), 4)
    records = []

    for (model_id, split), sub in predictions.groupby(["model_id", "split"]):
        sub = sub.copy()

        total_rows = len(sub)
        total_bad = sub["y_true"].sum()

        for threshold in thresholds:
            approved = sub[sub["y_pred"] < threshold]
            rejected = sub[sub["y_pred"] >= threshold]

            records.append(
                {
                    "model_id": model_id,
                    "split": split,
                    "threshold": threshold,
                    "approval_rate": len(approved) / total_rows if total_rows else np.nan,
                    "rejection_rate": len(rejected) / total_rows if total_rows else np.nan,
                    "approved_bad_rate": approved["y_true"].mean() if len(approved) else np.nan,
                    "rejected_bad_rate": rejected["y_true"].mean() if len(rejected) else np.nan,
                    "captured_bad_share_rejected": rejected["y_true"].sum() / total_bad if total_bad else np.nan,
                    "avg_approved_pd": approved["y_pred"].mean() if len(approved) else np.nan,
                    "avg_rejected_pd": rejected["y_pred"].mean() if len(rejected) else np.nan,
                }
            )

    pd.DataFrame(records).to_csv(PERF_DIR / "threshold_analysis.csv", index=False)


# --------------------------------------------------
# Top bucket summary
# --------------------------------------------------

def build_top_bucket_summary(predictions: pd.DataFrame) -> None:
    records = []

    for (model_id, split), sub in predictions.groupby(["model_id", "split"]):
        sub = sub.sort_values("y_pred", ascending=False).reset_index(drop=True)

        total_bad = sub["y_true"].sum()

        for pct in [0.05, 0.10, 0.20]:
            n = max(1, int(len(sub) * pct))
            top = sub.head(n)

            records.append(
                {
                    "model_id": model_id,
                    "split": split,
                    "top_population_pct": pct,
                    "top_bucket_bad_rate": top["y_true"].mean(),
                    "top_bucket_default_capture": top["y_true"].sum() / total_bad if total_bad else np.nan,
                    "top_bucket_avg_pd": top["y_pred"].mean(),
                    "row_count": len(top),
                }
            )

    pd.DataFrame(records).to_csv(PERF_DIR / "top_bucket_summary.csv", index=False)


# --------------------------------------------------
# Calibration error summary
# --------------------------------------------------

def build_calibration_error_summary() -> None:
    frames = []

    for path in [
        BASE_DIAG_DIR / "calibration_tables.csv",
        ENSEMBLE_DIAG_DIR / "calibration_tables.csv",
    ]:
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)
    df["abs_calibration_error"] = (df["predicted_prob"] - df["actual_rate"]).abs()
    df["squared_calibration_error"] = (df["predicted_prob"] - df["actual_rate"]) ** 2

    out = (
        df.groupby(["model_id", "split"], as_index=False)
        .agg(
            mean_abs_calibration_error=("abs_calibration_error", "mean"),
            max_abs_calibration_error=("abs_calibration_error", "max"),
            mean_squared_calibration_error=("squared_calibration_error", "mean"),
        )
    )

    out.to_csv(PERF_DIR / "calibration_error_summary.csv", index=False)


# --------------------------------------------------
# Train-validation metric gaps
# --------------------------------------------------

def build_metric_gaps(base_summary: pd.DataFrame, ensemble_summary: pd.DataFrame) -> None:
    frames = [base_summary.copy()]

    if not ensemble_summary.empty:
        frames.append(ensemble_summary.copy())

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["model_id"], keep="last")

    if {"train_auc", "validation_auc"}.issubset(data.columns):
        data["auc_gap"] = data["train_auc"] - data["validation_auc"]

    if {"train_ks", "validation_ks"}.issubset(data.columns):
        data["ks_gap"] = data["train_ks"] - data["validation_ks"]

    keep_cols = [
        "model_id",
        "model_family",
        "model_name",
        "train_auc",
        "validation_auc",
        "auc_gap",
        "train_ks",
        "validation_ks",
        "ks_gap",
        "validation_brier_score",
        "validation_log_loss",
    ]

    keep_cols = [c for c in keep_cols if c in data.columns]

    data[keep_cols].to_csv(PERF_DIR / "train_validation_metric_gaps.csv", index=False)


# --------------------------------------------------
# Manifest
# --------------------------------------------------

def write_manifest() -> None:
    manifest = {
        "output_root": str(PERF_DIR),
        "files": {
            "all_predictions": str(PERF_DIR / "model_predictions.csv"),
            "validation_predictions": str(PERF_DIR / "model_validation_predictions.csv"),
            "threshold_analysis": str(PERF_DIR / "threshold_analysis.csv"),
            "top_bucket_summary": str(PERF_DIR / "top_bucket_summary.csv"),
            "calibration_error_summary": str(PERF_DIR / "calibration_error_summary.csv"),
            "train_validation_metric_gaps": str(PERF_DIR / "train_validation_metric_gaps.csv"),
        },
        "score_interpretation": "Higher y_pred means higher predicted probability of default. Threshold analysis treats y_pred below threshold as approved.",
    }

    with open(PERF_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    (
        datasets,
        base_summary,
        base_artifact_paths,
        ensemble_summary,
        ensemble_artifact_paths,
    ) = load_inputs()

    print("Building base prediction cache...")
    base_prediction_cache = build_base_prediction_cache(
        datasets=datasets,
        base_summary=base_summary,
        base_artifact_paths=base_artifact_paths,
    )

    print(f"Base prediction cache entries: {len(base_prediction_cache)}")

    print("Building prediction table...")
    predictions = build_prediction_table(
        base_prediction_cache=base_prediction_cache,
        ensemble_summary=ensemble_summary,
        ensemble_artifact_paths=ensemble_artifact_paths,
    )

    print(f"Prediction rows exported: {len(predictions):,}")

    print("Building threshold analysis...")
    build_threshold_analysis(predictions)

    print("Building top bucket summary...")
    build_top_bucket_summary(predictions)

    print("Building calibration error summary...")
    build_calibration_error_summary()

    print("Building train-validation metric gaps...")
    build_metric_gaps(base_summary, ensemble_summary)

    write_manifest()

    print(f"\nPerformance export complete → {PERF_DIR}")


if __name__ == "__main__":
    main()