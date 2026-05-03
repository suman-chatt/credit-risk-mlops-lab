import pandas as pd
import json
from pathlib import Path

import joblib

def create_empty_registry() -> dict:
    """
    Create empty model registry container.

    Summary tables stay compact.
    Large diagnostics are stored separately and keyed by model_id.
    """

    return {
        "summary": [],
        "metrics": [],
        "lift_tables": {},
        "pva_tables": {},
        "roc_curves": {},
        "ks_curves": {},
        "calibration_tables": {},
        "vif_tables": {},
        "coefficient_tables": {},
        "feature_importance_tables": {},
        "models": {},
        "artifact_paths": {},
        "split_metadata": {},
    }


def add_model_record(
    registry: dict,
    model_id: str,
    model_family: str,
    model_name: str,
    dataset_type: str,
    features: list[str],
    params: dict,
    train_metrics: dict,
    validation_metrics: dict,
    model_object=None,
    notes: str = "",
    selected_flag: bool = False,
) -> dict:
    """
    Add one model to the registry.
    """

    summary_row = {
        "model_id": model_id,
        "model_family": model_family,
        "model_name": model_name,
        "dataset_type": dataset_type,
        "feature_count": len(features),
        "features": ", ".join(features),
        "params": str(params),
        "selected_flag": selected_flag,
        "notes": notes,
    }

    for key, value in train_metrics.items():
        summary_row[f"train_{key}"] = value

    for key, value in validation_metrics.items():
        summary_row[f"validation_{key}"] = value

    registry["summary"].append(summary_row)

    registry["metrics"].append(
        {
            "model_id": model_id,
            "split": "train",
            **train_metrics,
        }
    )

    registry["metrics"].append(
        {
            "model_id": model_id,
            "split": "validation",
            **validation_metrics,
        }
    )

    if model_object is not None:
        registry["models"][model_id] = model_object

    return registry


def add_diagnostics(
    registry: dict,
    model_id: str,
    split: str,
    lift_table: pd.DataFrame | None = None,
    pva_table: pd.DataFrame | None = None,
    roc_curve: pd.DataFrame | None = None,
    ks_curve: pd.DataFrame | None = None,
    calibration_table: pd.DataFrame | None = None,
) -> dict:
    """
    Add model diagnostics tables.
    """

    key = f"{model_id}_{split}"

    if lift_table is not None:
        registry["lift_tables"][key] = lift_table.copy()

    if pva_table is not None:
        registry["pva_tables"][key] = pva_table.copy()

    if roc_curve is not None:
        registry["roc_curves"][key] = roc_curve.copy()

    if ks_curve is not None:
        registry["ks_curves"][key] = ks_curve.copy()

    if calibration_table is not None:
        registry["calibration_tables"][key] = calibration_table.copy()

    return registry


def add_vif_table(
    registry: dict,
    model_id: str,
    vif_table: pd.DataFrame,
) -> dict:
    """
    Add VIF table for models where VIF is meaningful.
    """

    registry["vif_tables"][model_id] = vif_table.copy()
    return registry


def add_coefficient_table(
    registry: dict,
    model_id: str,
    coefficient_table: pd.DataFrame,
) -> dict:
    """
    Add coefficient / odds-ratio table.
    """

    registry["coefficient_tables"][model_id] = coefficient_table.copy()
    return registry


def add_feature_importance_table(
    registry: dict,
    model_id: str,
    feature_importance_table: pd.DataFrame,
) -> dict:
    """
    Add feature importance table for tree-style models.
    """

    registry["feature_importance_tables"][model_id] = (
        feature_importance_table.copy()
    )

    return registry


def add_artifact_path(
    registry: dict,
    model_id: str,
    artifact_path: str,
) -> dict:
    """
    Store saved model artifact path.
    """

    registry["artifact_paths"][model_id] = artifact_path
    return registry


def add_split_metadata(
    registry: dict,
    metadata: dict,
) -> dict:
    """
    Store train/validation split metadata.
    """

    registry["split_metadata"] = metadata.copy()
    return registry


def registry_to_summary_dataframe(registry: dict) -> pd.DataFrame:
    """
    Convert registry summary records to dataframe.
    """

    return pd.DataFrame(registry["summary"])


def registry_to_metrics_dataframe(registry: dict) -> pd.DataFrame:
    """
    Convert registry metric records to dataframe.
    """

    return pd.DataFrame(registry["metrics"])


def flatten_diagnostic_tables(tables: dict, table_name: str) -> pd.DataFrame:
    """
    Convert dictionary of diagnostic tables into one long dataframe.
    """

    frames = []

    for key, df in tables.items():
        model_id, split = key.rsplit("_", 1)

        temp = df.copy()
        temp.insert(0, "split", split)
        temp.insert(0, "model_id", model_id)
        temp.insert(0, "table_name", table_name)

        frames.append(temp)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def export_registry_to_excel(
    registry: dict,
    output_path,
) -> None:
    """
    Export lightweight audit registry tables to Excel.

    Large diagnostics such as lift, ROC, KS, and calibration should be
    exported separately as CSV/parquet to avoid oversized Excel files.
    """

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        registry_to_summary_dataframe(registry).to_excel(
            writer,
            sheet_name="model_summary",
            index=False,
        )

        registry_to_metrics_dataframe(registry).to_excel(
            writer,
            sheet_name="model_metrics",
            index=False,
        )

        flatten_simple_tables(
            registry["vif_tables"],
            "model_id",
        ).to_excel(writer, sheet_name="vif_tables", index=False)

        flatten_simple_tables(
            registry["coefficient_tables"],
            "model_id",
        ).to_excel(writer, sheet_name="coefficients", index=False)

        flatten_simple_tables(
            registry["feature_importance_tables"],
            "model_id",
        ).to_excel(writer, sheet_name="feature_importance", index=False)

        pd.DataFrame(
            [
                {"model_id": key, "artifact_path": value}
                for key, value in registry["artifact_paths"].items()
            ]
        ).to_excel(writer, sheet_name="artifact_paths", index=False)

        pd.DataFrame([registry["split_metadata"]]).to_excel(
            writer,
            sheet_name="split_metadata",
            index=False,
        )

def flatten_simple_tables(tables: dict, id_col: str) -> pd.DataFrame:
    """
    Flatten tables keyed by model_id.
    """

    frames = []

    for key, df in tables.items():
        temp = df.copy()
        temp.insert(0, id_col, key)
        frames.append(temp)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

def add_feature_importance_table(
    registry: dict,
    model_id: str,
    feature_importance_table: pd.DataFrame,
) -> dict:
    """
    Store feature importance table for a given model.
    """

    if "feature_importance_tables" not in registry:
        registry["feature_importance_tables"] = {}

    registry["feature_importance_tables"][model_id] = feature_importance_table

    return registry


def export_registry_artifacts(
    registry: dict,
    output_dir,
    save_models: bool = True,
) -> None:
    """
    Export registry in machine-readable form.

    This saves:
    - summary tables
    - metrics tables
    - diagnostics tables for Streamlit
    - model artifacts
    - metadata

    Excel remains for human audit.
    Parquet/JSON/joblib are the reusable system artifacts.
    """

    output_dir = Path(output_dir)

    registry_dir = output_dir / "registry"
    diagnostics_dir = output_dir / "diagnostics"
    models_dir = output_dir / "models"

    registry_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Core registry tables
    # -------------------------

    summary_df = registry_to_summary_dataframe(registry)
    metrics_df = registry_to_metrics_dataframe(registry)

    summary_df.to_parquet(registry_dir / "model_summary.parquet", index=False)
    summary_df.to_csv(registry_dir / "model_summary.csv", index=False)

    metrics_df.to_parquet(registry_dir / "model_metrics.parquet", index=False)
    metrics_df.to_csv(registry_dir / "model_metrics.csv", index=False)

    # -------------------------
    # Diagnostics for Streamlit
    # -------------------------

    diagnostic_exports = {
        "lift_tables": flatten_diagnostic_tables(
            registry["lift_tables"],
            "lift",
        ),
        "pva_tables": flatten_diagnostic_tables(
            registry["pva_tables"],
            "pva",
        ),
        "roc_curves": flatten_diagnostic_tables(
            registry["roc_curves"],
            "roc",
        ),
        "ks_curves": flatten_diagnostic_tables(
            registry["ks_curves"],
            "ks",
        ),
        "calibration_tables": flatten_diagnostic_tables(
            registry["calibration_tables"],
            "calibration",
        ),
    }

    for name, df in diagnostic_exports.items():
        if not df.empty:
            df.to_parquet(diagnostics_dir / f"{name}.parquet", index=False)
            df.to_csv(diagnostics_dir / f"{name}.csv", index=False)

    # -------------------------
    # Model explanation tables
    # -------------------------

    vif_df = flatten_simple_tables(registry["vif_tables"], "model_id")
    coef_df = flatten_simple_tables(registry["coefficient_tables"], "model_id")
    importance_df = flatten_simple_tables(
        registry["feature_importance_tables"],
        "model_id",
    )

    if not vif_df.empty:
        vif_df.to_parquet(diagnostics_dir / "vif_tables.parquet", index=False)
        vif_df.to_csv(diagnostics_dir / "vif_tables.csv", index=False)

    if not coef_df.empty:
        coef_df.to_parquet(diagnostics_dir / "coefficient_tables.parquet", index=False)
        coef_df.to_csv(diagnostics_dir / "coefficient_tables.csv", index=False)

    if not importance_df.empty:
        importance_df.to_parquet(
            diagnostics_dir / "feature_importance_tables.parquet",
            index=False,
        )
        importance_df.to_csv(
            diagnostics_dir / "feature_importance_tables.csv",
            index=False,
        )

    # -------------------------
    # Metadata
    # -------------------------

    with open(registry_dir / "split_metadata.json", "w") as f:
        json.dump(registry["split_metadata"], f, indent=4)

    artifact_paths = {}

    # -------------------------
    # Save fitted models
    # -------------------------

    if save_models:
        for model_id, model in registry["models"].items():
            model_path = models_dir / f"{model_id}.joblib"
            joblib.dump(model, model_path)

            artifact_paths[model_id] = str(model_path)

    with open(registry_dir / "artifact_paths.json", "w") as f:
        json.dump(artifact_paths, f, indent=4)