import numbers
from pathlib import Path

import mlflow
import mlflow.sklearn


TRACKING_DIR = Path("mlruns_local").resolve()


def _safe_param_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_metric_value(value):
    if isinstance(value, numbers.Number):
        return float(value)
    return None


def log_model_to_mlflow(
    model_id: str,
    model,
    params: dict,
    train_metrics: dict,
    validation_metrics: dict,
    model_family: str,
    model_name: str,
    dataset_type: str,
    experiment_name: str = "credit-risk-models",
) -> None:
    mlflow.set_tracking_uri(f"file://{TRACKING_DIR}")
    mlflow.set_experiment(experiment_name)

    print(f"[MLflow] logging {model_id} to {experiment_name}")

    with mlflow.start_run(run_name=model_id):
        mlflow.log_param("model_id", model_id)
        mlflow.log_param("model_family", model_family)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset_type", dataset_type)

        for key, value in params.items():
            mlflow.log_param(key, _safe_param_value(value))

        for key, value in train_metrics.items():
            metric_value = _safe_metric_value(value)
            if metric_value is not None:
                mlflow.log_metric(f"train_{key}", metric_value)

        for key, value in validation_metrics.items():
            metric_value = _safe_metric_value(value)
            if metric_value is not None:
                mlflow.log_metric(f"validation_{key}", metric_value)
