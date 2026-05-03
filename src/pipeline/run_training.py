import pandas as pd

from src.config import OUTPUTS_SRC_DIR
from src.models.data_split import prepare_training_datasets
from src.models.model_registry import (
    create_empty_registry,
    export_registry_artifacts,
    export_registry_to_excel,
)
from src.models.train import (
    train_bernoulli_nb_models,
    train_gaussian_nb_models,
    train_neural_network_models,
    train_regularized_logistic,
    train_staged_binned_logistic,
    train_staged_woe_logistic,
    train_tree_models,
    train_woe_gaussian_nb_models,
)


MODEL_READY_DIR = OUTPUTS_SRC_DIR / "model_ready"
IV_PATH = OUTPUTS_SRC_DIR / "woe" / "iv_values.json"
EXCEL_OUTPUT_PATH = OUTPUTS_SRC_DIR / "model_registry.xlsx"


def load_training_datasets() -> dict:
    """
    Load model-ready datasets and create shared train/validation splits.
    """

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

    print("\nDataset split check:")
    for dataset_name in ["tree", "scaled", "woe", "binned_logit", "binned_ohe"]:
        split = datasets[dataset_name]
        print(
            f"{dataset_name}: "
            f"X_train={split['X_train'].shape}, "
            f"X_val={split['X_val'].shape}, "
            f"features={len(split['features'])}"
        )

    return datasets


def main() -> None:
    OUTPUTS_SRC_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    datasets = load_training_datasets()

    registry = create_empty_registry()

    print("\nTraining WOE logistic...")
    registry = train_staged_woe_logistic(
        registry=registry,
        datasets=datasets,
        iv_path=IV_PATH,
    )

    print("\nTraining binned logistic...")
    registry = train_staged_binned_logistic(
        registry=registry,
        datasets=datasets,
        iv_path=IV_PATH,
    )

    print("\nTraining regularized logistic...")
    registry = train_regularized_logistic(
        registry=registry,
        datasets=datasets,
    )

    print("\nTraining Gaussian NB...")
    registry = train_gaussian_nb_models(
        registry=registry,
        datasets=datasets,
    )

    print("\nTraining Bernoulli NB...")
    registry = train_bernoulli_nb_models(
        registry=registry,
        datasets=datasets,
    )

    print("\nTraining WOE Gaussian NB...")
    registry = train_woe_gaussian_nb_models(
        registry=registry,
        datasets=datasets,
    )

    print("\nTraining neural networks...")
    registry = train_neural_network_models(
        registry=registry,
        datasets=datasets,
    )

    print("\nTraining tree / boosting models...")
    registry = train_tree_models(
        registry=registry,
        datasets=datasets,
    )

    print("\nSaving Excel registry...")
    export_registry_to_excel(
        registry=registry,
        output_path=EXCEL_OUTPUT_PATH,
    )

    print("\nSaving machine-readable registry artifacts...")
    export_registry_artifacts(
        registry=registry,
        output_dir=OUTPUTS_SRC_DIR,
        save_models=True,
    )

    summary = pd.DataFrame(registry["summary"])

    print("\nTraining pipeline complete.")
    print(f"Models trained: {len(summary)}")
    print(f"Excel registry saved to: {EXCEL_OUTPUT_PATH}")

    print("\nTop 10 models by validation AUC:")
    print(
        summary[
            [
                "model_id",
                "model_family",
                "model_name",
                "dataset_type",
                "feature_count",
                "validation_auc",
                "validation_ks",
                "validation_log_loss",
                "validation_brier_score",
            ]
        ]
        .sort_values("validation_auc", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()