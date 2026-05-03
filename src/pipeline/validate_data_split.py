import pandas as pd

from src.config import OUTPUTS_SRC_DIR
from src.models.data_split import prepare_training_datasets


MODEL_READY_DIR = OUTPUTS_SRC_DIR / "model_ready"


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
    )

    print("\n=== SHARED SPLIT VALIDATION ===\n")
    print(f"Train rows: {len(datasets['train_idx'])}")
    print(f"Validation rows: {len(datasets['val_idx'])}")

    for name in ["tree", "scaled", "woe", "binned_logit", "binned_ohe"]:
        split = datasets[name]

        print(f"\n{name}")
        print(f"  X_train: {split['X_train'].shape}")
        print(f"  X_val:   {split['X_val'].shape}")
        print(f"  y_train mean: {split['y_train'].mean():.6f}")
        print(f"  y_val mean:   {split['y_val'].mean():.6f}")
        print(f"  feature count: {len(split['features'])}")


if __name__ == "__main__":
    main()