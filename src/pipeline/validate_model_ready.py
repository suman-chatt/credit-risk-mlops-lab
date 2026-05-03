import pandas as pd

from src.config import OUTPUTS_SRC_DIR


MODEL_READY_DIR = OUTPUTS_SRC_DIR / "model_ready"

FILES = {
    "tree": "df_tree_model_ready_src.csv",
    "scaled": "df_scaled_model_ready_src.csv",
    "woe": "df_logit_woe_ready_src.csv",
    "binned_logit": "df_logit_bins_ready_src.csv",
}


def main():
    print("\n=== MODEL-READY DATASET VALIDATION ===\n")

    for name, filename in FILES.items():
        path = MODEL_READY_DIR / filename
        df = pd.read_csv(path)

        print(f"{name}:")
        print(f"  path: {path}")
        print(f"  shape: {df.shape}")
        print(f"  missing values: {df.isna().sum().sum()}")
        print(f"  duplicate rows: {df.duplicated().sum()}")

        if "SeriousDlqin2yrs" not in df.columns:
            print("  ERROR: missing target column")
        else:
            print(f"  target mean: {df['SeriousDlqin2yrs'].mean():.6f}")

        print(f"  feature count: {df.shape[1] - 1}")
        print()

    print("Validation complete.")


if __name__ == "__main__":
    main()