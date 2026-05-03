import pandas as pd

from src.config import DATA_DIR, OUTPUTS_SRC_DIR


def main():
    notebook_path = DATA_DIR / "df_scaled_model_ready.csv"
    src_path = OUTPUTS_SRC_DIR / "features" / "df_model_scaled_src.csv"

    df_notebook = pd.read_csv(notebook_path)
    df_src = pd.read_csv(src_path)

    print("\n=== SCALED DATASET SHAPES ===")
    print(f"Notebook: {df_notebook.shape}")
    print(f"src:      {df_src.shape}")

    notebook_scaled_cols = sorted(
        [col for col in df_notebook.columns if col.endswith("_scaled")]
    )

    src_scaled_cols = sorted(
        [col for col in df_src.columns if col.endswith("_scaled")]
    )

    print("\n=== SCALED COLUMN CHECK ===")
    print("Only in notebook:")
    print(sorted(set(notebook_scaled_cols) - set(src_scaled_cols)))

    print("\nOnly in src:")
    print(sorted(set(src_scaled_cols) - set(notebook_scaled_cols)))

    common_scaled_cols = sorted(set(notebook_scaled_cols) & set(src_scaled_cols))

    print("\n=== SCALED VALUE CHECK ===")

    mismatches = []

    for col in common_scaled_cols:
        max_abs_diff = (df_notebook[col] - df_src[col]).abs().max()

        if max_abs_diff > 1e-10:
            mismatches.append((col, max_abs_diff))

    if not mismatches:
        print("All common scaled columns match.")
    else:
        print("Mismatched scaled columns:")
        for col, diff in mismatches:
            print(f"{col}: max abs diff = {diff}")

    print("\n=== BASIC SCALING SANITY CHECK SRC ===")

    summary = df_src[src_scaled_cols].agg(["mean", "std"]).T
    print(summary)


if __name__ == "__main__":
    main()