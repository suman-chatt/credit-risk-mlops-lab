import pandas as pd

from src.config import DATA_DIR, PREPROCESSING_OUTPUT_DIR


def main():
    notebook_path = DATA_DIR / "df_model_preprocessed.csv"
    src_path = PREPROCESSING_OUTPUT_DIR / "df_model_preprocessed_src.csv"

    df_notebook = pd.read_csv(notebook_path)
    df_src = pd.read_csv(src_path)

    print("Notebook shape:", df_notebook.shape)
    print("src shape:", df_src.shape)

    notebook_cols = set(df_notebook.columns)
    src_cols = set(df_src.columns)

    print("\nColumns only in notebook:")
    print(sorted(notebook_cols - src_cols))

    print("\nColumns only in src:")
    print(sorted(src_cols - notebook_cols))

    common_cols = sorted(notebook_cols & src_cols)

    mismatched_cols = []
    for col in common_cols:
        if not df_notebook[col].equals(df_src[col]):
            mismatched_cols.append(col)

    print("\nCommon columns with mismatched values:")
    print(mismatched_cols)


if __name__ == "__main__":
    main()