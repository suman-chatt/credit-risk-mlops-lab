import pandas as pd

from src.config import PREPROCESSING_OUTPUT_DIR, OUTPUTS_SRC_DIR
from src.features.binning import apply_final_binning, validate_final_bins
from src.utils.io import save_dataframe


BINNING_OUTPUT_DIR = OUTPUTS_SRC_DIR / "features"


def main():
    input_path = PREPROCESSING_OUTPUT_DIR / "df_model_preprocessed_src.csv"
    output_path = BINNING_OUTPUT_DIR / "df_model_binned_src.csv"

    df_preprocessed = pd.read_csv(input_path)

    df_binned = apply_final_binning(df_preprocessed)

    validate_final_bins(df_binned)

    save_dataframe(df_binned, output_path)
    print("Saving to:", output_path)

if __name__ == "__main__":
    main()