import pandas as pd

from src.config import OUTPUTS_SRC_DIR
from src.features.scaling import fit_transform_scaler
from src.utils.io import save_dataframe, save_pickle


SCALING_OUTPUT_DIR = OUTPUTS_SRC_DIR / "features"


def main():
    input_path = OUTPUTS_SRC_DIR / "woe" / "df_model_woe_src.csv"
    output_path = SCALING_OUTPUT_DIR / "df_model_scaled_src.csv"
    scaler_path = SCALING_OUTPUT_DIR / "tree_standard_scaler_src.joblib"

    df = pd.read_csv(input_path)

    df_scaled, scaler = fit_transform_scaler(df)

    save_dataframe(df_scaled, output_path)
    save_pickle(scaler, scaler_path)


if __name__ == "__main__":
    main()