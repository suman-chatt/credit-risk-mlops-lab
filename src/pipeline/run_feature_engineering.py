import pandas as pd

from src.config import OUTPUTS_SRC_DIR
from src.features.feature_engineering import build_model_ready_datasets
from src.utils.io import save_dataframe


FEATURE_OUTPUT_DIR = OUTPUTS_SRC_DIR / "model_ready"


def main():
    input_path = OUTPUTS_SRC_DIR / "features" / "df_model_scaled_src.csv"

    df_engineered = pd.read_csv(input_path)

    datasets = build_model_ready_datasets(df_engineered)

    save_dataframe(
        datasets["tree"],
        FEATURE_OUTPUT_DIR / "df_tree_model_ready_src.csv",
    )

    save_dataframe(
        datasets["scaled"],
        FEATURE_OUTPUT_DIR / "df_scaled_model_ready_src.csv",
    )

    save_dataframe(
        datasets["woe"],
        FEATURE_OUTPUT_DIR / "df_logit_woe_ready_src.csv",
    )

    save_dataframe(
        datasets["binned_logit"],
        FEATURE_OUTPUT_DIR / "df_logit_bins_ready_src.csv",
    )


if __name__ == "__main__":
    main()