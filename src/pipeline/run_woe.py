import pandas as pd

from src.config import OUTPUTS_SRC_DIR
from src.features.binning import FINAL_BIN_COLS
from src.features.woe import build_woe_dataset
from src.utils.io import save_dataframe, save_json


WOE_OUTPUT_DIR = OUTPUTS_SRC_DIR / "woe"


def main():
    input_path = OUTPUTS_SRC_DIR / "features" / "df_model_binned_src.csv"

    df_binned = pd.read_csv(input_path)

    target = "SeriousDlqin2yrs"

    df_woe, woe_dict, iv_dict, woe_features = build_woe_dataset(
        df_binned,
        FINAL_BIN_COLS,
        target,
    )

    save_dataframe(df_woe, WOE_OUTPUT_DIR / "df_model_woe_src.csv")
    save_json(woe_dict, WOE_OUTPUT_DIR / "woe_mappings.json")
    save_json(iv_dict, WOE_OUTPUT_DIR / "iv_values.json")


if __name__ == "__main__":
    main()