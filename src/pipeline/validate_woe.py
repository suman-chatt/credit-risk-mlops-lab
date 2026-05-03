import json

import pandas as pd

from src.config import OUTPUTS_SRC_DIR
from src.features.binning import FINAL_BIN_COLS


def main():
    binned_path = OUTPUTS_SRC_DIR / "features" / "df_model_binned_src.csv"
    woe_path = OUTPUTS_SRC_DIR / "woe" / "woe_mappings.json"

    df_binned = pd.read_csv(binned_path)

    with open(woe_path, "r") as f:
        woe_dict = json.load(f)

    print("\n=== BIN vs WOE CONSISTENCY CHECK ===\n")

    all_ok = True

    for col in FINAL_BIN_COLS:
        unique_bins = sorted(df_binned[col].dropna().astype(str).unique())
        woe_bins = sorted(str(key) for key in woe_dict[col].keys())

        num_bins = len(unique_bins)
        num_woe = len(woe_bins)

        print(col)
        print(f"  unique bins: {num_bins}")
        print(f"  WOE entries: {num_woe}")

        missing_in_woe = sorted(set(unique_bins) - set(woe_bins))
        extra_in_woe = sorted(set(woe_bins) - set(unique_bins))

        if missing_in_woe or extra_in_woe:
            all_ok = False
            print("  MISMATCH")
            print(f"  missing in WOE: {missing_in_woe}")
            print(f"  extra in WOE: {extra_in_woe}")
        else:
            print("  OK")

        print()

    if all_ok:
        print("All bin columns match their WOE mappings.")


if __name__ == "__main__":
    main()