import numpy as np


def calculate_woe_iv(df, feature, target):
    """
    Calculate WOE and IV for a single binned feature.
    """

    grouped = df.groupby(feature)[target].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]

    grouped["good"] = grouped["total"] - grouped["bad"]

    # Avoid division by zero
    grouped["dist_good"] = grouped["good"] / grouped["good"].sum()
    grouped["dist_bad"] = grouped["bad"] / grouped["bad"].sum()

    grouped["dist_good"] = grouped["dist_good"].replace(0, 1e-6)
    grouped["dist_bad"] = grouped["dist_bad"].replace(0, 1e-6)

    grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
    grouped["iv"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]

    iv_value = grouped["iv"].sum()

    return grouped.reset_index(), iv_value


def build_woe_mappings(df, bin_cols, target):
    """
    Build WOE mappings for all bin columns.
    """

    woe_dict = {}
    iv_dict = {}

    for col in bin_cols:
        woe_table, iv_value = calculate_woe_iv(df, col, target)

        mapping = dict(zip(woe_table[col], woe_table["woe"]))

        woe_dict[col] = mapping
        iv_dict[col] = iv_value

    return woe_dict, iv_dict


def apply_woe_transformation(df, woe_dict):
    """
    Apply WOE transformation using precomputed mappings.
    """

    df_out = df.copy()

    for col, mapping in woe_dict.items():
        df_out[col + "_woe"] = df_out[col].map(mapping)

    return df_out


def build_woe_dataset(df, bin_cols, target):
    """
    Full WOE pipeline:
    - calculate mappings
    - apply mappings
    - return dataset + metadata
    """

    woe_dict, iv_dict = build_woe_mappings(df, bin_cols, target)

    df_woe = apply_woe_transformation(df, woe_dict)

    woe_features = [col + "_woe" for col in bin_cols]

    return df_woe, woe_dict, iv_dict, woe_features