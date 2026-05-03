import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(
    X: pd.DataFrame,
    vif_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Calculate VIF for a numeric design matrix.

    VIF is only meaningful when there are at least two numeric features.
    """

    X_numeric = X.select_dtypes(include="number").copy()

    if X_numeric.empty:
        return pd.DataFrame(
            columns=["feature", "vif", "vif_flag", "vif_threshold"]
        )

    if X_numeric.shape[1] == 1:
        return pd.DataFrame(
            [
                {
                    "feature": X_numeric.columns[0],
                    "vif": 1.0,
                    "vif_flag": "single_feature",
                    "vif_threshold": vif_threshold,
                }
            ]
        )

    vif_rows = []

    for idx, col in enumerate(X_numeric.columns):
        vif_value = variance_inflation_factor(X_numeric.values, idx)

        if vif_value <= vif_threshold:
            flag = "preferred"
        elif vif_value <= 5:
            flag = "caution"
        else:
            flag = "high_concern"

        vif_rows.append(
            {
                "feature": col,
                "vif": float(vif_value),
                "vif_flag": flag,
                "vif_threshold": vif_threshold,
            }
        )

    return pd.DataFrame(vif_rows).sort_values("vif", ascending=False)


def summarize_vif(
    vif_table: pd.DataFrame,
    vif_threshold: float = 3.0,
) -> dict:
    """
    Summarize VIF table for model registry.
    """

    if vif_table.empty:
        return {
            "max_vif": None,
            "vif_flag": "not_available",
            "num_features_above_vif_threshold": 0,
        }

    max_vif = float(vif_table["vif"].max())

    if "single_feature" in set(vif_table["vif_flag"]):
        flag = "single_feature"
    elif max_vif <= vif_threshold:
        flag = "preferred"
    elif max_vif <= 5:
        flag = "caution"
    else:
        flag = "high_concern"

    return {
        "max_vif": max_vif,
        "vif_flag": flag,
        "num_features_above_vif_threshold": int(
            (vif_table["vif"] > vif_threshold).sum()
        ),
    }