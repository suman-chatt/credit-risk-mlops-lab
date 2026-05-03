import numpy as np
import pandas as pd


def sample_for_explainability(
    df: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample rows for SHAP explainability.

    SHAP can be expensive, so we explain a representative sample.
    """

    if len(df) <= sample_size:
        return df.copy()

    return df.sample(
        n=sample_size,
        random_state=random_state,
    ).copy()


def compute_tree_shap_values(
    model,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute SHAP values for tree/boosting models.

    Requires shap to be installed.
    """

    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is not installed. Install it with: pip install shap"
        ) from exc

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Some classifiers return list[class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame(
        shap_values,
        columns=X.columns,
        index=X.index,
    )

    return shap_df


def create_global_shap_importance(
    shap_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create global SHAP importance table.
    """

    importance = shap_df.abs().mean().reset_index()
    importance.columns = ["feature", "mean_abs_shap"]

    return importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)


def create_applicant_shap_reasons(
    shap_df: pd.DataFrame,
    X: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Create applicant-level top SHAP drivers.

    Positive SHAP = increases model score.
    Negative SHAP = decreases model score.
    """

    rows = []

    for idx in shap_df.index:
        applicant_shap = shap_df.loc[idx].sort_values(
            key=np.abs,
            ascending=False,
        )

        top_features = applicant_shap.head(top_n)

        for rank, (feature, shap_value) in enumerate(top_features.items(), start=1):
            rows.append(
                {
                    "row_id": idx,
                    "rank": rank,
                    "feature": feature,
                    "feature_value": X.loc[idx, feature],
                    "shap_value": shap_value,
                    "direction": "risk_increasing"
                    if shap_value > 0
                    else "risk_decreasing",
                }
            )

    return pd.DataFrame(rows)