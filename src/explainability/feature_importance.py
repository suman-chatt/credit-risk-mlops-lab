import pandas as pd


def load_feature_importance(path) -> pd.DataFrame:
    """
    Load exported tree/boosting feature importance table.
    """

    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)

    return pd.read_csv(path)


def get_model_feature_importance(
    feature_importance_df: pd.DataFrame,
    model_id: str,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Return feature importance for one model.
    """

    df = feature_importance_df[
        feature_importance_df["model_id"] == model_id
    ].copy()

    df = df.sort_values("importance", ascending=False)

    if top_n is not None:
        df = df.head(top_n)

    return df


def get_top_features_across_models(
    feature_importance_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Aggregate feature importance across tree/boosting models.
    """

    summary = (
        feature_importance_df.groupby("feature", observed=False)
        .agg(
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
            model_count=("model_id", "nunique"),
        )
        .reset_index()
        .sort_values("mean_importance", ascending=False)
    )

    return summary.head(top_n)


def create_feature_importance_summary(
    feature_importance_df: pd.DataFrame,
    champion_model_id: str,
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Create feature importance tables for reporting/dashboarding.
    """

    champion_importance = get_model_feature_importance(
        feature_importance_df,
        model_id=champion_model_id,
        top_n=top_n,
    )

    global_importance = get_top_features_across_models(
        feature_importance_df,
        top_n=top_n,
    )

    return {
        "champion_feature_importance": champion_importance,
        "global_feature_importance": global_importance,
    }