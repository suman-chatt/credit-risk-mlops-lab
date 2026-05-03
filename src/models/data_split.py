import pandas as pd

from sklearn.model_selection import train_test_split


TARGET_COL = "SeriousDlqin2yrs"


def create_shared_train_validation_indices(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    validation_size: float = 0.30,
    random_state: int = 42,
) -> tuple[pd.Index, pd.Index]:
    """
    Create one shared stratified train/validation split.

    The same row indices should be reused across all model-ready datasets:
    - tree
    - scaled
    - WOE logistic
    - binned logistic
    - BernoulliNB one-hot dataset

    This prevents split drift between model families.
    """

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    train_idx, val_idx = train_test_split(
        df.index,
        test_size=validation_size,
        random_state=random_state,
        stratify=df[target_col],
    )

    return train_idx, val_idx


def split_xy(
    df: pd.DataFrame,
    train_idx: pd.Index,
    val_idx: pd.Index,
    target_col: str = TARGET_COL,
) -> dict:
    """
    Split one model-ready dataframe into train/validation X/y.
    """

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    feature_cols = [col for col in df.columns if col != target_col]

    X_train = df.loc[train_idx, feature_cols].copy()
    X_val = df.loc[val_idx, feature_cols].copy()

    y_train = df.loc[train_idx, target_col].copy()
    y_val = df.loc[val_idx, target_col].copy()

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "features": feature_cols,
    }


def one_hot_encode_binned_dataset(
    df_binned: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    One-hot encode binned categorical features.

    Used for BernoulliNB.

    Notebook logic used:
    pd.get_dummies(..., drop_first=False, dtype=int)

    We keep all levels because BernoulliNB expects binary indicator variables.
    """

    if target_col not in df_binned.columns:
        raise ValueError(f"Missing target column: {target_col}")

    X = df_binned.drop(columns=[target_col])
    y = df_binned[target_col]

    X_ohe = pd.get_dummies(
        X,
        drop_first=False,
        dtype=int,
    )

    df_ohe = pd.concat([y, X_ohe], axis=1)

    return df_ohe


def prepare_training_datasets(
    df_tree: pd.DataFrame,
    df_scaled: pd.DataFrame,
    df_woe: pd.DataFrame,
    df_binned_logit: pd.DataFrame,
    target_col: str = TARGET_COL,
    validation_size: float = 0.30,
    random_state: int = 42,
) -> dict:
    """
    Prepare all model-family datasets using one shared train/validation split.
    """

    train_idx, val_idx = create_shared_train_validation_indices(
        df=df_tree,
        target_col=target_col,
        validation_size=validation_size,
        random_state=random_state,
    )

    df_binned_ohe = one_hot_encode_binned_dataset(
        df_binned=df_binned_logit,
        target_col=target_col,
    )

    datasets = {
        "tree": split_xy(df_tree, train_idx, val_idx, target_col),
        "scaled": split_xy(df_scaled, train_idx, val_idx, target_col),
        "woe": split_xy(df_woe, train_idx, val_idx, target_col),
        "binned_logit": split_xy(df_binned_logit, train_idx, val_idx, target_col),
        "binned_ohe": split_xy(df_binned_ohe, train_idx, val_idx, target_col),
        "train_idx": train_idx,
        "val_idx": val_idx,
    }

    return datasets