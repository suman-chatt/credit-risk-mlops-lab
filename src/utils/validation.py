def validate_required_columns(df, required_columns):
    """Raise an error if required columns are missing."""
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_dataframe_not_empty(df, name="dataframe"):
    """Raise an error if dataframe is empty."""
    if df.empty:
        raise ValueError(f"{name} is empty.")