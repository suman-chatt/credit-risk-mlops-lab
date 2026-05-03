import json
import pickle
from pathlib import Path


def ensure_directory(path):
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_dataframe(df, file_path):
    """Save dataframe based on file extension."""
    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    if file_path.suffix == ".csv":
        df.to_csv(file_path, index=False)
    elif file_path.suffix == ".parquet":
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Unsupported dataframe file type: {file_path.suffix}")


def save_json(obj, file_path):
    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    with open(file_path, "w") as f:
        json.dump(obj, f, indent=4)


def save_pickle(obj, file_path):
    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)