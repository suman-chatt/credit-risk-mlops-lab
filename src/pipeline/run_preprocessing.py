from src.config import RAW_TRAIN_FILE, PREPROCESSING_OUTPUT_DIR
from src.data.load_data import load_training_data
from src.preprocessing.preprocessing import preprocess_training_data
from src.utils.io import save_dataframe, save_json


def main():
    df_raw = load_training_data(RAW_TRAIN_FILE)

    df_preprocessed, preprocessing_values = preprocess_training_data(df_raw)

    save_dataframe(
        df_preprocessed,
        PREPROCESSING_OUTPUT_DIR / "df_model_preprocessed_src.csv",
    )

    save_json(
        preprocessing_values,
        PREPROCESSING_OUTPUT_DIR / "preprocessing_values_src.json",
    )


if __name__ == "__main__":
    main()