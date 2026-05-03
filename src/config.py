from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_SRC_DIR = PROJECT_ROOT / "outputs_src"
PREPROCESSING_OUTPUT_DIR = OUTPUTS_SRC_DIR / "preprocessing"

RAW_TRAIN_FILE = DATA_DIR / "cs-training.csv"
RAW_TEST_FILE = DATA_DIR / "cs-test.csv"
DATA_DICTIONARY_FILE = DATA_DIR / "Data Dictionary.xls"

TARGET_COL = "SeriousDlqin2yrs"
ID_COL = "Unnamed: 0"