import pandas as pd


def load_csv_data(file_path):
    """Load a CSV file into a dataframe."""
    return pd.read_csv(file_path)


def load_training_data(file_path):
    """Load raw training data."""
    return pd.read_csv(file_path)


def load_test_data(file_path):
    """Load raw test data."""
    return pd.read_csv(file_path)


def load_data_dictionary(file_path):
    """Load data dictionary Excel file."""
    return pd.read_excel(file_path)