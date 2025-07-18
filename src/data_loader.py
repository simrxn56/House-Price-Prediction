import pandas as pd
import os

def load_data(filepath):
    """
    Load a CSV file into a DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)


def save_data(df, filepath):
    """
    Save a DataFrame to a CSV file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
