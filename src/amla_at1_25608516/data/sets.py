import pandas as pd
from pathlib import Path
from typing import Tuple

def load_data(data_path: str) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
]:
    """Load train, validation, and test splits into X/y.

    Parameters
    ----------
    data_path : str
        Path to directory containing the split files.
        Expected filenames (but not all required): 
        - X_train.csv, y_train.csv
        - X_val.csv,   y_val.csv
        - X_test.csv,  y_test.csv

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : tuple
        Features (DataFrame) and targets (Series).
        If a file is missing, returns an empty DataFrame (for X) 
        or empty Series (for y).
    """
    data_path = Path(data_path)
    expected_files = [
        "X_train.csv", "y_train.csv",
        "X_val.csv",   "y_val.csv",
        "X_test.csv",  "y_test.csv",
    ]

    splits = {}
    for fname in expected_files:
        file = data_path / fname
        if file.exists():
            df = pd.read_csv(file)
            # If it's a y_ file → flatten to Series
            if fname.startswith("y_"):
                if df.shape[1] == 1:
                    df = df.iloc[:, 0]
            splits[fname] = df
        else:
            # Return empty placeholder
            if fname.startswith("y_"):
                splits[fname] = pd.Series([], name=fname.replace(".csv", ""))
            else:
                splits[fname] = pd.DataFrame()

    return (
        splits["X_train.csv"], splits["y_train.csv"],
        splits["X_val.csv"],   splits["y_val.csv"],
        splits["X_test.csv"],  splits["y_test.csv"],
    )