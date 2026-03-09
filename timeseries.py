"""Simple helper for loading the sales data.

This is a small script that demonstrates a robust CSV loader that tries a
couple of common encodings (UTF-8 / Latin-1 / CP1252).

To run a user-friendly web UI for forecasting, run:

    streamlit run app.py
"""

import os

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file trying a few common encodings."""

    encodings = ["utf-8", "latin1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue

    # If none of the encodings worked, raise the error from pandas.
    return pd.read_csv(path)


if __name__ == "__main__":
    path = "train.csv"
    if not os.path.exists(path):
        raise SystemExit(f"Missing data file: {path}. Place it in the project root.")

    df = load_csv(path)
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print(df.info())
