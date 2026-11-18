import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

files = list(RAW_PATH.glob("*.csv"))

print("=" * 60)
print("INITIAL DATA PREPARATION")
print("=" * 60)

for file in files:

    print("\nProcessing:", file.name)

    df = pd.read_csv(file, low_memory=False)

    original_rows = len(df)

    # Convert date and timestamp columns
    date_columns = [
        col for col in df.columns
        if "date" in col.lower() or "timestamp" in col.lower()
    ]

    for col in date_columns:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )

    # Remove only exact duplicates in geolocation
    if file.name == "olist_geolocation_dataset.csv":

        before = len(df)

        df = df.drop_duplicates()

        removed = before - len(df)

        print("Exact duplicates removed:", removed)

    # Save processed dataset
    output_file = PROCESSED_PATH / file.name

    df.to_csv(output_file, index=False)

    print("Original rows:", original_rows)
    print("Processed rows:", len(df))
    print("Date columns converted:", date_columns)
    print("Saved:", output_file)

print("\n" + "=" * 60)
print("INITIAL DATA PREPARATION COMPLETED")
print("=" * 60)