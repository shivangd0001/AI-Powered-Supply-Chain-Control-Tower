import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw")

files = list(DATA_PATH.glob("*.csv"))

print("=" * 65)
print("ECOMMERCE DATA QUALITY ANALYSIS")
print("=" * 65)

for file in files:

    df = pd.read_csv(file, low_memory=False)

    print("\n\nFILE:", file.name)
    print("-" * 65)

    # Basic information
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # Missing values
    print("\nMISSING VALUES:")
    missing = df.isnull().sum()

    print(missing[missing > 0])

    # Duplicate records
    print("\nDUPLICATE ROWS:", df.duplicated().sum())

    # Data types
    print("\nDATA TYPES:")
    print(df.dtypes)

    # Unique values
    print("\nUNIQUE VALUES:")
    print(df.nunique())

    print("\n" + "=" * 65)

print("\nDATA QUALITY ANALYSIS COMPLETED")