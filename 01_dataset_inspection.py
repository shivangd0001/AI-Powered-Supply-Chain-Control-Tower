import pandas as pd
from pathlib import Path

# Dataset folder
DATA_PATH = Path("data/raw")

# All dataset files
files = list(DATA_PATH.glob("*.csv"))

print("TOTAL CSV FILES:", len(files))
print("-" * 60)

for file in files:

    df = pd.read_csv(file, low_memory=False)

    print("\nFILE:", file.name)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn Names:")
    print(list(df.columns))

    print("\nMissing Values:")
    print(df.isnull().sum().sum())

    print("Duplicate Rows:", df.duplicated().sum())

    print("-" * 60)