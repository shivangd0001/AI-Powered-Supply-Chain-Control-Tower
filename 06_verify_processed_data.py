import pandas as pd
from pathlib import Path

folder = Path("data/processed")

files = list(folder.glob("*.csv"))

print("Total processed files:", len(files))

for file in files:
    df = pd.read_csv(file, low_memory=False)

    print("\n", file.name)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

print("\nVerification completed!")