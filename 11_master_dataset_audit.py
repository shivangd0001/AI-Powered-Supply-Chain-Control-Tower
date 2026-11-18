import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/processed")
REPORT_PATH = Path("reports")

REPORT_PATH.mkdir(parents=True, exist_ok=True)

# Load master dataset
df = pd.read_csv(
    DATA_PATH / "delivery_prediction_master.csv"
)

print("=" * 60)
print("MASTER DATASET AUDIT")
print("=" * 60)

# 1. Dataset dimensions
print("\n1. DATASET INFORMATION")

print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Unique orders:", df["order_id"].nunique())
print("Duplicate order IDs:", df["order_id"].duplicated().sum())

# 2. Target variable
print("\n2. TARGET VARIABLE ANALYSIS")

print(df["is_late"].value_counts())

print("\nTarget percentages:")

print(
    (df["is_late"].value_counts(normalize=True) * 100)
    .round(2)
)

# 3. Missing values
print("\n3. MISSING VALUE ANALYSIS")

missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percentage": df.isnull().mean() * 100
})

missing = missing[missing["missing_count"] > 0]

print(missing.sort_values(
    "missing_percentage",
    ascending=False
))

missing.to_csv(
    REPORT_PATH / "master_missing_values.csv"
)

# 4. Numeric feature summary
print("\n4. NUMERIC FEATURE SUMMARY")

print(df.describe().T)

df.describe().T.to_csv(
    REPORT_PATH / "master_numeric_summary.csv"
)

# 5. Check impossible values
print("\n5. LOGICAL VALIDATION")

print(
    "Negative promised delivery days:",
    (df["promised_delivery_days"] < 0).sum()
)

print(
    "Negative product prices:",
    (df["total_price"] < 0).sum()
)

print(
    "Negative freight values:",
    (df["total_freight"] < 0).sum()
)

print(
    "Negative product weights:",
    (df["total_product_weight"] < 0).sum()
)

# 6. Potential leakage columns
print("\n6. POTENTIAL DATA LEAKAGE COLUMNS")

leakage_columns = [
    "delivery_delay_days",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "is_late"
]

print([
    col for col in leakage_columns
    if col in df.columns
])

# 7. Feature correlation
print("\n7. NUMERIC FEATURE CORRELATION")

numeric_df = df.select_dtypes(include=np.number)

correlation = numeric_df.corr()

if "is_late" in correlation.columns:
    target_correlation = (
        correlation["is_late"]
        .drop("is_late")
        .sort_values(key=abs, ascending=False)
    )

    print(target_correlation)

    target_correlation.to_csv(
        REPORT_PATH / "feature_target_correlation.csv"
    )

print("\nAUDIT COMPLETED")