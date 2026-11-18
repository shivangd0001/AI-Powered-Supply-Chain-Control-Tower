import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

DATA_PATH = Path("data/processed")
REPORT_PATH = Path("reports")

REPORT_PATH.mkdir(parents=True, exist_ok=True)

# Load master dataset
df = pd.read_csv(
    DATA_PATH / "delivery_prediction_master.csv"
)

print("Master dataset loaded:", df.shape)

# Target
y = df["is_late"]

# Remove leakage and identifiers
excluded_columns = [
    "order_id",
    "customer_id",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "delivery_delay_days",
    "is_late"
]

X = df.drop(columns=excluded_columns)

# Categorical features
categorical_features = [
    "customer_state",
    "seller_state",
    "product_category"
]

# Numeric features
numeric_features = [
    col for col in X.columns
    if col not in categorical_features
]

print("\nNumeric features:", numeric_features)
print("\nCategorical features:", categorical_features)

# Preprocessing for numeric features
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA:", X_train.shape)
print("TESTING DATA:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True) * 100)

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

# Save datasets
from scipy.sparse import save_npz

save_npz(
    DATA_PATH / "X_train.npz",
    X_train_processed
)

save_npz(
    DATA_PATH / "X_test.npz",
    X_test_processed
)

pd.DataFrame({
    "is_late": y_train
}).to_csv(DATA_PATH / "y_train.csv", index=False)

pd.DataFrame({
    "is_late": y_test
}).to_csv(DATA_PATH / "y_test.csv", index=False)

print("\nML DATA PREPARATION COMPLETED")