
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# 1. Load dataset
print("Loading master dataset...")

df = pd.read_csv(
    "data/processed/delivery_prediction_master.csv"
)

# 2. Create chronological split
df["year_month"] = (
    df["purchase_year"].astype(int).astype(str)
    + "-"
    + df["purchase_month"].astype(int).astype(str).str.zfill(2)
)

df = df.sort_values("year_month")

train_df = df[df["year_month"] < "2018-04"].copy()
test_df = df[df["year_month"] >= "2018-04"].copy()

print("\nTRAINING PERIOD:")
print(train_df["year_month"].min(), "to",
      train_df["year_month"].max())

print("Training records:", len(train_df))

print("\nTESTING PERIOD:")
print(test_df["year_month"].min(), "to",
      test_df["year_month"].max())

print("Testing records:", len(test_df))

# 3. Define features
numeric_features = [
    "customer_zip_code_prefix",
    "total_items",
    "total_price",
    "total_freight",
    "average_item_price",
    "average_product_weight",
    "total_product_weight",
    "average_product_length",
    "average_product_height",
    "average_product_width",
    "unique_products",
    "unique_sellers",
    "purchase_year",
    "purchase_month",
    "purchase_day",
    "purchase_dayofweek",
    "purchase_hour",
    "is_weekend",
    "promised_delivery_days"
]

categorical_features = [
    "customer_state",
    "seller_state",
    "product_category"
]

X_train = train_df[numeric_features + categorical_features]
y_train = train_df["is_late"]

X_test = test_df[numeric_features + categorical_features]
y_test = test_df["is_late"]

# 4. Build preprocessing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 5. Transform training and testing data
print("\nPreparing features...")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

# 6. Train Random Forest
print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_processed, y_train)

# 7. Predictions
predictions = model.predict(X_test_processed)
probabilities = model.predict_proba(X_test_processed)[:, 1]

# 8. Evaluation
print("\n" + "=" * 60)
print("TEMPORAL VALIDATION RESULTS")
print("=" * 60)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nPerformance Metrics:")

metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions, zero_division=0),
    "Recall": recall_score(y_test, predictions, zero_division=0),
    "F1 Score": f1_score(y_test, predictions, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, probabilities),
    "PR-AUC": average_precision_score(y_test, probabilities)
}

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# 9. Save model and results
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(model, "models/temporal_random_forest.pkl")
joblib.dump(preprocessor, "models/temporal_preprocessor.pkl")

pd.DataFrame([metrics]).to_csv(
    "reports/temporal_validation_results.csv",
    index=False
)

print("\nTEMPORAL MODEL VALIDATION COMPLETED!")
