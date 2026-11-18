import os
import joblib
import numpy as np
import pandas as pd

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

# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading master dataset...")

df = pd.read_csv(
    "data/processed/delivery_prediction_master.csv"
)

df["year_month"] = (
    df["purchase_year"].astype(int).astype(str)
    + "-"
    + df["purchase_month"].astype(int).astype(str).str.zfill(2)
)

# ==========================================
# 2. CHRONOLOGICAL SPLIT
# ==========================================

train_df = df[df["year_month"] <= "2017-12"].copy()

val_df = df[
    (df["year_month"] >= "2018-01") &
    (df["year_month"] <= "2018-03")
].copy()

test_df = df[df["year_month"] >= "2018-04"].copy()

print("\nDATASET DISTRIBUTION")

for name, data in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:
    print(
        f"{name}: {len(data)} orders | "
        f"Late rate: {data['is_late'].mean()*100:.2f}%"
    )

# ==========================================
# 3. DEFINE FEATURES
# ==========================================

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

feature_columns = numeric_features + categorical_features

X_train = train_df[feature_columns]
y_train = train_df["is_late"]

X_val = val_df[feature_columns]
y_val = val_df["is_late"]

X_test = test_df[feature_columns]
y_test = test_df["is_late"]

# ==========================================
# 4. PREPROCESSING
# ==========================================

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

print("\nPreparing features...")

X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

X_test_processed = preprocessor.transform(X_test)

# ==========================================
# 5. TRAIN RANDOM FOREST
# ==========================================

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

# ==========================================
# 6. VALIDATION PREDICTIONS
# ==========================================

val_probabilities = model.predict_proba(
    X_val_processed
)[:, 1]

thresholds = np.arange(0.05, 0.81, 0.01)

threshold_results = []

for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_val, val_predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, val_predictions, zero_division=0
        ),
        "F1": f1_score(
            y_val, val_predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

best_row = threshold_df.loc[
    threshold_df["F1"].idxmax()
]

best_threshold = best_row["Threshold"]

print("\nBEST VALIDATION THRESHOLD")
print(best_row)

# ==========================================
# 7. FINAL TEST EVALUATION
# ==========================================

print("\nEvaluating untouched test dataset...")

test_probabilities = model.predict_proba(
    X_test_processed
)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nFINAL CLASSIFICATION REPORT")

print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

print("\nCONFUSION MATRIX")

print(confusion_matrix(
    y_test,
    test_predictions
))

# ==========================================
# 8. FINAL METRICS
# ==========================================

metrics = {
    "Threshold": best_threshold,
    "Accuracy": accuracy_score(y_test, test_predictions),
    "Precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "F1 Score": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(
        y_test, test_probabilities
    ),
    "PR-AUC": average_precision_score(
        y_test, test_probabilities
    )
}

print("\nFINAL TEST PERFORMANCE")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# ==========================================
# 9. SAVE OUTPUTS
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    model,
    "models/final_temporal_random_forest.pkl"
)

joblib.dump(
    preprocessor,
    "models/final_temporal_preprocessor.pkl"
)

threshold_df.to_csv(
    "reports/validation_threshold_results.csv",
    index=False
)

pd.DataFrame([metrics]).to_csv(
    "reports/final_temporal_metrics.csv",
    index=False
)

print("\nPROPER TEMPORAL VALIDATION COMPLETED!")