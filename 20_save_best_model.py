import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# =====================================
# 1. LOAD DATA
# =====================================

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/delivery_prediction_master.csv"
)

df["year_month"] = (
    df["purchase_year"].astype(int).astype(str)
    + "-"
    + df["purchase_month"].astype(int).astype(str).str.zfill(2)
)

# =====================================
# 2. CHRONOLOGICAL SPLIT
# =====================================

train_df = df[df["year_month"] <= "2017-12"].copy()

val_df = df[
    (df["year_month"] >= "2018-01") &
    (df["year_month"] <= "2018-03")
].copy()

test_df = df[df["year_month"] >= "2018-04"].copy()

# =====================================
# 3. FEATURES
# =====================================

numeric_features = [
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
    "customer_zip_code_prefix",
    "customer_state",
    "seller_state",
    "product_category"
]

features = numeric_features + categorical_features

X_train = train_df[features]
y_train = train_df["is_late"]

X_val = val_df[features]
y_val = val_df["is_late"]

X_test = test_df[features]
y_test = test_df["is_late"]

# =====================================
# 4. PREPROCESSING
# =====================================

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

X_train = preprocessor.fit_transform(X_train).toarray()
X_val = preprocessor.transform(X_val).toarray()
X_test = preprocessor.transform(X_test).toarray()

# =====================================
# 5. TRAIN WINNING MODEL
# =====================================

print("\nTraining HistGradientBoosting...")

model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

model.fit(
    X_train,
    y_train,
    sample_weight=weights
)

# =====================================
# 6. SELECT VALIDATION THRESHOLD
# =====================================

val_probabilities = model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.05, 0.81, 0.01)

best_threshold = 0.5
best_f1 = -1

for threshold in thresholds:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print("\nSelected threshold:", best_threshold)
print("Validation F1:", round(best_f1, 4))

# =====================================
# 7. FINAL TEST EVALUATION
# =====================================

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nFINAL TEST REPORT")

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

print("\nFINAL METRICS")

print("Precision:", precision_score(
    y_test, test_predictions, zero_division=0
))

print("Recall:", recall_score(
    y_test, test_predictions, zero_division=0
))

print("F1 Score:", f1_score(
    y_test, test_predictions, zero_division=0
))

print("ROC-AUC:", roc_auc_score(
    y_test, test_probabilities
))

print("PR-AUC:", average_precision_score(
    y_test, test_probabilities
))

# =====================================
# 8. SAVE MODEL
# =====================================

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    model,
    "models/best_delivery_model.pkl"
)

joblib.dump(
    preprocessor,
    "models/best_delivery_preprocessor.pkl"
)

joblib.dump(
    best_threshold,
    "models/best_delivery_threshold.pkl"
)

# =====================================
# 9. SAVE TEST PREDICTIONS
# =====================================

predictions_df = pd.DataFrame({
    "order_id": test_df["order_id"].values,
    "actual_late": y_test.values,
    "late_probability": test_probabilities,
    "predicted_late": test_predictions
})

predictions_df.to_csv(
    "reports/delivery_predictions.csv",
    index=False
)

print("\nBEST MODEL SAVED SUCCESSFULLY!")
print("Predictions saved successfully!")
