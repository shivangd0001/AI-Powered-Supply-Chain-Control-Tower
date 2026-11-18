

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

# 1. Load dataset
print("Loading dataset...")

df = pd.read_csv(
    "data/processed/delivery_prediction_master.csv"
)

# 2. Recreate temporal test data
df["year_month"] = (
    df["purchase_year"].astype(int).astype(str)
    + "-"
    + df["purchase_month"].astype(int).astype(str).str.zfill(2)
)

test_df = df[df["year_month"] >= "2018-04"].copy()

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

X_test = test_df[numeric_features + categorical_features]
y_test = test_df["is_late"]

# 4. Load temporal model and preprocessor
print("Loading temporal model...")

model = joblib.load(
    "models/temporal_random_forest.pkl"
)

preprocessor = joblib.load(
    "models/temporal_preprocessor.pkl"
)

# 5. Generate probabilities
X_test_processed = preprocessor.transform(X_test)

probabilities = model.predict_proba(
    X_test_processed
)[:, 1]

# 6. Test different thresholds
thresholds = [
    0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50
]

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Flagged Orders": int(predictions.sum()),
        "Actual Late Orders": int(y_test.sum())
    })

results_df = pd.DataFrame(results)

# 7. Display results
print("\nTHRESHOLD COMPARISON")
print("=" * 85)

print(results_df.round(4).to_string(index=False))

# 8. Save results
os.makedirs("reports", exist_ok=True)

results_df.to_csv(
    "reports/threshold_comparison.csv",
    index=False
)

# 9. Plot precision and recall
plt.figure(figsize=(10, 6))

plt.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["F1 Score"],
    marker="o",
    label="F1 Score"
)

plt.xlabel("Prediction Threshold")
plt.ylabel("Score")
plt.title("Delivery Risk Threshold Optimization")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "reports/threshold_optimization.png",
    dpi=300
)

plt.show()

# 10. Display best F1 threshold
best_row = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\nBEST THRESHOLD BY F1 SCORE")
print(best_row.to_string())

print("\nTHRESHOLD OPTIMIZATION COMPLETED!")
