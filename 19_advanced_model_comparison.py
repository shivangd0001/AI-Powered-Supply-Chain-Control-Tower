import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
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

print("Loading dataset...")

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

print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

# ==========================================
# 3. FEATURES
# ==========================================

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

X_train = preprocessor.fit_transform(X_train).toarray()
X_val = preprocessor.transform(X_val).toarray()
X_test = preprocessor.transform(X_test).toarray()

print("Processed training:", X_train.shape)
print("Processed validation:", X_val.shape)
print("Processed testing:", X_test.shape)

# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )
}

# ==========================================
# 6. TRAIN AND VALIDATE
# ==========================================

results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 55)
    print("TRAINING:", name)

    if name == "Hist Gradient Boosting":
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights
        )

    else:
        model.fit(X_train, y_train)

    # Validation probabilities
    val_prob = model.predict_proba(X_val)[:, 1]

    # Select threshold using validation F1
    thresholds = np.arange(0.05, 0.81, 0.01)

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:

        val_pred = (val_prob >= threshold).astype(int)

        score = f1_score(
            y_val,
            val_pred,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    # Final evaluation on test
    test_prob = model.predict_proba(X_test)[:, 1]

    test_pred = (
        test_prob >= best_threshold
    ).astype(int)

    precision = precision_score(
        y_test, test_pred, zero_division=0
    )

    recall = recall_score(
        y_test, test_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, test_pred, zero_division=0
    )

    results.append({
        "Model": name,
        "Threshold": best_threshold,
        "Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc_score(y_test, test_prob),
        "PR-AUC": average_precision_score(y_test, test_prob)
    })

    trained_models[name] = model

    print("Best validation threshold:", best_threshold)
    print("Test precision:", round(precision, 4))
    print("Test recall:", round(recall, 4))
    print("Test F1:", round(f1, 4))

# ==========================================
# 7. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="PR-AUC",
    ascending=False
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))

# ==========================================
# 8. SAVE RESULTS
# ==========================================

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

results_df.to_csv(
    "reports/advanced_model_comparison.csv",
    index=False
)

joblib.dump(
    preprocessor,
    "models/advanced_preprocessor.pkl"
)

print("\nADVANCED MODEL COMPARISON COMPLETED!")