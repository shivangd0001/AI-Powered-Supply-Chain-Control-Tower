
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Load master dataset
print("Loading master dataset...")

df = pd.read_csv(
    "data/processed/delivery_prediction_master.csv"
)

# 2. Define features
numeric_features = [
    'customer_zip_code_prefix', 'total_items',
    'total_price', 'total_freight',
    'average_item_price', 'average_product_weight',
    'total_product_weight', 'average_product_length',
    'average_product_height', 'average_product_width',
    'unique_products', 'unique_sellers',
    'purchase_year', 'purchase_month',
    'purchase_day', 'purchase_dayofweek',
    'purchase_hour', 'is_weekend',
    'promised_delivery_days'
]

categorical_features = [
    'customer_state',
    'seller_state',
    'product_category'
]

X = df[numeric_features + categorical_features]
y = df['is_late']

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# 4. Build preprocessing pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_features),
    ('categorical', categorical_pipeline, categorical_features)
])

# 5. Transform data
print("\nPreparing features...")

X_train_processed = preprocessor.fit_transform(X_train)

# 6. Train Random Forest
print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_processed, y_train)

# 7. Get feature names and importance
feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

# 8. Display top 20 features
print("\n" + "=" * 55)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 55)

print(importance_df.head(20).to_string(index=False))

# 9. Save files
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(model, "models/random_forest.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")

importance_df.to_csv(
    "reports/feature_importance.csv",
    index=False
)

# 10. Plot feature importance
top_features = importance_df.head(20).sort_values(
    by='Importance'
)

plt.figure(figsize=(12, 8))

plt.barh(
    top_features['Feature'],
    top_features['Importance']
)

plt.title("Random Forest - Top 20 Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig(
    "reports/feature_importance.png",
    dpi=300
)

plt.show()

print("\nFEATURE IMPORTANCE ANALYSIS COMPLETED!")
print("Model and preprocessor saved successfully.")
