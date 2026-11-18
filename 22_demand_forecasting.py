import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

REPORTS.mkdir(exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading datasets...")

orders = pd.read_csv("data/processed/olist_orders_dataset.csv")
items = pd.read_csv("data/processed/olist_order_items_dataset.csv")
products = pd.read_csv("data/processed/olist_products_dataset.csv")
translation = pd.read_csv("data/processed/product_category_name_translation.csv")

# ==========================================
# MERGE DATA
# ==========================================

print("Combining datasets...")

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

df = items.merge(
    orders[["order_id", "order_purchase_timestamp"]],
    on="order_id",
    how="inner"
)

df = df.merge(
    products[["product_id", "product_category_name"]],
    on="product_id",
    how="left"
)

df = df.merge(
    translation,
    on="product_category_name",
    how="left"
)

df["category"] = df[
    "product_category_name_english"
].fillna("unknown")

df = df.dropna(
    subset=["order_purchase_timestamp"]
)

# ==========================================
# MONTHLY CATEGORY DEMAND
# ==========================================

df["month"] = (
    df["order_purchase_timestamp"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

monthly = (
    df.groupby(["month", "category"])
    .size()
    .reset_index(name="demand")
)

all_months = pd.date_range(
    monthly["month"].min(),
    monthly["month"].max(),
    freq="MS"
)

all_categories = sorted(
    monthly["category"].unique()
)

full_index = pd.MultiIndex.from_product(
    [all_months, all_categories],
    names=["month", "category"]
)

monthly = (
    monthly.set_index(["month", "category"])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

monthly = monthly.sort_values(
    ["category", "month"]
)

print("Monthly demand dataset created.")
print("Total records:", len(monthly))
print("Categories:", monthly["category"].nunique())

# ==========================================
# FEATURE ENGINEERING
# ==========================================

monthly["lag_1"] = (
    monthly.groupby("category")["demand"].shift(1)
)

monthly["lag_2"] = (
    monthly.groupby("category")["demand"].shift(2)
)

monthly["lag_3"] = (
    monthly.groupby("category")["demand"].shift(3)
)

monthly["rolling_mean_3"] = (
    monthly.groupby("category")["demand"]
    .transform(lambda x: x.shift(1).rolling(3).mean())
)

monthly["month_number"] = monthly["month"].dt.month

monthly = monthly.dropna().copy()

# ==========================================
# CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================

dates = sorted(monthly["month"].unique())

test_dates = dates[-3:]

train = monthly[
    ~monthly["month"].isin(test_dates)
].copy()

test = monthly[
    monthly["month"].isin(test_dates)
].copy()

features = [
    "category",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "month_number"
]

X_train = pd.get_dummies(
    train[features],
    columns=["category"]
)

X_test = pd.get_dummies(
    test[features],
    columns=["category"]
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

y_train = train["demand"]
y_test = test["demand"]

# ==========================================
# TRAIN RANDOM FOREST
# ==========================================

print("\nTraining Random Forest Demand Model...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# MODEL EVALUATION
# ==========================================

predictions = np.maximum(
    model.predict(X_test),
    0
)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

print("\n========== MODEL PERFORMANCE ==========")

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# ==========================================
# THREE-MONTH FUTURE FORECAST
# ==========================================

print("\nGenerating future demand forecasts...")

history = monthly.copy()

future_dates = pd.date_range(
    monthly["month"].max() + pd.offsets.MonthBegin(1),
    periods=3,
    freq="MS"
)

forecast_records = []

for future_date in future_dates:

    for category in all_categories:

        category_data = history[
            history["category"] == category
        ].sort_values("month")

        values = category_data["demand"].tail(3).tolist()

        lag_1 = values[-1]
        lag_2 = values[-2]
        lag_3 = values[-3]

        rolling_mean = np.mean(values)

        row = pd.DataFrame([{
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "rolling_mean_3": rolling_mean,
            "month_number": future_date.month,
            "category": category
        }])

        row_encoded = pd.get_dummies(
            row,
            columns=["category"]
        )

        row_encoded = row_encoded.reindex(
            columns=X_train.columns,
            fill_value=0
        )

        forecast = max(
            0,
            model.predict(row_encoded)[0]
        )

        forecast = int(round(forecast))

        forecast_records.append({
            "month": future_date,
            "category": category,
            "forecast_demand": forecast
        })

        history = pd.concat([
            history,
            pd.DataFrame([{
                "month": future_date,
                "category": category,
                "demand": forecast
            }])
        ], ignore_index=True)

forecast_df = pd.DataFrame(forecast_records)

# ==========================================
# SAVE RESULTS
# ==========================================

forecast_df.to_csv(
    REPORTS / "demand_forecast.csv",
    index=False
)

monthly.to_csv(
    REPORTS / "monthly_category_demand.csv",
    index=False
)

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

feature_importance.to_csv(
    REPORTS / "demand_feature_importance.csv",
    index=False
)

print("\n========== FORECAST COMPLETE ==========")

print("\nFuture Demand Forecast:")
print(forecast_df.head(15).to_string(index=False))

print("\nFiles saved successfully:")
print("1. reports/demand_forecast.csv")
print("2. reports/monthly_category_demand.csv")
print("3. reports/demand_feature_importance.csv")