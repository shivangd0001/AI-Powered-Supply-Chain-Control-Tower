import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/processed")
OUTPUT_PATH = Path("data/processed")

# ------------------------------------------------
# 1. LOAD DATASETS
# ------------------------------------------------

orders = pd.read_csv(DATA_PATH / "olist_orders_dataset.csv")
items = pd.read_csv(DATA_PATH / "olist_order_items_dataset.csv")
customers = pd.read_csv(DATA_PATH / "olist_customers_dataset.csv")
products = pd.read_csv(DATA_PATH / "olist_products_dataset.csv")
sellers = pd.read_csv(DATA_PATH / "olist_sellers_dataset.csv")

print("Datasets loaded successfully.")

# ------------------------------------------------
# 2. CONVERT DATE COLUMNS
# ------------------------------------------------

date_columns = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_columns:
    orders[col] = pd.to_datetime(
        orders[col], errors="coerce"
    )

# ------------------------------------------------
# 3. CREATE TARGET DATASET
# ------------------------------------------------

model_orders = orders[
    (orders["order_status"] == "delivered") &
    (orders["order_delivered_customer_date"].notna()) &
    (orders["order_estimated_delivery_date"].notna())
].copy()

model_orders["delivery_delay_days"] = (
    model_orders["order_delivered_customer_date"] -
    model_orders["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

model_orders["is_late"] = (
    model_orders["delivery_delay_days"] > 0
).astype(int)

# ------------------------------------------------
# 4. CUSTOMER INFORMATION
# ------------------------------------------------

customer_features = customers[
    [
        "customer_id",
        "customer_state",
        "customer_zip_code_prefix"
    ]
].copy()

# ------------------------------------------------
# 5. PRODUCT AND SELLER INFORMATION
# ------------------------------------------------

item_details = items.merge(
    products[
        [
            "product_id",
            "product_category_name",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm"
        ]
    ],
    on="product_id",
    how="left"
)

item_details = item_details.merge(
    sellers[
        [
            "seller_id",
            "seller_state",
            "seller_zip_code_prefix"
        ]
    ],
    on="seller_id",
    how="left"
)

# ------------------------------------------------
# 6. AGGREGATE ITEM INFORMATION AT ORDER LEVEL
# ------------------------------------------------

def most_common(series):
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else np.nan

order_features = item_details.groupby("order_id").agg(
    total_items=("order_item_id", "count"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
    average_item_price=("price", "mean"),
    average_product_weight=("product_weight_g", "mean"),
    total_product_weight=("product_weight_g", "sum"),
    average_product_length=("product_length_cm", "mean"),
    average_product_height=("product_height_cm", "mean"),
    average_product_width=("product_width_cm", "mean"),
    unique_products=("product_id", "nunique"),
    unique_sellers=("seller_id", "nunique"),
    product_category=("product_category_name", most_common),
    seller_state=("seller_state", most_common)
).reset_index()

# ------------------------------------------------
# 7. COMBINE ORDER AND CUSTOMER INFORMATION
# ------------------------------------------------

master = model_orders[
    [
        "order_id",
        "customer_id",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "delivery_delay_days",
        "is_late"
    ]
].merge(
    customer_features,
    on="customer_id",
    how="left"
)

master = master.merge(
    order_features,
    on="order_id",
    how="left"
)

# ------------------------------------------------
# 8. CREATE TIME-BASED FEATURES
# ------------------------------------------------

purchase = master["order_purchase_timestamp"]

master["purchase_year"] = purchase.dt.year
master["purchase_month"] = purchase.dt.month
master["purchase_day"] = purchase.dt.day
master["purchase_dayofweek"] = purchase.dt.dayofweek
master["purchase_hour"] = purchase.dt.hour

master["is_weekend"] = (
    purchase.dt.dayofweek >= 5
).astype(int)

master["promised_delivery_days"] = (
    master["order_estimated_delivery_date"] -
    master["order_purchase_timestamp"]
).dt.total_seconds() / 86400

# ------------------------------------------------
# 9. SAVE MASTER DATASET
# ------------------------------------------------

master = master.replace([np.inf, -np.inf], np.nan)

output_file = OUTPUT_PATH / "delivery_prediction_master.csv"

master.to_csv(output_file, index=False)

# ------------------------------------------------
# 10. DATASET SUMMARY
# ------------------------------------------------

print("\nMASTER DATASET CREATED")

print("Total records:", len(master))
print("Total columns:", len(master.columns))

print("\nTarget distribution:")
print(master["is_late"].value_counts())

print("\nTarget percentage:")
print(
    (master["is_late"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nMissing values:")
print(
    master.isnull().sum()[master.isnull().sum() > 0]
)

print("\nDuplicate order IDs:", master["order_id"].duplicated().sum())

print("\nDataset columns:")
print(list(master.columns))

print("\nSaved at:", output_file)