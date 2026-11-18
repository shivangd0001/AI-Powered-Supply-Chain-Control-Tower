import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw")

# Load datasets
orders = pd.read_csv(DATA_PATH / "olist_orders_dataset.csv")
payments = pd.read_csv(DATA_PATH / "olist_order_payments_dataset.csv")
reviews = pd.read_csv(DATA_PATH / "olist_order_reviews_dataset.csv")
items = pd.read_csv(DATA_PATH / "olist_order_items_dataset.csv")
geo = pd.read_csv(DATA_PATH / "olist_geolocation_dataset.csv")

# 1. Order status analysis
print("\n1. ORDER STATUS ANALYSIS")

print(orders["order_status"].value_counts(dropna=False))

print("\nMissing timestamps by order status:")

timestamp_cols = [
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date"
]

print(
    orders.groupby("order_status")[timestamp_cols]
    .apply(lambda x: x.isnull().sum())
)

# 2. Multiple payments
print("\n2. PAYMENT ANALYSIS")

payment_count = payments.groupby("order_id").size()

print("Orders with multiple payment records:",
      (payment_count > 1).sum())

print("Maximum payments for one order:",
      payment_count.max())

# 3. Multiple reviews
print("\n3. REVIEW ANALYSIS")

review_count = reviews.groupby("order_id").size()

print("Orders with multiple review records:",
      (review_count > 1).sum())

print("Maximum reviews for one order:",
      review_count.max())

# 4. Order items
print("\n4. ORDER ITEM ANALYSIS")

item_count = items.groupby("order_id").size()

print("Orders with multiple products:",
      (item_count > 1).sum())

print("Maximum items in one order:",
      item_count.max())

# 5. Geolocation analysis
print("\n5. GEOLOCATION ANALYSIS")

print("Total records:", len(geo))

print("Exact duplicate records:",
      geo.duplicated().sum())

print("Unique ZIP prefixes:",
      geo["geolocation_zip_code_prefix"].nunique())

print("Unique ZIP prefixes with multiple coordinates:")

coordinates = geo.groupby(
    "geolocation_zip_code_prefix"
).size()

print((coordinates > 1).sum())

print("\nDIAGNOSTICS COMPLETED")