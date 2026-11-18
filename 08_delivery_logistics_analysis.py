import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/processed")
REPORT_PATH = Path("reports")

REPORT_PATH.mkdir(parents=True, exist_ok=True)

# Load datasets
orders = pd.read_csv(DATA_PATH / "olist_orders_dataset.csv")
items = pd.read_csv(DATA_PATH / "olist_order_items_dataset.csv")

# Convert timestamps
date_columns = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_columns:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

# Keep delivered orders with valid delivery dates
delivered = orders[
    (orders["order_status"] == "delivered") &
    orders["order_delivered_customer_date"].notna() &
    orders["order_estimated_delivery_date"].notna()
].copy()

# Calculate actual delivery time
delivered["actual_delivery_days"] = (
    delivered["order_delivered_customer_date"] -
    delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400

# Calculate estimated delivery time
delivered["estimated_delivery_days"] = (
    delivered["order_estimated_delivery_date"] -
    delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400

# Calculate delivery delay
delivered["delivery_delay_days"] = (
    delivered["order_delivered_customer_date"] -
    delivered["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

# Late delivery flag
delivered["is_late"] = (
    delivered["delivery_delay_days"] > 0
)

# Overall delivery performance
print("\nOVERALL DELIVERY PERFORMANCE")

print("Total eligible delivered orders:", len(delivered))

print(
    "Average delivery time:",
    round(delivered["actual_delivery_days"].mean(), 2),
    "days"
)

print(
    "Average estimated delivery time:",
    round(delivered["estimated_delivery_days"].mean(), 2),
    "days"
)

print(
    "Late deliveries:",
    delivered["is_late"].sum()
)

print(
    "Late delivery percentage:",
    round(delivered["is_late"].mean() * 100, 2),
    "%"
)

# Monthly delivery performance
delivered["month"] = (
    delivered["order_purchase_timestamp"]
    .dt.to_period("M").astype(str)
)

monthly_delivery = delivered.groupby("month").agg(
    total_orders=("order_id", "nunique"),
    average_delivery_days=("actual_delivery_days", "mean"),
    late_orders=("is_late", "sum"),
    late_percentage=("is_late", "mean")
).reset_index()

monthly_delivery["late_percentage"] *= 100

print("\nMONTHLY DELIVERY PERFORMANCE")
print(monthly_delivery)

monthly_delivery.to_csv(
    REPORT_PATH / "monthly_delivery_analysis.csv",
    index=False
)

# Monthly late delivery chart
plt.figure(figsize=(12, 6))

plt.plot(
    monthly_delivery["month"],
    monthly_delivery["late_percentage"],
    marker="o"
)

plt.title("Monthly Late Delivery Percentage")
plt.xlabel("Month")
plt.ylabel("Late Deliveries (%)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(REPORT_PATH / "monthly_late_delivery.png")
plt.show()

# Seller performance
seller_items = items[
    ["order_id", "seller_id", "freight_value"]
].drop_duplicates()

seller_orders = seller_items.merge(
    delivered[
        [
            "order_id",
            "actual_delivery_days",
            "is_late"
        ]
    ],
    on="order_id",
    how="inner"
)

seller_performance = seller_orders.groupby("seller_id").agg(
    total_orders=("order_id", "nunique"),
    average_delivery_days=("actual_delivery_days", "mean"),
    late_percentage=("is_late", "mean"),
    total_freight=("freight_value", "sum")
).reset_index()

seller_performance["late_percentage"] *= 100

# Consider sellers with at least 20 orders
seller_performance = seller_performance[
    seller_performance["total_orders"] >= 20
]

seller_performance = seller_performance.sort_values(
    "late_percentage",
    ascending=False
)

print("\nTOP 10 SELLERS BY LATE DELIVERY PERCENTAGE")
print(seller_performance.head(10))

seller_performance.to_csv(
    REPORT_PATH / "seller_delivery_performance.csv",
    index=False
)

print("\nDELIVERY ANALYSIS COMPLETED")