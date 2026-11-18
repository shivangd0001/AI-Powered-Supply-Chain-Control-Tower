import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# File locations
DATA_PATH = Path("data/processed")
REPORT_PATH = Path("reports")

REPORT_PATH.mkdir(parents=True, exist_ok=True)

# Load datasets
orders = pd.read_csv(DATA_PATH / "olist_orders_dataset.csv")
items = pd.read_csv(DATA_PATH / "olist_order_items_dataset.csv")
products = pd.read_csv(DATA_PATH / "olist_products_dataset.csv")
translation = pd.read_csv(
    DATA_PATH / "product_category_name_translation.csv"
)

# Convert purchase timestamp
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"]
)

# Keep delivered orders for realised sales analysis
delivered_orders = orders[
    orders["order_status"] == "delivered"
].copy()

# Add monthly period
delivered_orders["month"] = (
    delivered_orders["order_purchase_timestamp"]
    .dt.to_period("M").astype(str)
)

# Merge orders and items
sales = delivered_orders[
    ["order_id", "month"]
].merge(items, on="order_id", how="inner")

# Monthly sales analysis
monthly_sales = sales.groupby("month").agg(
    total_orders=("order_id", "nunique"),
    total_revenue=("price", "sum"),
    total_freight=("freight_value", "sum")
).reset_index()

monthly_sales["average_order_value"] = (
    monthly_sales["total_revenue"] /
    monthly_sales["total_orders"]
)

print("\nMONTHLY SALES ANALYSIS")
print(monthly_sales)

# Save monthly results
monthly_sales.to_csv(
    REPORT_PATH / "monthly_sales_analysis.csv",
    index=False
)

# Monthly revenue chart
plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["month"],
    monthly_sales["total_revenue"],
    marker="o"
)

plt.title("Monthly Merchandise Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(REPORT_PATH / "monthly_revenue.png")
plt.show()

# Product category analysis
sales = sales.merge(
    products[["product_id", "product_category_name"]],
    on="product_id",
    how="left"
)

sales = sales.merge(
    translation,
    on="product_category_name",
    how="left"
)

sales["category"] = sales[
    "product_category_name_english"
].fillna("Unknown")

category_sales = sales.groupby("category").agg(
    total_revenue=("price", "sum"),
    units_sold=("order_item_id", "count")
).reset_index()

category_sales = category_sales.sort_values(
    "total_revenue", ascending=False
)

print("\nTOP 10 PRODUCT CATEGORIES")
print(category_sales.head(10))

category_sales.to_csv(
    REPORT_PATH / "category_sales_analysis.csv",
    index=False
)

# Top 10 category chart
top_categories = category_sales.head(10)

plt.figure(figsize=(12, 6))

plt.barh(
    top_categories["category"],
    top_categories["total_revenue"]
)

plt.title("Top 10 Product Categories by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Category")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(REPORT_PATH / "top_categories.png")
plt.show()

print("\nSALES ANALYSIS COMPLETED")