import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/processed")
REPORT_PATH = Path("reports")

REPORT_PATH.mkdir(parents=True, exist_ok=True)

# Load datasets
orders = pd.read_csv(DATA_PATH / "olist_orders_dataset.csv")
reviews = pd.read_csv(DATA_PATH / "olist_order_reviews_dataset.csv")
items = pd.read_csv(DATA_PATH / "olist_order_items_dataset.csv")

# Convert dates
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"], errors="coerce"
)

orders["order_delivered_customer_date"] = pd.to_datetime(
    orders["order_delivered_customer_date"], errors="coerce"
)

orders["order_estimated_delivery_date"] = pd.to_datetime(
    orders["order_estimated_delivery_date"], errors="coerce"
)

# 1. Review score distribution
print("\n1. CUSTOMER REVIEW ANALYSIS")

print(reviews["review_score"].value_counts().sort_index())

print("\nAverage review score:",
      round(reviews["review_score"].mean(), 2))

# Aggregate multiple reviews per order
order_reviews = reviews.groupby("order_id").agg(
    average_review_score=("review_score", "mean"),
    review_count=("review_score", "count")
).reset_index()

# 2. Prepare delivered orders
delivered = orders[
    (orders["order_status"] == "delivered") &
    orders["order_delivered_customer_date"].notna() &
    orders["order_estimated_delivery_date"].notna()
].copy()

delivered["delivery_delay_days"] = (
    delivered["order_delivered_customer_date"] -
    delivered["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

delivered["is_late"] = (
    delivered["delivery_delay_days"] > 0
)

# 3. Aggregate freight at order level
order_freight = items.groupby("order_id").agg(
    total_freight=("freight_value", "sum"),
    total_item_value=("price", "sum"),
    total_items=("order_item_id", "count")
).reset_index()

# 4. Combine order, review and freight information
analysis = delivered.merge(
    order_reviews,
    on="order_id",
    how="left"
)

analysis = analysis.merge(
    order_freight,
    on="order_id",
    how="left"
)

analysis["month"] = (
    analysis["order_purchase_timestamp"]
    .dt.to_period("M").astype(str)
)

print("\n2. LATE DELIVERY VS CUSTOMER SATISFACTION")

late_analysis = analysis.groupby("is_late").agg(
    total_orders=("order_id", "nunique"),
    average_review_score=("average_review_score", "mean")
).reset_index()

print(late_analysis)

# 5. Monthly customer satisfaction
monthly_reviews = analysis.groupby("month").agg(
    total_orders=("order_id", "nunique"),
    average_review_score=("average_review_score", "mean")
).reset_index()

print("\n3. MONTHLY CUSTOMER SATISFACTION")
print(monthly_reviews)

monthly_reviews.to_csv(
    REPORT_PATH / "monthly_customer_satisfaction.csv",
    index=False
)

# 6. Freight analysis
print("\n4. FREIGHT ANALYSIS")

print("Average freight per order:",
      round(analysis["total_freight"].mean(), 2))

print("Average item value per order:",
      round(analysis["total_item_value"].mean(), 2))

print("Total freight:",
      round(analysis["total_freight"].sum(), 2))

# Save combined analysis
analysis.to_csv(
    REPORT_PATH / "customer_satisfaction_analysis.csv",
    index=False
)

# 7. Review score distribution chart
plt.figure(figsize=(8, 5))

reviews["review_score"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Customer Review Score Distribution")
plt.xlabel("Review Score")
plt.ylabel("Number of Reviews")
plt.tight_layout()

plt.savefig(REPORT_PATH / "review_score_distribution.png")
plt.show()

# 8. Late delivery vs review score chart
plt.figure(figsize=(8, 5))

late_analysis["delivery_type"] = late_analysis[
    "is_late"
].map({
    True: "Late",
    False: "On Time"
})

plt.bar(
    late_analysis["delivery_type"],
    late_analysis["average_review_score"]
)

plt.title("Delivery Performance vs Customer Satisfaction")
plt.xlabel("Delivery Status")
plt.ylabel("Average Review Score")
plt.ylim(0, 5)
plt.tight_layout()

plt.savefig(REPORT_PATH / "delivery_vs_satisfaction.png")
plt.show()

# 9. Monthly satisfaction chart
plt.figure(figsize=(12, 5))

plt.plot(
    monthly_reviews["month"],
    monthly_reviews["average_review_score"],
    marker="o"
)

plt.title("Monthly Customer Satisfaction Trend")
plt.xlabel("Month")
plt.ylabel("Average Review Score")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(REPORT_PATH / "monthly_customer_satisfaction.png")
plt.show()

print("\nCUSTOMER SATISFACTION ANALYSIS COMPLETED")