
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Load master dataset
df = pd.read_csv(
    "data/processed/delivery_prediction_master.csv"
)

# 2. Create year-month column
df["year_month"] = (
    df["purchase_year"].astype(int).astype(str)
    + "-"
    + df["purchase_month"].astype(int).astype(str).str.zfill(2)
)

# 3. Group by year and month
monthly = df.groupby("year_month").agg(
    total_orders=("order_id", "count"),
    late_orders=("is_late", "sum"),
    average_delay=("delivery_delay_days", "mean")
).reset_index()

# 4. Calculate late delivery percentage
monthly["late_percentage"] = (
    monthly["late_orders"] /
    monthly["total_orders"]
) * 100

# 5. Sort chronologically
monthly = monthly.sort_values("year_month")

# 6. Display results
print("\nYEAR-WISE MONTHLY DELIVERY ANALYSIS")
print("=" * 75)

print(monthly.to_string(index=False))

# 7. Save results
os.makedirs("reports", exist_ok=True)

monthly.to_csv(
    "reports/yearwise_monthly_delivery.csv",
    index=False
)

# 8. Plot monthly late delivery rates
plt.figure(figsize=(14, 6))

plt.plot(
    monthly["year_month"],
    monthly["late_percentage"],
    marker="o"
)

plt.title("Year-wise Monthly Late Delivery Trend")
plt.xlabel("Year-Month")
plt.ylabel("Late Delivery Percentage (%)")

plt.xticks(rotation=60)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "reports/yearwise_monthly_delivery.png",
    dpi=300
)

plt.show()

print("\nYEAR-WISE MONTHLY ANALYSIS COMPLETED!")
