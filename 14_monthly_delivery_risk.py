
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load master dataset
df = pd.read_csv(
    "data/processed/delivery_prediction_master.csv"
)

# Create monthly analysis
monthly = df.groupby("purchase_month").agg(
    total_orders=("order_id", "count"),
    late_orders=("is_late", "sum"),
    average_delay=("delivery_delay_days", "mean")
).reset_index()

# Calculate late delivery percentage
monthly["late_percentage"] = (
    monthly["late_orders"] /
    monthly["total_orders"]
) * 100

monthly = monthly.sort_values("purchase_month")

# Display results
print("\nMONTHLY DELIVERY RISK ANALYSIS")
print("=" * 65)

print(monthly.to_string(index=False))

# Identify highest-risk month
highest_risk = monthly.loc[
    monthly["late_percentage"].idxmax()
]

print("\nHIGHEST DELIVERY RISK MONTH:")
print("Month:", int(highest_risk["purchase_month"]))
print("Late Delivery Rate:",
      round(highest_risk["late_percentage"], 2), "%")

# Save results
os.makedirs("reports", exist_ok=True)

monthly.to_csv(
    "reports/monthly_delivery_risk.csv",
    index=False
)

# Plot late delivery percentage
plt.figure(figsize=(10, 5))

plt.bar(
    monthly["purchase_month"],
    monthly["late_percentage"]
)

plt.title("Monthly Late Delivery Percentage")
plt.xlabel("Purchase Month")
plt.ylabel("Late Deliveries (%)")
plt.xticks(range(1, 13))
plt.tight_layout()

plt.savefig(
    "reports/monthly_delivery_risk.png",
    dpi=300
)

plt.show()

print("\nMONTHLY DELIVERY RISK ANALYSIS COMPLETED!")