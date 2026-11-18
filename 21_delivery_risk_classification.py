import os
import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD PREDICTIONS
# ==========================================

print("Loading AI delivery predictions...")

predictions = pd.read_csv(
    "reports/delivery_predictions.csv"
)

master = pd.read_csv(
    "data/processed/delivery_prediction_master.csv"
)

# ==========================================
# 2. ADD ORDER INFORMATION
# ==========================================

columns_to_add = [
    "order_id",
    "customer_state",
    "seller_state",
    "total_price",
    "total_freight",
    "total_items",
    "product_category",
    "promised_delivery_days",
    "purchase_month"
]

master_subset = master[columns_to_add].copy()

risk_df = predictions.merge(
    master_subset,
    on="order_id",
    how="left"
)

# ==========================================
# 3. CLASSIFY DELIVERY RISK
# ==========================================

def classify_risk(probability):

    if probability < 0.20:
        return "Low"

    elif probability < 0.40:
        return "Medium"

    elif probability < 0.60:
        return "High"

    else:
        return "Critical"


risk_df["risk_category"] = (
    risk_df["late_probability"].apply(classify_risk)
)

# ==========================================
# 4. RECOMMEND BUSINESS ACTIONS
# ==========================================

def recommend_action(risk):

    actions = {
        "Low": "Continue normal shipment monitoring",

        "Medium": "Monitor shipment and track delivery progress",

        "High": "Investigate shipment status and contact logistics partner",

        "Critical": "Prioritize shipment investigation and customer communication"
    }

    return actions[risk]


risk_df["recommended_action"] = (
    risk_df["risk_category"].apply(recommend_action)
)

# ==========================================
# 5. ASSIGN PRIORITY SCORE
# ==========================================

risk_df["priority_score"] = (
    risk_df["late_probability"] * 100
).round(2)

risk_df["priority_rank"] = (
    risk_df["priority_score"]
    .rank(method="first", ascending=False)
    .astype(int)
)

# ==========================================
# 6. RISK SUMMARY
# ==========================================

print("\nDELIVERY RISK SUMMARY")
print("=" * 60)

summary = risk_df.groupby(
    "risk_category"
).agg(
    total_orders=("order_id", "count"),
    average_risk=("late_probability", "mean"),
    average_order_value=("total_price", "mean")
).round(3)

summary["percentage_of_orders"] = (
    summary["total_orders"] / len(risk_df) * 100
).round(2)

print(summary)

# ==========================================
# 7. ACTUAL PERFORMANCE BY RISK CATEGORY
# ==========================================

print("\nACTUAL LATE DELIVERY RATE BY RISK CATEGORY")
print("=" * 60)

performance = risk_df.groupby(
    "risk_category"
).agg(
    total_orders=("order_id", "count"),
    actual_late_orders=("actual_late", "sum"),
    actual_late_rate=("actual_late", "mean")
)

performance["actual_late_rate"] *= 100

print(performance.round(2))

# ==========================================
# 8. HIGH PRIORITY ORDERS
# ==========================================

high_priority = risk_df[
    risk_df["risk_category"].isin(["High", "Critical"])
].sort_values(
    by="late_probability",
    ascending=False
)

print("\nTOP 20 HIGH-PRIORITY ORDERS")
print("=" * 60)

print(
    high_priority[
        [
            "order_id",
            "late_probability",
            "risk_category",
            "customer_state",
            "seller_state",
            "recommended_action"
        ]
    ].head(20).to_string(index=False)
)

# ==========================================
# 9. SAVE OUTPUTS
# ==========================================

os.makedirs("reports", exist_ok=True)

risk_df.to_csv(
    "reports/delivery_risk_control_tower.csv",
    index=False
)

summary.to_csv(
    "reports/delivery_risk_summary.csv"
)

performance.to_csv(
    "reports/risk_category_performance.csv"
)

high_priority.to_csv(
    "reports/high_priority_orders.csv",
    index=False
)

print("\nDELIVERY RISK CLASSIFICATION COMPLETED!")

print("\nFiles generated:")
print("1. delivery_risk_control_tower.csv")
print("2. delivery_risk_summary.csv")
print("3. risk_category_performance.csv")
print("4. high_priority_orders.csv")