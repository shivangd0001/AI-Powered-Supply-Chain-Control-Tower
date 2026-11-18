import pandas as pd
import numpy as np
from pathlib import Path

print("Loading project data...")

BASE = Path(__file__).resolve().parent
REPORTS = BASE / "reports"

# Load datasets
demand = pd.read_csv(REPORTS / "demand_forecast.csv")
delivery = pd.read_csv(REPORTS / "delivery_risk_control_tower.csv")

# Clean category names
demand["category"] = demand["category"].astype(str).str.strip()

# =====================================================
# PART 1: DEMAND-BASED INVENTORY PLANNING
# =====================================================

print("\nGenerating inventory planning recommendations...")

# Simulation assumptions
LEAD_TIME_DAYS = 7
SAFETY_STOCK_PERCENT = 0.20

inventory = demand.groupby("category", as_index=False).agg(
    forecast_demand=("forecast_demand", "sum")
)

inventory["average_daily_demand"] = (
    inventory["forecast_demand"] / 90
)

inventory["lead_time_demand"] = (
    inventory["average_daily_demand"] * LEAD_TIME_DAYS
)

inventory["safety_stock"] = (
    inventory["lead_time_demand"] * SAFETY_STOCK_PERCENT
)

inventory["reorder_point"] = (
    inventory["lead_time_demand"] + inventory["safety_stock"]
)

inventory["recommended_stock_target"] = np.ceil(
    inventory["reorder_point"]
).astype(int)

inventory.to_csv(
    REPORTS / "inventory_planning_recommendations.csv",
    index=False
)

# =====================================================
# PART 2: DELIVERY RISK OPTIMIZATION
# =====================================================

print("Simulating delivery risk reduction scenarios...")

# Baseline expected late orders
baseline_risk = delivery["late_probability"].sum()

scenarios = [
    {
        "scenario": "Current Operations",
        "high_reduction": 0,
        "medium_reduction": 0
    },
    {
        "scenario": "Moderate Intervention",
        "high_reduction": 0.15,
        "medium_reduction": 0.10
    },
    {
        "scenario": "Aggressive Intervention",
        "high_reduction": 0.30,
        "medium_reduction": 0.20
    }
]

results = []

for scenario in scenarios:

    adjusted_probability = delivery["late_probability"].copy()

    high_mask = delivery["risk_category"].isin(
        ["High", "Critical"]
    )

    medium_mask = delivery["risk_category"] == "Medium"

    adjusted_probability.loc[high_mask] *= (
        1 - scenario["high_reduction"]
    )

    adjusted_probability.loc[medium_mask] *= (
        1 - scenario["medium_reduction"]
    )

    expected_late = adjusted_probability.sum()

    results.append({
        "scenario": scenario["scenario"],
        "baseline_expected_late_orders": round(baseline_risk, 2),
        "optimized_expected_late_orders": round(expected_late, 2),
        "expected_late_orders_reduced": round(
            baseline_risk - expected_late, 2
        ),
        "expected_risk_reduction_percent": round(
            (baseline_risk - expected_late)
            / baseline_risk * 100, 2
        )
    })

scenario_results = pd.DataFrame(results)

scenario_results.to_csv(
    REPORTS / "optimization_scenario_results.csv",
    index=False
)

# =====================================================
# PART 3: CATEGORY-LEVEL PRIORITY
# =====================================================

# =====================================================
# PART 3: CATEGORY-LEVEL PRIORITY
# =====================================================

print("Calculating category-level priorities...")

translation = pd.read_csv(
    BASE / "data" / "processed" /
    "product_category_name_translation.csv"
)

delivery = delivery.merge(
    translation,
    left_on="product_category",
    right_on="product_category_name",
    how="left"
)

delivery["category"] = delivery[
    "product_category_name_english"
].fillna(delivery["product_category"])

category_risk = delivery.groupby(
    "category", as_index=False
).agg(
    total_orders=("order_id", "count"),
    average_late_probability=("late_probability", "mean"),
    high_priority_orders=(
        "risk_category",
        lambda x: x.isin(["High", "Critical"]).sum()
    )
)

category_risk = category_risk.merge(
    inventory,
    on="category",
    how="outer"
)

category_risk.to_csv(
    REPORTS / "category_supply_chain_priority.csv",
    index=False
)
# =====================================================
# FINAL RESULTS
# =====================================================

print("\n========== OPTIMIZATION RESULTS ==========")

print("\nInventory Planning:")
print(inventory.head(10).to_string(index=False))

print("\nDelivery Risk Scenarios:")
print(scenario_results.to_string(index=False))

print("\nTop Supply Chain Priority Categories:")
print(
    category_risk.sort_values(
        "average_late_probability",
        ascending=False
    ).head(10).to_string(index=False)
)

print("\nAll optimization files saved successfully:")
print("1. inventory_planning_recommendations.csv")
print("2. optimization_scenario_results.csv")
print("3. category_supply_chain_priority.csv")