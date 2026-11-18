import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Supply Chain Control Tower",
    page_icon="📦",
    layout="wide"
)

st.title("📦 AI-Powered Supply Chain Control Tower")

st.caption(
    "Predictive E-Commerce Logistics | Delivery Risk Intelligence "
    "| Demand Forecasting | Inventory Optimization"
)

# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"


# ==========================================
# DATA LOADING
# ==========================================

@st.cache_data
def load_csv(filename):
    path = REPORTS / filename

    if not path.exists():
        return None

    return pd.read_csv(path)


@st.cache_data
def load_delivery_data():

    data = load_csv("delivery_risk_control_tower.csv")

    if data is None:
        raise FileNotFoundError(
            "delivery_risk_control_tower.csv not found."
        )

    required_columns = [
        "order_id",
        "customer_state",
        "seller_state",
        "late_probability",
        "risk_category",
        "actual_late",
        "purchase_month",
        "recommended_action"
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for col in [
        "late_probability",
        "purchase_month",
        "actual_late"
    ]:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    return data.dropna(
        subset=[
            "late_probability",
            "actual_late",
            "purchase_month"
        ]
    )


try:
    df = load_delivery_data()

except Exception as error:
    st.error("Unable to load delivery dashboard data.")
    st.exception(error)
    st.stop()


# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("Dashboard Filters")

risk_order = [
    "Low",
    "Medium",
    "High",
    "Critical"
]

selected_risks = st.sidebar.multiselect(
    "Select Risk Category",
    risk_order,
    default=risk_order
)

states = sorted(
    df["customer_state"].dropna().unique()
)

selected_states = st.sidebar.multiselect(
    "Select Customer State",
    states,
    default=states
)

filtered_df = df[
    df["risk_category"].isin(selected_risks)
    & df["customer_state"].isin(selected_states)
].copy()

if filtered_df.empty:
    st.warning("No orders found for the selected filters.")
    st.stop()


# ==========================================
# EXECUTIVE OVERVIEW
# ==========================================

st.header("Executive Overview")

total_orders = len(filtered_df)

actual_late = int(filtered_df["actual_late"].sum())

actual_late_rate = actual_late / total_orders * 100

high_risk = filtered_df[
    filtered_df["risk_category"].isin(
        ["High", "Critical"]
    )
].shape[0]

average_risk = filtered_df["late_probability"].mean() * 100

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Orders", f"{total_orders:,}")
c2.metric("Actual Late Orders", f"{actual_late:,}")
c3.metric("Actual Late Rate", f"{actual_late_rate:.2f}%")
c4.metric("High/Critical Risk Orders", f"{high_risk:,}")

st.metric(
    "Average Predicted Late Risk",
    f"{average_risk:.2f}%"
)

st.divider()


# ==========================================
# AI DELIVERY RISK ANALYSIS
# ==========================================

st.header("AI Delivery Risk Analysis")

col1, col2 = st.columns(2)

risk_summary = (
    filtered_df["risk_category"]
    .value_counts()
    .reindex(risk_order, fill_value=0)
    .reset_index()
)

risk_summary.columns = ["Risk Category", "Orders"]

with col1:
    fig = px.pie(
        risk_summary,
        names="Risk Category",
        values="Orders",
        title="Delivery Risk Distribution",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)


performance = (
    filtered_df.groupby("risk_category")
    .agg(
        Total_Orders=("order_id", "count"),
        Actual_Late_Rate=("actual_late", "mean"),
        Average_Predicted_Risk=("late_probability", "mean")
    )
    .reindex(risk_order)
    .reset_index()
)

performance["Actual_Late_Rate"] *= 100
performance["Average_Predicted_Risk"] *= 100

with col2:
    fig = px.bar(
        performance,
        x="risk_category",
        y="Actual_Late_Rate",
        title="Actual Late Delivery Rate by Risk Category",
        labels={
            "risk_category": "Risk Category",
            "Actual_Late_Rate": "Late Delivery (%)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# MONTHLY DELIVERY ANALYSIS
# ==========================================

st.divider()

st.header("Monthly Delivery Performance")

monthly = (
    filtered_df.groupby("purchase_month")
    .agg(
        Total_Orders=("order_id", "count"),
        Late_Orders=("actual_late", "sum")
    )
    .reset_index()
    .sort_values("purchase_month")
)

monthly["Late_Percentage"] = (
    monthly["Late_Orders"]
    / monthly["Total_Orders"] * 100
)

fig = px.line(
    monthly,
    x="purchase_month",
    y="Late_Percentage",
    markers=True,
    title="Monthly Actual Late Delivery Trend"
)

fig.update_layout(
    xaxis_title="Purchase Month",
    yaxis_title="Late Delivery (%)",
    xaxis=dict(tickmode="linear", dtick=1)
)

st.plotly_chart(fig, use_container_width=True)


# ==========================================
# HIGH PRIORITY ORDER ALERTS
# ==========================================

st.divider()

st.header("High-Priority Order Alerts")

priority_df = filtered_df[
    filtered_df["risk_category"].isin(
        ["High", "Critical"]
    )
].copy()

priority_df = priority_df.sort_values(
    "late_probability",
    ascending=False
)

display_columns = [
    "order_id",
    "customer_state",
    "seller_state",
    "late_probability",
    "risk_category",
    "recommended_action"
]

if not priority_df.empty:

    priority_display = priority_df[
        display_columns
    ].head(100).copy()

    priority_display["late_probability"] = (
        priority_display["late_probability"] * 100
    ).round(2)

    priority_display = priority_display.rename(
        columns={
            "late_probability": "Predicted Late Risk (%)"
        }
    )

    st.dataframe(
        priority_display,
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("No High or Critical risk orders in the selected filters.")


# ==========================================
# DEMAND FORECASTING
# ==========================================

st.divider()

st.header("📈 AI Demand Forecasting")

forecast_df = load_csv("demand_forecast.csv")

if forecast_df is not None and not forecast_df.empty:

    st.caption(
        "Machine-learning demand forecast by product category."
    )

    st.subheader("Forecast Dataset")

    st.dataframe(
        forecast_df.head(100),
        use_container_width=True,
        hide_index=True
    )

    numeric_cols = forecast_df.select_dtypes(
        include="number"
    ).columns.tolist()

    category_col = next(
        (
            col for col in forecast_df.columns
            if col.lower() in [
                "category",
                "product_category",
                "product_category_name"
            ]
        ),
        None
    )

    demand_col = next(
        (
            col for col in numeric_cols
            if "forecast" in col.lower()
            or "demand" in col.lower()
        ),
        None
    )

    month_col = next(
        (
            col for col in forecast_df.columns
            if "month" in col.lower()
            or "date" in col.lower()
        ),
        None
    )

    if category_col and demand_col:

        forecast_chart = forecast_df.copy()

        if month_col:
            fig = px.line(
                forecast_chart,
                x=month_col,
                y=demand_col,
                color=category_col,
                markers=True,
                title="Forecasted Demand by Product Category"
            )
        else:
            top_categories = (
                forecast_chart.groupby(category_col)[demand_col]
                .sum()
                .nlargest(15)
                .reset_index()
            )

            fig = px.bar(
                top_categories,
                x=category_col,
                y=demand_col,
                title="Top 15 Categories by Forecast Demand"
            )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            "Forecast chart could not be generated automatically. "
            "Review the forecast table above."
        )

    st.download_button(
        "Download Demand Forecast",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="demand_forecast.csv",
        mime="text/csv"
    )

else:
    st.warning("Demand forecast file not found.")


# ==========================================
# INVENTORY PLANNING
# ==========================================

st.divider()

st.header("📦 Inventory Planning Recommendations")

inventory_df = load_csv(
    "inventory_planning_recommendations.csv"
)

if inventory_df is not None and not inventory_df.empty:

    st.caption(
        "Simulated inventory recommendations based on forecast demand, "
        "assumed lead time and safety stock."
    )

    st.dataframe(
        inventory_df,
        use_container_width=True,
        hide_index=True
    )

    numeric_cols = inventory_df.select_dtypes(
        include="number"
    ).columns.tolist()

    category_col = next(
        (
            col for col in inventory_df.columns
            if "category" in col.lower()
        ),
        None
    )

    reorder_col = next(
        (
            col for col in numeric_cols
            if "reorder" in col.lower()
            or "stock" in col.lower()
        ),
        None
    )

    if category_col and reorder_col:

        chart_data = (
            inventory_df
            .groupby(category_col)[reorder_col]
            .sum()
            .nlargest(15)
            .reset_index()
        )

        fig = px.bar(
            chart_data,
            x=category_col,
            y=reorder_col,
            title="Top 15 Categories by Inventory Recommendation"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Inventory Recommendations",
        data=inventory_df.to_csv(index=False).encode("utf-8"),
        file_name="inventory_planning_recommendations.csv",
        mime="text/csv"
    )

else:
    st.warning("Inventory planning file not found.")


# ==========================================
# DELIVERY OPTIMIZATION SCENARIOS
# ==========================================

st.divider()

st.header("🎯 Delivery Optimization Scenarios")

scenario_df = load_csv(
    "optimization_scenario_results.csv"
)

if scenario_df is not None and not scenario_df.empty:

    st.caption(
        "Comparison of simulated delivery-risk intervention scenarios."
    )

    st.dataframe(
        scenario_df,
        use_container_width=True,
        hide_index=True
    )

    numeric_cols = scenario_df.select_dtypes(
        include="number"
    ).columns.tolist()

    scenario_col = next(
        (
            col for col in scenario_df.columns
            if "scenario" in col.lower()
        ),
        None
    )

    reduction_col = next(
        (
            col for col in numeric_cols
            if "reduction" in col.lower()
            or "improvement" in col.lower()
        ),
        None
    )

    if scenario_col and reduction_col:

        fig = px.bar(
            scenario_df,
            x=scenario_col,
            y=reduction_col,
            text_auto=".2f",
            title="Expected Risk Reduction by Scenario"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Optimization Scenarios",
        data=scenario_df.to_csv(index=False).encode("utf-8"),
        file_name="optimization_scenario_results.csv",
        mime="text/csv"
    )

else:
    st.warning("Optimization scenario file not found.")


# ==========================================
# CATEGORY SUPPLY CHAIN PRIORITIES
# ==========================================

st.divider()

st.header("🏭 Category-Level Supply Chain Priorities")

category_df = load_csv(
    "category_supply_chain_priority.csv"
)

if category_df is not None and not category_df.empty:

    st.caption(
        "Product categories ranked using the available supply-chain "
        "risk indicators."
    )

    st.dataframe(
        category_df,
        use_container_width=True,
        hide_index=True
    )

    numeric_cols = category_df.select_dtypes(
        include="number"
    ).columns.tolist()

    category_col = next(
        (
            col for col in category_df.columns
            if "category" in col.lower()
        ),
        None
    )

    score_col = next(
        (
            col for col in numeric_cols
            if "risk" in col.lower()
            or "priority" in col.lower()
            or "score" in col.lower()
        ),
        None
    )

    if category_col and score_col:

        chart_data = category_df.nlargest(
            15, score_col
        )

        fig = px.bar(
            chart_data,
            x=category_col,
            y=score_col,
            title="Top 15 Categories by Supply Chain Risk/Priority"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Category Priority Report",
        data=category_df.to_csv(index=False).encode("utf-8"),
        file_name="category_supply_chain_priority.csv",
        mime="text/csv"
    )

else:
    st.warning("Category priority file not found.")


# ==========================================
# DOWNLOAD DELIVERY REPORT
# ==========================================

st.divider()

st.header("Download Delivery Risk Report")

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Delivery Risk Report",
    data=csv_data,
    file_name="delivery_risk_report.csv",
    mime="text/csv"
)


# ==========================================
# FOOTER
# ==========================================

st.success(
    "AI Supply Chain Control Tower loaded successfully!"
)

st.caption(
    "Historical dataset-based predictions and simulated inventory "
    "and optimization scenarios. This is not a live logistics feed. "
    "Inventory lead time, safety stock and intervention benefits "
    "are assumption-based."
)