# AI-Powered Predictive E-Commerce Supply Chain Control Tower

## 1. Project Overview

The **AI-Powered Predictive E-Commerce Supply Chain Control Tower** is a machine-learning-based analytics project designed to improve supply chain visibility and support logistics decision-making.

It combines delivery risk prediction, demand forecasting, inventory planning, and optimization scenario analysis into one interactive Streamlit dashboard.

The project uses historical Brazilian e-commerce data to identify delivery risks, forecast product demand, and generate simulated inventory and operational recommendations.

## 2. Business Problem

E-commerce companies face several supply chain challenges:

- Unexpected delivery delays.
- Poor demand visibility.
- Inventory planning difficulties.
- Customer dissatisfaction.
- Inefficient logistics decisions.

This project uses historical data and machine learning to help identify potential risks and support better operational decisions.

## 3. Project Objectives

1. Predict the probability of late deliveries.
2. Identify high-risk customer orders.
3. Forecast future product category demand.
4. Generate inventory planning recommendations.
5. Compare different delivery risk intervention scenarios.
6. Build an interactive supply chain control tower.

## 4. Dataset

**Dataset:** Brazilian E-Commerce Public Dataset by Olist.

The dataset contains approximately 100,000 orders from 2016–2018.

It includes information about:

- Orders and order items.
- Customers and sellers.
- Products and categories.
- Payments and reviews.
- Delivery timestamps.
- Customer and seller locations.

The dataset is publicly available on Kaggle.

## 5. Technologies Used

| Technology | Application |
|---|---|
| Python | Programming |
| Pandas | Data processing |
| NumPy | Numerical analysis |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data analysis |
| Scikit-learn | Machine learning |
| Plotly | Interactive visualizations |
| Streamlit | Dashboard development |

## 6. Project Workflow

### Phase 1: Data Understanding and Preparation
- Dataset inspection.
- Data quality analysis.
- Missing value and duplicate analysis.
- Relationship and data integration checks.
- Master dataset creation.

### Phase 2: Exploratory Data Analysis
- Sales and revenue analysis.
- Delivery performance analysis.
- Customer satisfaction analysis.
- Monthly and yearly delivery trends.
- Late delivery patterns.

### Phase 3: Delivery Risk Prediction

Machine learning models were trained and evaluated using chronological data splits.

The final selected model was **HistGradientBoostingClassifier**.

Test performance:

| Metric | Result |
|---|---:|
| Accuracy | 63.51% |
| Precision | 10.48% |
| Recall | 67.05% |
| F1 Score | 18.12% |
| ROC-AUC | 0.703 |
| PR-AUC | 0.129 |

The model helps identify a substantial proportion of late deliveries, although its precision is low. Predictions should therefore be treated as risk signals requiring operational review, not confirmed delays.

### Phase 4: Demand Forecasting

A Random Forest regression model was developed to forecast monthly product category demand.

Results:

- 72 product categories.
- 1,800 forecast records.
- MAE: 50.31.
- RMSE: 123.56.

The demand measure uses product order-item counts as a proxy for demand.

### Phase 5: Inventory Planning

A simulated inventory planning framework was developed using forecasted demand.

Planning assumptions:
- 90-day planning horizon.
- 7-day assumed lead time.
- 20% safety stock.

The framework generates category-level inventory planning recommendations.

These are simulated recommendations because actual warehouse stock and supplier lead-time records were not available in the dataset.

### Phase 6: Supply Chain Optimization

Three scenarios were compared:

- Baseline scenario.
- Moderate intervention.
- Aggressive intervention.

Simulated expected delivery risk reduction:

| Scenario | Expected Risk Reduction |
|---|---:|
| Baseline | 0% |
| Moderate Intervention | 12.3% |
| Aggressive Intervention | 24.6% |

These percentages represent scenario assumptions, not measured real-world improvements.

## 7. Interactive Dashboard

The Streamlit dashboard provides:

1. Executive Overview.
2. AI Delivery Risk Analysis.
3. Monthly Delivery Performance.
4. High-Priority Order Alerts.
5. AI Demand Forecasting.
6. Inventory Planning Recommendations.
7. Delivery Optimization Scenarios.
8. Category-Level Supply Chain Priorities.
9. CSV Report Downloads.

## 8. Project Structure

```text
Ecommerce_Supply_Chain_AI/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── models/
│
├── reports/
│   ├── delivery_predictions.csv
│   ├── delivery_risk_control_tower.csv
│   ├── demand_forecast.csv
│   ├── inventory_planning_recommendations.csv
│   ├── optimization_scenario_results.csv
│   └── category_supply_chain_priority.csv
│
├── dashboard/
│   └── app.py
│
├── 01_dataset_inspection.py
├── 02_relationship_analysis.py
├── 03_data_quality_analysis.py
├── 04_detailed_data_diagnostics.py
├── 05_initial_data_preparation.py
├── 06_verify_processed_data.py
├── 07_sales_analysis.py
├── 08_delivery_logistics_analysis.py
├── 09_customer_satisfaction_analysis.py
├── 10_build_master_dataset.py
├── 11_master_dataset_audit.py
├── 12_prepare_ml_dataset.py
├── 12_train_delivery_models.py
├── 13_feature_importance.py
├── 14_monthly_delivery_risk.py
├── 15_yearwise_delivery_analysis.py
├── 16_temporal_model_validation.py
├── 17_delivery_risk_threshold.py
├── 18_proper_temporal_validation.py
├── 19_advanced_model_comparison.py
├── 20_save_best_model.py
├── 21_delivery_risk_classification.py
├── 22_demand_forecasting.py
├── 23_supply_chain_optimization.py
│
└── README.md
```

## 9. Installation and Execution

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the project folder

```bash
cd Ecommerce_Supply_Chain_AI
```

### Step 3: Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly streamlit
```

### Step 4: Run the dashboard

```bash
streamlit run dashboard/app.py
```

## 10. Business Value

This project demonstrates how data analytics and machine learning can support supply chain operations.

Potential business applications include:

- Early identification of delivery risks.
- Better demand visibility.
- Data-driven inventory planning.
- Prioritization of operational attention.
- Improved logistics decision support.
- Better supply chain monitoring.

## 11. Limitations

- The dataset represents historical transactions from 2016–2018.
- Delivery predictions are based on historical patterns, not a live logistics feed.
- Model precision is limited and requires human review.
- Demand forecasts are category-level estimates.
- Actual warehouse inventory and supplier data are unavailable.
- Inventory planning uses assumed lead time and safety stock.
- Optimization improvements are simulated rather than experimentally measured.

## 12. Conclusion

The project integrates machine learning, demand forecasting, inventory planning, and operational analytics into a unified supply chain control tower.

It demonstrates the practical application of data-driven decision-making to e-commerce logistics and supply chain management.

**Project Domain:** Supply Chain Management | Operations Research | Machine Learning | Business Analytics