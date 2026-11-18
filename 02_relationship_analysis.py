import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw")

# Load datasets
orders = pd.read_csv(DATA_PATH / "olist_orders_dataset.csv")
customers = pd.read_csv(DATA_PATH / "olist_customers_dataset.csv")
items = pd.read_csv(DATA_PATH / "olist_order_items_dataset.csv")
payments = pd.read_csv(DATA_PATH / "olist_order_payments_dataset.csv")
reviews = pd.read_csv(DATA_PATH / "olist_order_reviews_dataset.csv")
products = pd.read_csv(DATA_PATH / "olist_products_dataset.csv")
sellers = pd.read_csv(DATA_PATH / "olist_sellers_dataset.csv")

def check_relationship(child, parent, key, relationship_name):

    unmatched = (~child[key].isin(parent[key])).sum()

    print(f"\n{relationship_name}")
    print("Total records:", len(child))
    print("Unmatched records:", unmatched)
    print("Matched records:", len(child) - unmatched)


print("=" * 55)
print("DATASET RELATIONSHIP ANALYSIS")
print("=" * 55)

check_relationship(
    orders, customers,
    "customer_id",
    "Orders -> Customers"
)

check_relationship(
    items, orders,
    "order_id",
    "Order Items -> Orders"
)

check_relationship(
    items, products,
    "product_id",
    "Order Items -> Products"
)

check_relationship(
    items, sellers,
    "seller_id",
    "Order Items -> Sellers"
)

check_relationship(
    payments, orders,
    "order_id",
    "Payments -> Orders"
)

check_relationship(
    reviews, orders,
    "order_id",
    "Reviews -> Orders"
)

print("\nAnalysis completed.")