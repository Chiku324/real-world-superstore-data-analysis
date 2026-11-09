import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/superstore.csv"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Full dataset not found. Download the public Sample Superstore CSV and save it as data/superstore.csv."
    )

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

# Date parsing
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

# Data-quality checks
print("Shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
if "Order ID" in df:
    print("Duplicate Order IDs:", df["Order ID"].duplicated().sum())

# Derived metrics
df["Profit Margin"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"], np.nan)
df["Order Year"] = df["Order Date"].dt.year
df["Order Month"] = df["Order Date"].dt.to_period("M").astype(str)

sales = df["Sales"].sum()
profit = df["Profit"].sum()
quantity = df["Quantity"].sum()
orders = df["Order ID"].nunique()
customers = df["Customer ID"].nunique()
products = df["Product ID"].nunique()
margin = profit / sales if sales else np.nan

print("\nKPI SUMMARY")
print(f"Sales: ${sales:,.2f}")
print(f"Profit: ${profit:,.2f}")
print(f"Profit margin: {margin:.2%}")
print(f"Quantity: {quantity:,}")
print(f"Orders: {orders:,}")
print(f"Customers: {customers:,}")
print(f"Products: {products:,}")

def save_bar(series, title, xlabel, filename, top_n=None):
    s = series.sort_values(ascending=False)
    if top_n:
        s = s.head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=s.values, y=s.index)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, filename), dpi=200)
    plt.close()

monthly = df.groupby("Order Month")["Sales"].sum()
save_bar(monthly, "Monthly Sales Trend", "Sales ($)", "monthly_sales.png")

category_sales = df.groupby("Category")["Sales"].sum()
save_bar(category_sales, "Sales by Category", "Sales ($)", "sales_by_category.png")

region_sales = df.groupby("Region")["Sales"].sum()
save_bar(region_sales, "Sales by Region", "Sales ($)", "sales_by_region.png")

subcategory_profit = df.groupby("Sub-Category")["Profit"].sum().sort_values()
save_bar(subcategory_profit, "Profit by Sub-Category", "Profit ($)", "profit_by_subcategory.png")

plt.figure(figsize=(9, 6))
sns.scatterplot(data=df.sample(min(3000, len(df)), random_state=42),
                x="Discount", y="Profit", hue="Category", alpha=0.7)
plt.title("Discount vs Profit")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "discount_vs_profit.png"), dpi=200)
plt.close()

summary = pd.DataFrame({
    "Metric": ["Sales", "Profit", "Profit Margin", "Quantity", "Orders", "Customers", "Products"],
    "Value": [sales, profit, margin, quantity, orders, customers, products]
})
summary.to_csv(os.path.join(OUT, "kpi_summary.csv"), index=False)

print("\nTop 10 sub-categories by profit:")
print(df.groupby("Sub-Category")["Profit"].sum().sort_values(ascending=False).head(10))
print("\nBottom 10 sub-categories by profit:")
print(df.groupby("Sub-Category")["Profit"].sum().sort_values().head(10))

print("\nAnalysis complete. Charts saved in outputs/.")
