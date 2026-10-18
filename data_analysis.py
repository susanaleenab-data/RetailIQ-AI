
import pandas as pd

# Load the retail dataset
print("Loading RetailIQ AI dataset...")

file_path = "data/online_retail_II.xlsx"

# Read the first worksheet
df = pd.read_excel(file_path, sheet_name=0)

# Display basic information
print("\nDataset loaded successfully!")

print("\nTotal rows and columns:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

# STEP 8: Data Quality Analysis

print("\n========== DATA QUALITY REPORT ==========")

# Total number of rows
print("\nTotal rows:", len(df))

# Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate rows:", df.duplicated().sum())

# Check data types
print("\nData types:")
print(df.dtypes)

# Check negative and zero quantities
print("\nNegative quantities:", (df["Quantity"] < 0).sum())
print("Zero quantities:", (df["Quantity"] == 0).sum())

# Check negative and zero prices
print("\nNegative prices:", (df["Price"] < 0).sum())
print("Zero prices:", (df["Price"] == 0).sum())

print("\n========== REPORT COMPLETE ==========")

# STEP 9: Investigate unusual transactions

print("\n===== NEGATIVE QUANTITY EXAMPLES =====")
print(
    df.loc[df["Quantity"] < 0,
           ["Invoice", "Description", "Quantity", "Price"]].head(10)
)

print("\n===== NEGATIVE PRICE EXAMPLES =====")
print(
    df.loc[df["Price"] < 0,
           ["Invoice", "Description", "Quantity", "Price"]]
)

print("\n===== ZERO PRICE EXAMPLES =====")
print(
    df.loc[df["Price"] == 0,
           ["Invoice", "Description", "Quantity", "Price"]].head(10)
)

print("\n===== MISSING CUSTOMER IDS =====")
print(df["Customer ID"].isna().sum())

print("\n===== DUPLICATE RECORDS =====")
print(df.duplicated().sum())

# STEP 10: Clean retail purchase data

from pathlib import Path

print("\n===== STARTING DATA CLEANING =====")

# Make a copy to preserve the original data
clean_df = df.copy()

# Remove exact duplicate records
clean_df = clean_df.drop_duplicates()

# Keep transactions with valid customer IDs
clean_df = clean_df.dropna(subset=["Customer ID"])

# Keep positive quantities and prices
clean_df = clean_df[
    (clean_df["Quantity"] > 0) &
    (clean_df["Price"] > 0)
].copy()

# Exclude invoices marked as cancellations
clean_df = clean_df[
    ~clean_df["Invoice"].astype(str).str.upper().str.startswith("C")
].copy()

# Create a revenue column
clean_df["Revenue"] = (
    clean_df["Quantity"] * clean_df["Price"]
)

# Convert customer IDs to strings
clean_df["Customer ID"] = (
    clean_df["Customer ID"].astype("int64").astype(str)
)

# Save the cleaned dataset
output_path = Path("data/cleaned_retail.csv")
clean_df.to_csv(output_path, index=False)

print("\nOriginal rows:", len(df))
print("Cleaned rows:", len(clean_df))
print("Total revenue:", round(clean_df["Revenue"].sum(), 2))
print("Unique customers:", clean_df["Customer ID"].nunique())

print("\nCleaned dataset saved to:", output_path)
print("\n===== DATA CLEANING COMPLETE =====")

# STEP 11: Retail Business KPI Analysis

print("\n===== RETAIL BUSINESS KPI REPORT =====")

# Total purchase revenue
total_revenue = clean_df["Revenue"].sum()

# Unique customers
total_customers = clean_df["Customer ID"].nunique()

# Number of unique invoices
total_orders = clean_df["Invoice"].nunique()

# Average order value
average_order_value = total_revenue / total_orders

# Average revenue per customer
revenue_per_customer = total_revenue / total_customers

print(f"\nTotal Revenue: £{total_revenue:,.2f}")
print(f"Total Customers: {total_customers:,}")
print(f"Total Orders: {total_orders:,}")
print(f"Average Order Value: £{average_order_value:,.2f}")
print(f"Revenue per Customer: £{revenue_per_customer:,.2f}")

# Top 10 products by revenue
top_products = (
    clean_df.groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n===== TOP 10 PRODUCTS BY REVENUE =====")
print(top_products)

# Revenue by country
country_sales = (
    clean_df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n===== TOP 10 COUNTRIES BY REVENUE =====")
print(country_sales)

print("\n===== KPI ANALYSIS COMPLETE =====")

# STEP 12: Visualize revenue by country

import matplotlib.pyplot as plt

# Get top 10 countries by revenue
top_countries = (
    clean_df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# Create chart
plt.figure(figsize=(12, 6))

top_countries.sort_values().plot(
    kind="barh",
    color="steelblue"
)

plt.title("RetailIQ AI - Top 10 Countries by Revenue")
plt.xlabel("Revenue (£)")
plt.ylabel("Country")

plt.tight_layout()

# Save chart
plt.savefig("data/country_revenue.png", dpi=300)

# Display chart
plt.show()

print("\nCountry revenue chart created successfully!")

# STEP 13: Monthly Sales Trend Analysis

print("\n===== MONTHLY SALES ANALYSIS =====")

# Create a copy for time-series analysis
monthly_df = clean_df.copy()

# Ensure invoice dates are datetime values
monthly_df["InvoiceDate"] = pd.to_datetime(
    monthly_df["InvoiceDate"]
)

# Group revenue by month
monthly_sales = (
    monthly_df.set_index("InvoiceDate")
    .resample("MS")["Revenue"]
    .sum()
)

print("\nMonthly Revenue:")
print(monthly_sales)

# Create line chart
plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o",
    linewidth=2,
    color="royalblue"
)

plt.title("RetailIQ AI - Monthly Revenue Trends")
plt.xlabel("Month")
plt.ylabel("Revenue (£)")

plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Save chart
plt.savefig("data/monthly_sales.png", dpi=300)

plt.show()

print("\nMonthly sales chart created successfully!")
