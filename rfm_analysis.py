
import pandas as pd
import sqlite3

print("Starting RetailIQ AI RFM Analysis...")

# Connect to database
conn = sqlite3.connect("database/retailiq.db")

# Load customer purchase records
df = pd.read_sql_query(
    "SELECT * FROM retail_transactions",
    conn
)

conn.close()

# Convert invoice dates
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Reference date: one day after final transaction
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Calculate RFM values for each customer
rfm = df.groupby("Customer ID").agg(
    Recency=("InvoiceDate",
             lambda x: (reference_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("Revenue", "sum")
).reset_index()

# Round monetary values
rfm["Monetary"] = rfm["Monetary"].round(2)

# Save RFM dataset
rfm.to_csv("data/customer_rfm.csv", index=False)

print("\n===== RFM ANALYSIS RESULTS =====")
print(rfm.head(10))

print("\nTotal customers analyzed:", len(rfm))

print("\n===== RFM STATISTICS =====")
print(rfm[["Recency", "Frequency", "Monetary"]].describe())

print("\nRFM dataset saved successfully!")
