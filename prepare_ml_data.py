
import pandas as pd

print("Preparing RetailIQ AI machine learning data...")

# Load cleaned transactions
df = pd.read_csv(
    "data/full_cleaned_retail.csv",
    dtype={"Customer ID": "string"},
    parse_dates=["InvoiceDate"],
    low_memory=False
)

# Historical cutoff and prediction horizon
cutoff = pd.Timestamp("2011-09-01")
end_date = cutoff + pd.Timedelta(days=90)

# Historical data only
history = df[df["InvoiceDate"] < cutoff].copy()

# Future purchases used only for labels
future = df[
    (df["InvoiceDate"] >= cutoff) &
    (df["InvoiceDate"] < end_date)
].copy()

# Create customer features
features = history.groupby("Customer ID").agg(
    Last_Purchase=("InvoiceDate", "max"),
    First_Purchase=("InvoiceDate", "min"),
    Frequency=("Invoice", "nunique"),
    Monetary=("Revenue", "sum"),
    Total_Items=("Quantity", "sum")
).reset_index()

# Calculate recency and customer tenure
features["Recency"] = (
    cutoff - features["Last_Purchase"]
).dt.days

features["Tenure_Days"] = (
    cutoff - features["First_Purchase"]
).dt.days

# Identify customers who purchased again
returning_customers = set(future["Customer ID"].dropna())

# Target: 1 means no purchase in next 90 days
features["Churn"] = (
    ~features["Customer ID"].isin(returning_customers)
).astype(int)

# Select model-ready columns
ml_df = features[[
    "Customer ID",
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Items",
    "Tenure_Days",
    "Churn"
]].copy()

# Save dataset
ml_df.to_csv(
    "data/churn_ml_dataset.csv",
    index=False
)

print("\n===== MACHINE LEARNING DATASET =====")
print("Customers:", len(ml_df))
print("Churned:", (ml_df["Churn"] == 1).sum())
print("Returned:", (ml_df["Churn"] == 0).sum())

print("\nChurn distribution:")
print(ml_df["Churn"].value_counts(normalize=True))

print("\nFirst five rows:")
print(ml_df.head())

print("\nSaved: data/churn_ml_dataset.csv")
