
import pandas as pd
from pathlib import Path

print("Loading combined retail dataset...")

df = pd.read_csv(
    "data/combined_retail.csv",
    low_memory=False
)

print("\nOriginal records:", len(df))

# Separate cancellation and adjustment records
returns_df = df[
    (df["Quantity"] < 0) |
    (df["Invoice"].astype(str).str.upper().str.startswith("C"))
].copy()

# Clean customer purchase data
clean_df = df.dropna(subset=["Customer ID"]).copy()

clean_df = clean_df[
    (clean_df["Quantity"] > 0) &
    (clean_df["Price"] > 0) &
    (~clean_df["Invoice"].astype(str).str.upper().str.startswith("C"))
].copy()

# Convert customer IDs to strings
clean_df["Customer ID"] = (
    pd.to_numeric(clean_df["Customer ID"])
    .astype("int64")
    .astype(str)
)

# Calculate revenue
clean_df["Revenue"] = (
    clean_df["Quantity"] * clean_df["Price"]
)

# Convert dates
clean_df["InvoiceDate"] = pd.to_datetime(
    clean_df["InvoiceDate"]
)

# Save datasets
Path("data").mkdir(exist_ok=True)

clean_df.to_csv(
    "data/full_cleaned_retail.csv",
    index=False
)

returns_df.to_csv(
    "data/returns_retail.csv",
    index=False
)

print("\n===== FINAL CLEANING REPORT =====")
print("Clean purchase records:", len(clean_df))
print("Returns/adjustment records:", len(returns_df))
print("Unique customers:", clean_df["Customer ID"].nunique())
print("Total purchase revenue: £", round(clean_df["Revenue"].sum(), 2))

print("\nFiles saved successfully!")
