
import pandas as pd
from pathlib import Path

print("Preparing RetailIQ AI deployment data...")

output = Path("deployment_data")
output.mkdir(exist_ok=True)

# Load cleaned purchase data
df = pd.read_csv(
    "data/full_cleaned_retail.csv",
    parse_dates=["InvoiceDate"]
)

# Overall business statistics
summary = pd.DataFrame([{
    "Total_Revenue": df["Revenue"].sum(),
    "Total_Transactions": df["Invoice"].nunique(),
    "Total_Customers": df["Customer ID"].nunique()
}])

summary.to_csv(
    output / "business_summary.csv",
    index=False
)

# Monthly revenue
monthly = (
    df.assign(
        Month=df["InvoiceDate"].dt.to_period("M")
    )
    .groupby("Month")["Revenue"]
    .sum()
    .reset_index()
)

monthly["Month"] = monthly["Month"].astype(str)

monthly.to_csv(
    output / "monthly_revenue.csv",
    index=False
)

# Country revenue
country = (
    df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

country.to_csv(
    output / "country_revenue.csv",
    index=False
)

# Customer segment counts
customers = pd.read_csv("data/customer_segments.csv")

if "Segment" in customers.columns:
    segment_col = "Segment"
elif "Customer_Segment" in customers.columns:
    segment_col = "Customer_Segment"
else:
    segment_col = None

if segment_col:
    segments = (
        customers[segment_col]
        .value_counts()
        .rename_axis("Segment")
        .reset_index(name="Count")
    )

    segments.to_csv(
        output / "segment_counts.csv",
        index=False
    )
else:
    print("No customer segment label column found.")

print("Deployment analytics files created!")
print("Output folder:", output)
