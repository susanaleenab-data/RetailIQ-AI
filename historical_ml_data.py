
import pandas as pd

print("Building historical machine learning snapshots...")

df = pd.read_csv(
    "data/full_cleaned_retail.csv",
    dtype={"Customer ID": "string"},
    parse_dates=["InvoiceDate"],
    low_memory=False
)

# Each snapshot predicts the next 90 days.
# Dates are separated to avoid overlapping label windows.
cutoff_dates = [
    "2010-03-01",
    "2010-06-01",
    "2010-09-01",
    "2010-12-01",
    "2011-03-01",
    "2011-06-01",
    "2011-09-01"
]

all_snapshots = []

for date in cutoff_dates:
    cutoff = pd.Timestamp(date)
    end_date = cutoff + pd.Timedelta(days=90)

    history = df[df["InvoiceDate"] < cutoff]
    future = df[
        (df["InvoiceDate"] >= cutoff) &
        (df["InvoiceDate"] < end_date)
    ]

    if history.empty:
        continue

    features = history.groupby("Customer ID").agg(
        Last_Purchase=("InvoiceDate", "max"),
        First_Purchase=("InvoiceDate", "min"),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        Total_Items=("Quantity", "sum")
    ).reset_index()

    features["Recency"] = (
        cutoff - features["Last_Purchase"]
    ).dt.days

    features["Tenure_Days"] = (
        cutoff - features["First_Purchase"]
    ).dt.days

    returning = set(future["Customer ID"].dropna())

    features["Churn"] = (
        ~features["Customer ID"].isin(returning)
    ).astype(int)

    features["Snapshot_Date"] = date

    all_snapshots.append(features[[
        "Customer ID",
        "Snapshot_Date",
        "Recency",
        "Frequency",
        "Monetary",
        "Total_Items",
        "Tenure_Days",
        "Churn"
    ]])

    print(
        date,
        "| Customers:", len(features),
        "| Churn rate:", round(features["Churn"].mean(), 3)
    )

ml_data = pd.concat(
    all_snapshots,
    ignore_index=True
)

ml_data.to_csv(
    "data/historical_churn_dataset.csv",
    index=False
)

print("\nTotal snapshot records:", len(ml_data))
print("Unique customers:", ml_data["Customer ID"].nunique())
print("\nHistorical dataset saved successfully!")
