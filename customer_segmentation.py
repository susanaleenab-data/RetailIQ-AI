
import pandas as pd
import matplotlib.pyplot as plt

print("Starting customer segmentation...")

# Load RFM data
rfm = pd.read_csv(
    "data/customer_rfm.csv",
    dtype={"Customer ID": "string"}
)

# Calculate RFM scores from 1 to 5
# Lower recency is better
rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

# Higher frequency is better
rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Higher monetary value is better
rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Classify customers
def segment_customer(row):
    if (row["R_Score"] >= 4 and
        row["F_Score"] >= 4 and
        row["M_Score"] >= 4):
        return "Champions"

    elif (row["R_Score"] <= 2 and
          row["F_Score"] >= 4):
        return "At Risk"

    elif (row["R_Score"] >= 3 and
          row["F_Score"] >= 3):
        return "Loyal Customers"

    else:
        return "Others"

rfm["Segment"] = rfm.apply(
    segment_customer,
    axis=1
)

# Save customer segmentation results
rfm.to_csv(
    "data/customer_rfm.csv",
    index=False
)

print("Customer segmentation data saved successfully!")

# Display customer counts
segment_counts = rfm["Segment"].value_counts()

print("\n===== CUSTOMER SEGMENTATION RESULTS =====")
print(segment_counts)

# Plot customer segments
plt.figure(figsize=(10, 6))

segment_counts.plot(
    kind="bar",
    color="mediumseagreen"
)

plt.title("RetailIQ AI - Customer Segmentation")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "data/customer_segments.png",
    dpi=300
)

plt.show()

print("\nCustomer segmentation completed!")
