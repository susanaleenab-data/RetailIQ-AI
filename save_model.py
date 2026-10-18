
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

print("Saving RetailIQ AI Machine Learning model...")

# Load historical dataset
df = pd.read_csv("data/historical_churn_dataset.csv")

feature_columns = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Items",
    "Tenure_Days"
]

# Use all available historical snapshots
# for the final demonstration model
X = df[feature_columns]
y = df["Churn"]

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# Create model directory
Path("models").mkdir(exist_ok=True)

# Save model and feature order
joblib.dump(
    {
        "model": model,
        "features": feature_columns
    },
    "models/churn_model.pkl"
)

print("\nModel saved to: models/churn_model.pkl")
print("Training records:", len(df))
print("Features:", feature_columns)
print("\nMODEL SAVING COMPLETE!")
