import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

df = pd.read_csv("data/historical_churn_dataset.csv")

features = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Items",
    "Tenure_Days"
]

train = df[df["Snapshot_Date"] < "2011-09-01"]
test = df[df["Snapshot_Date"] == "2011-09-01"]

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(train[features], train["Churn"])

probabilities = model.predict_proba(test[features])[:, 1]

print(
    "Brier Score:",
    round(brier_score_loss(test["Churn"], probabilities), 4)
)

results = pd.DataFrame({
    "Actual": test["Churn"].to_numpy(),
    "Predicted_Probability": probabilities
})

results["Probability_Group"] = pd.cut(
    results["Predicted_Probability"],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    include_lowest=True
)

print("\nCalibration Results:")
print(
    results.groupby(
        "Probability_Group",
        observed=True
    ).agg(
        Customers=("Actual", "size"),
        Average_Prediction=("Predicted_Probability", "mean"),
        Actual_Churn_Rate=("Actual", "mean")
    )
)