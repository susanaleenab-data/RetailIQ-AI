
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("Starting RetailIQ AI time-based validation...")

df = pd.read_csv(
    "data/historical_churn_dataset.csv",
    parse_dates=["Snapshot_Date"]
)

features = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Items",
    "Tenure_Days"
]

# Historical test dates
test_dates = [
    "2010-12-01",
    "2011-03-01",
    "2011-06-01",
    "2011-09-01"
]

results = []

for date in test_dates:
    test_date = pd.Timestamp(date)

    # Only train on snapshots whose 90-day
    # outcome window is completely finished.
    eligible = (
        df["Snapshot_Date"] + pd.Timedelta(days=90)
        <= test_date
    )

    train = df[eligible].copy()
    test = df[
        df["Snapshot_Date"] == test_date
    ].copy()

    if train.empty or test.empty:
        continue

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[features],
        train["Churn"]
    )

    predictions = model.predict(test[features])
    probabilities = model.predict_proba(
        test[features]
    )[:, 1]

    result = {
        "Test_Date": date,
        "Accuracy": accuracy_score(
            test["Churn"], predictions
        ),
        "Precision": precision_score(
            test["Churn"], predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            test["Churn"], predictions,
            zero_division=0
        ),
        "F1": f1_score(
            test["Churn"], predictions,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            test["Churn"], probabilities
        )
    }

    results.append(result)

results_df = pd.DataFrame(results)

print("\n===== TIME-BASED VALIDATION RESULTS =====")
print(results_df.round(4).to_string(index=False))

print("\n===== AVERAGE PERFORMANCE =====")
print(
    results_df[
        ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    ].mean().round(4)
)

results_df.to_csv(
    "data/model_validation_results.csv",
    index=False
)

print("\nValidation results saved successfully!")
