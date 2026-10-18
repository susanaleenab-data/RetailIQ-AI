
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

print("Calibrating RetailIQ AI model...")

# Load historical customer snapshots
df = pd.read_csv("data/historical_churn_dataset.csv")
df["Snapshot_Date"] = pd.to_datetime(df["Snapshot_Date"])

features = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Items",
    "Tenure_Days"
]

# Chronological separation:
# Train: snapshots before March 2011
# Calibrate: March and June 2011
# Test: September 2011
train = df[df["Snapshot_Date"] < "2011-03-01"].copy()

calibration = df[
    (df["Snapshot_Date"] >= "2011-03-01") &
    (df["Snapshot_Date"] < "2011-09-01")
].copy()

test = df[df["Snapshot_Date"] == "2011-09-01"].copy()

for name, subset in [
    ("Training", train),
    ("Calibration", calibration),
    ("Testing", test)
]:
    print(f"{name} records: {len(subset)}")
    if subset.empty:
        raise ValueError(f"{name} dataset is empty!")

X_train = train[features]
y_train = train["Churn"]

X_cal = calibration[features]
y_cal = calibration["Churn"]

X_test = test[features]
y_test = test["Churn"]

# Train base Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Calibrate on separate historical data
from sklearn.frozen import FrozenEstimator

calibrated_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(model),
    method="sigmoid"
)

calibrated_model.fit(X_cal, y_cal)

# Compare original and calibrated probabilities
original_probs = model.predict_proba(X_test)[:, 1]
calibrated_probs = calibrated_model.predict_proba(X_test)[:, 1]
# Baseline: predict training churn rate for every test customer
from sklearn.metrics import brier_score_loss

baseline_probability = y_train.mean()

baseline_probs = [baseline_probability] * len(y_test)

print("\n===== BASELINE MODEL =====")
print("Training churn rate:", round(baseline_probability, 4))
print(
    "Baseline Brier Score:",
    round(brier_score_loss(y_test, baseline_probs), 4)
)
print("\n===== CALIBRATION COMPARISON =====")

for name, probabilities in [
    ("Original Random Forest", original_probs),
    ("Calibrated Random Forest", calibrated_probs)
]:
    print(f"\n{name}")
    print(
        "Brier Score:",
        round(brier_score_loss(y_test, probabilities), 4)
    )
    print(
        "ROC-AUC:",
        round(roc_auc_score(y_test, probabilities), 4)
    )

# Save calibrated model separately
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": calibrated_model,
        "features": features
    },
    "models/calibrated_churn_model.pkl"
)

print("\nCalibrated model saved successfully!")
