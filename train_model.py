
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("Training RetailIQ AI churn prediction models...")

df = pd.read_csv("data/historical_churn_dataset.csv")

feature_columns = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Items",
    "Tenure_Days"
]

# Train on earlier snapshots.
# The June 2011 training labels finish before
# the September 2011 test snapshot begins.
train = df[df["Snapshot_Date"] < "2011-09-01"].copy()
test = df[df["Snapshot_Date"] == "2011-09-01"].copy()

X_train = train[feature_columns]
y_train = train["Churn"]

X_test = test[feature_columns]
y_test = test["Churn"]

print("\nTraining records:", len(train))
print("Testing records:", len(test))

# Model 1: Logistic Regression
logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

# Model 2: Random Forest
forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": forest_model
}

for name, model in models.items():
    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))
    print("Precision:", round(
        precision_score(y_test, predictions, zero_division=0), 4
    ))
    print("Recall:", round(
        recall_score(y_test, predictions, zero_division=0), 4
    ))
    print("F1 Score:", round(
        f1_score(y_test, predictions, zero_division=0), 4
    ))
    print("ROC-AUC:", round(
        roc_auc_score(y_test, probabilities), 4
    ))

print("\nMODEL EVALUATION COMPLETE!")

# STEP 25: Random Forest Feature Importance

import matplotlib.pyplot as plt
import pandas as pd

print("\n===== FEATURE IMPORTANCE ANALYSIS =====")

# Get importance scores from trained Random Forest
importances = forest_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importances
}).sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

# Create chart
plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"],
    color="teal"
)

plt.title("RetailIQ AI - Churn Model Feature Importance")
plt.xlabel("Feature Importance")
plt.ylabel("Customer Features")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    "data/feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature importance chart saved!")

# STEP 26: Random Forest Confusion Matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("\n===== CONFUSION MATRIX =====")

# Predict on held-out test data
y_pred = forest_model.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print(cm)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Returned", "No Return"]
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("RetailIQ AI - Random Forest Confusion Matrix")
plt.tight_layout()

# Save chart
plt.savefig(
    "data/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved successfully!")
print("\nFeature importance chart saved!")
