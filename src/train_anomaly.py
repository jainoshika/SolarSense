import pandas as pd
import pickle

from sklearn.ensemble import IsolationForest


# ==========================
# Load dataset
# ==========================

df = pd.read_csv(
    "data/processed/final_dataset.csv"
)

print(
    "Dataset Shape:",
    df.shape
)


# ==========================
# Features for anomaly detection
# ==========================

X = df[
    [
        "VOLTAGE",
        "CURRENT",
        "Power",
        "AMBIENT",
        "HUMIDITY"
    ]
]


# ==========================
# Create model
# ==========================

model = IsolationForest(

    contamination=0.05,

    random_state=42
)


# ==========================
# Train model
# ==========================

model.fit(X)


# ==========================
# Test predictions
# ==========================

predictions = model.predict(X)

df["Anomaly"] = predictions


# Count anomalies

normal = (
    df["Anomaly"]==1
).sum()

abnormal = (
    df["Anomaly"]==-1
).sum()


print(
    "\nNormal Samples:",
    normal
)

print(
    "Anomalies:",
    abnormal
)


# ==========================
# Save model
# ==========================

with open(
    "models/anomaly.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )


print(
    "\nModel saved:"
)

print(
    "models/anomaly.pkl"
)


# ==========================
# Save anomaly results
# ==========================

df.to_csv(
    "data/processed/anomaly_results.csv",
    index=False
)

print(
    "\nResults saved:"
)

print(
    "data/processed/anomaly_results.csv"
)