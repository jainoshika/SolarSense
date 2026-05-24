import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor


# ==========================
# Load dataset
# ==========================

df = pd.read_csv(
    "data/processed/final_dataset.csv"
)


# ==========================
# Encode categorical features
# ==========================

categorical_cols = [
    "SEASON",
    "TimePeriod"
]

encoders = {}

for col in categorical_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col]
    )

    encoders[col] = le


# ==========================
# Features and Target
# ==========================

X = df[
    [
        "Hour",
        "AMBIENT",
        "TARGET",
        "HUMIDITY",
        "SEASON",
        "TempDiff",
        "TimePeriod"
    ]
]

y = df["Power"]


# ==========================
# Train test split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================
# Models
# ==========================

models = {

    "LinearRegression":
    LinearRegression(),

    "RandomForest":
    RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost":
    XGBRegressor(
        n_estimators=100,
        random_state=42
    )
}


results = {}


# ==========================
# Training loop
# ==========================

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = (
        mean_squared_error(
            y_test,
            pred
        )
    )**0.5


    r2 = r2_score(
        y_test,
        pred
    )


    results[name] = {

        "MAE":mae,
        "RMSE":rmse,
        "R2":r2
    }


    # Save model

    with open(
        f"models/{name}.pkl",
        "wb"
    ) as f:

        pickle.dump(
            model,
            f
        )


# ==========================
# Results
# ==========================

results_df = pd.DataFrame(
    results
).T

print("\nResults:\n")

print(results_df)


# Save results

results_df.to_csv(
    "models/model_results.csv",
    index=True
)

print(
    "\nSaved model results"
)