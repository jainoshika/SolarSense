import pandas as pd
import numpy as np

# ==========================
# Load engineered dataset
# ==========================

df = pd.read_csv(
    "data/processed/solar_features.csv"
)

print("Original Shape:", df.shape)


# ==========================
# Generate synthetic data
# ==========================

synthetic_rows = []

n_samples = 10000


for _ in range(n_samples):

    # Pick random row from real data
    row = (
        df.sample(1)
        .iloc[0]
        .copy()
    )

    # ----------------------
    # Add realistic noise
    # using EDA statistics
    # ----------------------

    row["VOLTAGE"] += np.random.normal(
        0,
        df["VOLTAGE"].std()*0.1
    )

    row["CURRENT"] += np.random.normal(
        0,
        df["CURRENT"].std()*0.1
    )

    row["AMBIENT"] += np.random.normal(
        0,
        df["AMBIENT"].std()*0.1
    )

    row["TARGET"] += np.random.normal(
        0,
        df["TARGET"].std()*0.1
    )

    row["HUMIDITY"] += np.random.normal(
        0,
        df["HUMIDITY"].std()*0.1
    )

    # ----------------------
    # Constraints
    # ----------------------

    row["CURRENT"] = max(
        row["CURRENT"],
        0.01
    )

    row["HUMIDITY"] = min(
        max(
            row["HUMIDITY"],
            0
        ),
        100
    )

    row["VOLTAGE"] = max(
        row["VOLTAGE"],
        0
    )

    # ----------------------
    # Recompute features
    # ----------------------

    row["Power"] = (
        row["VOLTAGE"]
        *
        row["CURRENT"]
    )

    row["TempDiff"] = (
        row["TARGET"]
        -
        row["AMBIENT"]
    )

    synthetic_rows.append(
        row
    )


# ==========================
# Convert to dataframe
# ==========================

synthetic_df = pd.DataFrame(
    synthetic_rows
)


# ==========================
# Combine with original
# ==========================

final_df = pd.concat(
    [
        df,
        synthetic_df
    ],
    ignore_index=True
)


# ==========================
# Save final dataset
# ==========================

final_df.to_csv(
    "data/processed/final_dataset.csv",
    index=False
)


# ==========================
# Print results
# ==========================

print(
    "\nOriginal rows:",
    len(df)
)

print(
    "Synthetic rows:",
    len(synthetic_df)
)

print(
    "Final rows:",
    len(final_df)
)

print(
    "\nSaved successfully:"
)

print(
    "data/processed/final_dataset.csv"
)

print(
    "\nLast few rows:\n"
)

print(
    final_df.tail()
)