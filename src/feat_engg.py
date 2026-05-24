import pandas as pd

# Load processed data

df = pd.read_csv(
    "data/processed/solar_processed.csv"
)

# ==========================
# Time Period Feature
# ==========================

def time_period(hour):

    if hour < 12:
        return "Morning"

    elif hour < 15:
        return "Afternoon"

    else:
        return "Evening"


df["TimePeriod"] = (
    df["Hour"]
    .apply(time_period)
)

# Save new file

df.to_csv(
    "data/processed/solar_features.csv",
    index=False
)

print(df.head())

print("\nFeature Engineering Done")