import pandas as pd
import re

df = pd.read_csv(
    "data/raw/solar.csv"
)

df["DATE"] = df["DATE"].ffill()
df["SEASON"] = df["SEASON"].ffill()

def clean_numeric(x):
    nums = re.findall(
        r"[-+]?\d*\.?\d+",
        str(x)
    )
    return float(nums[0])

cols = [
    "VOLTAGE",
    "CURRENT",
    "AMBIENT",
    "TARGET",
    "HUMIDITY"
]

for col in cols:
    df[col] = df[col].apply(clean_numeric)

# =====================
# Time conversion
# =====================

df["TIME"] = (
    df["TIME"]
    .astype(str)
    .str.replace(".", "", regex=False)
)

# Remove AM/PM if hour already >12

df["TIME"] = df["TIME"].str.replace(
    r'(\d{2}):(\d{2})\s(?:AM|PM)',
    lambda x: f"{x.group(1)}:{x.group(2)}",
    regex=True
)

df["TIME"] = pd.to_datetime(
    df["TIME"],
    format="%H:%M"
)

df["Hour"] = df["TIME"].dt.hour

df["Power"] = (
    df["VOLTAGE"]
    * df["CURRENT"]
)

df["TempDiff"] = (
    df["TARGET"]
    - df["AMBIENT"]
)

df.to_csv(
    "data/processed/solar_processed.csv",
    index=False
)

print(df.head())