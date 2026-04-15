"""
generate_data.py
Generates a realistic synthetic energy market dataset.
Run this once to create energy_data.csv in the data/ folder.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 8760  # one full year of hourly data

hours = np.arange(N)
timestamps = pd.date_range(start="2023-01-01", periods=N, freq="h")

# --- Weather features ---
# Temperature: seasonal + daily cycle + noise
day_of_year = (timestamps.dayofyear - 1) / 365
hour_of_day = timestamps.hour / 24
temperature = (
    25
    + 12 * np.sin(2 * np.pi * day_of_year - np.pi / 2)   # seasonal (summer peak)
    + 5  * np.sin(2 * np.pi * hour_of_day - np.pi / 2)   # daily (afternoon peak)
    + np.random.normal(0, 2, N)
)

# Sunlight index (0-1): zero at night, peak midday
sunlight = np.clip(
    np.sin(np.pi * (hour_of_day - 6/24) / (12/24)),
    0, 1
) * np.clip(1 - 0.3 * np.random.rand(N), 0.4, 1.0)  # cloud factor

# Wind speed (m/s): random with seasonal variation
wind_speed = np.abs(
    8
    + 3 * np.sin(2 * np.pi * day_of_year + np.pi)  # stronger in winter
    + np.random.normal(0, 2, N)
)

# --- Demand features ---
# Demand (MW): temperature, hour, day-of-week driven
is_weekend = (timestamps.dayofweek >= 5).astype(float)
demand = (
    220
    + 80  * np.abs(np.sin(2 * np.pi * hour_of_day))        # daily cycle
    + 30  * (temperature - 20) / 20                          # heat/cold load
    - 40  * is_weekend                                        # lower on weekends
    + np.random.normal(0, 15, N)
)
demand = np.clip(demand, 80, 450)

# --- Price (₹/unit): demand + supply shortage driven ---
renewable_supply = 100 * sunlight + 60 * (wind_speed / 15)
supply_gap = demand - renewable_supply
price = (
    35
    + 0.18 * supply_gap
    + 0.25 * temperature
    + np.random.normal(0, 4, N)
    + 20 * (np.random.rand(N) < 0.05)  # occasional price spikes (~5% of hours)
)
price = np.clip(price, 15, 140)

df = pd.DataFrame({
    "timestamp":    timestamps,
    "temperature":  temperature.round(2),
    "sunlight":     sunlight.round(3),
    "wind_speed":   wind_speed.round(2),
    "demand":       demand.round(1),
    "price":        price.round(2),
    "is_weekend":   is_weekend.astype(int),
    "hour":         timestamps.hour,
    "month":        timestamps.month,
})

out_path = "data/energy_data.csv"
df.to_csv(out_path, index=False)
print(f"Dataset generated: {out_path}  ({len(df)} rows)")
print(df.describe().round(2))
