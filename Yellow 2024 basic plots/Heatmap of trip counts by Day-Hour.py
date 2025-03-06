# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 23:58:15 2025

@author: 44746
"""

import pandas as pd
import dask.dataframe as dd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------
data_dir = r"C:\Users\44746\ADS taxi dataset\2024 yellow taxi datasets"

# Glob pattern assumes files are named like "yellow_tripdata_2024-01.parquet", etc.
yellow_pattern = f"{data_dir}/yellow_tripdata_*.parquet"

# Specify columns needed for filtering and grouping
necessary_columns = [
    "tpep_pickup_datetime",  # For extracting hour & day of week
    "DOLocationID",          # To filter by destination
    "VendorID"               # Just to have something to count; any column will do
]

# Define date range for 2024 only
start_date = "2024-01-01"
end_date = "2025-01-01"

# Choose your target drop-off location (change this as needed)
target_doloc = 79  # Change to the DOLocationID you want to analyze

# Dictionary to store trip counts grouped by (day_of_week, hour)
heatmap_data = {}

# ------------------------------------------------------------
# 2. Process Files One-by-One
# ------------------------------------------------------------
files = sorted(glob.glob(yellow_pattern))
print(f"Found {len(files)} files matching the pattern.")

for file in files:
    print(f"Processing file: {file}")
    
    # Read the Parquet file with only necessary columns
    df = dd.read_parquet(file, columns=necessary_columns, engine="pyarrow")
    
    # Convert pickup datetime to datetime type
    df["tpep_pickup_datetime"] = dd.to_datetime(df["tpep_pickup_datetime"])
    
    # Filter rows to keep only 2024 data
    df = df[
        (df["tpep_pickup_datetime"] >= start_date) &
        (df["tpep_pickup_datetime"] < end_date)
    ]
    
    # Filter to the target destination
    df_dest = df[df["DOLocationID"] == target_doloc]
    
    # If this file has no rows for the target destination in 2024, skip it
    if df_dest.shape[0].compute() == 0:
        continue
    
    # Extract day of the week and hour
    df_dest["hour"] = df_dest["tpep_pickup_datetime"].dt.hour
    df_dest["day_of_week"] = df_dest["tpep_pickup_datetime"].dt.day_name()

    # Group by (day_of_week, hour) and count trips
    grouped_data = df_dest.groupby(["day_of_week", "hour"])["VendorID"].count().compute()

    # Merge results into global heatmap dictionary
    for (day, hour), count in grouped_data.items():
        heatmap_data[(day, hour)] = heatmap_data.get((day, hour), 0) + count

# ------------------------------------------------------------
# 3. Convert the Dictionary to a Pivot Table
# ------------------------------------------------------------
# Convert dictionary to DataFrame
df_heatmap = pd.DataFrame.from_dict(heatmap_data, orient="index", columns=["trip_count"])
df_heatmap.index = pd.MultiIndex.from_tuples(df_heatmap.index, names=["day_of_week", "hour"])

# Convert to a pivot table format (rows=day_of_week, columns=hour, values=trip_count)
heatmap_pivot = df_heatmap.unstack(level="hour", fill_value=0)

# Reorder days to Monday-Sunday for a better visualization
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heatmap_pivot = heatmap_pivot.reindex(day_order)

# Extract the numeric matrix for heatmap plotting
heatmap_matrix = heatmap_pivot["trip_count"].values

# ------------------------------------------------------------
# 4. Plot the Heatmap
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_matrix, cmap="Reds", annot=False, xticklabels=range(24), yticklabels=day_order)

plt.title(f"Heatmap of Trip Counts by Day/Hour for Drop-off {target_doloc} (2024)")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.show()
