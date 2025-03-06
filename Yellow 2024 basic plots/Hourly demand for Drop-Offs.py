import pandas as pd
import dask.dataframe as dd
import glob
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------
data_dir = r"C:\Users\44746\ADS taxi dataset\2024 yellow taxi datasets"

# Glob pattern assumes files are named like "yellow_tripdata_2024-01.parquet", etc.
yellow_pattern = f"{data_dir}/yellow_tripdata_*.parquet"

# Specify columns needed for time-based grouping and destination filtering
necessary_columns = [
    "tpep_pickup_datetime",  # For extracting hour
    "DOLocationID",          # So we can filter by destination
    "VendorID"               # Just to have something to count; any column will do
]

# Define date range for 2024 only
start_date = "2024-01-01"
end_date = "2025-01-01"

# Choose your target drop-off location (change to any valid DOLocationID)
target_doloc = 24

# Dictionary to store hour-of-day trip counts for the target destination
trips_by_hour_dest_dict = {}

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
    
    # Extract the hour from pickup datetime
    df_dest["hour"] = df_dest["tpep_pickup_datetime"].dt.hour
    
    # Group by hour, count the rows
    hour_trips = df_dest.groupby("hour")["VendorID"].count().compute()
    
    # Merge into our global dictionary
    for hr, count in hour_trips.items():
        trips_by_hour_dest_dict[hr] = trips_by_hour_dest_dict.get(hr, 0) + count

# ------------------------------------------------------------
# 3. Convert the Dictionary to a Series and Plot
# ------------------------------------------------------------
hourly_trips_dest_series = pd.Series(trips_by_hour_dest_dict).sort_index()

plt.figure(figsize=(10, 6))
hourly_trips_dest_series.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title(f"Hourly Demand for Drop-off Location {target_doloc} (2024)")
plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Number of Trips")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.5)
plt.show()
