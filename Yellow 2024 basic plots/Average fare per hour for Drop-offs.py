

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

# Specify columns needed for fare calculation and filtering
necessary_columns = [
    "tpep_pickup_datetime",  # For extracting hour
    "DOLocationID",          # To filter by destination
    "total_amount"           # For fare calculations
]

# Define date range for 2024 only
start_date = "2024-01-01"
end_date = "2025-01-01"

# Choose your target drop-off location (change this as needed)
target_doloc = 19  # Change to the DOLocationID you want to analyze

# Dictionary to store total fare and trip count per hour
fare_by_hour_dict = {}
trip_count_by_hour_dict = {}

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
    
    # Group by hour and calculate total fare and trip count
    hourly_fare = df_dest.groupby("hour")["total_amount"].sum().compute()
    hourly_trip_count = df_dest.groupby("hour")["total_amount"].count().compute()
    
    # Merge into our global dictionaries
    for hr, fare_sum in hourly_fare.items():
        fare_by_hour_dict[hr] = fare_by_hour_dict.get(hr, 0) + fare_sum
    
    for hr, trip_count in hourly_trip_count.items():
        trip_count_by_hour_dict[hr] = trip_count_by_hour_dict.get(hr, 0) + trip_count

# ------------------------------------------------------------
# 3. Compute Average Fare by Hour
# ------------------------------------------------------------
# Convert dictionaries to Pandas Series
total_fare_series = pd.Series(fare_by_hour_dict).sort_index()
trip_count_series = pd.Series(trip_count_by_hour_dict).sort_index()

# Avoid division by zero
avg_fare_by_hour_series = total_fare_series / trip_count_series
avg_fare_by_hour_series.fillna(0, inplace=True)  # Replace NaN values with 0

# ------------------------------------------------------------
# 4. Plot the Average Fare by Hour
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
avg_fare_by_hour_series.plot(kind="line", marker="o", color="green")
plt.title(f"Average Fare by Hour for Drop-off Location {target_doloc} (2024)")
plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Average Fare ($)")
plt.xticks(range(0, 24))
plt.grid(alpha=0.5)
plt.show()
