import pandas as pd
import dask.dataframe as dd
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


data_dir = r"C:\Users\44746\ADS taxi dataset\2024 yellow taxi datasets" 


yellow_pattern = f"{data_dir}/yellow_tripdata_*.parquet"

# Only load these columns to minimize memory usage
necessary_columns = [
    "tpep_pickup_datetime",  # for time-based grouping
    "VendorID"               # used to count trips
]

# Date range for 2024 only
start_date = "2024-01-01"
end_date = "2025-01-01"  

# ------------------------------------------------------------
# 2. Initialize Aggregation Containers
# ------------------------------------------------------------
daily_trips_dict = {}       # for daily trip counts
trips_by_hour_dict = {}     # for hour-of-day trip counts
trips_by_dayofweek_dict = {}  # for day-of-week trip counts

# ------------------------------------------------------------
# 3. Process Files One-by-One
# ------------------------------------------------------------
files = sorted(glob.glob(yellow_pattern))
print(f"Found {len(files)} files matching the pattern.")

for file in files:
    print(f"Processing file: {file}")
    
    # Read one file using Dask with only the necessary columns
    df = dd.read_parquet(file, columns=necessary_columns, engine="pyarrow")
    
    # Convert pickup datetime to a proper datetime type
    df["tpep_pickup_datetime"] = dd.to_datetime(df["tpep_pickup_datetime"])
    
    # Filter rows to keep only 2024 data
    # This avoids aggregating unnecessary rows
    df = df[
        (df["tpep_pickup_datetime"] >= start_date) &
        (df["tpep_pickup_datetime"] < end_date)
    ]
    
    # If this file has no 2024 data, skip it
    if df.shape[0].compute() == 0:
        continue
    
    # Create derived columns (hour, day_of_week, pickup_date)
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    df["day_of_week"] = df["tpep_pickup_datetime"].dt.day_name()
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date
    
    # --- Aggregation A: Daily Trips ---
    daily_trips = df.groupby("pickup_date")["VendorID"].count().compute()
    for date, count in daily_trips.items():
        daily_trips_dict[date] = daily_trips_dict.get(date, 0) + count
    
    # --- Aggregation B: Trips by Hour of Day ---
    hour_trips = df.groupby("hour")["VendorID"].count().compute()
    for hr, count in hour_trips.items():
        trips_by_hour_dict[hr] = trips_by_hour_dict.get(hr, 0) + count
    
    # --- Aggregation C: Trips by Day of Week ---
    dayofweek_trips = df.groupby("day_of_week")["VendorID"].count().compute()
    for dow, count in dayofweek_trips.items():
        trips_by_dayofweek_dict[dow] = trips_by_dayofweek_dict.get(dow, 0) + count


# Daily trips: Convert to a Series with DateTimeIndex
daily_trips_series = pd.Series(daily_trips_dict)
daily_trips_series.index = pd.to_datetime(daily_trips_series.index)
daily_trips_series.sort_index(inplace=True)

# Trips by hour: Convert to Series (hour is just an integer 0-23)
trips_by_hour_series = pd.Series(trips_by_hour_dict).sort_index()

# Trips by day of week: Convert to Series
trips_by_dayofweek_series = pd.Series(trips_by_dayofweek_dict)

# For a nice bar plot, reorder to Monday-Sunday
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# Any missing day gets NaN; fill with 0
trips_by_dayofweek_series = trips_by_dayofweek_series.reindex(day_order, fill_value=0)

# ------------------------------------------------------------
# 5. Plot the Aggregated Results
# ------------------------------------------------------------

# A) Daily Trips Over Time (2024)
plt.figure(figsize=(12, 6))
daily_trips_series.plot(color="purple")
plt.title("Daily Trips Over Time (2024)")
plt.xlabel("Date")
plt.ylabel("Number of Trips")
plt.grid(alpha=0.5)
# Configure the x-axis to show monthly ticks
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())          # tick at the start of each month
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))   # abbreviated month names (Jan, Feb, etc.)
plt.show()

# B) Trips by Hour of the Day
plt.figure(figsize=(10, 6))
trips_by_hour_series.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Number of Trips by Hour of the Day (2024)")
plt.xlabel("Hour (0-23)")
plt.ylabel("Number of Trips")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.5)
plt.show()

# C) Trips by Day of the Week
plt.figure(figsize=(10, 6))
trips_by_dayofweek_series.plot(kind="bar", color="lightgreen", edgecolor="black")
plt.title("Number of Trips by Day of the Week (2024)")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Trips")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.5)
plt.show()


