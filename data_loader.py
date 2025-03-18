import os
import pyarrow.parquet as pq
import pandas as pd
import requests
from io import BytesIO

def get_from_url(url):
    """Fetch and load parquet data from a URL."""
    response = requests.get(url)
    response.raise_for_status()  # Checks for HTTP errors

    trips = pq.read_table(BytesIO(response.content))
    trips_pd = trips.to_pandas()

    return trips, trips_pd

def get_from_file(filename):
    """Load parquet data from a local file."""
    trips = pq.read_table(filename)
    trips_pd = trips.to_pandas()

    return trips, trips_pd

def get_all_months(directory="clean yellow taxis 2024", year=2024):
    """
    Load and concatenate all months of TLC data for a given year from a directory.

    Args:
        directory (str): Path to the directory containing parquet files (default: 'clean yellow taxis 2024').
        year (int): Year of the data to load (default: 2024).

    Returns:
        tuple: (pyarrow.Table, pandas.DataFrame) containing concatenated data for all months.
    """
    # Ensure directory exists
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found.")

    # Expected file pattern: cleaned_yellow_tripdata_2024-MM.parquet
    all_trips_pd = []
    for month in range(1, 13):  # Months 01 to 12
        filename = os.path.join(directory, f"cleaned_yellow_tripdata_{year}-{month:02d}.parquet")
        if os.path.isfile(filename):
            try:
                _, trips_pd = get_from_file(filename)
                all_trips_pd.append(trips_pd)
                print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"File not found: {filename}")

    if not all_trips_pd:
        raise ValueError(f"No valid parquet files found in '{directory}' for year {year}.")

    # Concatenate all months into a single DataFrame
    combined_trips_pd = pd.concat(all_trips_pd, ignore_index=True)

    print(f"Combined data: {len(combined_trips_pd)} total trips across {len(all_trips_pd)} months.")
    return combined_trips_pd

def yield_months(directory="clean yellow taxis 2024", year=2024):
    """
    Generator to yield one month's TLC data at a time from a directory.

    Args:
        directory (str): Path to the directory containing parquet files (default: 'clean yellow taxis 2024').
        year (int): Year of the data to load (default: 2024).

    Yields:
        pd.DataFrame: DataFrame for each month's data.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found.")

    for month in range(1, 13):  # Months 01 to 12
        filename = os.path.join(directory, f"cleaned_yellow_tripdata_{year}-{month:02d}.parquet")
        if os.path.isfile(filename):
            try:
                _, trips_pd = get_from_file(filename)
                print(f"Yielded: {filename} ({len(trips_pd)} rows)")
                yield trips_pd
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"File not found: {filename}")