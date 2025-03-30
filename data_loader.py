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

def get_month(filename):
    """Load parquet data from a local file."""
    trips = pq.read_table(filename)
    trips_pd = trips.to_pandas()
    return trips_pd

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

def get_year_samples(year=2024, sample_size=100000, directory="clean yellow taxis 2024"):
    """
    Get a sample of taxi trip data for each month of a given year and return a concatenated DataFrame.

    Args:
        year (int): The year to sample (default: 2024).
        sample_size (int): Number of rows to sample per month (default: 100,000).
        directory (str): Directory containing the parquet files (default: "clean yellow taxis 2024").

    Returns:
        pd.DataFrame: Concatenated DataFrame with samples from all available months.
    """
    year_samples = []
    for month in range(1, 13):
        filename = os.path.join(directory, f"cleaned_yellow_tripdata_{year}-{month:02d}.parquet")
        try:
            # Load the month's data
            month_df = pd.read_parquet(filename)
            # Sample the data (use replace=False to avoid duplicates, adjust if needed)
            if len(month_df) > sample_size:
                sampled_df = month_df.sample(n=sample_size, random_state=42)
            else:
                sampled_df = month_df  # If fewer rows than sample_size, take all
            year_samples.append(sampled_df)
            print(f"Sampled {len(sampled_df)} rows from {filename}")
        except FileNotFoundError:
            print(f"Warning: File not found - {filename}. Skipping month {month:02d}.")
        except Exception as e:
            print(f"Error processing {filename}: {e}. Skipping month {month:02d}.")

    # Concatenate all samples into a single DataFrame
    if year_samples:
        return pd.concat(year_samples, ignore_index=True)
    else:
        print("No data sampled for the year.")
        return pd.DataFrame()


def get_file_size(filename):
    """Get the number of rows in a parquet file."""
    try:
        df = pd.read_parquet(filename)
        return len(df)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return 0


def sample_all_cab_types_proportionally(year=2024, total_sample_size=5000000, base_dir=""):
    """
    Sample all four cab types (yellow, green, FHV, HVFHV) for a given year proportionally to their data sizes.

    Args:
        year (int): The year to sample (default: 2024).
        total_sample_size (int): Total number of rows to sample across all cab types (default: 100,000).
        base_dir (str): Base directory containing subdirectories for each cab type (default: "taxi_data").

    Returns:
        pd.DataFrame: Concatenated DataFrame with proportional samples from all cab types.
    """
    # Define cab types and their corresponding directories
    cab_types = {
        "yellow": f"clean yellow taxis {year}",
        "green": "clean green taxis",
        "fhv": "clean fhv",
        "hvfhv": "clean hvfhvs"
    }

    # Initialise dictionaries to store file sizes and samples
    total_sizes = {}
    all_samples = []

    for cab_type, directory in cab_types.items():
        if not os.path.isdir(directory):
            print(f"Warning: Directory '{directory}' not found. Skipping {cab_type}.")
            total_sizes[cab_type] = 0
            continue

        total_size = 0
        for month in range(1, 13):
            if cab_type != 'hvfhv':
                filename = os.path.join(directory, f"cleaned_{cab_type}_tripdata_{year}-{month:02d}.parquet")
            else:
                filename = os.path.join(directory, f"fhvhv_green_tripdata_{year}-{month:02d}.parquet")
            if os.path.isfile(filename):
                month_size = get_file_size(filename)
                total_size += month_size
                print(f"{cab_type} - {filename}: {month_size} rows")
            else:
                print(f"Warning: File not found - {filename}")

        total_sizes[cab_type] = total_size
        print(f"Total {cab_type} trips for {year}: {total_size}")


    overall_total = sum(total_sizes.values())
    if overall_total == 0:
        print("No data found across all cab types.")
        return pd.DataFrame()

    sample_sizes = {}
    for cab_type, size in total_sizes.items():
        if size > 0:
            proportion = size / overall_total
            sample_size = int(proportion * total_sample_size)
            sample_sizes[cab_type] = max(sample_size, 1)  # Ensure at least 1 sample
        else:
            sample_sizes[cab_type] = 0
        print(f"{cab_type} sample size: {sample_sizes[cab_type]} ({size / overall_total:.2%} of total)")

    for cab_type, directory in cab_types.items():
        if sample_sizes[cab_type] == 0:
            continue

        monthly_samples = []
        monthly_sizes = {}
        total_cab_size = total_sizes[cab_type]

        # Calculates size per month for proportional sampling within the cab type
        for month in range(1, 13):
            if cab_type != 'fhvhv':
                filename = os.path.join(directory, f"cleaned_{cab_type}_tripdata_{year}-{month:02d}.parquet")
            else:
                filename = os.path.join(directory, f"fhvhv_green_tripdata_{year}-{month:02d}.parquet")
            if os.path.isfile(filename):
                month_size = get_file_size(filename)
                monthly_sizes[month] = month_size
            else:
                monthly_sizes[month] = 0

        total_monthly_size = sum(monthly_sizes.values())
        if total_monthly_size == 0:
            continue

        # Allocate sample size across months
        for month, month_size in monthly_sizes.items():
            if month_size > 0:
                month_proportion = month_size / total_monthly_size
                month_sample_size = int(month_proportion * sample_sizes[cab_type])
                if month_sample_size > 0:
                    filename = os.path.join(directory, f"cleaned_{cab_type}_tripdata_{year}-{month:02d}.parquet")
                    month_df = pd.read_parquet(filename)
                    if len(month_df) > month_sample_size:
                        sampled_df = month_df.sample(n=month_sample_size, random_state=42)
                    else:
                        sampled_df = month_df
                    monthly_samples.append(sampled_df)
                    print(f"Sampled {len(sampled_df)} rows from {filename}")

        # Concatenate monthly samples for this cab type
        if monthly_samples:
            cab_sample = pd.concat(monthly_samples, ignore_index=True)
            all_samples.append(cab_sample)
            print(f"Total {cab_type} sampled: {len(cab_sample)} rows")

    # Combines all samples
    if all_samples:
        combined_df = pd.concat(all_samples, ignore_index=True)
        # Adjust to exact total_sample_size if needed
        if len(combined_df) > total_sample_size:
            combined_df = combined_df.sample(n=total_sample_size, random_state=42)
        print(f"Final combined sample size: {len(combined_df)} rows")
        return combined_df
    else:
        print("No data sampled across any cab types.")
        return pd.DataFrame()


# Test usage
if __name__ == "__main__":
    sampled_data = sample_all_cab_types_proportionally()
    print(f"Total rows sampled: {len(sampled_data)}")
    if not sampled_data.empty:
        print(sampled_data.columns)
        print(sampled_data.head())