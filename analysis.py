import pandas as pd
from data_loader import get_year_samples
import geopandas as gpd

# Change the filename and URL strings as applicable

# Copied URL from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
filename = "yellow_tripdata_2024-01.parquet"

# Set pandas display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Use full width

# Load data (choose one method)
# trips, df = get_from_url(url)  # Uncomment to use URL
df = get_year_samples()

# Print first few rows
print(df.head())


def calculate_trip_percentages(trips_df, num_routes=10):
    """
    Calculate the percentage of total trips for the top N routes.

    Args:
        trips_df (pd.DataFrame): DataFrame containing trip data with PULocationID and DOLocationID
        num_routes (int): Number of top routes to return (default: 10)

    Returns:
        pd.DataFrame: DataFrame with route info and percentage of total trips
    """
    # Count total number of trips in the dataset
    total_trips = len(trips_df)

    # Group by route (bidirectional) and count trips
    directional_counts = trips_df.groupby(['PULocationID', 'DOLocationID']).size().to_dict()
    edges = pd.DataFrame([
        {'PULocationID': pu, 'DOLocationID': do, 'Forward_Count': count}
        for (pu, do), count in directional_counts.items()
    ])
    edges['Reverse_Count'] = edges.apply(
        lambda row: directional_counts.get((row['DOLocationID'], row['PULocationID']), 0), axis=1)
    edges['Pair_Key'] = edges.apply(lambda row: tuple(sorted([row['PULocationID'], row['DOLocationID']])), axis=1)

    # Aggregate bidirectional counts
    route_counts = edges.groupby('Pair_Key').agg({
        'Forward_Count': 'sum',
        'Reverse_Count': 'sum',
        'PULocationID': 'first',
        'DOLocationID': 'first'
    }).reset_index()
    route_counts['Total_Trips'] = route_counts['Forward_Count'] + route_counts['Reverse_Count']

    # Determine dominant direction
    route_counts['Dominant_PU'] = route_counts.apply(
        lambda row: row['PULocationID'] if row['Forward_Count'] >= row['Reverse_Count'] else row['DOLocationID'],
        axis=1)
    route_counts['Dominant_DO'] = route_counts.apply(
        lambda row: row['DOLocationID'] if row['Forward_Count'] >= row['Reverse_Count'] else row['PULocationID'],
        axis=1)
    route_counts['Dominant_Count'] = route_counts[['Forward_Count', 'Reverse_Count']].max(axis=1)
    route_counts['Reverse_Count'] = route_counts[['Forward_Count', 'Reverse_Count']].min(axis=1)

    # Get top N routes
    top_routes = route_counts.nlargest(num_routes, 'Total_Trips')
    top_routes = top_routes[['Dominant_PU', 'Dominant_DO', 'Total_Trips', 'Forward_Count', 'Reverse_Count']]
    top_routes.columns = ['PULocationID', 'DOLocationID', 'Total_Trips', 'Forward_Count', 'Reverse_Count']

    # Calculate percentage of total trips
    top_routes['Percentage'] = (top_routes['Total_Trips'] / total_trips) * 100

    # Load zone names for readability (assuming taxi_zones is available)
    temp_zones = gpd.read_file("maps/taxi_zones/taxi_zones.shp")
    zone_names = temp_zones.set_index('LocationID')['zone'].to_dict()

    # Print results
    print(f"\nTop {num_routes} Routes by Percentage of Total Trips (Total Trips: {total_trips}):")
    print("---------------------------------------------------------------")
    for i, row in top_routes.iterrows():
        start_id = row['PULocationID']
        end_id = row['DOLocationID']
        start_name = zone_names.get(start_id, f"Zone {start_id}")
        end_name = zone_names.get(end_id, f"Zone {end_id}")
        total_trips_route = int(row['Total_Trips'])
        forward_trips = int(row['Forward_Count'])
        reverse_trips = int(row['Reverse_Count'])
        percentage = round(row['Percentage'], 2)
        print(f"{i + 1}. {start_name} <-> {end_name}: {total_trips_route} trips "
              f"({forward_trips} from.pm to {end_name}, {reverse_trips} reverse), "
              f"{percentage}% of total trips")

    return top_routes

calculate_trip_percentages(df, num_routes=50)