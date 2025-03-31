import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

from data_loader import sample_all_cab_types_proportionally


def load_game_schedule(games_csv_path):
    """Loads and prepares the Yankees game schedule."""
    games_df = pd.read_csv(games_csv_path)
    games_df['game_datetime'] = pd.to_datetime(games_df['Date'] + ' ' + games_df['Start Time (EDT)'],
                                               format='%Y-%m-%d %I:%M %p').astype('datetime64[ns]')
    return games_df


def prepare_trips_data(trips_df, stadium_zone, window_hours_before, window_hours_after, games_df):
    """Prepares taxi trips data and adds flags to game-related trips."""
    # Input validation
    required_cols = ['pickup_datetime', 'dropoff_datetime', 'PULocationID', 'DOLocationID']
    if not all(col in trips_df.columns for col in required_cols):
        raise ValueError(f"trips_df must contain {required_cols}")

    # Prepare trips
    trips_df = trips_df.copy()
    trips_df['pickup_time'] = pd.to_datetime(trips_df['pickup_datetime']).astype('datetime64[ns]')
    trips_df['dropoff_time'] = pd.to_datetime(trips_df['dropoff_datetime']).astype('datetime64[ns]')
    trips_df['duration_hours'] = (trips_df['dropoff_time'] - trips_df['pickup_time']).dt.total_seconds() / 3600
    trips_df['date'] = trips_df['pickup_time'].dt.date

    # Filter trips to/from stadium
    stadium_trips = trips_df[
        (trips_df['PULocationID'] == stadium_zone) | (trips_df['DOLocationID'] == stadium_zone)
        ].copy()
    if stadium_trips.empty:
        print(f"No trips found to/from Yankee Stadium (zone {stadium_zone}).")
        return pd.DataFrame()

    # Add game window columns
    games_df['window_start'] = games_df['game_datetime'] - pd.Timedelta(hours=window_hours_before)
    games_df['window_end'] = games_df['game_datetime'] + pd.Timedelta(hours=window_hours_after)
    game_dates = games_df['game_datetime'].dt.date.unique()

    # Merge with game schedule
    stadium_trips = stadium_trips.sort_values('pickup_time')
    games_df = games_df.sort_values('window_start')
    merged_trips = pd.merge_asof(
        stadium_trips,
        games_df[['window_start', 'window_end', 'game_datetime']],
        left_on='pickup_time',
        right_on='window_start',
        direction='backward'
    )
    merged_trips['during_game'] = (
            (merged_trips['pickup_time'] >= merged_trips['window_start']) &
            (merged_trips['pickup_time'] <= merged_trips['window_end'])
    )
    merged_trips['is_game_day'] = merged_trips['date'].isin(game_dates)
    merged_trips = merged_trips.drop(columns=['window_start', 'window_end'])

    # Log basic stats
    print(f"Total trips to/from stadium: {len(merged_trips)}")
    print(f"Trips during game windows: {merged_trips['during_game'].sum()}")
    print(f"Trips on game days: {merged_trips['is_game_day'].sum()}")

    return merged_trips


def compute_distances_and_speeds(trips_df, taxi_zones):
    """Compute trip distances and speeds using zone centroids."""
    zone_centroids = taxi_zones.set_index('LocationID')['geometry'].centroid.to_dict()

    trips_df['pu_centroid'] = trips_df['PULocationID'].map(zone_centroids)
    trips_df['do_centroid'] = trips_df['DOLocationID'].map(zone_centroids)
    trips_df['distance_feet'] = trips_df.apply(
        lambda row: row['pu_centroid'].distance(row['do_centroid'])
        if pd.notnull(row['pu_centroid']) and pd.notnull(row['do_centroid']) else 0,
        axis=1
    )
    trips_df['distance_miles'] = trips_df['distance_feet'] / 5280
    trips_df['speed_mph'] = np.where(
        trips_df['duration_hours'] > 0,
        trips_df['distance_miles'] / trips_df['duration_hours'],
        0
    )
    return trips_df


def filter_valid_trips(trips_df, stadium_zone):
    """Filter out invalid trips and add route metadata."""
    valid_trips = trips_df[
        (trips_df['duration_hours'] > 0) &
        (trips_df['distance_miles'] > 0)
        ].copy()
    valid_trips['route'] = valid_trips['PULocationID'].astype(str) + '-' + valid_trips['DOLocationID'].astype(str)
    valid_trips['direction'] = np.where(
        valid_trips['DOLocationID'] == stadium_zone, 'To_Stadium', 'From_Stadium'
    )
    return valid_trips


def compare_game_vs_non_game(trips_df):
    """Split trips into game and non-game sets with common routes."""
    game_trips = trips_df[trips_df['during_game']].copy()
    non_game_trips = trips_df[~trips_df['is_game_day']].copy()

    common_routes = set(game_trips['route']).intersection(set(non_game_trips['route']))
    game_trips = game_trips[game_trips['route'].isin(common_routes)]
    non_game_trips = non_game_trips[non_game_trips['route'].isin(common_routes)]

    return game_trips, non_game_trips


def perform_statistical_tests(game_trips, non_game_trips):
    """Performs t-test and Mann-Whitney U test on trip durations."""
    game_durations = game_trips['duration_hours'].dropna()
    non_game_durations = non_game_trips['duration_hours'].dropna()

    t_stat, p_value_t = ttest_ind(game_durations, non_game_durations, equal_var=False)
    mw_stat, p_value_mw = mannwhitneyu(game_durations, non_game_durations, alternative='two-sided')

    return {
        't_statistic': round(t_stat, 4),
        'p_value_t': round(p_value_t, 4),
        'mw_statistic': round(mw_stat, 4),
        'p_value_mw': round(p_value_mw, 4)
    }


def analyse_routes(trips_df):
    """Analyses journey times by route for game vs. non-game trips."""
    route_analysis = trips_df.groupby(['route', 'during_game'])['duration_hours'].agg(['mean', 'count']).reset_index()
    route_pivot = route_analysis.pivot_table(
        index='route', columns='during_game', values='mean', fill_value=0
    ).rename(columns={False: 'non_game_time', True: 'game_time'}) * 60
    route_pivot['time_difference'] = route_pivot['game_time'] - route_pivot['non_game_time']
    return route_pivot


def print_analysis_summary(results, route_pivot, trips_df):
    """Prints a concise summary of the analysis."""
    overall = results['overall']
    print("\nOverall Journey Time Impact:")
    print(f"Game Time: {overall['game_time_mean_minutes']} min ({overall['game_trip_count']} trips)")
    print(f"Non-Game Time: {overall['non_game_time_mean_minutes']} min ({overall['non_game_trip_count']} trips)")
    print(f"Time Difference: {overall['time_difference_minutes']} min ({overall['percent_time_change']}%)")
    print(f"T-test: t={overall['t_statistic']}, p={overall['p_value_t']}")
    print(f"Mann-Whitney U: U={overall['mw_statistic']}, p={overall['p_value_mw']}")

    print("\nRoute Analysis (Top 5 by trip count):")
    route_counts = trips_df['route'].value_counts()
    top_routes = route_pivot.loc[route_counts.head(5).index]
    print(top_routes[['game_time', 'non_game_time', 'time_difference']])

    print("\nConclusion:")
    if overall['p_value_t'] < 0.05:
        print(f"Significant effect on journey times (t-test p={overall['p_value_t']:.4f}).")
    else:
        print(f"No significant effect on journey times (t-test p={overall['p_value_t']:.4f}).")


def analyse_yankees_game_impact(
        trips_df,
        games_csv_path,
        taxi_zones_path="taxi_zones/taxi_zones.shp",
        stadium_zone=247,
        window_hours_before=2,
        window_hours_after=2
):
    """Analyses journey times around Yankee games vs. average to/from same locations."""
    # Load data
    games_df = load_game_schedule(games_csv_path)
    taxi_zones = gpd.read_file(taxi_zones_path)

    # Prepare and process trips
    stadium_trips = prepare_trips_data(trips_df, stadium_zone, window_hours_before, window_hours_after, games_df)
    if stadium_trips.empty:
        return {"error": f"No relevant trips found for zone {stadium_zone}"}

    stadium_trips = compute_distances_and_speeds(stadium_trips, taxi_zones)
    valid_trips = filter_valid_trips(stadium_trips, stadium_zone)
    game_trips, non_game_trips = compare_game_vs_non_game(valid_trips)

    # Statistical analysis
    stats = perform_statistical_tests(game_trips, non_game_trips)
    route_pivot = analyse_routes(valid_trips)

    # Compile overall results
    game_time_mean = game_trips['duration_hours'].mean() * 60
    non_game_time_mean = non_game_trips['duration_hours'].mean() * 60
    results = {
        "overall": {
            "game_time_mean_minutes": round(game_time_mean, 2),
            "non_game_time_mean_minutes": round(non_game_time_mean, 2),
            "time_difference_minutes": round(game_time_mean - non_game_time_mean, 2),
            "percent_time_change": round((game_time_mean - non_game_time_mean) / non_game_time_mean * 100,
                                         2) if non_game_time_mean > 0 else 0,
            "game_trip_count": len(game_trips),
            "non_game_trip_count": len(non_game_trips),
            **stats
        },
        "route_analysis": route_pivot.reset_index().to_dict(orient='records')
    }

    print_analysis_summary(results, route_pivot, valid_trips)
    return results


if __name__ == "__main__":
    year_data = sample_all_cab_types_proportionally()
    games_csv_path = "extra data/mlb-2024-yankees.csv"
    analysis_results = analyse_yankees_game_impact(
        trips_df=year_data,
        games_csv_path=games_csv_path,
        taxi_zones_path="maps/taxi_zones/taxi_zones.shp",
        stadium_zone=247,
        window_hours_before=1,
        window_hours_after=1
    )
