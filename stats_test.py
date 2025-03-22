import os
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.stats import shapiro, mannwhitneyu, kruskal, spearmanr
from statsmodels.stats.multitest import multipletests

from data_loader import get_all_months, get_from_file


class JourneyTimeStats:
    """A class to perform statistical tests on journey time in the NYC-TLC dataset."""

    def __init__(self, trips_df, sample_size=5000):
        """
        Initialize with a preprocessed TLC trips DataFrame.

        Args:
            trips_df (pd.DataFrame): DataFrame with columns like 'duration_hours', 'PULocationID', 'DOLocationID', 'tpep_pickup_datetime'.
            sample_size (int): Number of samples for tests requiring subsampling (default: 5000).
        """
        self.trips_df = trips_df.copy()
        self.sample_size = sample_size
        # Ensure necessary temporal columns are present
        if 'duration_hours' not in self.trips_df.columns:
            self.trips_df['pickup_time'] = pd.to_datetime(self.trips_df['pickup_datetime'])
            self.trips_df['dropoff_time'] = pd.to_datetime(self.trips_df['dropoff_datetime'])
            self.trips_df['duration_hours'] = (self.trips_df['dropoff_time'] - self.trips_df['pickup_time']).dt.total_seconds() / 3600
        if 'hour' not in self.trips_df.columns:
            self.trips_df['hour'] = self.trips_df['pickup_datetime'].dt.hour
        if 'day_of_week' not in self.trips_df.columns:
            self.trips_df['day_of_week'] = self.trips_df['pickup_time'].dt.dayofweek

    def test_normality(self):
        """Test if journey times are normally distributed using Shapiro-Wilk."""
        journey_times = self.trips_df['duration_hours'].dropna()
        if len(journey_times) > self.sample_size:
            journey_times = journey_times.sample(self.sample_size, random_state=42)
        stat, p_value = shapiro(journey_times)
        print(f"Shapiro-Wilk Test for Normality:")
        print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("Result: Reject H₀ - Journey times are not normally distributed.")
        else:
            print("Result: Fail to reject H₀ - Journey times may be normally distributed.")
        return stat, p_value

    def compare_zones(self, zone_pair_1, zone_pair_2):
        """Compare journey times between two zone pairs using Mann-Whitney U Test."""
        pu1, do1 = zone_pair_1
        pu2, do2 = zone_pair_2
        zone1_trips = self.trips_df[
            (self.trips_df['PULocationID'] == pu1) & (self.trips_df['DOLocationID'] == do1)
        ]['duration_hours'].dropna()
        zone2_trips = self.trips_df[
            (self.trips_df['PULocationID'] == pu2) & (self.trips_df['DOLocationID'] == do2)
        ]['duration_hours'].dropna()
        if len(zone1_trips) == 0 or len(zone2_trips) == 0:
            print(f"Insufficient data for zones {zone_pair_1} or {zone_pair_2}")
            return None, None
        stat, p_value = mannwhitneyu(zone1_trips, zone2_trips, alternative='two-sided')
        print(f"Mann-Whitney U Test: {pu1}->{do1} vs {pu2}->{do2}")
        print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("Result: Reject H₀ - Journey times differ between these zone pairs.")
        else:
            print("Result: Fail to reject H₀ - No significant difference in journey times.")
        return stat, p_value

    def test_time_of_day(self):
        """Test journey time variation by time of day using Kruskal-Wallis Test."""
        morning = self.trips_df[self.trips_df['hour'].between(6, 10)]['duration_hours'].dropna()
        afternoon = self.trips_df[self.trips_df['hour'].between(11, 15)]['duration_hours'].dropna()
        evening = self.trips_df[self.trips_df['hour'].between(16, 20)]['duration_hours'].dropna()
        if any(len(group) == 0 for group in [morning, afternoon, evening]):
            print("Insufficient data for one or more time periods.")
            return None, None
        stat, p_value = kruskal(morning, afternoon, evening)
        print(f"Kruskal-Wallis Test for Time of Day (Morning, Afternoon, Evening):")
        print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("Result: Reject H₀ - Journey times differ across times of day.")
        else:
            print("Result: Fail to reject H₀ - No significant difference across times of day.")
        return stat, p_value

    def correlate_distance_time(self):
        """Test correlation between journey time and trip distance using Spearman’s Rank."""
        if 'trip_distance' not in self.trips_df.columns:
            print("Trip distance not available in dataset.")
            return None, None
        journey_times = self.trips_df['duration_hours'].dropna()
        trip_distances = self.trips_df['trip_distance'].dropna()
        if len(journey_times) != len(trip_distances):
            # Align indices if necessary
            aligned_df = self.trips_df[['duration_hours', 'trip_distance']].dropna()
            journey_times = aligned_df['duration_hours']
            trip_distances = aligned_df['trip_distance']
        correlation, p_value = spearmanr(journey_times, trip_distances)
        print(f"Spearman’s Rank Correlation (Journey Time vs. Trip Distance):")
        print(f"Coefficient: {correlation:.4f}, p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"Result: Reject H₀ - Significant correlation ({'positive' if correlation > 0 else 'negative'}) exists.")
        else:
            print("Result: Fail to reject H₀ - No significant correlation.")
        return correlation, p_value

    def test_day_of_week(self):
        """Test journey time differences between weekdays and weekends using Mann-Whitney U Test."""
        weekdays = self.trips_df[self.trips_df['day_of_week'] < 5]['duration_hours'].dropna()  # Mon-Fri
        weekends = self.trips_df[self.trips_df['day_of_week'] >= 5]['duration_hours'].dropna()  # Sat-Sun
        if len(weekdays) == 0 or len(weekends) == 0:
            print("Insufficient data for weekdays or weekends.")
            return None, None
        stat, p_value = mannwhitneyu(weekdays, weekends, alternative='two-sided')
        print(f"Mann-Whitney U Test (Weekdays vs. Weekends):")
        print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("Result: Reject H₀ - Journey times differ between weekdays and weekends.")
        else:
            print("Result: Fail to reject H₀ - No significant difference between weekdays and weekends.")
        return stat, p_value

    def detect_outliers(self):
        """Detect outliers in journey times using the IQR method."""
        Q1 = self.trips_df['duration_hours'].quantile(0.25)
        Q3 = self.trips_df['duration_hours'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.trips_df[
            (self.trips_df['duration_hours'] < lower_bound) | (self.trips_df['duration_hours'] > upper_bound)
        ]
        print(f"Outlier Detection (IQR Method):")
        print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}] hours")
        print(f"Number of outliers: {len(outliers)} trips")
        return outliers

    def run_all_tests(self, zone_pair_1=(132, 230), zone_pair_2=(230, 132)):
        """Run all statistical tests with default zone pairs."""
        print("\n=== Running All Statistical Tests ===\n")
        self.test_normality()
        print("\n")
        self.compare_zones(zone_pair_1, zone_pair_2)
        print("\n")
        self.test_time_of_day()
        print("\n")
        self.correlate_distance_time()
        print("\n")
        self.test_day_of_week()
        print("\n")
        self.detect_outliers()


# Example usage with standalone data loading
if __name__ == "__main__":
    # Define file paths
    filename = "clean yellow taxis 2024/cleaned_yellow_tripdata_2024-01.parquet"
    shapefile_path = "heatmaps/taxi_zones/taxi_zones.shp"

    # Load shapefile to get valid zone IDs
    temp_zones = gpd.read_file(shapefile_path)
    valid_zone_ids = set(temp_zones['LocationID'])

    # Load trip data
    _, trips_pd = get_from_file(filename)
    # Basic preprocessing

    # Initialize and run tests
    stats = JourneyTimeStats(trips_pd, sample_size=5000)
    stats.run_all_tests(zone_pair_1=(132, 230), zone_pair_2=(230, 132))  # JFK -> Times Sq. vs reverse