import os
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import AntPath
from shapely.geometry import shape, mapping
from pyproj import Transformer
import numpy as np
from data_loader import get_from_file


class SpeedRoutesVisualizer:
    """A class to visualize top N taxi routes in NYC with edges colored by speed or journey time."""

    def __init__(self, parquet_file, shapefile_path="taxi_zones/taxi_zones.shp", output_dir="output"):
        """
        Initialize the SpeedRoutesVisualizer with file paths and settings.

        Args:
            parquet_file (str): Path to the TLC parquet file (e.g., 'yellow_tripdata_2024-01.parquet').
            shapefile_path (str): Path to the taxi zones shapefile (default: 'taxi_zones/taxi_zones.shp').
            output_dir (str): Directory to save the output HTML file (default: 'heatmaps').
        """
        self.parquet_file = parquet_file
        self.shapefile_path = shapefile_path
        self.output_dir = output_dir
        self.transformer = Transformer.from_crs("epsg:2263", "epsg:4326", always_xy=True)
        self.distance_transformer = Transformer.from_crs("epsg:4326", "epsg:2263",
                                                         always_xy=True)  # For distance in feet
        self.nyc_center = [40.7128, -74.0060]  # NYC coordinates (lat, lon)

    def load_and_process_data(self, valid_zone_ids):
        """Load trip data and compute location counts and durations, filtering by valid zones."""
        _, trips_pd = get_from_file(self.parquet_file)
        # Calculate trip duration in hours
        trips_pd['pickup_time'] = pd.to_datetime(trips_pd['tpep_pickup_datetime'])
        trips_pd['dropoff_time'] = pd.to_datetime(trips_pd['tpep_dropoff_datetime'])
        trips_pd['duration_hours'] = (trips_pd['dropoff_time'] - trips_pd['pickup_time']).dt.total_seconds() / 3600
        # Filter out invalid durations and zones not in shapefile
        trips_pd = trips_pd[
            (trips_pd['duration_hours'] > 0) &
            (trips_pd['duration_hours'] < 24) &
            (trips_pd['PULocationID'].isin(valid_zone_ids)) &
            (trips_pd['DOLocationID'].isin(valid_zone_ids))
            ]
        pickup_counts = self._count_trips(trips_pd, 'PULocationID', 'Pickup_Count')
        dropoff_counts = self._count_trips(trips_pd, 'DOLocationID', 'Dropoff_Count')
        location_counts = self._merge_counts(pickup_counts, dropoff_counts)
        return trips_pd, location_counts

    def _count_trips(self, df, column, count_name):
        """Count trips for a given location column and return as a DataFrame."""
        counts = df[column].value_counts().reset_index()
        counts.columns = ['LocationID', count_name]
        return counts

    def _merge_counts(self, pickup_counts, dropoff_counts):
        """Merge pickup and dropoff counts into a single DataFrame."""
        location_counts = pd.merge(pickup_counts, dropoff_counts, on='LocationID', how='outer').fillna(0)
        location_counts['Total_Trips'] = location_counts['Pickup_Count'] + location_counts['Dropoff_Count']
        return location_counts

    def load_and_transform_zones(self, location_counts):
        """Load taxi zones shapefile, merge with counts, and transform to EPSG:4326."""
        taxi_zones = gpd.read_file(self.shapefile_path)
        taxi_zones = taxi_zones.merge(location_counts, on='LocationID', how='left')
        taxi_zones['Total_Trips'] = taxi_zones['Total_Trips'].fillna(0)
        print("Original CRS:", taxi_zones.crs)
        print("Any geometries with Z:", taxi_zones.geometry.has_z.any())

        # Compute centroids in the original projected CRS (EPSG:2263) for distance
        taxi_zones['centroid_proj'] = taxi_zones.geometry.centroid
        # Transform to EPSG:4326 for mapping
        taxi_zones['centroid'] = taxi_zones['centroid_proj'].apply(self._transform_geometry)
        taxi_zones.geometry = taxi_zones.geometry.apply(self._transform_geometry)
        taxi_zones.crs = "epsg:4326"
        print("New CRS:", taxi_zones.crs)

        return taxi_zones

    def _transform_geometry(self, geom):
        """Transform a geometry from EPSG:2263 to EPSG:4326."""
        if geom.is_empty:
            return geom
        geojson = mapping(geom)
        if geom.geom_type == 'Polygon':
            new_coords = [self.transformer.transform(x, y) for x, y in geojson['coordinates'][0]]
            geojson['coordinates'] = [new_coords]
        elif geom.geom_type == 'MultiPolygon':
            new_coords = [[self.transformer.transform(x, y) for x, y in ring]
                          for ring in geojson['coordinates'][0]]
            geojson['coordinates'] = [new_coords]
        elif geom.geom_type == 'Point':
            new_coords = self.transformer.transform(geom.x, geom.y)
            return shape({'type': 'Point', 'coordinates': new_coords})
        return shape(geojson)

    def get_top_routes(self, trips_pd, taxi_zones, num_routes=10, metric="speed"):
        """Identify and print the top N routes with speed or journey time."""
        # Compute directional trip counts and average durations
        directional_counts = trips_pd.groupby(['PULocationID', 'DOLocationID']).size().to_dict()
        directional_durations = trips_pd.groupby(['PULocationID', 'DOLocationID'])['duration_hours'].mean().to_dict()

        # Create a DataFrame with all pairs and their counts/durations
        edges = pd.DataFrame([
            {'PULocationID': pu, 'DOLocationID': do, 'Forward_Count': count,
             'Forward_Duration': directional_durations.get((pu, do), 0)}
            for (pu, do), count in directional_counts.items()
        ])

        # Add reverse counts and durations
        edges['Reverse_Count'] = edges.apply(
            lambda row: directional_counts.get((row['DOLocationID'], row['PULocationID']), 0), axis=1)
        edges['Reverse_Duration'] = edges.apply(
            lambda row: directional_durations.get((row['DOLocationID'], row['PULocationID']), 0), axis=1)

        # Create a unique key for each pair (sorted to combine A-B and B-A)
        edges['Pair_Key'] = edges.apply(lambda row: tuple(sorted([row['PULocationID'], row['DOLocationID']])), axis=1)

        # Group by pair to sum total trips
        route_counts = edges.groupby('Pair_Key').agg({
            'Forward_Count': 'sum',
            'Reverse_Count': 'sum',
            'PULocationID': 'first',
            'DOLocationID': 'first',
            'Forward_Duration': 'first',
            'Reverse_Duration': 'first'
        }).reset_index()

        # Manually calculate total trips (Forward + Reverse)
        route_counts['Total_Trips'] = route_counts['Forward_Count'] + route_counts['Reverse_Count']

        # Determine dominant direction based on trip count
        route_counts['Dominant_PU'] = route_counts.apply(
            lambda row: row['PULocationID'] if row['Forward_Count'] >= row['Reverse_Count'] else row['DOLocationID'],
            axis=1)
        route_counts['Dominant_DO'] = route_counts.apply(
            lambda row: row['DOLocationID'] if row['Forward_Count'] >= row['Reverse_Count'] else row['PULocationID'],
            axis=1)
        route_counts['Dominant_Count'] = route_counts[['Forward_Count', 'Reverse_Count']].max(axis=1)
        route_counts['Reverse_Count'] = route_counts[['Forward_Count', 'Reverse_Count']].min(axis=1)
        route_counts['Dominant_Duration'] = route_counts.apply(
            lambda row: row['Forward_Duration'] if row['Forward_Count'] >= row['Reverse_Count'] else row[
                'Reverse_Duration'], axis=1)

        # Calculate distance and metric (speed or time)
        zone_centroids_proj = taxi_zones.set_index('LocationID')['centroid_proj'].to_dict()
        route_counts['Distance_Feet'] = route_counts.apply(
            lambda row: zone_centroids_proj[row['Dominant_PU']].distance(zone_centroids_proj[row['Dominant_DO']]),
            axis=1)
        route_counts['Distance_Miles'] = route_counts['Distance_Feet'] / 5280  # Convert feet to miles

        if metric == "speed":
            route_counts['Metric_Value'] = route_counts['Distance_Miles'] / route_counts[
                'Dominant_Duration']  # Speed in mph
            route_counts['Metric_Value'] = route_counts['Metric_Value'].replace([np.inf, -np.inf], np.nan).fillna(0)
            metric_label = "Speed"
            metric_unit = "mph"
        elif metric == "time":
            route_counts['Metric_Value'] = route_counts['Dominant_Duration']  # Time in hours
            metric_label = "Journey Time"
            metric_unit = "hours"
        else:
            raise ValueError("Metric must be 'speed' or 'time'")

        # Get top N routes by total trips
        top_routes = route_counts.nlargest(num_routes, 'Total_Trips')
        top_routes = top_routes[
            ['Dominant_PU', 'Dominant_DO', 'Total_Trips', 'Forward_Count', 'Reverse_Count', 'Metric_Value']]
        top_routes.columns = ['PULocationID', 'DOLocationID', 'Total_Trips', 'Forward_Count', 'Reverse_Count',
                              'Metric_Value']

        # Map LocationID to zone names
        zone_names = taxi_zones.set_index('LocationID')['zone'].to_dict()

        # Print the top N routes with the chosen metric
        print(f"\nTop {num_routes} Most Popular Routes (Bidirectional Combined) with {metric_label}:")
        print("--------------------------------------------------------------------")
        for i, row in top_routes.iterrows():
            start_id = row['PULocationID']
            end_id = row['DOLocationID']
            start_name = zone_names.get(start_id, f"Zone {start_id}")
            end_name = zone_names.get(end_id, f"Zone {end_id}")
            total_trips = int(row['Total_Trips'])
            forward_trips = int(row['Forward_Count'])
            reverse_trips = int(row['Reverse_Count'])
            metric_value = round(row['Metric_Value'], 2)
            print(f"{i + 1}. {start_name} <-> {end_name}: {total_trips} total trips "
                  f"({forward_trips} from {start_name} to {end_name}, {reverse_trips} reverse), "
                  f"{metric_label}: {metric_value} {metric_unit}")

        return top_routes, metric

    def create_routes_map(self, taxi_zones, top_routes, metric="speed"):
        """Create a Folium map with edges colored by the chosen metric (speed or time)."""
        m = folium.Map(location=self.nyc_center, zoom_start=11, tiles="cartodbdark_matter")

        # Add taxi zone boundaries as a light background
        folium.GeoJson(
            taxi_zones.drop(columns=['centroid', 'centroid_proj']),
            style_function=lambda x: {'fillOpacity': 0.1, 'weight': 1, 'color': 'gray'}
        ).add_to(m)

        # Get unique zones involved in top routes
        top_zone_ids = set(top_routes['PULocationID']).union(top_routes['DOLocationID'])
        top_zones = taxi_zones[taxi_zones['LocationID'].isin(top_zone_ids)]

        # Map LocationID to centroids and zone names
        zone_centroids = top_zones.set_index('LocationID')['centroid'].to_dict()
        zone_names = top_zones.set_index('LocationID')['zone'].to_dict()

        # Determine metric range and label
        max_value = top_routes['Metric_Value'].max()
        min_value = top_routes['Metric_Value'].min()
        value_range = max_value - min_value if max_value > min_value else 1  # Avoid division by zero
        if metric == "speed":
            metric_label = "Speed"
            metric_unit = "mph"
            # Higher speed = green, lower speed = red
            norm_direction = 1  # Direct: low to high
        elif metric == "time":
            metric_label = "Journey Time"
            metric_unit = "hours"
            # Shorter time = green, longer time = red (invert)
            norm_direction = -1  # Inverse: high to low
        else:
            raise ValueError("Metric must be 'speed' or 'time'")

        # Add edges with metric-based coloring
        for _, edge in top_routes.iterrows():
            start_id = edge['PULocationID']
            end_id = edge['DOLocationID']
            if start_id in zone_centroids and end_id in zone_centroids:
                start = [zone_centroids[start_id].y, zone_centroids[start_id].x]
                end = [zone_centroids[end_id].y, zone_centroids[end_id].x]
                metric_value = edge['Metric_Value']
                # Normalize value to 0-1 range
                norm = (metric_value - min_value) / value_range if value_range > 0 else 0.5
                if norm_direction == -1:  # Invert for time
                    norm = 1 - norm
                # Interpolate from red (low) to blue (medium) to green (high)
                if norm < 0.5:
                    r = 255
                    g = int(255 * (norm * 2))  # 0 to 255
                    b = 0
                else:
                    r = int(255 * (1 - (norm - 0.5) * 2))  # 255 to 0
                    g = 255
                    b = int(255 * ((norm - 0.5) * 2))  # 0 to 255
                color = f'rgb({r}, {g}, {b})'
                weight = 5  # Fixed weight for visibility
                start_name = zone_names.get(start_id, f"Zone {start_id}")
                end_name = zone_names.get(end_id, f"Zone {end_id}")
                popup_text = (f"{start_name} <-> {end_name}: {int(edge['Total_Trips'])} total trips "
                              f"({int(edge['Forward_Count'])} to {end_name}, {int(edge['Reverse_Count'])} reverse), "
                              f"{metric_label}: {round(metric_value, 2)} {metric_unit}")
                AntPath(
                    locations=[start, end],
                    color=color,
                    weight=weight,
                    opacity=0.7,
                    popup=folium.Popup(popup_text, max_width=300),
                    dash_array=[10, 20],
                    delay=1000,
                    pulse_color='white'
                ).add_to(m)

        # Add nodes at centroids (above edges)
        max_trips = top_zones['Total_Trips'].max() if not top_zones.empty else 1
        for _, row in top_zones.iterrows():
            centroid = [row['centroid'].y, row['centroid'].x]
            size = (row['Total_Trips'] / max_trips) * 10 + 2  # Scale size (2-12 range)
            zone_name = row['zone'] if pd.notna(row['zone']) else f"Zone {row['LocationID']}"
            popup_text = f"{zone_name}: {int(row['Total_Trips'])} trips"
            folium.CircleMarker(
                location=centroid,
                radius=size,
                popup=folium.Popup(popup_text, max_width=300),
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.75
            ).add_to(m)

        folium.LayerControl().add_to(m)
        return m

    def save_map(self, map_obj, suffix="speed_routes"):
        """Save the Folium map to an HTML file based on the parquet filename."""
        base_name = os.path.splitext(os.path.basename(self.parquet_file))[0]
        output_filename = os.path.join(self.output_dir, f"{base_name}_{suffix}.html")
        os.makedirs(self.output_dir, exist_ok=True)
        map_obj.save(output_filename)
        print(f"Map saved as: {output_filename}")

    def generate(self, num_routes=10, metric="speed"):
        """Generate and save the routes map, printing the list, with chosen metric."""
        # Load zones first to get valid LocationIDs
        temp_zones = gpd.read_file(self.shapefile_path)
        valid_zone_ids = set(temp_zones['LocationID'])
        trips_pd, location_counts = self.load_and_process_data(valid_zone_ids)
        taxi_zones = self.load_and_transform_zones(location_counts)
        top_routes, metric_used = self.get_top_routes(trips_pd, taxi_zones, num_routes=num_routes, metric=metric)
        routes_map = self.create_routes_map(taxi_zones, top_routes, metric=metric_used)
        suffix = f"{metric_used}_routes"
        self.save_map(routes_map, suffix=suffix)


if __name__ == "__main__":
    filename = "../clean yellow taxis 2024/cleaned_yellow_tripdata_2024-01.parquet"
    visualizer = SpeedRoutesVisualizer(filename)
    # Generate speed-based map (default)
    visualizer.generate(num_routes=100, metric="speed")
    # Generate time-based map
    visualizer.generate(num_routes=100, metric="time")