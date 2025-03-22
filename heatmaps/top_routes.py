import os
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import AntPath
from shapely.geometry import shape, mapping
from pyproj import Transformer
from data_loader import get_from_file


class TopRoutesVisualizer:
    """A class to visualize and list the top N most popular taxi routes in NYC with combined bidirectional edges."""

    def __init__(self, parquet_file, shapefile_path="taxi_zones/taxi_zones.shp", output_dir="output"):
        """
        Initialize the TopRoutesVisualizer with file paths and settings.

        Args:
            parquet_file (str): Path to the TLC parquet file (e.g., 'yellow_tripdata_2024-01.parquet').
            shapefile_path (str): Path to the taxi zones shapefile (default: 'taxi_zones/taxi_zones.shp').
            output_dir (str): Directory to save the output HTML file (default: 'heatmaps').
        """
        self.parquet_file = parquet_file
        self.shapefile_path = shapefile_path
        self.output_dir = output_dir
        self.transformer = Transformer.from_crs("epsg:2263", "epsg:4326", always_xy=True)
        self.nyc_center = [40.7128, -74.0060]  # NYC coordinates (lat, lon)

    def load_and_process_data(self):
        """Load trip data and compute location counts."""
        _, trips_pd = get_from_file(self.parquet_file)
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

        # Compute centroids in the original projected CRS (EPSG:2263)
        taxi_zones['centroid'] = taxi_zones.geometry.centroid

        # Transform geometries and centroids to EPSG:4326
        taxi_zones.geometry = taxi_zones.geometry.apply(self._transform_geometry)
        taxi_zones['centroid'] = taxi_zones['centroid'].apply(self._transform_geometry)
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
        elif geom.geom_type == 'Point':  # For centroids
            new_coords = self.transformer.transform(geom.x, geom.y)
            return shape({'type': 'Point', 'coordinates': new_coords})
        return shape(geojson)

    def get_top_routes(self, trips_pd, taxi_zones, num_routes=10):
        """Identify and print the top N most popular routes, combining bidirectional trips."""
        # Compute edges (trips between zones)
        edges = trips_pd.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='Trip_Count')

        # Create a unique key for each pair (sorted to combine A-B and B-A)
        edges['Pair_Key'] = edges.apply(lambda row: tuple(sorted([row['PULocationID'], row['DOLocationID']])), axis=1)

        # Group by pair to sum total trips and determine dominant direction
        route_counts = edges.groupby('Pair_Key').agg({
            'Trip_Count': 'sum',
            'PULocationID': 'first',  # Temporary, will adjust direction later
            'DOLocationID': 'first'
        }).reset_index()

        # For each pair, calculate directional counts
        directional_counts = edges.groupby(['PULocationID', 'DOLocationID'])['Trip_Count'].sum().to_dict()

        # Determine dominant direction and net counts
        route_counts['Forward_Count'] = route_counts.apply(
            lambda row: directional_counts.get((row['PULocationID'], row['DOLocationID']), 0), axis=1)
        route_counts['Reverse_Count'] = route_counts.apply(
            lambda row: directional_counts.get((row['DOLocationID'], row['PULocationID']), 0), axis=1)

        # Set direction based on higher count
        route_counts['Dominant_PU'] = route_counts.apply(
            lambda row: row['PULocationID'] if row['Forward_Count'] >= row['Reverse_Count'] else row['DOLocationID'],
            axis=1)
        route_counts['Dominant_DO'] = route_counts.apply(
            lambda row: row['DOLocationID'] if row['Forward_Count'] >= row['Reverse_Count'] else row['PULocationID'],
            axis=1)
        route_counts['Dominant_Count'] = route_counts[['Forward_Count', 'Reverse_Count']].max(axis=1)
        route_counts['Reverse_Count'] = route_counts[['Forward_Count', 'Reverse_Count']].min(
            axis=1)  # Update reverse to lesser count

        # Get top N routes by total trips (Forward + Reverse)
        top_routes = route_counts.nlargest(num_routes, 'Trip_Count')
        top_routes = top_routes[['Dominant_PU', 'Dominant_DO', 'Trip_Count', 'Dominant_Count', 'Reverse_Count']]
        top_routes.columns = ['PULocationID', 'DOLocationID', 'Total_Trips', 'Forward_Count', 'Reverse_Count']

        # Map LocationID to zone names
        zone_names = taxi_zones.set_index('LocationID')['zone'].to_dict()

        # Print the top N routes
        print(f"\nTop {num_routes} Most Popular Routes (Bidirectional Combined):")
        print("---------------------------------------------")
        for i, row in top_routes.iterrows():
            start_id = row['PULocationID']
            end_id = row['DOLocationID']
            start_name = zone_names.get(start_id, f"Zone {start_id}")
            end_name = zone_names.get(end_id, f"Zone {end_id}")
            total_trips = int(row['Total_Trips'])
            forward_trips = int(row['Forward_Count'])
            reverse_trips = int(row['Reverse_Count'])
            print(f"{i + 1}. {start_name} <-> {end_name}: {total_trips} total trips "
                  f"({forward_trips} from {start_name} to {end_name}, {reverse_trips} reverse)")

        return top_routes

    def create_top_routes_map(self, taxi_zones, top_routes):
        """Create a Folium map showing the top N routes as edges with arrows and nodes."""
        m = folium.Map(location=self.nyc_center, zoom_start=11, tiles="cartodbdark_matter")

        # Add taxi zone boundaries as a light background
        folium.GeoJson(
            taxi_zones.drop(columns=['centroid']),
            style_function=lambda x: {'fillOpacity': 0.1, 'weight': 1, 'color': 'gray'}
        ).add_to(m)

        # Get unique zones involved in top routes
        top_zone_ids = set(top_routes['PULocationID']).union(top_routes['DOLocationID'])
        top_zones = taxi_zones[taxi_zones['LocationID'].isin(top_zone_ids)]

        # Map LocationID to centroids and zone names
        zone_centroids = top_zones.set_index('LocationID')['centroid'].to_dict()
        zone_names = top_zones.set_index('LocationID')['zone'].to_dict()

        # Add edges with arrows using AntPath
        max_trip_count = top_routes['Total_Trips'].max()
        for _, edge in top_routes.iterrows():
            start_id = edge['PULocationID']
            end_id = edge['DOLocationID']
            if start_id in zone_centroids and end_id in zone_centroids:
                start = [zone_centroids[start_id].y, zone_centroids[start_id].x]
                end = [zone_centroids[end_id].y, zone_centroids[end_id].x]
                weight = (edge['Total_Trips'] / max_trip_count) * 10 + 2  # Scale weight (2-12 range)
                start_name = zone_names.get(start_id, f"Zone {start_id}")
                end_name = zone_names.get(end_id, f"Zone {end_id}")
                popup_text = (f"{start_name} <-> {end_name}: {int(edge['Total_Trips'])} total trips "
                              f"({int(edge['Forward_Count'])} to {end_name}, {int(edge['Reverse_Count'])} reverse)")
                AntPath(
                    locations=[start, end],
                    color='red',
                    weight=weight,
                    opacity=0.7,
                    popup=folium.Popup(popup_text, max_width=300),
                    dash_array=[10, 20],  # Dashed line for animation
                    delay=1000,  # Animation speed (ms)
                    pulse_color='yellow'  # Arrow highlight color
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

    def save_map(self, map_obj, suffix="top_routes"):
        """Save the Folium map to an HTML file based on the parquet filename."""
        base_name = os.path.splitext(os.path.basename(self.parquet_file))[0]
        output_filename = os.path.join(self.output_dir, f"{base_name}_{suffix}.html")
        os.makedirs(self.output_dir, exist_ok=True)
        map_obj.save(output_filename)
        print(f"Map saved as: {output_filename}")

    def generate(self, num_routes=10):
        """Generate and save the top routes map with nodes, printing the list."""
        trips_pd, location_counts = self.load_and_process_data()
        taxi_zones = self.load_and_transform_zones(location_counts)
        top_routes = self.get_top_routes(trips_pd, taxi_zones, num_routes=num_routes)
        routes_map = self.create_top_routes_map(taxi_zones, top_routes)
        self.save_map(routes_map, suffix="top_routes")


if __name__ == "__main__":
    filename = "../clean yellow taxis 2024/cleaned_yellow_tripdata_2024-01.parquet"
    visualizer = TopRoutesVisualizer(filename)
    visualizer.generate(num_routes=50)  # Default to 10, change as needed