import os
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import shape, mapping
from pyproj import Transformer
from data_loader import get_from_file


class TaxiHeatmap:
    """A class to create taxi trip visualizations (heatmap or graph) for NYC based on TLC data."""

    def __init__(self, parquet_file, shapefile_path="taxi_zones/taxi_zones.shp", output_dir="heatmaps"):
        """
        Initialize the TaxiHeatmap with file paths and settings.

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
        return self._merge_counts(pickup_counts, dropoff_counts), trips_pd

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
        """Load taxi zones shapefile, merge with counts, compute centroids, and transform to EPSG:4326."""
        taxi_zones = gpd.read_file(self.shapefile_path)
        taxi_zones = taxi_zones.merge(location_counts, on='LocationID', how='left')
        taxi_zones['Total_Trips'] = taxi_zones['Total_Trips'].fillna(0)

        # Debug CRS and geometry
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

    def create_heatmap(self, taxi_zones):
        """Create and return a Folium choropleth map."""
        m = folium.Map(location=self.nyc_center, zoom_start=11)
        folium.Choropleth(
            geo_data=taxi_zones,
            name='choropleth',
            data=taxi_zones,
            columns=['LocationID', 'Total_Trips'],
            key_on='feature.properties.LocationID',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Total Taxi Trips'
        ).add_to(m)
        return m

    def create_graph_map(self, taxi_zones, trips_pd):
        """Create a Folium map with nodes (zones) and edges (trips between zones), nodes above edges."""
        m = folium.Map(location=self.nyc_center, zoom_start=11, tiles="cartodbdark_matter")

        # Adds taxi zone boundaries as a light background
        folium.GeoJson(
            taxi_zones.drop(columns=['centroid']),
            style_function=lambda x: {'fillOpacity': 0.1, 'weight': 1, 'color': 'gray'}
        ).add_to(m)

        # Creates a dictionary mapping LocationID to zone names
        zone_names = taxi_zones.set_index('LocationID')['zone'].to_dict()

        # Computes edges (trips between zones) and add them first (underneath nodes)
        edges = trips_pd.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='Trip_Count')
        max_edge_weight = edges['Trip_Count'].max()
        zone_centroids = taxi_zones.set_index('LocationID')['centroid'].to_dict()

        for _, edge in edges.iterrows():
            start_id = edge['PULocationID']
            end_id = edge['DOLocationID']
            if start_id in zone_centroids and end_id in zone_centroids:
                start = [zone_centroids[start_id].y, zone_centroids[start_id].x]
                end = [zone_centroids[end_id].y, zone_centroids[end_id].x]
                weight = (edge['Trip_Count'] / max_edge_weight) * 10.5 + 0.5  # Scale weight (0.5-10.5 range)
                start_name = zone_names.get(start_id, f"Zone {start_id}")
                end_name = zone_names.get(end_id, f"Zone {end_id}")
                popup_text = f"From {start_name} to {end_name}: {int(edge['Trip_Count'])} trips"
                folium.PolyLine(
                    locations=[start, end],
                    weight=weight,
                    color='red',
                    opacity=0.05,
                    popup=folium.Popup(popup_text, max_width=300)  # Set popup width
                ).add_to(m)

        # Adds nodes at centroids after edges (so they appear above)
        max_trips = taxi_zones['Total_Trips'].max()
        for _, row in taxi_zones.iterrows():
            centroid = [row['centroid'].y, row['centroid'].x]  # Latitude, Longitude
            size = (row['Total_Trips'] / max_trips) * 10.5 + 0.5  # Scale size (0.5-10.5 range)
            zone_name = row['zone'] if pd.notna(row['zone']) else f"Zone {row['LocationID']}"
            popup_text = f"{zone_name}: {int(row['Total_Trips'])} trips"
            folium.CircleMarker(
                location=centroid,
                radius=size,
                popup=folium.Popup(popup_text, max_width=300),  # Set popup width
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.75
            ).add_to(m)

        folium.LayerControl().add_to(m)

        return m

    def save_map(self, map_obj, suffix="heatmap"):
        """Save the Folium map to an HTML file based on the parquet filename."""
        base_name = os.path.splitext(os.path.basename(self.parquet_file))[0]
        output_filename = os.path.join(self.output_dir, f"{base_name}_{suffix}.html")
        os.makedirs(self.output_dir, exist_ok=True)
        map_obj.save(output_filename)
        print(f"Map saved as: {output_filename}")

    def generate_heatmap(self):
        """Generate and save the heatmap."""
        location_counts, _ = self.load_and_process_data()
        taxi_zones = self.load_and_transform_zones(location_counts)
        heatmap = self.create_heatmap(taxi_zones)
        self.save_map(heatmap, suffix="heatmap")

    def generate_graph(self):
        """Generate and save the graph map."""
        location_counts, trips_pd = self.load_and_process_data()
        taxi_zones = self.load_and_transform_zones(location_counts)
        graph_map = self.create_graph_map(taxi_zones, trips_pd)
        self.save_map(graph_map, suffix="graph")


if __name__ == "__main__":
    filename = "yellow_tripdata_2024-01.parquet"
    heatmap = TaxiHeatmap(filename)
    # heatmap.generate_heatmap()  # Uncomment to generate heatmap
    heatmap.generate_graph()    # Generate graph map