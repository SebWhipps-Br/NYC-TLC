import os
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import shape, mapping
from pyproj import Transformer
from data_loader import get_from_file


class TaxiHeatmap:
    """A class to create a taxi trip heatmap for NYC based on TLC data."""

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
        """Loads trip data and compute location counts."""
        _, trips_pd = get_from_file(self.parquet_file)
        pickup_counts = self._count_trips(trips_pd, 'PULocationID', 'Pickup_Count')
        dropoff_counts = self._count_trips(trips_pd, 'DOLocationID', 'Dropoff_Count')
        return self._merge_counts(pickup_counts, dropoff_counts)

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
        """Loads taxi zones shapefile, merges with counts, and transforms to EPSG:4326."""
        taxi_zones = gpd.read_file(self.shapefile_path)
        taxi_zones = taxi_zones.merge(location_counts, on='LocationID', how='left')
        taxi_zones['Total_Trips'] = taxi_zones['Total_Trips'].fillna(0)

        # Debug CRS and geometry
        print("Original CRS:", taxi_zones.crs)
        print("Any geometries with Z:", taxi_zones.geometry.has_z.any())

        # Transform geometries to EPSG:4326
        taxi_zones.geometry = taxi_zones.geometry.apply(self._transform_geometry)
        taxi_zones.crs = "epsg:4326"
        print("New CRS:", taxi_zones.crs)

        return taxi_zones

    def _transform_geometry(self, geom):
        """Transforms the geometry from EPSG:2263 to EPSG:4326."""
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
        return shape(geojson)

    def create_heatmap(self, taxi_zones):
        """Creates and return a Folium choropleth map."""
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

    def save_map(self, map_obj):
        """Saves the Folium map to an HTML file based on the parquet filename."""
        base_name = os.path.splitext(os.path.basename(self.parquet_file))[0]
        output_filename = os.path.join(self.output_dir, f"{base_name}.html")
        os.makedirs(self.output_dir, exist_ok=True)  # Create output directory if it doesn’t exist
        map_obj.save(output_filename)
        print(f"Map saved as: {output_filename}")

    def generate(self):
        """Generates and saves the heatmap."""
        location_counts = self.load_and_process_data()
        taxi_zones = self.load_and_transform_zones(location_counts)
        heatmap = self.create_heatmap(taxi_zones)
        self.save_map(heatmap)


if __name__ == "__main__":
    filename = "yellow_tripdata_2024-01.parquet"
    heatmap = TaxiHeatmap(filename)
    heatmap.generate()