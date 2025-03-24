import os
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import shape, mapping
from pyproj import Transformer
from data_loader import get_year_samples


class TaxiHeatmap:
    """A class to create taxi trip visualizations (heatmap or graph) for NYC based on TLC data."""

    def __init__(self, year=2024, sample_size=100000, directory="../clean yellow taxis 2024",
                 shapefile_path="taxi_zones/taxi_zones.shp", output_dir="output"):
        self.year = year
        self.sample_size = sample_size
        self.directory = directory
        self.shapefile_path = shapefile_path
        self.output_dir = output_dir
        self.transformer = Transformer.from_crs("epsg:2263", "epsg:4326", always_xy=True)
        self.nyc_center = [40.7128, -74.0060]

    def load_and_process_data(self):
        trips_pd = get_year_samples(year=self.year, sample_size=self.sample_size, directory=self.directory)
        print(f"Total sampled rows: {len(trips_pd)}")
        pickup_counts = self._count_trips(trips_pd, 'PULocationID', 'Pickup_Count')
        dropoff_counts = self._count_trips(trips_pd, 'DOLocationID', 'Dropoff_Count')
        return self._merge_counts(pickup_counts, dropoff_counts), trips_pd

    def _count_trips(self, df, column, count_name):
        counts = df[column].value_counts().reset_index()
        counts.columns = ['LocationID', count_name]
        return counts

    def _merge_counts(self, pickup_counts, dropoff_counts):
        location_counts = pd.merge(pickup_counts, dropoff_counts, on='LocationID', how='outer').fillna(0)
        location_counts['Total_Trips'] = location_counts['Pickup_Count'] + location_counts['Dropoff_Count']
        return location_counts

    def load_and_transform_zones(self, location_counts):
        taxi_zones = gpd.read_file(self.shapefile_path)
        taxi_zones = taxi_zones.merge(location_counts, on='LocationID', how='left')
        taxi_zones['Total_Trips'] = taxi_zones['Total_Trips'].fillna(0)
        print("Original CRS:", taxi_zones.crs)
        print("Any geometries with Z:", taxi_zones.geometry.has_z.any())
        taxi_zones['centroid'] = taxi_zones.geometry.centroid
        taxi_zones.geometry = taxi_zones.geometry.apply(self._transform_geometry)
        taxi_zones['centroid'] = taxi_zones['centroid'].apply(self._transform_geometry)
        taxi_zones.crs = "epsg:4326"
        print("New CRS:", taxi_zones.crs)
        return taxi_zones

    def _transform_geometry(self, geom):
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

    def create_heatmap_original(self, taxi_zones, palette='RdYlBu'):
        """Create and return a Folium choropleth map using the original method."""
        m = folium.Map(location=self.nyc_center, zoom_start=11, tiles="cartodbpositron",
                       attr="CartoDB Positron")

        taxi_zones_for_choropleth = taxi_zones.drop(columns=['centroid'])

        folium.Choropleth(
            geo_data=taxi_zones_for_choropleth,
            name='choropleth',
            data=taxi_zones_for_choropleth,
            columns=['LocationID', 'Total_Trips'],
            key_on='feature.properties.LocationID',
            fill_color=palette,
            fill_opacity=0.8,
            line_opacity=0.1,
            legend_name='Total Taxi Trips',
            nan_fill_color='gray',
            nan_fill_opacity=0
        ).add_to(m)
        return m

    def create_heatmap_updated(self, taxi_zones, palette='viridis'):
        """Create and return a Folium choropleth map using the updated method with custom bins."""
        m = folium.Map(location=self.nyc_center, zoom_start=11, tiles="cartodbpositron", attr="CartoDB Positron")

        taxi_zones_for_choropleth = taxi_zones.drop(columns=['centroid'])
        print("Total_Trips stats:", taxi_zones_for_choropleth['Total_Trips'].describe())
        print("Percentiles:",
              taxi_zones_for_choropleth['Total_Trips'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict())

        # Replace 0 with NaN to make them gray
        taxi_zones_for_choropleth['Total_Trips'] = taxi_zones_for_choropleth['Total_Trips'].replace(0, float('nan'))
        non_zero_trips = taxi_zones_for_choropleth['Total_Trips'].dropna()
        print("Non-zero Total_Trips stats:", non_zero_trips.describe())

        # Define custom bins based on data distribution
        if not non_zero_trips.empty:
            bins = [
                non_zero_trips.min(),
                non_zero_trips.quantile(0.05),
                non_zero_trips.quantile(0.20),
                non_zero_trips.quantile(0.50),
                non_zero_trips.quantile(0.75),
                non_zero_trips.quantile(0.95),
                non_zero_trips.max()
            ]
            print("Custom bins:", bins)
        else:
            bins = [0, 1]

        folium.Choropleth(
            geo_data=taxi_zones_for_choropleth,
            name='choropleth',
            data=taxi_zones_for_choropleth,
            columns=['LocationID', 'Total_Trips'],
            key_on='feature.properties.LocationID',
            fill_color=palette,
            fill_opacity=0.8,
            line_opacity=0.1,
            legend_name='Total Taxi Trips',
            nan_fill_color='white',
            nan_fill_opacity=1,
            bins=32
            #threshold_scale=bins
        ).add_to(m)
        return m

    def create_graph_map(self, taxi_zones, trips_pd, threshold_percentage=1.0):
        m = folium.Map(location=self.nyc_center, zoom_start=11, tiles="cartodbpositron",
                       attr="CartoDB Positron")
        folium.GeoJson(
            taxi_zones.drop(columns=['centroid']),
            style_function=lambda x: {'fillOpacity': 0.05, 'weight': 0.5, 'color': '#d3d3d3'}
        ).add_to(m)
        total_trips_all = taxi_zones['Total_Trips'].sum()
        trip_threshold = (threshold_percentage / 100) * total_trips_all
        print(f"Total trips: {total_trips_all}, Threshold ({threshold_percentage}%): {trip_threshold}")
        filtered_zones = taxi_zones[taxi_zones['Total_Trips'] >= trip_threshold]
        valid_zone_ids = set(filtered_zones['LocationID'])
        zone_names = filtered_zones.set_index('LocationID')['zone'].to_dict()
        zone_centroids = filtered_zones.set_index('LocationID')['centroid'].to_dict()
        edges = trips_pd.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='Trip_Count')
        edges = edges[(edges['PULocationID'].isin(valid_zone_ids)) &
                      (edges['DOLocationID'].isin(valid_zone_ids))]
        max_edge_weight = edges['Trip_Count'].max() if not edges.empty else 1
        for _, edge in edges.iterrows():
            start_id = edge['PULocationID']
            end_id = edge['DOLocationID']
            start = [zone_centroids[start_id].y, zone_centroids[start_id].x]
            end = [zone_centroids[end_id].y, zone_centroids[end_id].x]
            weight = (edge['Trip_Count'] / max_edge_weight) * 10.5 + 0.5
            start_name = zone_names.get(start_id, f"Zone {start_id}")
            end_name = zone_names.get(end_id, f"Zone {end_id}")
            popup_text = f"From {start_name} to {end_name}: {int(edge['Trip_Count'])} trips"
            folium.PolyLine(
                locations=[start, end],
                weight=weight,
                color='#FF073A',
                opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
        max_trips = filtered_zones['Total_Trips'].max() if not filtered_zones.empty else 1
        for _, row in filtered_zones.iterrows():
            centroid = [row['centroid'].y, row['centroid'].x]
            size = (row['Total_Trips'] / max_trips) * 10.5 + 0.5
            zone_name = row['zone'] if pd.notna(row['zone']) else f"Zone {row['LocationID']}"
            popup_text = f"{zone_name}: {int(row['Total_Trips'])} trips"
            folium.CircleMarker(
                location=centroid,
                radius=size,
                popup=folium.Popup(popup_text, max_width=300),
                color='#00FFFF',
                fill=True,
                fill_color='#00FFFF',
                fill_opacity=0.9
            ).add_to(m)
        folium.LayerControl().add_to(m)
        return m

    def save_map(self, map_obj, suffix="heatmap"):
        output_filename = os.path.join(self.output_dir, f"yellow_tripdata_{self.year}_{suffix}.html")
        os.makedirs(self.output_dir, exist_ok=True)
        map_obj.save(output_filename)
        print(f"Map saved as: {output_filename}")

    def generate_heatmap(self, palette='RdYlBu', use_updated_method=False):
        """Generate and save the heatmap with a specified color palette and method."""
        location_counts, _ = self.load_and_process_data()
        taxi_zones = self.load_and_transform_zones(location_counts)
        if use_updated_method:
            heatmap = self.create_heatmap_updated(taxi_zones, palette=palette)
        else:
            heatmap = self.create_heatmap_original(taxi_zones, palette=palette)
        self.save_map(heatmap, suffix="heatmap")

    def generate_graph(self, threshold_percentage=1.0):
        location_counts, trips_pd = self.load_and_process_data()
        taxi_zones = self.load_and_transform_zones(location_counts)
        graph_map = self.create_graph_map(taxi_zones, trips_pd, threshold_percentage=threshold_percentage)
        self.save_map(graph_map, suffix="graph")


if __name__ == "__main__":
    heatmap = TaxiHeatmap(year=2024, sample_size=100000, directory="../clean yellow taxis 2024")
    # Use original method
    #heatmap.generate_heatmap(palette='viridis', use_updated_method=False)
    # Use updated method
    heatmap.generate_heatmap(palette='YlOrRd', use_updated_method=True)
    # heatmap.generate_graph(threshold_percentage=1)  # Uncomment to generate graph