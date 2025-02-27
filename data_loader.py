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