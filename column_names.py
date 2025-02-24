import pyarrow.parquet as pq
import pandas as pd
import requests
from io import BytesIO

# Copied URL from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
response = requests.get(url)
response.raise_for_status()  # Checks for HTTP errors

trips = pq.read_table(BytesIO(response.content))
trip = trips.to_pandas() # Unused so far

print(trips.column_names)