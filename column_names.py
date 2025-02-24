import pyarrow.parquet as pq
trips = pq.read_table('yellow_tripdata_2024-01.parquet') #
trip = trips.to_pandas()
print(trips.column_names)