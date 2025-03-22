import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import requests

for month in range(1, 13):
    month_name = f"{month:02d}"
    input = f"fhv_tripdata_2024-{month_name}.parquet"
    output = f"cleaned_fhv_tripdata_2024-{month_name}.parquet"
    #trips = pq.read_table('fhv_tripdata_2024-01.parquet')
    trips = pq.read_table(input)
    trips_df = trips.to_pandas()

    #Delete unwanted columns (can add more)
    #trips_df.drop(['VendorID', 'store_and_fwd_flag', 'improvement_surcharge', 'mta_tax', 'extra', 'congestion_surcharge', 'ehail_fee'], axis=1, inplace=True)
    print(trips_df.columns)

    #Original rows pickup_datetime, dropOff_datetime, PUlocationID, DOlocationID, SR_Flag, Affiliated_base_number, dispatching_base_num
    original_rows = len(trips_df)
    print(f"Original rows:{original_rows}")

    #Delete duplicate dates and incorrect pickup month/year values
    trips_df = trips_df.rename(columns={'dropOff_datetime': 'dropoff_datetime'})
    pickup_month = trips_df['pickup_datetime'].dt.month
    pickup_year = trips_df['dropoff_datetime'].dt.year
        
    correct_month = (pickup_month == month) & (pickup_year == 2024)
    trips_df = trips_df[correct_month]
    incorrect_months = (~correct_month).sum()
    print(f"Incorrect month/year rows removed: {incorrect_months}")
        
    duplicate_time = trips_df['pickup_datetime'] != trips_df['dropoff_datetime']
    trips_df = trips_df[duplicate_time]
    time_rows_removed = (~duplicate_time).sum()
    print(f"Duplicate time rows removed:{time_rows_removed}")

    #Function for removing outliers using IQR
    def IQR_cleaning (df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        filter = (df[column] >= lower) & (df[column] <= upper)
        removed = (~filter).sum()
        cleaned = df[filter]
        print(f"{column} rows removed: {removed}")
        return cleaned

   #PUlocationID
    trips_df = trips_df.rename(columns={'PUlocationID': 'PULocationID'})

   #DOlocationID
    trips_df = trips_df.rename(columns={'DOlocationID': 'DOLocationID'})

   #SR_Flag: 1 if shared, null if not

   #Affiliated_base_number: remove if no value
    affiliated_filter = trips_df['Affiliated_base_number'].notnull()
    print(f"Affiliated rows removed: {(~affiliated_filter).sum()}")
    trips_df = trips_df[affiliated_filter]

   #dispatching_base_num: remove if no value
    dispatch_filter = trips_df['dispatching_base_num'].notnull()
    print(f"Dispatch rows removed: {(~dispatch_filter).sum()}")
    trips_df = trips_df[dispatch_filter]

    trips_df.to_parquet(output, index=False)
    print(f"Cleaned Columns:", list(trips_df.columns))
    print(f"Final rows: {len(trips_df)}")
