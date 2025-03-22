import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import requests

for month in range(1, 13):
    month_name = f"{month:02d}"
    input = f"green_tripdata_2024-{month_name}.parquet"
    output = f"cleaned_green_tripdata_2024-{month_name}.parquet"
    #trips = pq.read_table('green_tripdata_2024-01.parquet')
    trips = pq.read_table(input)
    trips_df = trips.to_pandas()

    #Delete unwanted columns (can add more)
    trips_df.drop(['VendorID', 'store_and_fwd_flag', 'improvement_surcharge', 'mta_tax', 'extra', 'congestion_surcharge', 'ehail_fee', 'tolls_amount'], axis=1, inplace=True)
    print(trips_df.columns)

    #Original rows
    original_rows = len(trips_df)
    print(f"Original rows:{original_rows}")

    #Delete duplicate dates and incorrect pickup month/year values
    trips_df = trips_df.rename(columns={'lpep_pickup_datetime': 'pickup_datetime'})
    trips_df = trips_df.rename(columns={'lpep_dropoff_datetime': 'dropoff_datetime'})
    pickup_month = trips_df['pickup_datetime'].dt.month
    pickup_year = trips_df['pickup_datetime'].dt.year
        
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


    #airport trips exempt from iqr filtering
    def airports_exempt_cleaning(df, column):
        airport_mask = ((trips_df['PULocationID'] == 132) | (trips_df['DOLocationID'] == 132) | (trips_df['PULocationID'] == 138) | (trips_df['DOLocationID'] == 138))
        normal_trips = trips_df[~airport_mask]
        airport_trips = trips_df[airport_mask]
        cleaned_normal_trips = IQR_cleaning(normal_trips, column)
        return pd.concat([cleaned_normal_trips, airport_trips])

    #trip_distance cleaning: Absolute, >0.1, IQR
    trips_df['trip_distance'] = np.abs(trips_df['trip_distance'])
    trip_filter = (trips_df['trip_distance'] >= 0.1)
    trips_df = airports_exempt_cleaning(trips_df, 'trip_distance')

    #fare_amount cleaning: Remove values < $3.00, then Iqr
    trips_df['fare_amount'] = np.abs(trips_df['fare_amount'])
    fare_filtered = (trips_df['fare_amount'] >= 3.00)
    trips_df = airports_exempt_cleaning(trips_df, 'fare_amount')

    #tip_amount cleaning: Absolute, IQR
    trips_df['tip_amount'] = np.abs(trips_df['tip_amount'])
    trips_df = airports_exempt_cleaning(trips_df, 'tip_amount')
    
    #total abs IQR
    trips_df['total_amount'] = np.abs(trips_df['total_amount'])
    trips_df = airports_exempt_cleaning(trips_df,'total_amount')

    #Not related to location/time/cost but may be interesting 
    #passenger_count 0-6
    passenger_filter = (trips_df['passenger_count'] >= 1) & (trips_df['passenger_count'] <= 6)
    trips_df = trips_df[passenger_filter]
    print(f"passenger_filter rows removed:{(~passenger_filter).sum()}")

    #RatecodeID: Keep 1-6 2-JFK
    ratecode_filter = (trips_df['RatecodeID'] >= 1) & (trips_df['RatecodeID'] <= 6)
    trips_df = trips_df[ratecode_filter]
    print(f"RatecodeID rows removed:{(~ratecode_filter).sum()}")

    #payment_type: Keep 1-6 no need to clean

    #trip_type: Only accept 1 or 2
    trip_filter = (trips_df['trip_type'] == 1) | (trips_df['trip_type'] == 2) 
    print(f"Trip rows removed: {(~trip_filter).sum()}")
    trips_df = trips_df[trip_filter]

    trips_df.to_parquet(output, index=False)
    print(f"Cleaned Columns:", list(trips_df.columns))
    print(f"Final rows: {len(trips_df)}")
