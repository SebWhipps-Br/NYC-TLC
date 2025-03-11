import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import requests

for month in range(1, 13):
    month_name = f"{month:02d}"
    input = f"fhvhv_tripdata_2024-{month_name}.parquet"
    output = f"fhvhv_green_tripdata_2024-{month_name}.parquet"
    #trips = pq.read_table('green_tripdata_2024-01.parquet')
    trips = pq.read_table(input)
    trips_df = trips.to_pandas()

    #Columns (* = removed): hvfhs_license_num, dispatching_base_num, originating_base_num, request_datetime, on_scene_datetime, pickup_datetimex, dropoff_datetimex, pulocationidx
    #dolocationidx, trip_milesx, trip_time, base_passenger_farex, tollsx, bcf*, sales_tax*, congestion_surcharge*, airport_feex, tipsx, driver_payx, shared_request_flag*
    #shared_match_flag*, access_a_ride_flag*, wav_request_flag*, wav_match_flag*

    #Delete unwanted columns (can add more)
    trips_df.drop(['bcf', 'sales_tax', 'wav_request_flag', 'wav_match_flag', 'shared_request_flag', 'shared_match_flag', 'access_a_ride_flag', 'congestion_surcharge', 'tolls' ], axis=1, inplace=True)
    print(trips_df.columns)

    #Original rows
    original_rows = len(trips_df)
    print(f"Original rows:{original_rows}")

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

    #hvfhs_license_num: remove if no value
    hvfhs_filter = trips_df['hvfhs_license_num'].notnull()
    print(f"Hvfhs rows removed: {(~hvfhs_filter).sum()}")
    trips_df = trips_df[hvfhs_filter]

    #Affiliated_base_number: remove if no value
    trips_df = trips_df.rename(columns={'originating_base_num': 'Affiliated_base_number'})
    affiliated_filter = trips_df['Affiliated_base_number'].notnull()
    print(f"Affiliated rows removed: {(~affiliated_filter).sum()}")
    trips_df = trips_df[affiliated_filter]

    #dispatching_base_num: remove if no value
    dispatch_filter = trips_df['dispatching_base_num'].notnull()
    print(f"Dispatch rows removed: {(~dispatch_filter).sum()}")
    trips_df = trips_df[dispatch_filter]

    #Delete duplicate dates and incorrect pickup month/year values
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

    #PUlocationID
    trips_df = trips_df.rename(columns={'PUlocationID': 'PULocationID'})

    #DOlocationID
    trips_df = trips_df.rename(columns={'DOlocationID': 'DOLocationID'})

    #tip_amount cleaning: Absolute, IQR
    trips_df = trips_df.rename(columns={'tips': 'tip_amount'})
    trips_df['tip_amount'] = np.abs(trips_df['tip_amount'])
    trips_df = IQR_cleaning (trips_df, 'tip_amount')

    #Airport_fee: absolute, only take 0, 1.25, 1.75
    trips_df = trips_df.rename(columns={'airport_fee': 'Airport_fee'})
    trips_df['Airport_fee'] = np.abs(trips_df['Airport_fee'])
    airport_filter = (trips_df['Airport_fee'] == 0) | (trips_df['Airport_fee'] == 1.25) | (trips_df['Airport_fee'] == 1.75)
    print(f"Airport rows removed: {(~airport_filter).sum()}")
    trips_df = trips_df[airport_filter]

    #trip_distance cleaning: Absolute, >0.1, IQR
    trips_df = trips_df.rename(columns={'trip_miles': 'trip_distance'})
    trips_df['trip_distance'] = np.abs(trips_df['trip_distance'])
    trip_filter = (trips_df['trip_distance'] >= 0.1)
    trips_df = trips_df[trip_filter]
    trips_df = IQR_cleaning (trips_df, 'trip_distance')

    #fare_amount cleaning: Remove values < $3.00, then Iqr
    trips_df = trips_df.rename(columns={'base_passenger_fare': 'fare_amount'})
    trips_df['fare_amount'] = np.abs(trips_df['fare_amount'])
    fare_filtered = (trips_df['fare_amount'] >= 3.00)
    trips_df = trips_df[fare_filtered]
    trips_df = IQR_cleaning (trips_df, 'fare_amount')

    #tolls_amount: Absolute, IQR
    #trips_df = trips_df.rename(columns={'tolls': 'tolls_amount'})
    #trips_df['tolls_amount'] = np.abs(trips_df['tolls_amount'])
    #trips_df = IQR_cleaning (trips_df, 'tolls_amount')

    #total abs IQR
    trips_df = trips_df.rename(columns={'driver_pay': 'total_amount'})
    trips_df['total_amount'] = np.abs(trips_df['total_amount'])
    trips_df = IQR_cleaning (trips_df, 'total_amount')

    #Not related to location/time/cost but may be interesting 
    #Delete duplicate dates and incorrect pickup month/year values for request_datetime, on_scene_datetime
    request_month = trips_df['request_datetime'].dt.month
    request_year = trips_df['request_datetime'].dt.year
        
    correct_request = (request_month == month) & (request_year == 2024)
    trips_df = trips_df[correct_request]
    incorrect_requests = (~correct_request).sum()
    print(f"Incorrect request month/year rows removed: {incorrect_requests}")

    scene_month = trips_df['on_scene_datetime'].dt.month
    scene_year = trips_df['on_scene_datetime'].dt.year
        
    correct_scene = (scene_month == month) & (scene_year == 2024)
    trips_df = trips_df[correct_scene]
    incorrect_scenes = (~correct_scene).sum()
    print(f"Incorrect scene month/year rows removed: {incorrect_scenes}")

    trips_df.to_parquet(output, index=False)
    print(f"Cleaned Columns:", list(trips_df.columns))
    print(f"Final rows: {len(trips_df)}")