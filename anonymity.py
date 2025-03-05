from data_loader import get_from_url
import pandas as pd

from pycanon.anonymity import k_anonymity # https://github.com/IFCA-Advanced-Computing/pycanon
from anjana.anonymity import k_anonymity as k_anonymiser # https://github.com/IFCA-Advanced-Computing/anjana
from anjana.anonymity.utils import generate_intervals


def data_cleaner():
    # gets January 2024 yellow taxi data from the source and removes most columns
    # depreciate when cleaned data is available

    # importing data
    url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet'
    _, data = get_from_url(url)

    # removing columns that we probably aren't interested in
    to_drop = ['VendorID', 'passenger_count', 'RatecodeID', 'store_and_fwd_flag',
               'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
               'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge',
               'Airport_fee', 'tpep_dropoff_datetime', 'trip_distance']

    data = data.drop(to_drop, axis=1).iloc[:100000, :]

    # converting the numpy datetime64 format to pandas datetime64
    # these lines take a short while to run
    data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'].to_list())
    # data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'].to_list())

    # list columns included in final set
    columns = data.columns.to_list()

    # save file so function only has to be run once
    data.to_csv('jan_small.csv', index=False)

    return columns, data


# replace with importing cleaned data when available
# qi are quasi-identifiers, corresponding to columns
quasi_identifiers, data = data_cleaner()


print('Data cleaned')
print('Columns:', *quasi_identifiers)

# creating hierarquies for k-anonymity computing
# truncating time stamps to group trip times together
pu_time_second = list(data['tpep_pickup_datetime'])
pu_time_minute = list(map(lambda x: x.replace(second=0).strftime('%d/%m/%y %H:%M:%S'), pu_time_second))
pu_time_hour = list(map(lambda x: x.replace(second=0, minute=0).strftime('%d/%m/%y %H:%M:%S'), pu_time_second))
pu_time_day = list(map(lambda x: x.replace(second=0, minute=0, hour=0).strftime('%d/%m/%y %H:%M:%S'), pu_time_second))
pu_time_second = list(data['tpep_pickup_datetime'].map(lambda x: x.strftime('%d/%m/%y %H:%M:%S')).astype('string'))



# doing the same for dropoff times
# do_time_second = list(data['tpep_dropoff_datetime'])
# do_time_minute = list(map(lambda x: x.replace(second=0).strftime('%d/%m/%y %H:%M:%S'), do_time_second))
# do_time_hour = list(map(lambda x: x.replace(second=0, minute=0).strftime('%d/%m/%y %H:%M:%S'), do_time_second))
# do_time_day = list(map(lambda x: x.replace(second=0, minute=0, hour=0).strftime('%d/%m/%y %H:%M:%S'), do_time_second))
# do_time_second = list(data['tpep_dropoff_datetime'].map(lambda x: x.strftime('%d/%m/%y %H:%M:%S')).astype('string'))


# grouping trip lengths together
# trip_distance = data['trip_distance']
# trip_distance_quarter = generate_intervals(trip_distance, 0, 350000, 0.25)
# trip_distance_half = generate_intervals(trip_distance, 0, 350000, 0.5)
# trip_distance_one = generate_intervals(trip_distance, 0, 350000, 1)

# hierarchies are what the k_anonymiser uses to group data by to eliminate unique results
hierarchies = {
    'tpep_pickup_datetime': {0: pu_time_second,
                             1: pu_time_minute,
                             2: pu_time_hour,
                             3: pu_time_day}
    }

if set(data['tpep_pickup_datetime'].values).issubset(set(pu_time_second)):
    print('success')


print('Data anonymising beginning')

k_desired = 2 # minimum k to be achieved by anonymiser method
suppression = 50 # maximum % of records to suppress to achieve desired k (0-100)

# k_anonymiser doesn't like the timestamp data type so convert it to a string
pickup_data = list(map(lambda x: x.strftime('%d/%m/%y %H:%M:%S'), data['tpep_pickup_datetime']))

data['tpep_pickup_datetime'] = pickup_data

# data['tpep_dropoff_datetime'] = data['tpep_dropoff_datetime'].map(lambda x: x.strftime('%d/%m/%y %H:%M:%S')).astype('string')

print(data.info())

k_original = k_anonymity(data, quasi_identifiers)
data_anonymised = k_anonymiser(data, [], quasi_identifiers, k_desired, suppression, hierarchies)
k_anonymised = k_anonymity(data_anonymised, quasi_identifiers)

print(f'Original dataset k: {k_original}')
print(f'New dataset k: {k_anonymised}')

data_anonymised.to_csv('small_anon.csv', index=False)

print(data.info())
print(data_anonymised.info())



