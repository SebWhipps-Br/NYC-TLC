from data_loader import get_from_url, get_from_file

import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq

from pycanon.anonymity import k_anonymity # https://github.com/IFCA-Advanced-Computing/pycanon
from anjana.anonymity import k_anonymity as k_anonymiser # https://github.com/IFCA-Advanced-Computing/anjana
from anjana.anonymity.utils import generate_intervals


def time_hierarchy(data, title):
    """ construct the hierarchy dictionary for time based quasi identifiers """
    hierarchy = {}
    hierarchy[0] = np.datetime_as_string(data[title])
    hierarchy[1] = np.datetime_as_string(data[title], unit='m')
    hierarchy[2] = np.datetime_as_string(data[title], unit='h')

    return hierarchy


def anonymiser(k_desired, suppression, url=None, file=None):
    """


    Parameters
    ----------
    k_desired : int
        Desired k-anonymity value to target
    suppression : int (0-100)
        Maximum % of rows to remove to achieve desired k
    url : string, optional
        URL of parquet file to use. Depreciated with clean data available. The default is None.
    file : string, optional
        Absolute file path of parquet file to use. Should be cleaned data. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if url:
        _, data = get_from_url(url)
    elif file:
        _, data = get_from_file(file)

    quasi_identifiers = ['tpep_pickup_datetime', 'tpep_dropoff_datetime',
                         'PULocationID', 'DOLocationID']

    # truncate data for testing
    # data = data.iloc[:100000, :]

    print('Data cleaned successfully')
    print('Columns included:', ', '.join(data.columns.tolist()))
    print('Quasi identifiers:', ', '.join(quasi_identifiers))

    # generate the hierarchy dictionary based on the quasi identifier columns
    hierarchy = {}
    # for qi in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
    #     hierarchy[qi] = time_hierarchy(data, qi)
    
    # in testing hierarchies always went to the highest level so manually set them before anonymising

    # k_anonymiser doesn't like the timestamp data type so convert it to a string
    pickup_data = np.datetime_as_string(np.array(data['tpep_pickup_datetime'], dtype='datetime64[h]'))
    dropoff_data = np.datetime_as_string(np.array(data['tpep_dropoff_datetime'], dtype='datetime64[h]'))
    data['tpep_pickup_datetime'] = pickup_data
    data['tpep_dropoff_datetime'] = dropoff_data

    print(f'Hierarchy constructed. k={k_desired} anonymisation beginning')

    # calculates new k-anonymous dataset and corresponding k for both
    k_original = k_anonymity(data, quasi_identifiers)

    try:
        data_anonymised = k_anonymiser(data, [], quasi_identifiers, k_desired, suppression, hierarchy).drop(['index'], axis=1)
    except AttributeError:
        print(f'Anonymisation could not be carried out for k={k_desired}. Lower k, increase suppression, or include larger sized groups')
        return pd.DataFrame()

    # calculate k for new anonymised dataset
    k_anonymised = k_anonymity(data_anonymised, quasi_identifiers)
    
    # turn datetime columns back into the datetime64 format
    data_anonymised['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'], yearfirst=True)
    data_anonymised['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'], yearfirst=True)

    print('Data anonymised')
    print(f'Original k={k_original}, anonymised k={k_anonymised}')

    print(f'Original entries={data.shape[0]}, anonymised entries={data_anonymised.shape[0]}')

    return data_anonymised


def main():
    k = 2
    suppression = 99
    load_path = os.path.join(os.path.dirname(__file__), 'clean yellow taxis 2024')
    save_path = os.path.join(os.path.dirname(__file__), 'clean anon yt')    

    # create list of all files in data directory
    directory = os.fsencode(load_path)
        
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".parquet"):
            parquet_path = os.path.join(load_path, filename)
            anon_data = anonymiser(k, suppression, file=parquet_path)
            anon_data.to_parquet(os.path.join(save_path, filename), index=False)
            continue
        else:
            continue




if __name__ == '__main__':
    main()





