from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR


def download_one_file_of_raw_data(year: int, month: int) -> Path:
    """
    Downloads Parquet file with historical taxi rides for the given `year` and `month`
    """

    URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    response = requests.get(URL)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        open(path, "wb").write(response.content)
        return path
    else:
        raise Exception(f"{URL} is not available")
    
def validate_raw_data(
        rides: pd.DataFrame,
        year: int,
        month: int,
    ) -> pd.DataFrame:
    """
    Removes rows with pickup_datetimes outside their valid range
    """

    # keep only rides for this month
    this_month_start = f"{year}-{month:02d}-01"
    next_month_start = f"{year}-{month+1:02d}-01" if month < 12 else f"{year+1}-{month-11:02d}-01"
    rides = rides[rides.pickup_datetime >= this_month_start]
    rides = rides[rides.pickup_datetime < next_month_start]

    return rides


# to load raw data for several months in a given year
def load_raw_data(
        year: int,
        months: Optional[List[int]] = None
    ) -> pd.DataFrame:
    """"""

    rides = pd.DataFrame()

    if months is None:
        # download data only for the months specified by `months`
        months = list(range(1, 13))
    elif isinstance(months, int):
        # download data for the entire year (all months)
        months = [months]

    for month in months:

        local_file = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        if not local_file.exists():
            try:
                # download the file from the NYC website
                print(f"Downloading file {year}-{month:02d}")
                download_one_file_of_raw_data(year, month)
            except:
                print(f"{year}-{month:02d} file is not available")
                continue
        else:
            print(f"File {year}-{month:02d} was already in local storage")

        # load the file into pandas
        rides_one_month = pd.read_parquet(local_file)

        #rename columns
        rides_one_month = rides_one_month[['tpep_pickup_datetime', 'PULocationID']]
        rides_one_month.rename(columns={
            'tpep_pickup_datetime': 'pickup_datetime',
            'PULocationID': 'pickup_location_id',
        }, inplace=True)

        # validate the file
        rides_one_month = validate_raw_data(rides_one_month, year, month)

        # append to existing data
        rides = pd.concat([rides, rides_one_month])
    
    if rides.empty:
        # no data, return empty dataframe
        return pd.DataFrame()
    else:
        # keep only time and origin of the ride
        rides = rides[['pickup_datetime', 'pickup_location_id']]
        return rides


# adding 0s whereever there are no rides at a given hour
def add_missing_slots(rides: pd.DataFrame) -> pd.DataFrame:
    # location_ids = rides['pickup_location_id'].unique()
    location_ids = range(1, rides['pickup_location_id'].max() + 1)
    full_range = pd.date_range(rides['pickup_hour'].min(),
                               rides['pickup_hour'].max(),
                               freq='h')
    output = pd.DataFrame()

    for location_id in tqdm(location_ids):

        # keep only rides for this `location_id`
        rides_i = rides.loc[rides.pickup_location_id == location_id, ['pickup_hour', 'rides']]

        if rides_i.empty:
            # no rides for this location, add a dummy entry with 0 rides
            rides_i = pd.DataFrame.from_dict([{
                'pickup_hour': rides['pickup_hour'].max(),
                'rides': 0
            }])

        # quick way to add missing dates with 0 in a series
        rides_i.set_index('pickup_hour', inplace=True)
        rides_i.index = pd.DatetimeIndex(rides_i.index)
        rides_i = rides_i.reindex(full_range, fill_value=0)

        # add back `location_id` columns
        rides_i['pickup_location_id'] = location_id

        output = pd.concat([output, rides_i])

    output = output.reset_index().rename(columns={
        'index': 'pickup_hour'
    })

    return output

# transform raw data into timeseries data
def transform_raw_data_into_ts_data(
        rides: pd.DataFrame
) -> pd.DataFrame:
    """"""

    # sum rides per location and pickup_hour
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('h')
    agg_rides = rides.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index()
    agg_rides.rename(columns={0: 'rides'}, inplace=True)

    # add rows for (locations, pickup_hours)s with 0 rides
    agg_rides_all_slots = add_missing_slots(agg_rides)

    return agg_rides_all_slots


def get_cutoff_indices(
        data: pd.DataFrame,
        n_features: int,
        step_size: int
        ) -> list:
    
    stop_position = len(data) - 1

    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0
    subseq_mid_idx = n_features
    subseq_last_idx = n_features + 1
    indices = []

    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))

        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size

    return indices

# tranfrom timeseries data into features and targets
def transform_ts_data_into_features_and_target(
        ts_data: pd.DataFrame,
        input_seq_len: int,
        step_size: int
        ) -> pd.DataFrame:
    """
    Slices and transposes data from time-series format into a (features, target)
    format that we can use to train Supervised ML models
    """

    assert set(ts_data.columns) == {'pickup_hour', 'rides', 'pickup_location_id'}

    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()

    for location_id in tqdm(location_ids):

        # keep only ts_data for this `location_id`
        ts_data_one_location = ts_data.loc[
            ts_data.pickup_location_id == location_id,
            ['pickup_hour', 'rides']
        ].sort_values(by=['pickup_hour'])

        # pre-compute cutoff indices to split dataframe rows
        indices = get_cutoff_indices(
            ts_data_one_location,
            input_seq_len,
            step_size
        )

        # slice and transpose data into numpy arrays for features and targets
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        pickup_hours = []
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values[0]
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        # features: numpy array -> pandas dataframe
        features_one_location = pd.DataFrame(
            x,
            columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['pickup_hour'] = pickup_hours
        features_one_location['pickup_location_id'] = location_id

        # target: numpy array -> pandas dataframe
        target_one_location = pd.DataFrame(
            y,
            columns=[f'target_rides_next_hour']
        )

        # concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, target_one_location])

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets['target_rides_next_hour']