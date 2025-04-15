import os
from dotenv import load_dotenv

from src.feature_store_api import FeatureGroupConfig, FeatureViewConfig
from src.paths import PARENT_DIR

# load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR / ".env")

HOPSWORKS_PROJECT_NAME = "nyc_taxi_demand_forecast"
try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception("Create an .env file on the project root with the HOPSWORKS_API_KEY")


FEATURE_GROUP_NAME = "time_series_hourly_feature_group"
FEATURE_GROUP_VERSION = 1
FEATURE_GROUP_METADATA = FeatureGroupConfig(
    name='time_series_hourly_feature_group',
    version=1,
    description='Feature group with hourly time-series data of historical taxi rides',
    primary_key=['pickup_location_id', 'pickup_ts'],
    event_time='pickup_ts',
    online_enabled=True,
)

FEATURE_VIEW_NAME = "time_series_hourly_feature_view"
FEATURE_VIEW_VERSION = 1
FEATURE_VIEW_METADATA = FeatureViewConfig(
    name='time_series_hourly_feature_view',
    version=1,
    feature_group=FEATURE_GROUP_METADATA,
)

N_FEATURES = 24 * 28

MODEL_NAME = "taxi_demand_forecaster_next_hour"
MODEL_VERSION = 1