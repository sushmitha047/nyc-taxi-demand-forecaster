import pandas as pd

from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline


import lightgbm as lgb

def avg_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one column with the average rodes from
    - 7 days ago
    - 14 days ago
    - 21 days ago
    - 28 days ago
    """
    X['avg_rides_last_4_weeks'] = 0.25*(
        X[f"rides_previous_{7*24}_hour"] + \
        X[f"rides_previous_{2*7*24}_hour"] + \
        X[f"rides_previous_{3*7*24}_hour"] + \
        X[f"rides_previous_{4*7*24}_hour"]
    )
    return X


class TemporalFeatureEngineering(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()

        # generate numeric columns from datetime
        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek

        return X_.drop(columns=["pickup_hour"])
    

def get_pipeline(**hyperparams) -> Pipeline:

    # sklearn transform
    add_feature_avg_rides_last_4_weeks = FunctionTransformer(
        avg_rides_last_4_weeks, validate=False
    )

    # sklearn transform
    add_temporal_features = TemporalFeatureEngineering()

    # sklearn pipeline
    return make_pipeline(
        add_feature_avg_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyperparams)
    )