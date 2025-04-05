from typing import Optional
from datetime import timedelta

import plotly.express as px
import pandas as pd

def plot_one_sample(
        features: pd.DataFrame,
        targets: pd.Series,
        example_id: int, # row number
        predictions: Optional[pd.Series] = None,
        display_title: Optional[bool] = True,
        location_id_to_zone: Optional[dict] = None, # for mapping location_id to location name
):
    """"""
    features_ = features.iloc[example_id]

    if targets is not None:
        targets_ = targets.iloc[example_id]
    else:
        targets_ = None

    ts_columns = [c for c in features.columns if c.startswith('rides_previous_')]
    ts_values = [features_[c] for c in ts_columns] + [targets_]
    ts_dates = pd.date_range(
        features_['pickup_hour'] - timedelta(hours=len(ts_columns)),
        features_['pickup_hour'],
        freq='h'
    )


    pick_hour_formatted = features_['pickup_hour'].strftime('%Y-%m-%d %H:%M:%S')

    pickup_location_id = features_['pickup_location_id']
    zone_name = location_id_to_zone.get(pickup_location_id, str(pickup_location_id)) if location_id_to_zone else str(pickup_location_id)

    # line plot with past values
    title = f'Pick up hour={pick_hour_formatted}, location =[{pickup_location_id}] {zone_name}' if display_title else None
    fig = px.line(
        x=ts_dates, y=ts_values,
        template='plotly_dark',
        markers=True, title=title
    )

    # Update axis labels
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Rides'
    )

    if targets is not None:
        # green dot for the value we want to predict
        fig.add_scatter(x=ts_dates[-1:], y=[targets_],
                        line_color='green',
                        mode='markers', marker_size=10, name='Actual Rides')
    
    if predictions is not None:
        # big red X for the prediction value, if passed
        prediction_ = predictions.iloc[example_id]
        fig.add_scatter(x=ts_dates[-1:], y=[prediction_],
                        line_color='red',
                        mode='markers',
                        marker_symbol='x',
                        marker_size=15,
                        name='Predicted Rides')
    
    return fig