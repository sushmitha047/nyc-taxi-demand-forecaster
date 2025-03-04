import zipfile
from datetime import datetime, timezone, UTC

import requests
import numpy as np
import pandas as pd

# plotting libraries
import streamlit as st
import geopandas as gpd
import pydeck as pdk

from src.inference import (
    load_batch_of_features_from_store,
    load_model_from_registry,
    get_model_predictions
)
from src.paths import DATA_DIR
from src.plot import plot_one_sample


st.set_page_config(layout="wide")

# title
current_date = pd.to_datetime(datetime.now(timezone.utc)).floor('h')
st.title(f"Taxi demand prediction :taxi:")
st.header(f'{current_date} UTC')

progress_bar = st.sidebar.header("o Working Progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 7


def load_shape_data_file():
    # download file

    URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    response = requests.get(URL)
    path = DATA_DIR / f'taxi_zones.zip'
    if response.status_code == 200:
        open(path, "wb").write(response.content)
    else:
        raise Exception(f"{URL} is not available")
    
    # unzip file
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR / 'taxi_zones')

    # load and return shape file
    return gpd.read_file(DATA_DIR / "taxi_zones/taxi_zones.shp").to_crs("epsg:4326")


with st.spinner(text="Downloading shape file to plot taxi zones"):
    geo_df = load_shape_data_file()
    st.sidebar.write(":white_check_mark: Shape file was downloaded")
    progress_bar.progress(1/N_STEPS)


# connecting to feature store to access data
with st.spinner(text="Fetching batch of inference data"):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write(":white_check_mark: Inference features fetched from the store")
    progress_bar.progress(2/N_STEPS)
    print(f"{features}")

# loading model from model registry
with st.spinner(text="Loading ML model from the registry"):
    model = load_model_from_registry()
    st.sidebar.write(":white_check_mark: ML model was loaded from the registry")
    progress_bar.progress(3/N_STEPS)

# generate predictions
with st.spinner(text="Computing Model Predictions"):
    results = get_model_predictions(model, features)
    st.sidebar.write(":white_check_mark: Model predictions arrived")
    progress_bar.progress(4/N_STEPS)

# preparing data to plot
with st.spinner(text="Preparing data to plot"):

    def psuedocolor(val, minval, maxval, startcolor, stopcolor):
        f = float(val - minval) / (maxval - minval)
        return tuple(f*(b-a)+a for (a,b) in zip(startcolor, stopcolor))
    
    df = pd.merge(geo_df, results, right_on="pickup_location_id", left_on="LocationID")

    BLACK, GREEN = (0, 0, 0), (0, 255, 0)
    df['color_scaling'] = df['predicted_demand']
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df["color_scaling"].apply(lambda x: psuedocolor(x, min_pred, max_pred))
    progress_bar.progress(5/N_STEPS)

# NYC map
with st.spinner(text="Generating NYC Map"):

    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=40.7831,
        longitude=73.9712,
        zoom=11,
        max_zoom=16,
        pitch=45,
        bearing=0
    )

    geojson = pdk.Layer(
        "GeoJsonLayer",
        df,
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_elevation=10,
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        auto_highlight=True,
        pickable=True,
    )

    tooltip = {"html": "<b>Zone:</b> [{LocationID}]{zone} <br /> <b> Predicted rides:</b> {predicted_demand}"}

    r = pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )

    st.pydeck_chart(r)
    progress_bar.progress(6/N_STEPS)


# plotting timeseries plot
with st.spinner(text="Plotting time-series data"):
    row_indices = np.argsort(results['predicted_demand'].values)[::-1]
    n_to_plot = 10

    # plot each time-series with the prediction
    for row_id in row_indices[:n_to_plot]:
        fig = plot_one_sample(
            features=features,
            targets=results['predicted_demand'],
            example_id=row_id,
            predictions=pd.Series(results['predicted_demand'])
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=100)

    progress_bar.progress(7/N_STEPS)